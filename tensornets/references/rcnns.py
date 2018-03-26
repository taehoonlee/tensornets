"""Collection of RCNN variants

The reference paper:

 - Faster R-CNN: Towards Real-Time Object Detection
   with Region Proposal Networks, NIPS 2015
 - Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
 - https://arxiv.org/abs/1506.01497

The reference implementation:

1. Caffe and Python utils
 - https://github.com/rbgirshick/py-faster-rcnn
2. RoI pooling in TensorFlow
 - https://github.com/deepsense-ai/roi-pooling
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from ..layers import conv2d
from ..layers import dropout
from ..layers import flatten
from ..layers import fc
from ..layers import max_pool2d
from ..layers import convrelu as conv

from ..ops import *
from ..utils import pad_info
from ..utils import set_args
from ..utils import var_scope

from .rpn_utils import filter_boxes
from .rpn_utils import get_anchors
from .rpn_utils import get_boxes
from .rpn_utils import get_shifts
from .rpn_utils import inv_boxes
from .rpn_utils import nms
from .rpn_utils import roi_pooling


def __args__(is_training):
    return [([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'scope': 'conv'}),
            ([dropout], {'is_training': is_training}),
            ([fc], {'activation_fn': None, 'scope': 'fc'}),
            ([max_pool2d], {'scope': 'pool'})]


@var_scope('stack')
def _stack(x, filters, blocks, pool_fn=max_pool2d, scope=None):
    for i in range(1, blocks+1):
        x = conv(x, filters, 3, scope=str(i))
    if pool_fn is not None:
        x = pool_fn(x, 2, stride=2)
    return x


@var_scope('rp_net')
def rp_net(x, filters, original_height, original_width, scales,
           anchors=9, feat_stride=16,
           nms_thresh=0.7,  # NMS threshold used on RPN proposals
           pre_nms_topN=6000,  # Number of top scoring boxes to keep before NMS
           post_nms_topN=300,  # Number of top scoring boxes to keep after NMS
           min_size=16,  # Minimum of box sizes at original scale
           scope=None):
    x = conv(x, filters, 3, padding='SAME', scope='0')

    height = tf.shape(x)[1]
    width = tf.shape(x)[2]

    x1 = conv2d(x, 2 * anchors, 1, scope='logits')
    x1 = tf.reshape(x1, (-1, height, width, 2, anchors))
    x1 = tf.nn.softmax(x1, dim=3)
    x1 = reshape(x1, (-1, height, width, 2 * anchors), name='probs')

    x2 = conv2d(x, 4 * anchors, 1, scope='boxes')

    # Force the following operations to use CPU
    # Note that inference time may increase up to 10x without this designation
    with tf.device('cpu:0'):
        # Enumerate all shifts
        shifts = get_shifts(width, height, feat_stride)

        # Enumerate all shifted anchors
        shifted_anchors = tf.expand_dims(get_anchors(), 0) + \
            tf.expand_dims(shifts, 1)
        shifted_anchors = tf.reshape(shifted_anchors, (-1, 4))

        # Same story for the scores
        scores = tf.reshape(x1[:, :, :, anchors:],
                            (-1, height * width * anchors))
        bbox_deltas = tf.reshape(x2, (-1, height * width * anchors, 4))

        # Convert anchors into proposals via bbox transformations
        # 2. clip predicted boxes to image
        proposals = inv_boxes(shifted_anchors, bbox_deltas,
                              original_height, original_width)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = filter_boxes(proposals, min_size * scales[0])
        scores = gather(scores, keep, axis=1, name='filtered/probs')
        proposals = gather(proposals, keep, axis=1, name='filtered/boxes')

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        _, order = tf.nn.top_k(scores[0], k=tf.shape(scores)[1])
        order = order[:pre_nms_topN]
        scores = gather(scores, order, axis=1, name='topk/probs')
        proposals = gather(proposals, order, axis=1, name='topk/boxes')

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(proposals[0], scores[0], nms_thresh)
        keep = keep[:post_nms_topN]
        scores = gather(scores, keep, axis=1, name='nms/probs')
        proposals = gather(proposals, keep, axis=1, name='nms/boxes')

    return proposals


@var_scope('roi_pool')
def roi_pool2d(x, kernel_size, rois, spatial_scale=0.0625, scope=None):
    rois = tf.cast(tf.round(rois * spatial_scale), dtype=tf.int32)
    rois = tf.pad(rois[0], [[0, 0], [1, 0]])
    return roi_pooling(x, rois, kernel_size, kernel_size)


def rcnn(x, stem_fn, roi_pool_fn, is_training, classes,
         scope=None, reuse=None):
    x = stem_fn(x)
    x, rois = roi_pool_fn(x)
    x = flatten(x)
    x = fc(x, 4096, scope='fc6')
    x = relu(x, name='relu6')
    x = dropout(x, keep_prob=0.5, scope='drop6')
    x = fc(x, 4096, scope='fc7')
    x = relu(x, name='relu7')
    x = dropout(x, keep_prob=0.5, scope='drop7')
    x = concat([softmax(fc(x, classes, scope='logits'), name='probs'),
                fc(x, 4 * classes, scope='boxes'),
                rois], axis=1, name='out')
    x.get_boxes = get_boxes
    return x


@var_scope('REFfasterrcnnZFvoc')
@set_args(__args__)
def faster_rcnn_zf_voc(x, is_training=False, classes=21,
                       scope=None, reuse=None):
    scales = tf.placeholder(tf.float32, [None])
    height = tf.cast(tf.shape(x)[1], dtype=tf.float32)
    width = tf.cast(tf.shape(x)[2], dtype=tf.float32)

    def stem_fn(x):
        x = pad(x, pad_info(7), name='pad1')
        x = conv(x, 96, 7, stride=2, padding='VALID', scope='conv1')
        x = srn(x, depth_radius=3, alpha=0.00005, beta=0.75, name='srn1')
        x = pad(x, pad_info(3, symmetry=False), name='pad2')
        x = max_pool2d(x, 3, stride=2, padding='VALID', scope='pool1')

        x = pad(x, pad_info(5), name='pad3')
        x = conv(x, 256, 5, stride=2, padding='VALID', scope='conv2')
        x = srn(x, depth_radius=3, alpha=0.00005, beta=0.75, name='srn2')
        x = pad(x, pad_info(3, symmetry=False), name='pad4')
        x = max_pool2d(x, 3, stride=2, padding='VALID', scope='pool2')

        x = conv(x, 384, 3, scope='conv3')
        x = conv(x, 384, 3, scope='conv4')
        x = conv(x, 256, 3, scope='conv5')
        return x

    def roi_pool_fn(x):
        rois = rp_net(x, 256, height, width, scales)
        x = roi_pool2d(x, 6, rois)
        return x, rois[0] / scales

    x = rcnn(x, stem_fn, roi_pool_fn, is_training, classes, scope, reuse)
    x.scales = scales
    return x


@var_scope('REFfasterrcnnVGG16voc')
@set_args(__args__)
def faster_rcnn_vgg16_voc(x, is_training=False, classes=21,
                          scope=None, reuse=None):
    scales = tf.placeholder(tf.float32, [None])
    height = tf.cast(tf.shape(x)[1], dtype=tf.float32)
    width = tf.cast(tf.shape(x)[2], dtype=tf.float32)

    def stem_fn(x):
        x = _stack(x, 64, 2, scope='conv1')
        x = _stack(x, 128, 2, scope='conv2')
        x = _stack(x, 256, 3, scope='conv3')
        x = _stack(x, 512, 3, scope='conv4')
        x = _stack(x, 512, 3, pool_fn=None, scope='conv5')
        return x

    def roi_pool_fn(x):
        rois = rp_net(x, 512, height, width, scales)
        x = roi_pool2d(x, 7, rois)
        return x, rois[0] / scales

    x = rcnn(x, stem_fn, roi_pool_fn, is_training, classes, scope, reuse)
    x.scales = scales
    return x


# Simple alias.
FasterRCNN_ZF_VOC = faster_rcnn_zf_voc
FasterRCNN_VGG16_VOC = faster_rcnn_vgg16_voc
