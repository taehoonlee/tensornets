"""Collection of generic object detection models

The reference papers:

1. YOLOv2
 - YOLO9000: Better, Faster, Stronger, CVPR 2017 (Best Paper Honorable Mention)
 - Joseph Redmon, Ali Farhadi
 - https://arxiv.org/abs/1612.08242
2. Faster R-CNN
 - Faster R-CNN: Towards Real-Time Object Detection
   with Region Proposal Networks, NIPS 2015
 - Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
 - https://arxiv.org/abs/1506.01497

The reference implementations:

1. Darknet
 - https://pjreddie.com/darknet/yolo/
2. darkflow
 - https://github.com/thtrieu/darkflow
3. Caffe and Python utils
 - https://github.com/rbgirshick/py-faster-rcnn
4. RoI pooling in TensorFlow
 - https://github.com/deepsense-ai/roi-pooling
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .layers import batch_norm
from .layers import bias_add
from .layers import conv2d
from .layers import darkconv
from .layers import dropout
from .layers import flatten
from .layers import fc
from .layers import max_pool2d

from .ops import *
from .utils import remove_head
from .utils import set_args
from .utils import var_scope

from .references.yolos import get_v2_boxes as yolo_boxes
from .references.yolos import opts
from .references.yolos import v2_inputs
from .references.yolos import v2_loss
from .references.rcnns import get_boxes as rcnn_boxes
from .references.rcnns import roi_pool2d
from .references.rcnns import rp_net


def __args_yolo__(is_training):
    return [([batch_norm], {'is_training': is_training}),
            ([bias_add, conv2d], {}),
            ([max_pool2d], {'padding': 'SAME'})]


def __args_rcnn__(is_training):
    return [([conv2d], {'activation_fn': None, 'scope': 'conv'}),
            ([dropout], {'is_training': is_training}),
            ([fc], {'activation_fn': None, 'scope': 'fc'})]


@var_scope('genYOLOv2')
@set_args(__args_yolo__)
def yolov2(x, stem_fn, stem_out=None, is_training=False, classes=20,
           scope=None, reuse=None):
    inputs = x
    opt = opts('yolov2' + data_name(classes))
    stem = x = stem_fn(x, is_training, stem=True, scope='stem')
    p = x.p

    if stem_out is not None:
        stem = x = remove_head(x, stem_out)

    x = darkconv(x, 1024, 3, scope='conv7')
    x = darkconv(x, 1024, 3, scope='conv8')

    p = darkconv(p, 64, 1, scope='conv5a')
    p = local_flatten(p, 2, name='flat5a')

    x = concat([p, x], axis=3, name='concat')
    x = darkconv(x, 1024, 3, scope='conv9')
    x = darkconv(x, (classes + 5) * 5, 1, onlyconv=True, scope='linear')
    x.aliases = []

    def get_boxes(*args, **kwargs):
        return yolo_boxes(opt, *args, **kwargs)
    x.get_boxes = get_boxes
    x.stem = stem
    x.inputs = [inputs]
    x.inputs += v2_inputs(x.shape[1:3], opt['num'], classes, x.dtype)
    if isinstance(is_training, tf.Tensor):
        x.inputs.append(is_training)
    x.loss = v2_loss(x, opt['anchors'], classes)
    return x


def data_name(classes):
    return 'voc' if classes == 20 else ''


@var_scope('genTinyYOLOv2')
@set_args(__args_yolo__)
def tinyyolov2(x, stem_fn, stem_out=None, is_training=False, classes=20,
               scope=None, reuse=None):
    inputs = x
    opt = opts('tinyyolov2' + data_name(classes))
    stem = x = stem_fn(x, is_training, stem=True, scope='stem')

    if stem_out is not None:
        stem = x = remove_head(x, stem_out)

    x = max_pool2d(x, 2, stride=1, scope='pool6')
    x = darkconv(x, 1024, 3, scope='conv7')
    x = darkconv(x, 1024 if classes == 20 else 512, 3, scope='conv8')
    x = darkconv(x, (classes + 5) * 5, 1, onlyconv=True, scope='linear')
    x.aliases = []

    def get_boxes(*args, **kwargs):
        return yolo_boxes(opt, *args, **kwargs)
    x.get_boxes = get_boxes
    x.stem = stem
    x.inputs = [inputs]
    x.inputs += v2_inputs(x.shape[1:3], opt['num'], classes, x.dtype)
    if isinstance(is_training, tf.Tensor):
        x.inputs.append(is_training)
    x.loss = v2_loss(x, opt['anchors'], classes)
    return x


@var_scope('genFasterRCNN')
@set_args(__args_rcnn__)
def fasterrcnn(x, stem_fn, stem_out=None, is_training=False, classes=21,
               scope=None, reuse=None):
    def roi_pool_fn(x, filters, kernel_size):
        rois = rp_net(x, filters, height, width, scales)
        x = roi_pool2d(x, kernel_size, rois)
        return x, rois[0] / scales

    scales = tf.placeholder(tf.float32, [None])
    height = tf.cast(tf.shape(x)[1], dtype=tf.float32)
    width = tf.cast(tf.shape(x)[2], dtype=tf.float32)

    stem = x = stem_fn(x, is_training, stem=True, scope='stem')

    if stem_out is not None:
        stem = x = remove_head(x, stem_out)

    if 'zf' in stem.model_name:
        x, rois = roi_pool_fn(x, 256, 6)
    else:
        x, rois = roi_pool_fn(x, 512, 7)

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
    x.get_boxes = rcnn_boxes
    x.scales = scales
    x.stem = stem
    return x


# Simple alias.
YOLOv2 = yolov2
TinyYOLOv2 = tinyyolov2
FasterRCNN = fasterrcnn
