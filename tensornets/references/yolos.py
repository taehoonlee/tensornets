"""Collection of YOLO variants

The reference papers:

1. YOLO9000
 - YOLO9000: Better, Faster, Stronger, CVPR 2017 (Best Paper Honorable Mention)
 - Joseph Redmon, Ali Farhadi
 - https://arxiv.org/abs/1612.08242
2. YOLOv3
 - YOLOv3: An Incremental Improvement
 - Joseph Redmon, Ali Farhadi
 - https://pjreddie.com/media/files/papers/YOLOv3.pdf

The reference implementations:

1. Darknet
 - https://pjreddie.com/darknet/yolo/
2. darkflow
 - https://github.com/thtrieu/darkflow
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from ..layers import batch_norm
from ..layers import bias_add
from ..layers import conv2d
from ..layers import darkconv as conv
from ..layers import max_pool2d

from ..ops import *
from ..utils import pad_info
from ..utils import set_args
from ..utils import var_scope

from .yolo_utils import opts
from .yolo_utils import get_v3_boxes
from .yolo_utils import get_v2_boxes
from .yolo_utils import v2_inputs
from .yolo_utils import v2_loss


def __args__(is_training):
    return [([batch_norm], {'is_training': is_training}),
            ([bias_add, conv2d], {}),
            ([max_pool2d], {'padding': 'SAME'})]


@var_scope('stack')
def _stack(x, filters, blocks, scope=None):
    for i in range(1, blocks+1):
        if i % 2 > 0:
            x = conv(x, filters, 3, scope=str(i))
        else:
            x = conv(x, filters // 2, 1, scope=str(i))
    return x


@var_scope('stackv3')
def stackv3(x, filters, blocks, kernel_size=3,
            conv_shortcut=True, scope=None):
    for i in range(1, blocks+1):
        shortcut = x
        p = conv(x, filters // 2, 1, scope="%d/1" % i)
        x = conv(p, filters, kernel_size, scope="%d/2" % i)
        if conv_shortcut is True:
            x = add(shortcut, x, name="%d/out" % i)
    if conv_shortcut is True:
        return x
    else:
        return x, p


@var_scope('down')
def down(x, filters, kernel_size=3, scope=None):
    x = pad(x, pad_info(kernel_size), name='pad')
    x = conv(x, filters, kernel_size, stride=2,
             padding='VALID', scope='conv')
    return x


@var_scope('up')
def up(x, filters, kernel_size=2, scope=None):
    x = conv(x, filters, 1, scope='conv')
    x = upsample(x, kernel_size, name='upsample')
    return x


def yolov3(x, blocks, is_training, classes, scope=None, reuse=None):
    x = conv(x, 32, 3, scope='conv1')
    x = down(x, 64, scope='down1')
    x = stackv3(x, 64, blocks[0], scope='conv2')
    x = down(x, 128, scope='down2')
    x = stackv3(x, 128, blocks[1], scope='conv3')
    x = down(x, 256, scope='down3')
    x = p0 = stackv3(x, 256, blocks[2], scope='conv4')
    x = down(x, 512, scope='down4')
    x = p1 = stackv3(x, 512, blocks[3], scope='conv5')
    x = down(x, 1024, scope='down5')
    x = stackv3(x, 1024, blocks[4], scope='conv6')

    x, p = stackv3(x, 1024, blocks[5], conv_shortcut=False, scope='conv7')
    out0 = conv(x, (classes + 5) * 3, 1, onlyconv=True, scope='linear7')
    p = up(p, 256, 2, scope='up7')
    x = concat([p, p1], axis=3, name='concat7')

    x, p = stackv3(x, 512, blocks[5], conv_shortcut=False, scope='conv8')
    out1 = conv(x, (classes + 5) * 3, 1, onlyconv=True, scope='linear8')
    p = up(p, 128, 2, scope='up8')
    x = concat([p, p0], axis=3, name='concat8')

    x, _ = stackv3(x, 256, blocks[5], conv_shortcut=False, scope='conv9')
    out2 = conv(x, (classes + 5) * 3, 1, onlyconv=True, scope='linear9')
    out2.aliases = []
    out2.preds = [out0, out1, out2]
    return out2


def yolo(x, blocks, is_training, classes, scope=None, reuse=None):
    x = _stack(x, 32, blocks[0], scope='conv1')
    x = max_pool2d(x, 2, stride=2, scope='pool1')
    x = _stack(x, 64, blocks[1], scope='conv2')
    x = max_pool2d(x, 2, stride=2, scope='pool2')
    x = _stack(x, 128, blocks[2], scope='conv3')
    x = max_pool2d(x, 2, stride=2, scope='pool3')
    x = _stack(x, 256, blocks[3], scope='conv4')
    x = max_pool2d(x, 2, stride=2, scope='pool4')
    x = p = _stack(x, 512, blocks[4], scope='conv5')
    x = max_pool2d(x, 2, stride=2, scope='pool5')
    x = _stack(x, 1024, blocks[5], scope='conv6')

    x = conv(x, 1024, 3, scope='conv7')
    x = conv(x, 1024, 3, scope='conv8')

    p = conv(p, 64, 1, scope='conv5a')
    p = local_flatten(p, 2, name='flat5a')

    x = concat([p, x], axis=3, name='concat')
    x = conv(x, 1024, 3, scope='conv9')
    x = conv(x, (classes + 5) * 5, 1, onlyconv=True, scope='linear')
    x.aliases = []
    return x


def tinyyolo(x, is_training, classes, scope=None, reuse=None):
    x = conv(x, 16, 3, scope='conv1')
    x = max_pool2d(x, 2, stride=2, scope='pool1')
    x = conv(x, 32, 3, scope='conv2')
    x = max_pool2d(x, 2, stride=2, scope='pool2')
    x = conv(x, 64, 3, scope='conv3')
    x = max_pool2d(x, 2, stride=2, scope='pool3')
    x = conv(x, 128, 3, scope='conv4')
    x = max_pool2d(x, 2, stride=2, scope='pool4')
    x = conv(x, 256, 3, scope='conv5')
    x = max_pool2d(x, 2, stride=2, scope='pool5')
    x = conv(x, 512, 3, scope='conv6')

    x = max_pool2d(x, 2, stride=1, scope='pool6')
    x = conv(x, 1024, 3, scope='conv7')
    x = conv(x, 1024 if classes == 20 else 512, 3, scope='conv8')
    x = conv(x, (classes + 5) * 5, 1, onlyconv=True, scope='linear')
    x.aliases = []
    return x


@var_scope('REFyolov3coco')
@set_args(__args__)
def yolov3coco(x, is_training=False, classes=80, scope=None, reuse=None):
    def _get_boxes(*args, **kwargs):
        return get_v3_boxes(opts('yolov3'), *args, **kwargs)
    x = yolov3(x, [1, 2, 8, 8, 4, 3], is_training, classes, scope, reuse)
    x.get_boxes = _get_boxes
    return x


@var_scope('REFyolov3voc')
@set_args(__args__)
def yolov3voc(x, is_training=False, classes=20, scope=None, reuse=None):
    def _get_boxes(*args, **kwargs):
        return get_v3_boxes(opts('yolov3voc'), *args, **kwargs)
    x = yolov3(x, [1, 2, 8, 8, 4, 3], is_training, classes, scope, reuse)
    x.get_boxes = _get_boxes
    return x


@var_scope('REFyolov2coco')
@set_args(__args__)
def yolov2coco(x, is_training=False, classes=80, scope=None, reuse=None):
    inputs = x
    opt = opts('yolov2')
    x = yolo(x, [1, 1, 3, 3, 5, 5], is_training, classes, scope, reuse)

    def _get_boxes(*args, **kwargs):
        return get_v2_boxes(opt, *args, **kwargs)
    x.get_boxes = _get_boxes
    x.inputs = [inputs]
    x.inputs += v2_inputs(x.shape[1:3], opt['num'], classes, x.dtype)
    if isinstance(is_training, tf.Tensor):
        x.inputs.append(is_training)
    x.loss = v2_loss(x, opt['anchors'], classes)
    return x


@var_scope('REFyolov2voc')
@set_args(__args__)
def yolov2voc(x, is_training=False, classes=20, scope=None, reuse=None):
    inputs = x
    opt = opts('yolov2voc')
    x = yolo(x, [1, 1, 3, 3, 5, 5], is_training, classes, scope, reuse)

    def _get_boxes(*args, **kwargs):
        return get_v2_boxes(opt, *args, **kwargs)
    x.get_boxes = _get_boxes
    x.inputs = [inputs]
    x.inputs += v2_inputs(x.shape[1:3], opt['num'], classes, x.dtype)
    if isinstance(is_training, tf.Tensor):
        x.inputs.append(is_training)
    x.loss = v2_loss(x, opt['anchors'], classes)
    return x


@var_scope('REFtinyyolov2coco')
@set_args(__args__)
def tinyyolov2coco(x, is_training=False, classes=80, scope=None, reuse=None):
    inputs = x
    opt = opts('tinyyolov2')
    x = tinyyolo(x, is_training, classes, scope, reuse)

    def _get_boxes(*args, **kwargs):
        return get_v2_boxes(opt, *args, **kwargs)
    x.get_boxes = _get_boxes
    x.inputs = [inputs]
    x.inputs += v2_inputs(x.shape[1:3], opt['num'], classes, x.dtype)
    if isinstance(is_training, tf.Tensor):
        x.inputs.append(is_training)
    x.loss = v2_loss(x, opt['anchors'], classes)
    return x


@var_scope('REFtinyyolov2voc')
@set_args(__args__)
def tinyyolov2voc(x, is_training=False, classes=20, scope=None, reuse=None):
    inputs = x
    opt = opts('tinyyolov2voc')
    x = tinyyolo(x, is_training, classes, scope, reuse)

    def _get_boxes(*args, **kwargs):
        return get_v2_boxes(opt, *args, **kwargs)
    x.get_boxes = _get_boxes
    x.inputs = [inputs]
    x.inputs += v2_inputs(x.shape[1:3], opt['num'], classes, x.dtype)
    if isinstance(is_training, tf.Tensor):
        x.inputs.append(is_training)
    x.loss = v2_loss(x, opt['anchors'], classes)
    return x


# Simple alias.
YOLOv3COCO = yolov3coco
YOLOv3VOC = yolov3voc
YOLOv2COCO = yolov2coco
YOLOv2VOC = yolov2voc
TinyYOLOv2COCO = tinyyolov2coco
TinyYOLOv2VOC = tinyyolov2voc
