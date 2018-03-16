"""Collection of YOLO variants

The reference paper:

 - YOLO9000: Better, Faster, Stronger, CVPR 2017 (Best Paper Honorable Mention)
 - Joseph Redmon, Ali Farhadi
 - https://arxiv.org/abs/1612.08242

The reference implementation:

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
from ..layers import fully_connected
from ..layers import max_pool2d

from ..ops import *
from ..utils import set_args
from ..utils import var_scope

from .yolo_utils import opts
from .yolo_utils import get_boxes


def __args__(is_training):
    return [([batch_norm], {'center': False, 'scale': True,
                            'is_training': is_training, 'epsilon': 1e-5,
                            'scope': 'bn'}),
            ([bias_add], {'scope': 'bias'}),
            ([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'biases_initializer': None, 'scope': 'conv'}),
            ([fully_connected], {'scope': 'fc'}),
            ([max_pool2d], {'padding': 'SAME', 'scope': 'pool'})]


def conv(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return leaky_relu(bias_add(batch_norm(conv2d(*args, **kwargs))),
                          alpha=0.1, name='lrelu')


@var_scope('stack')
def _stack(x, filters, blocks, scope=None):
    for i in range(1, blocks+1):
        if i % 2 > 0:
            x = conv(x, filters, 3, scope=str(i))
        else:
            x = conv(x, filters // 2, 1, scope=str(i))
    return x


@var_scope('localflatten')
def local_flatten(x, scope=None):
    x = concat([x[:, 0::2, 0::2], x[:, 0::2, 1::2],
                x[:, 1::2, 0::2], x[:, 1::2, 1::2]],
               axis=-1, name='concat')
    return x


def yolo(x, blocks, filters, is_training, classes, scope=None, reuse=None):
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
    p = local_flatten(p, scope='flat5a')

    x = concat([p, x], axis=3, name='concat')
    x = conv(x, 1024, 3, scope='conv9')
    x = conv2d(x, filters, 1, biases_initializer=tf.zeros_initializer(),
               scope='linear')
    x.get_boxes = get_boxes
    return x


def tinyyolo(x, filters, is_training, classes, scope=None, reuse=None):
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
    x = conv(x, filters[0], 3, scope='conv8')
    x = conv2d(x, filters[1], 1, biases_initializer=tf.zeros_initializer(),
               scope='linear')
    x.get_boxes = get_boxes
    return x


@var_scope('REFyolov2')
@set_args(__args__)
def yolov2(x, is_training=False, classes=1000, scope=None, reuse=None):
    x = yolo(x, [1, 1, 3, 3, 5, 5], 425, is_training, classes, scope, reuse)
    x.opts = opts('yolov2')
    return x


@var_scope('REFyolov2voc')
@set_args(__args__)
def yolov2voc(x, is_training=False, classes=1000, scope=None, reuse=None):
    x = yolo(x, [1, 1, 3, 3, 5, 5], 125, is_training, classes, scope, reuse)
    x.opts = opts('yolov2voc')
    return x


@var_scope('REFtinyyolov2')
@set_args(__args__)
def tinyyolov2(x, is_training=False, classes=1000, scope=None, reuse=None):
    x = tinyyolo(x, [512, 425], is_training, classes, scope, reuse)
    x.opts = opts('tinyyolov2')
    return x


@var_scope('REFtinyyolov2voc')
@set_args(__args__)
def tinyyolov2voc(x, is_training=False, classes=1000, scope=None, reuse=None):
    x = tinyyolo(x, [1024, 125], is_training, classes, scope, reuse)
    x.opts = opts('tinyyolov2voc')
    return x


# Simple alias.
YOLOv2 = yolov2
YOLOv2VOC = yolov2voc
TinyYOLOv2 = tinyyolov2
TinyYOLOv2VOC = tinyyolov2voc
