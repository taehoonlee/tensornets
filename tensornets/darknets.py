"""Darknet19 embedded in YOLO

The reference paper:

 - YOLO9000: Better, Faster, Stronger, CVPR 2017 (Best Paper Honorable Mention)
 - Joseph Redmon, Ali Farhadi
 - https://arxiv.org/abs/1612.08242

The reference implementation:

1. Darknet
 - https://pjreddie.com/darknet/yolo/
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .layers import batch_norm
from .layers import bias_add
from .layers import conv2d
from .layers import darkconv as conv
from .layers import fc
from .layers import max_pool2d as pool

from .ops import *
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([batch_norm], {'is_training': is_training}),
            ([bias_add, conv2d], {}),
            ([pool], {'padding': 'SAME'})]


@var_scope('stack')
def _stack(x, filters, blocks, scope=None):
    for i in range(1, blocks+1):
        if i % 2 > 0:
            x = conv(x, filters, 3, scope=str(i))
        else:
            x = conv(x, filters // 2, 1, scope=str(i))
    return x


@var_scope('darknet19')
@set_args(__args__)
def darknet19(x, is_training=False, classes=1000,
              stem=False, scope=None, reuse=None):
    x = _stack(x, 32, 1, scope='conv1')
    x = pool(x, 2, stride=2, scope='pool1')
    x = _stack(x, 64, 1, scope='conv2')
    x = pool(x, 2, stride=2, scope='pool2')
    x = _stack(x, 128, 3, scope='conv3')
    x = pool(x, 2, stride=2, scope='pool3')
    x = _stack(x, 256, 3, scope='conv4')
    x = pool(x, 2, stride=2, scope='pool4')
    x = _stack(x, 512, 5, scope='conv5')
    x = pool(x, 2, stride=2, scope='pool5')
    x = _stack(x, 1024, 5, scope='conv6')
    if stem: return x

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


@var_scope('tinydarknet19')
@set_args(__args__)
def tinydarknet19(x, is_training=False, classes=1000,
                  stem=False, scope=None, reuse=None):
    x = conv(x, 16, 3, scope='conv1')
    x = pool(x, 2, stride=2, scope='pool1')
    x = conv(x, 32, 3, scope='conv2')
    x = pool(x, 2, stride=2, scope='pool2')
    x = conv(x, 64, 3, scope='conv3')
    x = pool(x, 2, stride=2, scope='pool3')
    x = conv(x, 128, 3, scope='conv4')
    x = pool(x, 2, stride=2, scope='pool4')
    x = conv(x, 256, 3, scope='conv5')
    x = pool(x, 2, stride=2, scope='pool5')
    x = conv(x, 512, 3, scope='conv6')
    if stem: return x

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


# Simple alias.
Darknet19 = darknet19
TinyDarknet19 = tinydarknet19
