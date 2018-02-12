"""Collection of VGG variants

The reference paper:

 - Very Deep Convolutional Networks for Large-Scale
   Image Recognition, ICLR 2015
 - Karen Simonyan, Andrew Zisserman
 - https://arxiv.org/abs/1409.1556

The reference implementation:

1. Keras
 - https://github.com/keras-team/keras/blob/master/keras/applications/
   vgg{16,19}.py
2. Caffe VGG
 - http://www.robots.ox.ac.uk/~vgg/research/very_deep/
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .layers import conv2d
from .layers import dropout
from .layers import flatten
from .layers import fully_connected
from .layers import max_pool2d
from .layers import convrelu as conv

from .ops import *
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'scope': 'conv'}),
            ([dropout], {'is_training': is_training}),
            ([fully_connected], {'activation_fn': None, 'scope': 'fc'})]


def vgg(x, blocks, is_training, classes, scope=None, reuse=None):
    x = stack(x, 64, blocks[0], scope='conv1')
    x = stack(x, 128, blocks[1], scope='conv2')
    x = stack(x, 256, blocks[2], scope='conv3')
    x = stack(x, 512, blocks[3], scope='conv4')
    x = stack(x, 512, blocks[4], scope='conv5')
    x = flatten(x, scope='flatten')
    x = fully_connected(x, 4096, scope='fc6')
    x = relu(x, name='relu6')
    x = dropout(x, keep_prob=0.5, is_training=is_training, scope='drop6')
    x = fully_connected(x, 4096, scope='fc7')
    x = relu(x, name='relu7')
    x = dropout(x, keep_prob=0.5, is_training=is_training, scope='drop7')
    x = fully_connected(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


@var_scope('vgg16')
@set_args(__args__)
def vgg16(x, is_training=False, classes=1000, scope=None, reuse=None):
    return vgg(x, [2, 2, 3, 3, 3], is_training, classes, scope, reuse)


@var_scope('vgg19')
@set_args(__args__)
def vgg19(x, is_training=False, classes=1000, scope=None, reuse=None):
    return vgg(x, [2, 2, 4, 4, 4], is_training, classes, scope, reuse)


@var_scope('stack')
def stack(x, filters, blocks, scope=None):
    for i in range(1, blocks+1):
        x = conv(x, filters, 3, scope=str(i))
    x = max_pool2d(x, 2, stride=2, scope='pool')
    return x


# Simple alias.
VGG16 = vgg16
VGG19 = vgg19
