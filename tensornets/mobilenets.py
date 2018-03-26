"""Collection of MobileNet variants

The reference paper:

 - MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
   Applications, arXiv 2017
 - Andrew G. Howard et al.
 - https://arxiv.org/abs/1704.04861

The reference implementation:

1. TF Slim
 - https://github.com/tensorflow/models/blob/master/research/slim/nets/
   mobilenet_v1.py
"""
from __future__ import absolute_import

import tensorflow as tf

from .layers import batch_norm
from .layers import conv2d
from .layers import dropout
from .layers import fc
from .layers import separable_conv2d
from .layers import convbnrelu6 as conv
from .layers import sconvbnrelu6 as sconv

from .ops import *
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([batch_norm], {'decay': 0.9997, 'scale': True, 'epsilon': 0.001,
                            'is_training': is_training, 'scope': 'bn'}),
            ([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'biases_initializer': None, 'scope': 'conv'}),
            ([dropout], {'is_training': is_training, 'scope': 'dropout'}),
            ([fc], {'activation_fn': None, 'scope': 'fc'}),
            ([separable_conv2d],
             {'activation_fn': None, 'biases_initializer': None,
              'scope': 'sconv'})]


@var_scope('block')
def block(x, filters, stride=1, scope=None):
    x = sconv(x, None, 3, 1, stride=stride, scope='sconv')
    x = conv(x, filters, 1, stride=1, scope='conv')
    return x


def mobilenet(x, depth_multiplier, is_training, classes, stem,
              scope=None, reuse=None):
    def depth(d):
        return max(int(d * depth_multiplier), 8)
    x = conv(x, depth(32), 3, stride=2, scope='conv1')

    x = block(x, depth(64), scope='conv2')
    x = block(x, depth(128), stride=2, scope='conv3')

    x = block(x, depth(128), scope='conv4')
    x = block(x, depth(256), stride=2, scope='conv5')

    x = block(x, depth(256), scope='conv6')
    x = block(x, depth(512), stride=2, scope='conv7')

    x = block(x, depth(512), scope='conv8')
    x = block(x, depth(512), scope='conv9')
    x = block(x, depth(512), scope='conv10')
    x = block(x, depth(512), scope='conv11')
    x = block(x, depth(512), scope='conv12')
    x = block(x, depth(1024), stride=2, scope='conv13')

    x = block(x, depth(1024), scope='conv14')
    if stem: return x

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = dropout(x, keep_prob=0.999, is_training=is_training, scope='dropout')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


@var_scope('mobilenet25')
@set_args(__args__)
def mobilenet25(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return mobilenet(x, 0.25, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet50')
@set_args(__args__)
def mobilenet50(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return mobilenet(x, 0.5, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet75')
@set_args(__args__)
def mobilenet75(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return mobilenet(x, 0.75, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet100')
@set_args(__args__)
def mobilenet100(x, is_training=False, classes=1000,
                 stem=False, scope=None, reuse=None):
    return mobilenet(x, 1.0, is_training, classes, stem, scope, reuse)


# Simple alias.
MobileNet25 = mobilenet25
MobileNet50 = mobilenet50
MobileNet75 = mobilenet75
MobileNet100 = mobilenet100
