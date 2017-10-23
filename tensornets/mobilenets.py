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

from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import separable_conv2d

from .ops import *
from .utils import arg_scope
from .utils import collect_outputs
from .utils import var_scope


__layers__ = [batch_norm, conv2d, dropout, fully_connected, separable_conv2d]


def layers_common_args(func):
    def wrapper(*args, **kwargs):
        b_kwargs = {'decay': 0.9997,
                    'scale': True,
                    'epsilon': 0.001,
                    'is_training': args[2],
                    'scope': 'bn'}
        c_kwargs = {'padding': 'SAME',
                    'activation_fn': None,
                    'biases_initializer': None,
                    'scope': 'conv'}
        f_kwargs = {'activation_fn': None}
        s_kwargs = {'activation_fn': None,
                    'biases_initializer': None,
                    'scope': 'sconv'}
        with collect_outputs(__layers__), \
                arg_scope([batch_norm], **b_kwargs), \
                arg_scope([conv2d], **c_kwargs), \
                arg_scope([fully_connected], **f_kwargs), \
                arg_scope([separable_conv2d], **s_kwargs):
            return func(*args, **kwargs)
    return wrapper


def conv(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu6(batch_norm(conv2d(*args, **kwargs)))


def sconv(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu6(batch_norm(separable_conv2d(*args, **kwargs)))


@var_scope('block')
def block(x, filters, stride=1, scope=None):
    x = sconv(x, None, 3, 1, stride=stride, scope='sconv')
    x = conv(x, filters, 1, stride=1, scope='conv')
    return x


@layers_common_args
def mobilenet(x, depth_multiplier, is_training, classes,
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

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = dropout(x, keep_prob=0.999, is_training=is_training, scope='dropout')
    x = fully_connected(x, classes, scope='logits')
    x = softmax(x, name='probs')
    x.aliases = [tf.get_variable_scope().name]
    return x


@var_scope('mobilenet25')
def mobilenet25(x, is_training=False, classes=1000, scope=None, reuse=None):
    return mobilenet(x, 0.25, is_training, classes, scope, reuse)


@var_scope('mobilenet50')
def mobilenet50(x, is_training=False, classes=1000, scope=None, reuse=None):
    return mobilenet(x, 0.5, is_training, classes, scope, reuse)


@var_scope('mobilenet75')
def mobilenet75(x, is_training=False, classes=1000, scope=None, reuse=None):
    return mobilenet(x, 0.75, is_training, classes, scope, reuse)


@var_scope('mobilenet100')
def mobilenet100(x, is_training=False, classes=1000, scope=None, reuse=None):
    return mobilenet(x, 1.0, is_training, classes, scope, reuse)


# Simple alias.
MobileNet25 = mobilenet25
MobileNet50 = mobilenet50
MobileNet75 = mobilenet75
MobileNet100 = mobilenet100
