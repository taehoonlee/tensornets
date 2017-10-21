"""Collection of DenseNet variants

The reference paper:

 - Densely Connected Convolutional Networks, CVPR 2017 (Best Paper Award)
 - Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten
 - https://arxiv.org/abs/1608.06993

The reference implementation:

1. Torch DenseNets
 - https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua
"""
from __future__ import absolute_import

import tensorflow as tf

from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import max_pool2d

from .ops import *
from .utils import arg_scope
from .utils import collect_outputs
from .utils import var_scope


__layers__ = [avg_pool2d, batch_norm, conv2d, fully_connected, max_pool2d]


def layers_common_args(func):
    def wrapper(*args, **kwargs):
        b_kwargs = {'scale': True,
                    'is_training': args[2],
                    'scope': 'bn',
                    'epsilon': 1e-5}
        c_kwargs = {'padding': 'VALID',
                    'activation_fn': None,
                    'biases_initializer': None,
                    'scope': 'conv'}
        f_kwargs = {'activation_fn': None}
        with collect_outputs(__layers__), \
                arg_scope([batch_norm], **b_kwargs), \
                arg_scope([conv2d], **c_kwargs), \
                arg_scope([fully_connected], **f_kwargs):
            return func(*args, **kwargs)
    return wrapper


def conv(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu(batch_norm(conv2d(*args, **kwargs)))


@layers_common_args
def densenet(x, blocks, is_training, classes, scope=None, reuse=None):
    x = pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], name='conv1/pad')
    x = conv(x, 64, 7, stride=2, scope='conv1')
    x = pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pool1/pad')
    x = max_pool2d(x, 3, stride=2, scope='pool1')

    x = dense(x, blocks[0], scope='conv2')
    x = transition(x, scope='pool2')
    x = dense(x, blocks[1], scope='conv3')
    x = transition(x, scope='pool3')
    x = dense(x, blocks[2], scope='conv4')
    x = transition(x, scope='pool4')
    x = dense(x, blocks[3], scope='conv5')

    x = batch_norm(x)
    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fully_connected(x, classes, scope='logits')
    x = softmax(x, name='probs')
    x.aliases = [tf.get_variable_scope().name]
    return x


@var_scope('densenet121')
def densenet121(x, is_training=True, classes=1000, scope=None, reuse=None):
    return densenet(x, [6, 12, 24, 16], is_training, classes, scope, reuse)


@var_scope('densenet169')
def densenet169(x, is_training=True, classes=1000, scope=None, reuse=None):
    return densenet(x, [6, 12, 32, 32], is_training, classes, scope, reuse)


@var_scope('densenet201')
def densenet201(x, is_training=True, classes=1000, scope=None, reuse=None):
    return densenet(x, [6, 12, 48, 32], is_training, classes, scope, reuse)


@var_scope('dense')
def dense(x, blocks, scope=None):
    for i in range(blocks):
        x = block(x, scope="block%d" % (i + 1))
    return x


@var_scope('transition')
def transition(x, reduction=0.5, scope=None):
    x = batch_norm(x)
    x = relu(x)
    x = conv2d(x, x.shape[-1].value * reduction, 1, stride=1)
    x = avg_pool2d(x, 2, stride=2, scope='pool')
    return x


@var_scope('block')
def block(x, growth_rate=32, scope=None):
    x1 = batch_norm(x)
    x1 = relu(x1)
    x1 = conv(x1, 4 * growth_rate, 1, stride=1, scope='1')
    x1 = conv2d(x1, growth_rate, 3, stride=1, padding='SAME', scope='2/conv')
    x = concat([x, x1], axis=3, name='concat')
    return x


# Simple alias.
DenseNet121 = densenet121
DenseNet169 = densenet169
DenseNet201 = densenet201
