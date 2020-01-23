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

from .layers import avg_pool2d
from .layers import batch_norm
from .layers import conv2d
from .layers import fc
from .layers import max_pool2d
from .layers import convbnrelu as conv

from .ops import *
from .utils import pad_info
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([avg_pool2d, max_pool2d], {'scope': 'pool'}),
            ([batch_norm], {'scale': True, 'is_training': is_training,
                            'epsilon': 1e-5, 'scope': 'bn'}),
            ([conv2d], {'padding': 'VALID', 'activation_fn': None,
                        'biases_initializer': None, 'scope': 'conv'}),
            ([fc], {'activation_fn': None, 'scope': 'fc'})]


def densenet(x, blocks, is_training, classes, stem, scope=None, reuse=None):
    x = pad(x, pad_info(7), name='conv1/pad')
    x = conv(x, 64, 7, stride=2, scope='conv1')
    x = pad(x, pad_info(3), name='pool1/pad')
    x = max_pool2d(x, 3, stride=2, scope='pool1')

    x = dense(x, blocks[0], scope='conv2')
    x = transition(x, scope='pool2')
    x = dense(x, blocks[1], scope='conv3')
    x = transition(x, scope='pool3')
    x = dense(x, blocks[2], scope='conv4')
    x = transition(x, scope='pool4')
    x = dense(x, blocks[3], scope='conv5')

    x = batch_norm(x)
    x = relu(x)
    if stem: return x

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


@var_scope('densenet121')
@set_args(__args__)
def densenet121(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return densenet(x, [6, 12, 24, 16], is_training, classes,
                    stem, scope, reuse)


@var_scope('densenet169')
@set_args(__args__)
def densenet169(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return densenet(x, [6, 12, 32, 32], is_training, classes,
                    stem, scope, reuse)


@var_scope('densenet201')
@set_args(__args__)
def densenet201(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return densenet(x, [6, 12, 48, 32], is_training, classes,
                    stem, scope, reuse)


@var_scope('dense')
def dense(x, blocks, scope=None):
    for i in range(blocks):
        x = block(x, scope="block%d" % (i + 1))
    return x


@var_scope('transition')
def transition(x, reduction=0.5, scope=None):
    x = batch_norm(x)
    x = relu(x)
    infilters = int(x.shape[-1]) if tf_later_than('2') else x.shape[-1].value
    x = conv2d(x, int(infilters * reduction), 1, stride=1)
    x = avg_pool2d(x, 2, stride=2, scope='pool')
    return x


@var_scope('block')
def block(x, growth_rate=32, scope=None):
    x1 = batch_norm(x)
    x1 = relu(x1)
    x1 = conv(x1, 4 * growth_rate, 1, stride=1, scope='1')
    x1 = conv2d(x1, growth_rate, 3, stride=1, padding='SAME', scope='2/conv')
    x = concat([x, x1], axis=3, name='out')
    return x


# Simple alias.
DenseNet121 = densenet121
DenseNet169 = densenet169
DenseNet201 = densenet201
