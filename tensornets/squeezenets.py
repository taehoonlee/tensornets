"""Collection of SqueezeNet variants

The reference paper:

 - SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size, arXiv 2016
 - Forrest N. Iandola et al.
 - https://arxiv.org/abs/1602.07360

The reference implementation:

1. Caffe SqueezeNets
 - https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1
"""
from __future__ import absolute_import

import tensorflow as tf

from .layers import conv2d
from .layers import dropout
from .layers import fc
from .layers import max_pool2d
from .layers import convrelu as conv

from .ops import *
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'scope': 'conv'}),
            ([dropout], {'is_training': is_training, 'scope': 'dropout'}),
            ([fc], {'activation_fn': None, 'scope': 'fc'}),
            ([max_pool2d], {'scope': 'pool'})]


@var_scope('fire')
def fire(x, squeeze, expand, scope=None):
    x = conv(x, squeeze, 1, scope='squeeze1x1')
    x1 = conv(x, expand, 1, scope='expand1x1')
    x2 = conv(x, expand, 3, scope='expand3x3')
    x = concat([x1, x2], axis=3, name='concat')
    return x


@var_scope('squeezenet')
@set_args(__args__)
def squeezenet(x, is_training=False, classes=1000,
               stem=False, scope=None, reuse=None):
    x = conv(x, 64, 3, stride=2, padding='VALID', scope='conv1')
    x = max_pool2d(x, 3, stride=2, scope='pool1')

    x = fire(x, 16, 64, scope='fire2')
    x = fire(x, 16, 64, scope='fire3')
    x = max_pool2d(x, 3, stride=2, scope='pool3')

    x = fire(x, 32, 128, scope='fire4')
    x = fire(x, 32, 128, scope='fire5')
    x = max_pool2d(x, 3, stride=2, scope='pool5')

    x = fire(x, 48, 192, scope='fire6')
    x = fire(x, 48, 192, scope='fire7')
    x = fire(x, 64, 256, scope='fire8')
    x = fire(x, 64, 256, scope='fire9')
    if stem: return x
    x = dropout(x, keep_prob=0.5, scope='drop9')

    x = reduce_mean(x, [1, 2], name='pool10')
    x = fc(x, classes, scope='logits')  # the original name is `conv10`
    x = softmax(x, name='probs')
    return x


# Simple alias.
SqueezeNet = squeezenet
