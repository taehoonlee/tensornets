"""Collection of DenseNet variants

The reference paper:

 - SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB
   model size, arXiv 2016
 - Forrest N. Iandola et al.
 - https://arxiv.org/abs/1602.07360

The reference implementation:

1. Caffe SqueezeNets
 - https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1
"""
from __future__ import absolute_import

import tensorflow as tf

from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import max_pool2d

from .ops import *
from .utils import arg_scope
from .utils import collect_outputs
from .utils import var_scope


__layers__ = [conv2d, dropout, fully_connected, max_pool2d]


def layers_common_args(func):
    def wrapper(*args, **kwargs):
        c_kwargs = {'padding': 'SAME',
                    'activation_fn': None,
                    'scope': 'conv'}
        f_kwargs = {'activation_fn': None}
        with collect_outputs(__layers__), \
                arg_scope([conv2d], **c_kwargs), \
                arg_scope([fully_connected], **f_kwargs):
            return func(*args, **kwargs)
    return wrapper


def conv(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu(conv2d(*args, **kwargs))


@var_scope('fire')
def fire(x, squeeze, expand, scope=None):
    x = conv(x, squeeze, 1, scope='squeeze1x1')
    x1 = conv(x, expand, 1, scope='expand1x1')
    x2 = conv(x, expand, 3, scope='expand3x3')
    x = concat([x1, x2], axis=3, name='concat')
    return x


@var_scope('squeezenet')
@layers_common_args
def squeezenet(x, is_training=True, classes=1000, scope=None, reuse=None):
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
    x = dropout(x, keep_prob=0.5, is_training=is_training, scope='drop9')

    x = conv(x, classes, 1, scope='conv10')
    x = reduce_mean(x, [1, 2], name='pool10')
    x = softmax(x, name='probs')
    x.aliases = [tf.get_variable_scope().name]
    return x


# Simple alias.
SqueezeNet = squeezenet
