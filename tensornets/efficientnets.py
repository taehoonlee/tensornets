"""Collection of EfficientNet variants

The reference paper:

 - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, ICML 2019
 - Mingxing Tan, Quoc V. Le
 - https://arxiv.org/abs/1905.11946

The reference implementations:

1. Keras
 - https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
2. TF TPU
 - https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
"""
from __future__ import absolute_import
from __future__ import division

import math
import tensorflow as tf

from .layers import batch_norm
from .layers import conv2d
from .layers import convbn
from .layers import convbnswish as conv
from .layers import dropout
from .layers import fc
from .layers import sconv2d
from .layers import sconvbnswish as sconv

from .ops import *
from .ops import _swish
from .utils import pad_info
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([batch_norm], {'decay': 0.99, 'scale': True,
                            'is_training': is_training, 'scope': 'bn'}),
            ([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'biases_initializer': None, 'scope': 'conv'}),
            ([dropout], {'is_training': is_training, 'scope': 'dropout'}),
            ([fc], {'activation_fn': None, 'scope': 'fc'}),
            ([sconv2d], {'padding': 'SAME', 'activation_fn': None,
                         'biases_initializer': None, 'scope': 'sconv'})]


def blocks_args():
    return [
        {'blocks': 1, 'filters_in': 32, 'filters_out': 16, 'kernel_size': 3,
         'stride': 1, 'expand_ratio': 1, 'se_ratio': 0.25},
        {'blocks': 2, 'filters_in': 16, 'filters_out': 24, 'kernel_size': 3,
         'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
        {'blocks': 2, 'filters_in': 24, 'filters_out': 40, 'kernel_size': 5,
         'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
        {'blocks': 3, 'filters_in': 40, 'filters_out': 80, 'kernel_size': 3,
         'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
        {'blocks': 3, 'filters_in': 80, 'filters_out': 112, 'kernel_size': 5,
         'stride': 1, 'expand_ratio': 6, 'se_ratio': 0.25},
        {'blocks': 4, 'filters_in': 112, 'filters_out': 192, 'kernel_size': 5,
         'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
        {'blocks': 1, 'filters_in': 192, 'filters_out': 320, 'kernel_size': 3,
         'stride': 1, 'expand_ratio': 6, 'se_ratio': 0.25}
    ]


def efficientnet(x, width_coefficient, depth_coefficient,
                 default_size, is_training, classes, stem,
                 keep_prob=0.8, drop_rate=0.2, width_divisor=8,
                 scope=None, reuse=None):
    def width(w, coefficient=width_coefficient, divisor=width_divisor):
        w *= coefficient
        new_w = max(divisor, int(w + divisor / 2) // divisor * divisor)
        if new_w < 0.9 * w:
            new_w += divisor
        return int(new_w)

    def depth(d, coefficient=depth_coefficient):
        return int(math.ceil(d * coefficient))

    b = 0
    x = conv(x, width(32), 3, stride=2, scope='stem')
    blocks_total = float(sum(args['blocks'] for args in blocks_args()))
    for args in blocks_args():
        filters_in = width(args['filters_in'])
        filters_out = width(args['filters_out'])
        for j in range(depth(args['blocks'])):
            x = block(x, filters_in if j == 0 else filters_out, filters_out,
                      args['kernel_size'], 1 if j > 0 else args['stride'],
                      args['expand_ratio'], args['se_ratio'],
                      drop_rate * b / blocks_total,
                      scope="block{}".format(b))
            b += 1
    x = conv(x, width(1280), 1, scope='head')
    if stem: return x
    x = reduce_mean(x, [1, 2], name='avgpool')
    x = dropout(x, keep_prob=keep_prob, scope='dropout')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


@var_scope('se')
def se(i, filters_se, filters_out, scope=None):
    x = reduce_mean(i, [1, 2], keepdims=True, name='squeeze')
    x = conv2d(x, filters_se, 1, activation_fn=_swish,
               biases_initializer=tf.zeros_initializer(), scope='reduce')
    x = conv2d(x, filters_out, 1, activation_fn=tf.sigmoid,
               biases_initializer=tf.zeros_initializer(), scope='expand')
    x = multiply(i, x, name='excite')
    return x


@var_scope('block')
def block(i, filters_in=32, filters_out=16, kernel_size=3, stride=1,
          expand_ratio=1, se_ratio=0., drop_rate=0., scope=None):
    filters = filters_in * expand_ratio
    x = conv(i, filters, 1, scope='econv') if expand_ratio != 1 else i
    x = sconv(x, None, kernel_size, 1, stride=stride, scope='sconv')
    if 0 < se_ratio <= 1:
        x = se(x, max(1, int(filters_in * se_ratio)), filters, scope='se')
    x = convbn(x, filters_out, 1, scope='pconv')
    if (stride == 1) and (filters_in == filters_out):
        if drop_rate > 0:
            x = dropout(x, keep_prob=1 - drop_rate, scope='dropout')
        x = add(i, x, name='add')
    return x


@var_scope('efficientnetb0')
@set_args(__args__)
def efficientnetb0(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return efficientnet(x, 1.0, 1.0, 224, is_training, classes, stem,
                        keep_prob=0.8, scope=scope, reuse=reuse)


@var_scope('efficientnetb1')
@set_args(__args__)
def efficientnetb1(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return efficientnet(x, 1.0, 1.1, 240, is_training, classes, stem,
                        keep_prob=0.8, scope=scope, reuse=reuse)


@var_scope('efficientnetb2')
@set_args(__args__)
def efficientnetb2(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return efficientnet(x, 1.1, 1.2, 260, is_training, classes, stem,
                        keep_prob=0.7, scope=scope, reuse=reuse)


@var_scope('efficientnetb3')
@set_args(__args__)
def efficientnetb3(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return efficientnet(x, 1.2, 1.4, 300, is_training, classes, stem,
                        keep_prob=0.7, scope=scope, reuse=reuse)


@var_scope('efficientnetb4')
@set_args(__args__)
def efficientnetb4(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return efficientnet(x, 1.4, 1.8, 380, is_training, classes, stem,
                        keep_prob=0.6, scope=scope, reuse=reuse)


@var_scope('efficientnetb5')
@set_args(__args__)
def efficientnetb5(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return efficientnet(x, 1.6, 2.2, 456, is_training, classes, stem,
                        keep_prob=0.6, scope=scope, reuse=reuse)


@var_scope('efficientnetb6')
@set_args(__args__)
def efficientnetb6(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return efficientnet(x, 1.8, 2.6, 528, is_training, classes, stem,
                        keep_prob=0.5, scope=scope, reuse=reuse)


@var_scope('efficientnetb7')
@set_args(__args__)
def efficientnetb7(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return efficientnet(x, 2.0, 3.1, 600, is_training, classes, stem,
                        keep_prob=0.5, scope=scope, reuse=reuse)


# Simple alias.
EfficientNetB0 = efficientnetb0
EfficientNetB1 = efficientnetb1
EfficientNetB2 = efficientnetb2
EfficientNetB3 = efficientnetb3
EfficientNetB4 = efficientnetb4
EfficientNetB5 = efficientnetb5
EfficientNetB6 = efficientnetb6
EfficientNetB7 = efficientnetb7
