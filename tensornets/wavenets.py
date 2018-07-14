"""Collection of WaveNet variants

The reference paper:

 - WaveNet: A Generative Model for Raw Audio, arXiv 2016
 - Aaron van den Oord et al.
 - https://arxiv.org/abs/1609.03499

The reference implementations:

1. (initially and mainly) @ibab's repository
 - https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/model.py
2. (to improve readability) @basveeling's repository
 - https://github.com/basveeling/wavenet/blob/master/wavenet.py
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .layers import conv1d

from .ops import *
from .utils import pad_info
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([conv1d], {'padding': 'VALID', 'activation_fn': None})]


@var_scope('block')
def block(x, filters, skipfilters, dilation, scope=None):
    x = tf.pad(x, [[0, 0], [dilation, 0], [0, 0]])
    f = conv1d(x, filters, 2, rate=dilation, scope='filter')
    g = conv1d(x, filters, 2, rate=dilation, scope='gate')
    o = tanh(f, name='filter/tanh') * sigmoid(g, name='gate/sigmoid')
    d = conv1d(o, filters, 1, scope='dense')
    s = conv1d(o, skipfilters, 1, scope='skip')
    return x[:, dilation:] + d, s


@var_scope('wavenet')
@set_args(__args__)
def wavenet(x, filters=32, skipfilters=512,
            quantization=256, blocks=10, repeats=5,
            is_training=False, scope=None, reuse=None):
    x = one_hot(x, quantization, name='one_hot')
    x = tf.pad(x, [[0, 0], [1, 0], [0, 0]])
    x = conv1d(x, filters, 2, biases_initializer=None, scope='embedding')

    skips = []
    for i in range(blocks * repeats):
        x, s = block(x, filters, skipfilters, 2 ** (i % blocks), scope=str(i))
        skips.append(s)

    x = relu(sum(skips), name='skips')
    x = conv1d(x, skipfilters, 1, scope='fc')
    x = relu(x, name='fc/relu')
    x = conv1d(x, quantization, 1, scope='logits')
    x = softmax(x, name='probs')
    return x


# Simple alias.
WaveNet = wavenet
