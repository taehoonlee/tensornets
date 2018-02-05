"""Collection of CapsuleNet variants

The reference paper:

 - Dynamic Routing Between Capsules
 - Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton
 - https://arxiv.org/abs/1710.09829

The reference implementations:

1. TensorFlow CapsNet
 - https://github.com/naturomics/CapsNet-Tensorflow
2. Keras CapsNet
 - https://github.com/XifengGuo/CapsNet-Keras
"""
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .layers import batch_norm
from .layers import conv2d
from .layers import convrelu as conv

from .ops import *
from .utils import ops_to_outputs
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([batch_norm], {'scale': True, 'is_training': is_training,
                            'epsilon': 1e-5, 'scope': 'bn'}),
            ([conv2d], {'padding': 'VALID', 'activation_fn': None,
                        'biases_initializer': None, 'scope': 'conv'})]


@ops_to_outputs
def squash(x, epsilon=1e-9, name=None):
    norm = tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True)
    scale = norm / (1. + norm) / tf.sqrt(norm + epsilon)
    return tf.multiply(x, scale, name=name)


@var_scope('primary')
def primary(x, filters, length, kernel_size, stride, scope=None):
    x = conv(x, filters * length, kernel_size, stride=stride, scope='conv')
    pixels = np.prod(x.shape[1:-1].as_list())
    x = reshape(x, (-1, pixels * filters, length), name='out')
    return x


@var_scope('digit')
def digit(x, filters, length, iters=3, scope=None):
    filters0 = x.shape[1].value
    length0 = x.shape[2].value

    # fully-connected weights between capsules: [1152, 8, 10 * 16]
    w = tf.get_variable('weights', shape=(filters0, length0, filters * length),
                        dtype=tf.float32)

    # coupling logits: [1152, 10]
    b = tf.zeros((filters0, filters))

    # prediction vectors: [None, 1152, 10, 16]
    uhat = tf.scan(lambda a, b: tf.matmul(b, w), tf.expand_dims(x, 2),
                   initializer=tf.zeros([filters0, 1, filters * length]))
    uhat = reshape(uhat, (-1, filters0, filters, length), name='predvec')

    for r in range(iters):
        with tf.variable_scope("iter%d" % r):
            # coupling coefficients: [1152, 10]
            c = softmax(b, name='softmax')
            # activity vector: [None, 10, 16]
            v = squash(tf.reduce_sum(uhat * tf.expand_dims(c, -1), axis=1),
                       name='out')
            # agreement: [None, 1152, 10]
            a = reduce_sum(tf.multiply(uhat, tf.expand_dims(v, 1)), axis=-1,
                           name='agreement')
            # updates coupling logits
            b = b + reduce_sum(a, axis=0, name='delta')
    return v


@var_scope('capsulenet')
@set_args(__args__)
def capsulenet_mnist(x, is_training=False, classes=10, scope=None, reuse=None):
    x = conv(x, 256, 9, stride=1, scope='conv1')
    x = primary(x, 32, 8, 9, stride=2, scope='primary')
    x = digit(x, 10, 16, scope='digit')
    return x


# Simple alias.
CapsuleNet = capsulenet_mnist
