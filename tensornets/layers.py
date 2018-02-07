from __future__ import absolute_import

import tensorflow as tf

from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import separable_conv2d

from .ops import relu
from .ops import relu6
from .utils import remove_commons


def convbn(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return batch_norm(conv2d(*args, **kwargs))


def convrelu(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu(conv2d(*args, **kwargs))


def convrelu0(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    kwargs['biases_initializer'] = tf.zeros_initializer()
    with tf.variable_scope(scope):
        return relu(conv2d(*args, **kwargs))


def convbnrelu(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu(batch_norm(conv2d(*args, **kwargs)))


def convbnrelu6(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu6(batch_norm(conv2d(*args, **kwargs)))


def sconvbn(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return batch_norm(separable_conv2d(*args, **kwargs))


def sconvbnrelu6(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu6(batch_norm(separable_conv2d(*args, **kwargs)))


remove_commons(__name__)
