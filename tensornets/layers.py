from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import bias_add
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.layers import l2_regularizer as l2
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import separable_conv2d
from tensorflow.contrib.layers import variance_scaling_initializer

from .ops import conv2d_primitive
from .ops import leaky_relu
from .ops import relu
from .ops import relu6
from .ops import reshape
from .utils import arg_scope
from .utils import remove_commons


conv1d = conv2d


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


def gconvbn(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        x = separable_conv2d(*args, **kwargs)
        c = args[-1]
        f = x.shape[-1].value // c
        g = f // c
        kernel = np.zeros((1, 1, f * c, f), np.float32)
        for i in range(f):
            start = (i // c) * c * c + i % c
            end = start + c * c
            kernel[:, :, start:end:c, i] = 1.
        x = conv2d_primitive(x, tf.constant(kernel), strides=[1, 1, 1, 1],
                             padding='VALID', name='gconv')
        return batch_norm(x)


def sconvbn(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return batch_norm(separable_conv2d(*args, **kwargs))


def sconvbnrelu6(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu6(batch_norm(separable_conv2d(*args, **kwargs)))


def darkconv(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    onlyconv = kwargs.pop('onlyconv', False)
    with tf.variable_scope(scope):
        conv_kwargs = {
            'padding': 'SAME',
            'activation_fn': None,
            'weights_initializer': variance_scaling_initializer(1.53846),
            'weights_regularizer': l2(5e-4),
            'biases_initializer': None,
            'scope': 'conv'}
        if onlyconv:
            conv_kwargs.pop('biases_initializer')
        with arg_scope([conv2d], **conv_kwargs):
            x = conv2d(*args, **kwargs)
            if onlyconv: return x
            x = batch_norm(x, decay=0.99, center=False, scale=True,
                           epsilon=1e-5, scope='bn')
            x = bias_add(x, scope='bias')
            x = leaky_relu(x, alpha=0.1, name='lrelu')
            return x


remove_commons(__name__)
