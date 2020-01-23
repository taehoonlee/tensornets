from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .ops import conv2d_primitive
from .ops import leaky_relu
from .ops import relu
from .ops import relu6
from .ops import reshape
from .ops import swish
from .utils import arg_scope
from .utils import remove_commons
from .version_utils import tf_later_than


if tf_later_than('1.14'):
    tf = tf.compat.v1


if tf_later_than('2'):
    from .contrib_layers import avg_pool2d
    from .contrib_layers import batch_norm
    from .contrib_layers import bias_add
    from .contrib_layers import conv2d
    from .contrib_layers import dropout
    from .contrib_layers import flatten
    from .contrib_layers import fully_connected as fc
    from .contrib_layers import l2_regularizer as l2
    from .contrib_layers import max_pool2d
    from .contrib_layers import separable_conv2d as sconv2d
    from .contrib_layers import variance_scaling_initializer
else:
    from tensorflow.contrib.layers import avg_pool2d
    from tensorflow.contrib.layers import batch_norm
    from tensorflow.contrib.layers import bias_add
    from tensorflow.contrib.layers import conv2d
    from tensorflow.contrib.layers import dropout
    from tensorflow.contrib.layers import flatten
    from tensorflow.contrib.layers import fully_connected as fc
    from tensorflow.contrib.layers import l2_regularizer as l2
    from tensorflow.contrib.layers import max_pool2d
    from tensorflow.contrib.layers import separable_conv2d as sconv2d
    from tensorflow.contrib.layers import variance_scaling_initializer


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


def convbnswish(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return swish(batch_norm(conv2d(*args, **kwargs)))


def gconvbn(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        x = sconv2d(*args, **kwargs)
        c = args[-1]
        infilters = int(x.shape[-1]) if tf_later_than('2') else x.shape[-1].value
        f = infilters // c
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
        return batch_norm(sconv2d(*args, **kwargs))


def sconvbnrelu(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu(batch_norm(sconv2d(*args, **kwargs)))


def sconvbnrelu6(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu6(batch_norm(sconv2d(*args, **kwargs)))


def sconvbnswish(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return swish(batch_norm(sconv2d(*args, **kwargs)))


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
