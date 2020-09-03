from __future__ import absolute_import

import os
import tensorflow as tf

from .utils import ops_to_outputs
from .version_utils import tf_later_than


if tf_later_than('1.14'):
    tf = tf.compat.v1


try:
    reduce
except NameError:
    from functools import reduce


if tf_later_than('1.6'):
    # Note that `tf.nn.leaky_relu` has existed since 1.4.0,
    # but 1.4.0, 1.4.1, 1.5.0, 1.5.1 do not support float16.
    _leaky_relu = tf.nn.leaky_relu
else:
    def _leaky_relu(x, alpha=0.2, name=None):
        return tf.add(tf.nn.relu(x), -alpha * tf.nn.relu(-x), name=name)


TNETS_CUSTOM_SWISH = os.environ.get('TNETS_CUSTOM_SWISH', 'false').lower()
if TNETS_CUSTOM_SWISH != 'true':
    if TNETS_CUSTOM_SWISH != '1':
        TNETS_CUSTOM_SWISH = 'false'
    else:
        TNETS_CUSTOM_SWISH = 'true'

if tf_later_than('1.5') and not (TNETS_CUSTOM_SWISH == 'true'):
    # Note that `tf.nn.swish` has existed since 1.5.0.
    _swish = tf.nn.swish
else:
    def _swish(x, name=None):
        return tf.multiply(x, tf.sigmoid(x), name=name)


if tf_later_than('1.5'):
    # Note that `tf.reduce_mean` has existed since 1.0,
    # but the parameter name `keep_dims` has been changed to `keepdims`.
    _reduce_mean = tf.reduce_mean
else:
    def _reduce_mean(input_tensor, axis=None, keepdims=False, name=None):
        return tf.reduce_mean(input_tensor, axis=axis, keep_dims=keepdims,
                              name=name)


def _hard_sigmoid(x, name=None):
    return tf.divide(tf.nn.relu6(x + 3.), 6., name=name)


def _hard_swish(x, name=None):
    return tf.multiply(x, tf.nn.relu6(x + 3.) / 6., name=name)


argmax = ops_to_outputs(tf.argmax)
add = ops_to_outputs(tf.add)
concat = ops_to_outputs(tf.concat)
conv2d_primitive = ops_to_outputs(tf.nn.conv2d)
expand_dims = ops_to_outputs(tf.expand_dims)
gather = ops_to_outputs(tf.gather)
hard_sigmoid = ops_to_outputs(_hard_sigmoid)
hard_swish = ops_to_outputs(_hard_swish)
leaky_relu = ops_to_outputs(_leaky_relu)
lrn = ops_to_outputs(tf.nn.lrn)
maximum = ops_to_outputs(tf.maximum)
multiply = ops_to_outputs(tf.multiply)
one_hot = ops_to_outputs(tf.one_hot)
pad = ops_to_outputs(tf.pad)
reduce_max = ops_to_outputs(tf.reduce_max)
reduce_mean = ops_to_outputs(_reduce_mean)
reduce_sum = ops_to_outputs(tf.reduce_sum)
relu = ops_to_outputs(tf.nn.relu)
relu6 = ops_to_outputs(tf.nn.relu6)
reshape = ops_to_outputs(tf.reshape)
sigmoid = ops_to_outputs(tf.sigmoid)
softmax = ops_to_outputs(tf.nn.softmax)
sqrt = ops_to_outputs(tf.sqrt)
square = ops_to_outputs(tf.square)
squeeze = ops_to_outputs(tf.squeeze)
stack = ops_to_outputs(tf.stack)
swish = ops_to_outputs(_swish)
tanh = ops_to_outputs(tf.tanh)
to_int32 = ops_to_outputs(tf.to_int32)


@ops_to_outputs
def srn(x, depth_radius, alpha=1.0, beta=0.5, name=None):
    # Refer to the following code snippet
    # https://github.com/tensorflow/tensorflow/issues/1246#issuecomment-188588051
    squared_sum = tf.nn.depthwise_conv2d(
        tf.square(x),
        tf.ones([depth_radius] * 2 + [tf.shape(x)[3], 1], dtype=tf.float32),
        [1, 1, 1, 1],
        'SAME')
    alpha = tf.constant(alpha / (depth_radius ** 2), dtype=tf.float32)
    beta = tf.constant(beta, dtype=tf.float32)
    return tf.divide(x, (1.0 + alpha * squared_sum) ** beta, name=name)


@ops_to_outputs
def upsample(x, stride, name=None):
    if isinstance(stride, int):
        stride = (stride, stride)
    assert isinstance(stride, tuple)
    b = tf.shape(x)[0]
    h = tf.shape(x)[1] * stride[0]
    w = tf.shape(x)[2] * stride[1]
    c = int(x.shape[-1]) if tf_later_than('2') else x.shape[-1].value
    x = tf.expand_dims(x, 2)
    x = tf.expand_dims(x, 4)
    x = tf.tile(x, (1, 1, stride[0], 1, stride[1], 1))
    return tf.reshape(x, (b, h, w, c), name=name)


@ops_to_outputs
def local_flatten(x, kernel_size, name=None):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    assert isinstance(kernel_size, tuple)
    x = [[tf.strided_slice(x, (0, i, j), tf.shape(x)[:-1], (1,) + kernel_size)
          for j in range(kernel_size[1])] for i in range(kernel_size[0])]
    return tf.concat(reduce(lambda x, y: x + y, x), axis=-1, name=name)
