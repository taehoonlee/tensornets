from __future__ import absolute_import

import tensorflow as tf

from .utils import ops_to_outputs
from .utils import __later_tf_version__


argmax = ops_to_outputs(tf.argmax)
add = ops_to_outputs(tf.add)
concat = ops_to_outputs(tf.concat)
expand_dims = ops_to_outputs(tf.expand_dims)
gather = ops_to_outputs(tf.gather)
lrn = ops_to_outputs(tf.nn.lrn)
maximum = ops_to_outputs(tf.maximum)
pad = ops_to_outputs(tf.pad)
reduce_mean = ops_to_outputs(tf.reduce_mean)
reduce_sum = ops_to_outputs(tf.reduce_sum)
relu = ops_to_outputs(tf.nn.relu)
relu6 = ops_to_outputs(tf.nn.relu6)
reshape = ops_to_outputs(tf.reshape)
softmax = ops_to_outputs(tf.nn.softmax)
sqrt = ops_to_outputs(tf.sqrt)
square = ops_to_outputs(tf.square)
squeeze = ops_to_outputs(tf.squeeze)
stack = ops_to_outputs(tf.stack)
to_int32 = ops_to_outputs(tf.to_int32)


if __later_tf_version__:
    leaky_relu = ops_to_outputs(tf.nn.leaky_relu)
else:
    @ops_to_outputs
    def leaky_relu(x, alpha=0.2, name=None):
        return tf.add(tf.nn.relu(x), -alpha * tf.nn.relu(-x), name=name)


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
