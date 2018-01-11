from __future__ import absolute_import

import tensorflow as tf

from .utils import ops_to_outputs


argmax = ops_to_outputs(tf.argmax)
add = ops_to_outputs(tf.add)
concat = ops_to_outputs(tf.concat)
expand_dims = ops_to_outputs(tf.expand_dims)
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
to_int32 = ops_to_outputs(tf.to_int32)


@ops_to_outputs
def lrelu(x, alpha=0.3, name=None):
    return tf.add(tf.nn.relu(x), -alpha * tf.nn.relu(-x), name=name)
