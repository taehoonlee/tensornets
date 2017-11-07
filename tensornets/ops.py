from __future__ import absolute_import

import tensorflow as tf

from .utils import ops_to_outputs


add = ops_to_outputs(tf.add)
concat = ops_to_outputs(tf.concat)
lrn = ops_to_outputs(tf.nn.lrn)
pad = ops_to_outputs(tf.pad)
reduce_mean = ops_to_outputs(tf.reduce_mean)
relu = ops_to_outputs(tf.nn.relu)
relu6 = ops_to_outputs(tf.nn.relu6)
softmax = ops_to_outputs(tf.nn.softmax)
squeeze = ops_to_outputs(tf.squeeze)


@ops_to_outputs
def lrelu(x, alpha=0.3, name=None):
    return tf.add(tf.nn.relu(x), -alpha * tf.nn.relu(-x), name=name)
