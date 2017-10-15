"""Collection of Inception variants

The reference papers:

1. GoogLeNet, Inception (a.k.a. v1)
 - Going Deeper with Convolutions, CVPR 2015
 - Christian Szegedy et al.
 - https://arxiv.org/abs/1409.4842
2. BN-Inception (a.k.a. v2)
 - Batch Normalization: Accelerating Deep Network Training by Reducing Internal
   Covariate Shift, ICML 2015
 - Sergey Ioffe, Christian Szegedy
 - https://arxiv.org/abs/1502.03167
3. Inception3
 - Rethinking the Inception Architecture for Computer Vision, CVPR 2016
 - Christian Szegedy et al.
 - https://arxiv.org/abs/1512.00567
4. Inception4
 - Inception-v4, Inception-ResNet and the Impact of Residual Connections on
   Learning, AAAI 2017
 - Christian Szegedy et al.
 - https://arxiv.org/abs/1602.07261

The reference implementations:

1. (initially and mainly for v3) Keras
 - https://github.com/fchollet/keras/blob/master/keras/applications/
   inception_v3.py
2. (mainly for v1,2,4) TF Slim
 - https://github.com/tensorflow/models/blob/master/research/slim/nets/
   inception_v{1,2,3,4}.py
3. (to reproduce the original results) BAIR Caffe Model Zoo
 - https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/
   deploy.prototxt
"""
from __future__ import absolute_import

import tensorflow as tf

from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import separable_conv2d

from .ops import *
from .utils import arg_scope
from .utils import collect_outputs
from .utils import var_scope


__layers__ = [avg_pool2d, batch_norm, conv2d, dropout,
              fully_connected, max_pool2d, separable_conv2d]


def layers_common_args(func):
    def wrapper(*args, **kwargs):
        with collect_outputs(__layers__), \
             arg_scope([avg_pool2d], stride=1, padding='SAME', scope='pool'), \
             arg_scope([batch_norm], is_training=kwargs['is_training']), \
             arg_scope([conv2d], padding='SAME', activation_fn=None,
                       biases_initializer=None), \
             arg_scope([fully_connected], activation_fn=None), \
             arg_scope([max_pool2d], stride=1, padding='SAME', scope='pool'), \
             arg_scope([separable_conv2d], padding='SAME'):
            return func(*args, **kwargs)
    return wrapper


def conv0(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    kwargs['biases_initializer'] = tf.zeros_initializer()
    with tf.variable_scope(scope):
        return relu(conv2d(*args, **kwargs))


def conv(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return relu(batch_norm(conv2d(*args, **kwargs)))


@var_scope('inception1')
@layers_common_args
def inception1(x, is_training=True, classes=1000, scope=None, reuse=None):
    x = pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], name='pad')
    x = conv0(x, 64, 7, stride=2, padding='VALID', scope='block1')
    x = max_pool2d(x, 3, stride=2, scope='pool1')
    x = lrn(x, depth_radius=2, alpha=0.00002, beta=0.75, name='lrn1')

    x = conv0(x, 64, 1, scope='block2/3x3/r')
    x = conv0(x, 192, 3, scope='block2/3x3/1')
    x = lrn(x, depth_radius=2, alpha=0.00002, beta=0.75, name='lrn2')
    x = max_pool2d(x, 3, stride=2, scope='pool2')

    x = inception(x, [64, [96, 128], [16, 32], 32], scope='block3a')
    x = inception(x, [128, [128, 192], [32, 96], 64], scope='block3b')

    x = max_pool2d(x, 3, stride=2, scope='pool3')

    x = inception(x, [192, [96, 208], [16, 48], 64], scope='block4a')
    x = inception(x, [160, [112, 224], [24, 64], 64], scope='block4b')
    x = inception(x, [128, [128, 256], [24, 64], 64], scope='block4c')
    x = inception(x, [112, [144, 288], [32, 64], 64], scope='block4d')
    x = inception(x, [256, [160, 320], [32, 128], 128], scope='block4e')

    x = max_pool2d(x, 3, stride=2, scope='pool4')

    x = inception(x, [256, [160, 320], [32, 128], 128], scope='block5a')
    x = inception(x, [384, [192, 384], [48, 128], 128], scope='block5b')

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = dropout(x, keep_prob=0.8, is_training=is_training, scope='dropout')
    x = fully_connected(x, classes, scope='logits')
    x = softmax(x, name='probs')
    x.aliases = [tf.get_variable_scope().name]
    return x


@var_scope('inception2')
@layers_common_args
def inception2(x, is_training=True, classes=1000, scope=None, reuse=None):
    x = separable_conv2d(x, 64, 7, stride=2, depth_multiplier=8.,
                         activation_fn=None, scope='block1')
    x = max_pool2d(x, 3, stride=2, scope='pool1')

    x = conv(x, 64, 1, scope='block2/1')
    x = conv(x, 192, 3, scope='block2/2')
    x = max_pool2d(x, 3, stride=2, scope='pool2')

    x = inceptionA(x, [64, [64, 64], [64, 96], 32], scope='block3a')
    x = inceptionA(x, [64, [64, 96], [64, 96], 64], scope='block3b')

    x = reductionA(x, [[128, 160], [64, 96, 96]], padding='SAME',
                   scope='block3c')

    x = inceptionA(x, [224, [64, 96], [96, 128], 128], scope='block4a')
    x = inceptionA(x, [192, [96, 128], [96, 128], 128], scope='block4b')
    x = inceptionA(x, [160, [128, 160], [128, 160], 96], scope='block4c')
    x = inceptionA(x, [96, [128, 192], [160, 192], 96], scope='block4d')

    x = reductionA(x, [[128, 192], [192, 256, 256]], padding='SAME',
                   scope='block4e')

    x = inceptionA(x, [352, [192, 320], [160, 224], 128], scope='block5a')
    x = inceptionA(x, [352, [192, 320], [192, 224], 128],
                   pool_fn=max_pool2d, scope='block5b')

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = dropout(x, keep_prob=0.8, is_training=is_training, scope='dropout')
    x = fully_connected(x, classes, scope='logits')
    x = softmax(x, name='probs')
    x.aliases = [tf.get_variable_scope().name]
    return x


@var_scope('inception3')
@layers_common_args
def inception3(x, is_training=True, classes=1000, scope=None, reuse=None):
    x = conv(x, 32, 3, stride=2, padding='VALID', scope='block1a')
    x = conv(x, 32, 3, padding='VALID', scope='block2a')
    x = conv(x, 64, 3, scope='block2b')
    x = max_pool2d(x, 3, stride=2, padding='VALID', scope='pool3a')

    x = conv(x, 80, 1, padding='VALID', scope='block3b')
    x = conv(x, 192, 3, padding='VALID', scope='block4a')
    x = max_pool2d(x, 3, stride=2, padding='VALID', scope='pool5a')

    x = inceptionA(x, [64, [48, 64], [64, 96], 32], fs=5, scope='block5b')
    x = inceptionA(x, [64, [48, 64], [64, 96], 64], fs=5, scope='block5c')
    x = inceptionA(x, [64, [48, 64], [64, 96], 64], fs=5, scope='block5d')

    x = reductionA(x, [384, [64, 96, 96]], scope='block6a')

    x = inceptionB(x, [192, 128, 128, 192], scope='block6b')
    x = inceptionB(x, [192, 160, 160, 192], scope='block6c')
    x = inceptionB(x, [192, 160, 160, 192], scope='block6d')
    x = inceptionB(x, [192, 192, 192, 192], scope='block6e')

    x = reductionB(x, [[192, 320], 192], scope='block7a')

    x = inceptionC(x, [320, 384, [448, 384], 192], scope='block7b')
    x = inceptionC(x, [320, 384, [448, 384], 192], scope='block7c')

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fully_connected(x, classes, scope='logits')
    x = softmax(x, name='probs')
    x.aliases = [tf.get_variable_scope().name]
    return x


@var_scope('inception4')
@layers_common_args
def inception4(x, is_training=True, classes=1000, scope=None, reuse=None):
    x = conv(x, 32, 3, stride=2, padding='VALID', scope='block1a')

    x = conv(x, 32, 3, padding='VALID', scope='block2a')
    x = conv(x, 64, 3, scope='block2b')

    with tf.variable_scope('block3a'):
        x_1 = max_pool2d(x, 3, stride=2, padding='VALID', scope='pool')
        x_2 = conv(x, 96, 3, stride=2, padding='VALID', scope='conv')
        x = concat([x_1, x_2], axis=3, name='concat')

    with tf.variable_scope('block4a'):
        x_1 = conv(x, 64, 1, scope='1a')
        x_1 = conv(x_1, 96, 3, padding='VALID', scope='1b')
        x_2 = conv(x, 64, 1, scope='2a')
        x_2 = conv(x_2, 64, (1, 7), scope='2b')
        x_2 = conv(x_2, 64, (7, 1), scope='2c')
        x_2 = conv(x_2, 96, 3, padding='VALID', scope='2d')
        x = concat([x_1, x_2], axis=3, name='concat')

    with tf.variable_scope('block5a'):
        x_1 = conv(x, 192, 3, stride=2, padding='VALID', scope='conv')
        x_2 = max_pool2d(x, 3, stride=2, padding='VALID', scope='pool')
        x = concat([x_1, x_2], axis=3, name='concat')

    for i in range(4):
        x = inceptionA(x, [96, [64, 96], [64, 96], 96],
                       scope="block5%c" % (98 + i))

    x = reductionA(x, [384, [192, 224, 256]], scope='block6a')

    for i in range(7):
        x = inceptionB(x, [384, [192, 224, 256], [192, 224, 256], 128],
                       scope="block6%c" % (98 + i))

    x = reductionB(x, [[192, 192], [256, 320]], scope='block7a')

    for i in range(3):
        x = inceptionC(x, [256, [384, 256], [384, 448, 512, 256], 256],
                       scope="block7%c" % (98 + i))

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = dropout(x, keep_prob=0.8, is_training=is_training, scope='dropout')
    x = fully_connected(x, classes, scope='logits')
    x = softmax(x, name='probs')
    x.aliases = [tf.get_variable_scope().name]
    return x


@var_scope('inception')
def inception(x, filters, scope=None):
    conv1 = conv0(x, filters[0], 1, scope='1x1')

    conv2 = conv0(x, filters[1][0], 1, scope='3x3/r')
    conv2 = conv0(conv2, filters[1][1], 3, scope='3x3/1')

    conv3 = conv0(x, filters[2][0], 1, scope='5x5/r')
    conv3 = conv0(conv3, filters[2][1], 5, scope='5x5/1')

    pool = max_pool2d(x, 3)
    pool = conv0(pool, filters[3], 1, scope='proj')

    x = concat([conv1, conv2, conv3, pool], axis=3, name='concat')
    return x


@var_scope('inceptionA')
def inceptionA(x, filters, fs=3, pool_fn=avg_pool2d, scope=None):
    conv1 = conv(x, filters[0], 1, scope='1x1')

    conv2 = conv(x, filters[1][0], 1, scope='3x3/r')
    conv2 = conv(conv2, filters[1][1], fs, scope='3x3/1')

    conv3 = conv(x, filters[2][0], 1, scope='d3x3/r')
    conv3 = conv(conv3, filters[2][1], 3, scope='d3x3/1')
    conv3 = conv(conv3, filters[2][1], 3, scope='d3x3/2')

    pool = pool_fn(x, 3)
    pool = conv(pool, filters[3], 1, scope='proj')

    x = concat([conv1, conv2, conv3, pool], axis=3, name='concat')
    return x


@var_scope('reductionA')
def reductionA(x, filters, padding='VALID', scope=None):
    if padding == 'VALID':
        conv2 = conv(x, filters[0], 3, stride=2, padding=padding, scope='3x3')
    else:
        conv2 = conv(x, filters[0][0], 1, scope='3x3/r')
        conv2 = conv(conv2, filters[0][1], 3, stride=2, scope='3x3/1')

    f = filters[1]
    conv3 = conv(x, f[0], 1, scope='d3x3/r')
    conv3 = conv(conv3, f[1], 3, scope='d3x3/1')
    conv3 = conv(conv3, f[2], 3, stride=2, padding=padding, scope='d3x3/2')

    pool = max_pool2d(x, 3, stride=2, padding=padding)

    x = concat([conv2, conv3, pool], axis=3, name='concat')
    return x


@var_scope('inceptionB')
def inceptionB(x, filters, scope=None):
    conv1 = conv(x, filters[0], 1, scope='1x1')

    f = filters[1]
    if not isinstance(f, list):
        f = [f] * 2 + [filters[0]]
    conv2 = conv(x, f[0], 1, scope='7x7/r')
    conv2 = conv(conv2, f[1], (1, 7), scope='7x7/1')
    conv2 = conv(conv2, f[2], (7, 1), scope='7x7/2')

    f = filters[2]
    if isinstance(f, list):
        f = [f[0], f[0], f[1], f[1], f[2]]
    else:
        f = [f] * 4 + [filters[0]]
    conv3 = conv(x, f[0], 1, scope='d7x7/r')
    conv3 = conv(conv3, f[1], (7, 1), scope='d7x7/1')
    conv3 = conv(conv3, f[2], (1, 7), scope='d7x7/2')
    conv3 = conv(conv3, f[3], (7, 1), scope='d7x7/3')
    conv3 = conv(conv3, f[4], (1, 7), scope='d7x7/4')

    pool = avg_pool2d(x, 3)
    pool = conv(pool, filters[3], 1, scope='proj')

    x = concat([conv1, conv2, conv3, pool], axis=3, name='concat')
    return x


@var_scope('reductionB')
def reductionB(x, filters, scope=None):
    f = filters[0]
    conv2 = conv(x, f[0], 1, scope='3x3/r')
    conv2 = conv(conv2, f[1], 3, stride=2, padding='VALID', scope='3x3/1')

    f = filters[1]
    if isinstance(f, list):
        f = [f[0], f[0], f[1], f[1]]
    else:
        f = [f] * 4
    conv3 = conv(x, f[0], 1, scope='7x7/r')
    conv3 = conv(conv3, f[1], (1, 7), scope='7x7/1')
    conv3 = conv(conv3, f[2], (7, 1), scope='7x7/2')
    conv3 = conv(conv3, f[3], 3, stride=2, padding='VALID', scope='7x7/3')

    pool = max_pool2d(x, 3, stride=2, padding='VALID')

    x = concat([conv2, conv3, pool], axis=3, name='concat')
    return x


@var_scope('inceptionC')
def inceptionC(x, filters, scope=None):
    conv1 = conv(x, filters[0], 1, scope='1x1')

    f = filters[1]
    if isinstance(f, list):
        f = [f[0], f[1], f[1]]
    else:
        f = [f] * 3
    conv2 = conv(x, f[0], 1, scope='3x3/r')
    conv2_1 = conv(conv2, f[1], (1, 3), scope='3x3/1')
    conv2_2 = conv(conv2, f[2], (3, 1), scope='3x3/2')
    conv2 = concat([conv2_1, conv2_2], axis=3, name='3x3/c')

    conv3 = conv(x, filters[2][0], 1, scope='d3x3/r')
    if len(filters[2]) > 2:
        conv3 = conv(conv3, filters[2][1], (3, 1), scope='d3x3/11')
        conv3 = conv(conv3, filters[2][2], (1, 3), scope='d3x3/12')
        f = filters[2][3]
    else:
        f = filters[2][1]
        conv3 = conv(conv3, f, (3, 3), scope='d3x3/1')
    conv3_1 = conv(conv3, f, (1, 3), scope='d3x3/21')
    conv3_2 = conv(conv3, f, (3, 1), scope='d3x3/22')
    conv3 = concat([conv3_1, conv3_2], axis=3, name='d3x3/c')

    pool = avg_pool2d(x, 3)
    pool = conv(pool, filters[3], 1, scope='proj')

    x = concat([conv1, conv2, conv3, pool], axis=3, name='concat')
    return x


# Simple alias.
GoogLeNet = Inception1 = inception1
Inception2 = inception2
Inception3 = inception3
Inception4 = inception4
