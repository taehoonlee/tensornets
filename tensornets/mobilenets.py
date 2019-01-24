"""Collection of MobileNet variants

The reference papers:

1. V1
 - MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, arXiv 2017
 - Andrew G. Howard et al.
 - https://arxiv.org/abs/1704.04861
2. V2
 - MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018 (arXiv 2018)
 - Mark Sandler et al.
 - https://arxiv.org/abs/1801.04381

The reference implementations:

1. (for v1) TF Slim
 - https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
2. (for v2) TF Slim
 - https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
"""
from __future__ import absolute_import

import tensorflow as tf

from .layers import batch_norm
from .layers import conv2d
from .layers import dropout
from .layers import fc
from .layers import separable_conv2d
from .layers import convbn
from .layers import convbnrelu6 as conv
from .layers import sconvbnrelu6 as sconv

from .ops import *
from .utils import set_args
from .utils import var_scope


def __base_args__(is_training, decay):
    return [([batch_norm], {'decay': decay, 'scale': True, 'epsilon': 0.001,
                            'is_training': is_training, 'scope': 'bn'}),
            ([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'biases_initializer': None, 'scope': 'conv'}),
            ([dropout], {'is_training': is_training, 'scope': 'dropout'}),
            ([fc], {'activation_fn': None, 'scope': 'fc'}),
            ([separable_conv2d],
             {'activation_fn': None, 'biases_initializer': None,
              'scope': 'sconv'})]


def __args__(is_training):
    return __base_args__(is_training, 0.9997)


def __args_v2__(is_training):
    return __base_args__(is_training, 0.999)


@var_scope('block')
def block(x, filters, stride=1, scope=None):
    x = sconv(x, None, 3, 1, stride=stride, scope='sconv')
    x = conv(x, filters, 1, stride=1, scope='conv')
    return x


@var_scope('blockv2')
def block2(x, filters, first=False, stride=1, scope=None):
    shortcut = x
    x = conv(x, 6 * x.shape[-1].value, 1, scope='conv')
    x = sconv(x, None, 3, 1, stride=stride, scope='sconv')
    x = convbn(x, filters, 1, stride=1, scope='pconv')
    if stride == 1 and shortcut.shape[-1].value == filters:
        return add(shortcut, x, name='out')
    else:
        return x


def mobilenet(x, depth_multiplier, is_training, classes, stem,
              scope=None, reuse=None):
    def depth(d):
        return max(int(d * depth_multiplier), 8)
    x = conv(x, depth(32), 3, stride=2, scope='conv1')

    x = block(x, depth(64), scope='conv2')
    x = block(x, depth(128), stride=2, scope='conv3')

    x = block(x, depth(128), scope='conv4')
    x = block(x, depth(256), stride=2, scope='conv5')

    x = block(x, depth(256), scope='conv6')
    x = block(x, depth(512), stride=2, scope='conv7')

    x = block(x, depth(512), scope='conv8')
    x = block(x, depth(512), scope='conv9')
    x = block(x, depth(512), scope='conv10')
    x = block(x, depth(512), scope='conv11')
    x = block(x, depth(512), scope='conv12')
    x = block(x, depth(1024), stride=2, scope='conv13')

    x = block(x, depth(1024), scope='conv14')
    if stem: return x

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = dropout(x, keep_prob=0.999, is_training=is_training, scope='dropout')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


def mobilenetv2(x, depth_multiplier, is_training, classes, stem,
                scope=None, reuse=None):
    def depth(d):
        d *= depth_multiplier
        filters = max(8, int(d + 4) // 8 * 8)
        if filters < 0.9 * d:
            filters += 8
        return filters
    x = conv(x, depth(32), 3, stride=2, scope='conv1')
    x = sconv(x, None, 3, 1, scope='sconv1')
    x = convbn(x, depth(16), 1, scope='pconv1')

    x = block2(x, depth(24), stride=2, scope='conv2')
    x = block2(x, depth(24), scope='conv3')

    x = block2(x, depth(32), stride=2, scope='conv4')
    x = block2(x, depth(32), scope='conv5')
    x = block2(x, depth(32), scope='conv6')

    x = block2(x, depth(64), stride=2, scope='conv7')
    x = block2(x, depth(64), scope='conv8')
    x = block2(x, depth(64), scope='conv9')
    x = block2(x, depth(64), scope='conv10')

    x = block2(x, depth(96), scope='conv11')
    x = block2(x, depth(96), scope='conv12')
    x = block2(x, depth(96), scope='conv13')

    x = block2(x, depth(160), stride=2, scope='conv14')
    x = block2(x, depth(160), scope='conv15')
    x = block2(x, depth(160), scope='conv16')

    x = block2(x, depth(320), scope='conv17')
    x = conv(x, 1280 * depth_multiplier if depth_multiplier > 1. else 1280, 1,
             scope='conv18')
    if stem: return x

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


@var_scope('mobilenet25')
@set_args(__args__)
def mobilenet25(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return mobilenet(x, 0.25, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet50')
@set_args(__args__)
def mobilenet50(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return mobilenet(x, 0.5, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet75')
@set_args(__args__)
def mobilenet75(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return mobilenet(x, 0.75, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet100')
@set_args(__args__)
def mobilenet100(x, is_training=False, classes=1000,
                 stem=False, scope=None, reuse=None):
    return mobilenet(x, 1.0, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet35v2')
@set_args(__args_v2__)
def mobilenet35v2(x, is_training=False, classes=1000,
                  stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 0.35, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet50v2')
@set_args(__args_v2__)
def mobilenet50v2(x, is_training=False, classes=1000,
                  stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 0.50, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet75v2')
@set_args(__args_v2__)
def mobilenet75v2(x, is_training=False, classes=1000,
                  stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 0.75, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet100v2')
@set_args(__args_v2__)
def mobilenet100v2(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 1.0, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet130v2')
@set_args(__args_v2__)
def mobilenet130v2(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 1.3, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet140v2')
@set_args(__args_v2__)
def mobilenet140v2(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 1.4, is_training, classes, stem, scope, reuse)


# Simple alias.
MobileNet25 = mobilenet25
MobileNet50 = mobilenet50
MobileNet75 = mobilenet75
MobileNet100 = mobilenet100
MobileNet35v2 = mobilenet35v2
MobileNet50v2 = mobilenet50v2
MobileNet75v2 = mobilenet75v2
MobileNet100v2 = mobilenet100v2
MobileNet130v2 = mobilenet130v2
MobileNet140v2 = mobilenet140v2
