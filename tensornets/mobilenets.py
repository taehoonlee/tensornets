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
3. V3
 - Searching for MobileNetV3, ICCV 2019
 - Andrew Howard et al.
 - https://arxiv.org/abs/1905.02244

The reference implementations:

1. (for v1) TF Slim
 - https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
2. (for v2) TF Slim
 - https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
3. (for v3) TF Slim
 - https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
"""
from __future__ import absolute_import

import functools
import tensorflow as tf

from .layers import avg_pool2d
from .layers import batch_norm
from .layers import conv2d
from .layers import dropout
from .layers import fc
from .layers import sconv2d
from .layers import convbn
from .layers import convbnact
from .layers import convbnrelu6 as conv
from .layers import sconvbn
from .layers import sconvbnact
from .layers import sconvbnrelu6 as sconv

from .ops import *
from .utils import set_args
from .utils import var_scope


def __base_args__(is_training, decay):
    return [([avg_pool2d], {'padding': 'VALID', 'scope': 'pool'}),
            ([batch_norm], {'decay': decay, 'scale': True, 'epsilon': 0.001,
                            'is_training': is_training, 'scope': 'bn'}),
            ([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'biases_initializer': None, 'scope': 'conv'}),
            ([dropout], {'is_training': is_training, 'scope': 'dropout'}),
            ([fc], {'activation_fn': None, 'scope': 'fc'}),
            ([sconv2d],
             {'activation_fn': None, 'biases_initializer': None,
              'scope': 'sconv'})]


def __args_v1__(is_training):
    return __base_args__(is_training, 0.9997)


def __args__(is_training):
    return __base_args__(is_training, 0.999)


def _depth(d, divisor=8):
    filters = max(divisor, int(d + divisor // 2) // divisor * divisor)
    if filters < 0.9 * d:
        filters += divisor
    return filters


@var_scope('block')
def block(x, filters, stride=1, scope=None):
    x = sconv(x, None, 3, 1, stride=stride, scope='sconv')
    x = conv(x, filters, 1, stride=1, scope='conv')
    return x


@var_scope('blockv2')
def block2(x, filters, stride=1, scope=None):
    shortcut = x
    infilters = int(x.shape[-1]) if tf_later_than('2') else x.shape[-1].value
    x = conv(x, 6 * infilters, 1, scope='conv')
    x = sconv(x, None, 3, 1, stride=stride, scope='sconv')
    x = convbn(x, filters, 1, stride=1, scope='pconv')
    if stride == 1 and infilters == filters:
        return add(shortcut, x, name='out')
    else:
        return x


@var_scope('seblock')
def seblock(i, se, filters, scope=None):
    x = reduce_mean(i, [1, 2], keepdims=True, name='squeeze')
    x = conv2d(x, _depth(se * filters), 1, activation_fn=relu,
               biases_initializer=tf.zeros_initializer(), scope='reduce')
    x = conv2d(x, filters, 1, activation_fn=hard_sigmoid,
               biases_initializer=tf.zeros_initializer(), scope='expand')
    x = multiply(i, x, name='excite')
    return x


@var_scope('blockv3')
def block3(x, ex, se, filters, kernel_size, stride=1,
           activation_fn=relu, scope=None):
    shortcut = x
    infilters = int(x.shape[-1]) if tf_later_than('2') else x.shape[-1].value
    if ex > 1:
        x = convbnact(x, _depth(ex * infilters), 1,
                      activation_fn=activation_fn, scope='conv')
    x = sconvbnact(x, None, kernel_size, 1, stride=stride,
                   activation_fn=activation_fn, scope='sconv')
    if 0 < se <= 1:
        x = seblock(x, se, _depth(ex * infilters), scope='se')
    x = convbn(x, filters, 1, stride=1, scope='pconv')
    if stride == 1 and infilters == filters:
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
        return _depth(d * depth_multiplier)
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


def mobilenetv3large(x, depth_multiplier, kernel_size, se, activation_fn,
                     is_training, classes, stem, scope=None, reuse=None):
    def depth(d):
        return _depth(d * depth_multiplier)
    conva = functools.partial(convbnact, activation_fn=activation_fn)
    block3a = functools.partial(block3, activation_fn=activation_fn)

    x = conva(x, depth(16), 3, stride=2, scope='conv1')
    x = block3(x, 1, 0, depth(16), 3, scope='conv2')
    x = block3(x, 4, 0, depth(24), 3, stride=2, scope='conv3')
    x = block3(x, 3, 0, depth(24), 3, scope='conv4')

    x = block3(x, 3, se, depth(40), kernel_size, stride=2, scope='conv5')
    x = block3(x, 3, se, depth(40), kernel_size, scope='conv6')
    x = block3(x, 3, se, depth(40), kernel_size, scope='conv7')

    x = block3a(x, 6, 0, depth(80), 3, stride=2, scope='conv8')
    x = block3a(x, 2.5, 0, depth(80), 3, scope='conv9')
    x = block3a(x, 2.3, 0, depth(80), 3, scope='conv10')
    x = block3a(x, 2.3, 0, depth(80), 3, scope='conv11')

    x = block3a(x, 6, se, depth(112), 3, scope='conv12')
    x = block3a(x, 6, se, depth(112), 3, scope='conv13')

    x = block3a(x, 6, se, depth(160), kernel_size, stride=2, scope='conv14')
    x = block3a(x, 6, se, depth(160), kernel_size, scope='conv15')
    x = block3a(x, 6, se, depth(160), kernel_size, scope='conv16')

    x = conva(x, depth(960), 1, scope='conv17')
    x = avg_pool2d(x, 7, scope='pool')
    x = conv2d(x, depth(1280) if depth_multiplier > 1. else 1280, 1,
               biases_initializer=tf.zeros_initializer(), scope='conv18')
    x = activation_fn(x, 'conv18/out')
    if stem: return x

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


def mobilenetv3small(x, depth_multiplier, kernel_size, se, activation_fn,
                     is_training, classes, stem, scope=None, reuse=None):
    def depth(d):
        return _depth(d * depth_multiplier)
    conva = functools.partial(convbnact, activation_fn=activation_fn)
    block3a = functools.partial(block3, activation_fn=activation_fn)

    x = conva(x, depth(16), 3, stride=2, scope='conv1')
    x = block3(x, 1, se, depth(16), 3, stride=2, scope='conv2')

    x = block3(x, 72./16, 0, depth(24), 3, stride=2, scope='conv3')
    x = block3(x, 88./24, 0, depth(24), 3, scope='conv4')

    x = block3a(x, 4, se, depth(40), kernel_size, stride=2, scope='conv5')
    x = block3a(x, 6, se, depth(40), kernel_size, scope='conv6')
    x = block3a(x, 6, se, depth(40), kernel_size, scope='conv7')

    x = block3a(x, 3, se, depth(48), kernel_size, scope='conv8')
    x = block3a(x, 3, se, depth(48), kernel_size, scope='conv9')

    x = block3a(x, 6, se, depth(96), kernel_size, stride=2, scope='conv10')
    x = block3a(x, 6, se, depth(96), kernel_size, scope='conv11')
    x = block3a(x, 6, se, depth(96), kernel_size, scope='conv12')

    x = conva(x, depth(576), 1, scope='conv13')
    x = avg_pool2d(x, 7, scope='pool')
    x = conv2d(x, depth(1024) if depth_multiplier > 1. else 1024, 1,
               biases_initializer=tf.zeros_initializer(), scope='conv14')
    x = activation_fn(x, 'conv14/out')
    if stem: return x

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


@var_scope('mobilenet25')
@set_args(__args_v1__)
def mobilenet25(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return mobilenet(x, 0.25, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet50')
@set_args(__args_v1__)
def mobilenet50(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return mobilenet(x, 0.5, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet75')
@set_args(__args_v1__)
def mobilenet75(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    return mobilenet(x, 0.75, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet100')
@set_args(__args_v1__)
def mobilenet100(x, is_training=False, classes=1000,
                 stem=False, scope=None, reuse=None):
    return mobilenet(x, 1.0, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet35v2')
@set_args(__args__)
def mobilenet35v2(x, is_training=False, classes=1000,
                  stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 0.35, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet50v2')
@set_args(__args__)
def mobilenet50v2(x, is_training=False, classes=1000,
                  stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 0.50, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet75v2')
@set_args(__args__)
def mobilenet75v2(x, is_training=False, classes=1000,
                  stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 0.75, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet100v2')
@set_args(__args__)
def mobilenet100v2(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 1.0, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet130v2')
@set_args(__args__)
def mobilenet130v2(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 1.3, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet140v2')
@set_args(__args__)
def mobilenet140v2(x, is_training=False, classes=1000,
                   stem=False, scope=None, reuse=None):
    return mobilenetv2(x, 1.4, is_training, classes, stem, scope, reuse)


@var_scope('mobilenet75v3large')
@set_args(__args__)
def mobilenet75v3large(x, is_training=False, classes=1000,
                       stem=False, scope=None, reuse=None):
    return mobilenetv3large(x, 0.75, 5, 0.25, hard_swish,
                            is_training, classes, stem, scope, reuse)


@var_scope('mobilenet100v3large')
@set_args(__args__)
def mobilenet100v3large(x, is_training=False, classes=1000,
                        stem=False, scope=None, reuse=None):
    return mobilenetv3large(x, 1.0, 5, 0.25, hard_swish,
                            is_training, classes, stem, scope, reuse)


@var_scope('mobilenet100v3largemini')
@set_args(__args__)
def mobilenet100v3largemini(x, is_training=False, classes=1000,
                            stem=False, scope=None, reuse=None):
    return mobilenetv3large(x, 1.0, 3, 0, relu,
                            is_training, classes, stem, scope, reuse)


@var_scope('mobilenet75v3small')
@set_args(__args__)
def mobilenet75v3small(x, is_training=False, classes=1000,
                       stem=False, scope=None, reuse=None):
    return mobilenetv3small(x, 0.75, 5, 0.25, hard_swish,
                            is_training, classes, stem, scope, reuse)


@var_scope('mobilenet100v3small')
@set_args(__args__)
def mobilenet100v3small(x, is_training=False, classes=1000,
                        stem=False, scope=None, reuse=None):
    return mobilenetv3small(x, 1.0, 5, 0.25, hard_swish,
                            is_training, classes, stem, scope, reuse)


@var_scope('mobilenet100v3smallmini')
@set_args(__args__)
def mobilenet100v3smallmini(x, is_training=False, classes=1000,
                            stem=False, scope=None, reuse=None):
    return mobilenetv3small(x, 1.0, 3, 0, relu,
                            is_training, classes, stem, scope, reuse)


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
MobileNet75v3 = MobileNet75v3large = mobilenet75v3large
MobileNet100v3 = MobileNet100v3large = mobilenet100v3large
MobileNet100v3largemini = mobilenet100v3largemini
MobileNet75v3small = mobilenet75v3small
MobileNet100v3small = mobilenet100v3small
MobileNet100v3smallmini = mobilenet100v3smallmini
