"""Collection of ResNet variants

The reference papers:

1. Original (a.k.a. v1)
 - Deep Residual Learning for Image Recognition, CVPR 2016 (Best Paper Award)
 - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
 - https://arxiv.org/abs/1512.03385
2. Pre-activation (a.k.a. v2)
 - Identity Mappings in Deep Residual Networks, ECCV 2016
 - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
 - https://arxiv.org/abs/1603.05027
3. ResNeXt
 - Aggregated Residual Transformations for Deep Neural Networks, arXiv 2016
 - Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He
 - https://arxiv.org/abs/1611.05431
4. WideResNet
 - Wide Residual Networks
 - Sergey Zagoruyko, Nikos Komodakis
 - https://arxiv.org/abs/1605.07146

The reference implementations:

1. (initially and mainly) Keras
 - https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
2. (to reproduce the oricginal results) Caffe ResNet
 - https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt
3. (to factorize over v2) Torch ResNets
 - https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua
4. (to factorize over v3) Torch ResNeXts
 - https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
5. (mainly) Torch WideResNets
 - https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import max_pool2d

from .ops import *
from .utils import arg_scope
from .utils import collect_outputs
from .utils import var_scope


__layers__ = [batch_norm, conv2d, fully_connected, max_pool2d]


def layers_common_args(conv_bias):
    def real_layers_common_args(func):
        def wrapper(*args, **kwargs):
            is_training = kwargs.get('is_training', False)
            b_kwargs = {'scale': True,
                        'is_training': is_training,
                        'scope': 'bn',
                        'epsilon': 1e-5}
            c_kwargs = {'padding': 'VALID',
                        'activation_fn': None,
                        'scope': 'conv'}
            f_kwargs = {'activation_fn': None}
            if conv_bias is False:
                c_kwargs['biases_initializer'] = None
            with collect_outputs(__layers__), \
                    arg_scope([batch_norm], **b_kwargs), \
                    arg_scope([conv2d], **c_kwargs), \
                    arg_scope([fully_connected], **f_kwargs):
                return func(*args, **kwargs)
        return wrapper
    return real_layers_common_args


def conv(*args, **kwargs):
    scope = kwargs.pop('scope', None)
    with tf.variable_scope(scope):
        return batch_norm(conv2d(*args, **kwargs))


def resnet(x, preact, stack_fn, is_training, classes, scope=None, reuse=None):
    x = pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], name='conv1/pad')
    if preact:
        x = conv2d(x, 64, 7, stride=2, scope='conv1')
    else:
        x = conv(x, 64, 7, stride=2, scope='conv1')
        x = relu(x, name='conv1/relu')
    x = pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pool1/pad')
    x = max_pool2d(x, 3, stride=2, scope='pool1')
    x = stack_fn(x)
    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fully_connected(x, classes, scope='logits')
    x = softmax(x, name='probs')
    x.aliases = [tf.get_variable_scope().name]
    return x


@var_scope('resnet50')
@layers_common_args(True)
def resnet50(x, is_training=False, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block1, 64, 3, stride1=1, scope='conv2')
        x = _stack(x, _block1, 128, 4, scope='conv3')
        x = _stack(x, _block1, 256, 6, scope='conv4')
        x = _stack(x, _block1, 512, 3, scope='conv5')
        return x
    return resnet(x, False, stack, is_training, classes, scope, reuse)


@var_scope('resnet50v2')
@layers_common_args(True)
def resnet50v2(x, is_training=False, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stacks(x, 64, 3, scope='conv2')
        x = _stacks(x, 128, 4, scope='conv3')
        x = _stacks(x, 256, 6, scope='conv4')
        x = _stacks(x, 512, 3, stride1=1, scope='conv5')
        x = batch_norm(x, scope='postnorm')
        x = relu(x)
        return x
    return resnet(x, True, stack, is_training, classes, scope, reuse)


@var_scope('resnet101')
@layers_common_args(True)
def resnet101(x, is_training=False, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block1, 64, 3, stride1=1, scope='conv2')
        x = _stack(x, _block1, 128, 4, scope='conv3')
        x = _stack(x, _block1, 256, 23, scope='conv4')
        x = _stack(x, _block1, 512, 3, scope='conv5')
        return x
    return resnet(x, False, stack, is_training, classes, scope, reuse)


@var_scope('resnet101v2')
@layers_common_args(True)
def resnet101v2(x, is_training=False, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stacks(x, 64, 3, scope='conv2')
        x = _stacks(x, 128, 4, scope='conv3')
        x = _stacks(x, 256, 23, scope='conv4')
        x = _stacks(x, 512, 3, stride1=1, scope='conv5')
        x = batch_norm(x, scope='postnorm')
        x = relu(x)
        return x
    return resnet(x, True, stack, is_training, classes, scope, reuse)


@var_scope('resnet152')
@layers_common_args(True)
def resnet152(x, is_training=False, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block1, 64, 3, stride1=1, scope='conv2')
        x = _stack(x, _block1, 128, 8, scope='conv3')
        x = _stack(x, _block1, 256, 36, scope='conv4')
        x = _stack(x, _block1, 512, 3, scope='conv5')
        return x
    return resnet(x, False, stack, is_training, classes, scope, reuse)


@var_scope('resnet152v2')
@layers_common_args(True)
def resnet152v2(x, is_training=False, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stacks(x, 64, 3, scope='conv2')
        x = _stacks(x, 128, 8, scope='conv3')
        x = _stacks(x, 256, 36, scope='conv4')
        x = _stacks(x, 512, 3, stride1=1, scope='conv5')
        x = batch_norm(x, scope='postnorm')
        x = relu(x)
        return x
    return resnet(x, True, stack, is_training, classes, scope, reuse)


@var_scope('resnet200v2')
@layers_common_args(True)
def resnet200v2(x, is_training=False, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block2, 64, 3, stride1=1, scope='conv2')
        x = _stack(x, _block2, 128, 24, scope='conv3')
        x = _stack(x, _block2, 256, 36, scope='conv4')
        x = _stack(x, _block2, 512, 3, scope='conv5')
        x = batch_norm(x)
        x = relu(x)
        return x
    return resnet(x, False, stack, is_training, classes, scope, reuse)


@var_scope('resnext50')
@layers_common_args(False)
def resnext50(x, is_training=False, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block3, 128, 3, stride1=1, scope='conv2')
        x = _stack(x, _block3, 256, 4, scope='conv3')
        x = _stack(x, _block3, 512, 6, scope='conv4')
        x = _stack(x, _block3, 1024, 3, scope='conv5')
        return x
    return resnet(x, False, stack, is_training, classes, scope, reuse)


@var_scope('resnext101')
@layers_common_args(False)
def resnext101(x, is_training=False, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block3, 128, 3, stride1=1, scope='conv2')
        x = _stack(x, _block3, 256, 4, scope='conv3')
        x = _stack(x, _block3, 512, 23, scope='conv4')
        x = _stack(x, _block3, 1024, 3, scope='conv5')
        return x
    return resnet(x, False, stack, is_training, classes, scope, reuse)


@var_scope('wideresnet50')
@layers_common_args(False)
def wideresnet50(x, is_training=False, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _blockw, 128, 3, stride1=1, scope='conv2')
        x = _stack(x, _blockw, 256, 4, scope='conv3')
        x = _stack(x, _blockw, 512, 6, scope='conv4')
        x = _stack(x, _blockw, 1024, 3, scope='conv5')
        return x
    return resnet(x, False, stack, is_training, classes, scope, reuse)


@var_scope('stack')
def _stack(x, block_fn, filters, blocks, stride1=2, scope=None):
    x = block_fn(x, filters, stride=stride1, scope='block1')
    for i in range(2, blocks+1):
        x = block_fn(x, filters, conv_shortcut=False, scope="block%d" % i)
    return x


@var_scope('stack_tfslim')
def _stacks(x, filters, blocks, stride1=2, scope=None):
    x = _block2s(x, filters, conv_shortcut=True, scope='block1')
    for i in range(2, blocks):
        x = _block2s(x, filters, scope="block%d" % i)
    x = _block2s(x, filters, stride=stride1, scope="block%d" % blocks)
    return x


@var_scope('block1')
def _block1(x, filters, kernel_size=3, stride=1,
            conv_shortcut=True, scope=None):
    if conv_shortcut is True:
        shortcut = conv(x, 4 * filters, 1, stride=stride, scope='0')
    else:
        shortcut = x
    # Most reference implementations (e.g., TF-slim and Torch-ResNets)
    # apply a stride of 2 on the 3x3 conv kernel like the below `_block2`,
    # but here the stride 2 on the 1x1 to follow the original Caffe-ResNets.
    x = conv(x, filters, 1, stride=stride, scope='1')
    x = relu(x, name='1/relu')
    x = conv(x, filters, kernel_size, stride=1, padding='SAME', scope='2')
    x = relu(x, name='2/relu')
    x = conv(x, 4 * filters, 1, stride=1, scope='3')
    x = relu(shortcut + x, name='out')
    return x


@var_scope('block2')
def _block2(x, filters, kernel_size=3, stride=1,
            conv_shortcut=True, scope=None):
    if conv_shortcut is True:
        shortcut = conv(x, 4 * filters, 1, stride=stride, scope='0')
    else:
        shortcut = x
    x = batch_norm(x)
    x = relu(x)
    x = conv(x, filters, 1, stride=1, scope='1')
    x = relu(x, name='1/relu')
    x = pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], name='2/pad')
    x = conv(x, filters, kernel_size, stride=stride, scope='2')
    x = relu(x, name='2/relu')
    x = conv2d(x, 4 * filters, 1, stride=1, scope='3/conv')
    x = add(shortcut, x, name='out')
    return x


@var_scope('block2_tfslim')
def _block2s(x, filters, kernel_size=3, stride=1,
             conv_shortcut=False, scope=None):
    preact = batch_norm(x, scope='preact')
    preact = relu(preact)
    if conv_shortcut is True:
        shortcut = conv2d(preact, 4 * filters, 1, stride=stride, scope='0')
    else:
        shortcut = max_pool2d(x, 1, stride, scope='0') if stride > 1 else x
    x = conv(preact, filters, 1, stride=1, biases_initializer=None, scope='1')
    x = relu(x, name='1/relu')
    x = pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], name='2/pad')
    x = conv(x, filters, kernel_size, stride=stride, biases_initializer=None,
             scope='2')
    x = relu(x, name='2/relu')
    x = conv2d(x, 4 * filters, 1, stride=1, scope='3/conv')
    x = add(shortcut, x, name='out')
    return x


@var_scope('block3')
def _block3(x, filters, kernel_size=3, stride=1,
            conv_shortcut=True, scope=None):
    if conv_shortcut is True:
        shortcut = conv(x, 2 * filters, 1, stride=stride, scope='0')
    else:
        shortcut = x
    x = conv(x, filters, 1, stride=1, scope='1')
    x = relu(x, name='1/relu')
    x = pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], name='2/pad')
    groups = []
    channels = int(filters / 32)
    for c in range(32):
        group = conv2d(x[:, :, :, c*channels:(c+1)*channels], channels,
                       kernel_size, stride=stride,
                       biases_initializer=None, scope="2/%d" % c)
        groups.append(group)
    x = concat(groups, axis=3, name='concat')
    x = batch_norm(x, scope='2/bn')
    x = relu(x, name='2/relu')
    x = conv(x, 2 * filters, 1, stride=1, scope='3')
    x = relu(shortcut + x, name='out')
    return x


@var_scope('blockw')
def _blockw(x, filters, kernel_size=3, stride=1,
            conv_shortcut=True, scope=None):
    if conv_shortcut is True:
        shortcut = conv(x, 2 * filters, 1, stride=stride, scope='0')
    else:
        shortcut = x
    x = conv(x, filters, 1, stride=1, scope='1')
    x = relu(x, name='1/relu')
    x = pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], name='2/pad')
    x = conv(x, filters, kernel_size, stride=stride, scope='2')
    x = relu(x, name='2/relu')
    x = conv(x, 2 * filters, 1, stride=1, scope='3')
    x = relu(shortcut + x, name='out')
    return x


# Simple alias.
ResNet50 = resnet50
ResNet101 = resnet101
ResNet152 = resnet152
ResNet50v2 = resnet50v2
ResNet101v2 = resnet101v2
ResNet152v2 = resnet152v2
ResNet200v2 = resnet200v2
ResNeXt50 = resnext50
ResNeXt101 = resnext101
WideResNet50 = wideresnet50
