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
 - Aggregated Residual Transformations for Deep Neural Networks, CVPR 2017
 - Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He
 - https://arxiv.org/abs/1611.05431
4. WideResNet
 - Wide Residual Networks, BMVC 2016
 - Sergey Zagoruyko, Nikos Komodakis
 - https://arxiv.org/abs/1605.07146

The reference implementations:

1. (initially and mainly) Keras
 - https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py
2. (to reproduce the original results) Caffe ResNet
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

from .layers import batch_norm
from .layers import conv2d
from .layers import fc
from .layers import max_pool2d
from .layers import separable_conv2d
from .layers import convbn as conv
from .layers import gconvbn as gconv

from .ops import *
from .utils import pad_info
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([batch_norm], {'scale': True, 'is_training': is_training,
                            'epsilon': 1e-5, 'scope': 'bn'}),
            ([conv2d], {'padding': 'VALID', 'activation_fn': None,
                        'scope': 'conv'}),
            ([fc], {'activation_fn': None, 'scope': 'fc'}),
            ([max_pool2d], {'scope': 'pool'}),
            ([separable_conv2d], {'padding': 'VALID', 'activation_fn': None,
                                  'biases_initializer': None,
                                  'scope': 'sconv'})]


def resnet(x, preact, stack_fn, is_training, classes, stem,
           scope=None, reuse=None):
    x = pad(x, pad_info(7), name='conv1/pad')
    if preact:
        x = conv2d(x, 64, 7, stride=2, scope='conv1')
    else:
        x = conv(x, 64, 7, stride=2, scope='conv1')
        x = relu(x, name='conv1/relu')
    x = pad(x, pad_info(0 if stem else 3, symmetry=not preact),
            'SYMMETRIC', name='pool1/pad')
    x = max_pool2d(x, 3, stride=2, scope='pool1')
    x = stack_fn(x)
    if stem: return x
    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


@var_scope('resnet50')
@set_args(__args__)
def resnet50(x, is_training=False, classes=1000,
             stem=False, scope=None, reuse=None):
    def stack_fn(x):
        x = _stack(x, _block1, 64, 3, stride1=1, scope='conv2')
        x = _stack(x, _block1, 128, 4, scope='conv3')
        x = _stack(x, _block1, 256, 6, scope='conv4')
        x = _stack(x, _block1, 512, 3, scope='conv5')
        return x
    return resnet(x, False, stack_fn, is_training, classes, stem, scope, reuse)


@var_scope('resnet50v2')
@set_args(__args__)
def resnet50v2(x, is_training=False, classes=1000,
               stem=False, scope=None, reuse=None):
    def stack_fn(x):
        x = _stacks(x, 64, 3, scope='conv2')
        x = _stacks(x, 128, 4, scope='conv3')
        x = _stacks(x, 256, 6, scope='conv4')
        x = _stacks(x, 512, 3, stride1=1, scope='conv5')
        x = batch_norm(x, scope='postnorm')
        x = relu(x)
        return x
    return resnet(x, True, stack_fn, is_training, classes, stem, scope, reuse)


@var_scope('resnet101')
@set_args(__args__)
def resnet101(x, is_training=False, classes=1000,
              stem=False, scope=None, reuse=None):
    def stack_fn(x):
        x = _stack(x, _block1, 64, 3, stride1=1, scope='conv2')
        x = _stack(x, _block1, 128, 4, scope='conv3')
        x = _stack(x, _block1, 256, 23, scope='conv4')
        x = _stack(x, _block1, 512, 3, scope='conv5')
        return x
    return resnet(x, False, stack_fn, is_training, classes, stem, scope, reuse)


@var_scope('resnet101v2')
@set_args(__args__)
def resnet101v2(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    def stack_fn(x):
        x = _stacks(x, 64, 3, scope='conv2')
        x = _stacks(x, 128, 4, scope='conv3')
        x = _stacks(x, 256, 23, scope='conv4')
        x = _stacks(x, 512, 3, stride1=1, scope='conv5')
        x = batch_norm(x, scope='postnorm')
        x = relu(x)
        return x
    return resnet(x, True, stack_fn, is_training, classes, stem, scope, reuse)


@var_scope('resnet152')
@set_args(__args__)
def resnet152(x, is_training=False, classes=1000,
              stem=False, scope=None, reuse=None):
    def stack_fn(x):
        x = _stack(x, _block1, 64, 3, stride1=1, scope='conv2')
        x = _stack(x, _block1, 128, 8, scope='conv3')
        x = _stack(x, _block1, 256, 36, scope='conv4')
        x = _stack(x, _block1, 512, 3, scope='conv5')
        return x
    return resnet(x, False, stack_fn, is_training, classes, stem, scope, reuse)


@var_scope('resnet152v2')
@set_args(__args__)
def resnet152v2(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    def stack_fn(x):
        x = _stacks(x, 64, 3, scope='conv2')
        x = _stacks(x, 128, 8, scope='conv3')
        x = _stacks(x, 256, 36, scope='conv4')
        x = _stacks(x, 512, 3, stride1=1, scope='conv5')
        x = batch_norm(x, scope='postnorm')
        x = relu(x)
        return x
    return resnet(x, True, stack_fn, is_training, classes, stem, scope, reuse)


@var_scope('resnet200v2')
@set_args(__args__)
def resnet200v2(x, is_training=False, classes=1000,
                stem=False, scope=None, reuse=None):
    def stack_fn(x):
        x = _stack(x, _block2, 64, 3, stride1=1, scope='conv2')
        x = _stack(x, _block2, 128, 24, scope='conv3')
        x = _stack(x, _block2, 256, 36, scope='conv4')
        x = _stack(x, _block2, 512, 3, scope='conv5')
        x = batch_norm(x)
        x = relu(x)
        return x
    return resnet(x, False, stack_fn, is_training, classes, stem, scope, reuse)


@var_scope('resnext50c32')
@set_args(__args__, conv_bias=False)
def resnext50c32(x, is_training=False, classes=1000,
                 stem=False, scope=None, reuse=None):
    def stack_fn(x):
        def _block3c32(*args, **kwargs):
            kwargs.update({'groups': 32})
            return _block3(*args, **kwargs)
        x = _stack(x, _block3c32, 128, 3, stride1=1, scope='conv2')
        x = _stack(x, _block3c32, 256, 4, scope='conv3')
        x = _stack(x, _block3c32, 512, 6, scope='conv4')
        x = _stack(x, _block3c32, 1024, 3, scope='conv5')
        return x
    return resnet(x, False, stack_fn, is_training, classes, stem, scope, reuse)


@var_scope('resnext101c32')
@set_args(__args__, conv_bias=False)
def resnext101c32(x, is_training=False, classes=1000,
                  stem=False, scope=None, reuse=None):
    def stack_fn(x):
        def _block3c32(*args, **kwargs):
            kwargs.update({'groups': 32})
            return _block3(*args, **kwargs)
        x = _stack(x, _block3c32, 128, 3, stride1=1, scope='conv2')
        x = _stack(x, _block3c32, 256, 4, scope='conv3')
        x = _stack(x, _block3c32, 512, 23, scope='conv4')
        x = _stack(x, _block3c32, 1024, 3, scope='conv5')
        return x
    return resnet(x, False, stack_fn, is_training, classes, stem, scope, reuse)


@var_scope('resnext101c64')
@set_args(__args__, conv_bias=False)
def resnext101c64(x, is_training=False, classes=1000,
                  stem=False, scope=None, reuse=None):
    def stack_fn(x):
        def _block3c64(*args, **kwargs):
            kwargs.update({'groups': 64})
            return _block3(*args, **kwargs)
        x = _stack(x, _block3c64, 256, 3, stride1=1, scope='conv2')
        x = _stack(x, _block3c64, 512, 4, scope='conv3')
        x = _stack(x, _block3c64, 1024, 23, scope='conv4')
        x = _stack(x, _block3c64, 2048, 3, scope='conv5')
        return x
    return resnet(x, False, stack_fn, is_training, classes, stem, scope, reuse)


@var_scope('wideresnet50')
@set_args(__args__, conv_bias=False)
def wideresnet50(x, is_training=False, classes=1000,
                 stem=False, scope=None, reuse=None):
    def stack_fn(x):
        x = _stack(x, _blockw, 128, 3, stride1=1, scope='conv2')
        x = _stack(x, _blockw, 256, 4, scope='conv3')
        x = _stack(x, _blockw, 512, 6, scope='conv4')
        x = _stack(x, _blockw, 1024, 3, scope='conv5')
        return x
    return resnet(x, False, stack_fn, is_training, classes, stem, scope, reuse)


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
    x = pad(x, pad_info(kernel_size), name='2/pad')
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
    x = pad(x, pad_info(kernel_size), name='2/pad')
    x = conv(x, filters, kernel_size, stride=stride, biases_initializer=None,
             scope='2')
    x = relu(x, name='2/relu')
    x = conv2d(x, 4 * filters, 1, stride=1, scope='3/conv')
    x = add(shortcut, x, name='out')
    return x


@var_scope('block3')
def _block3(x, filters, kernel_size=3, stride=1, groups=32,
            conv_shortcut=True, scope=None):
    if conv_shortcut is True:
        shortcut = conv(x, (64 // groups) * filters, 1,
                        stride=stride, scope='0')
    else:
        shortcut = x
    x = conv(x, filters, 1, stride=1, scope='1')
    x = relu(x, name='1/relu')
    x = pad(x, pad_info(kernel_size), name='2/pad')
    x = gconv(x, None, kernel_size, filters // groups,
              stride=stride, scope='2')
    x = relu(x, name='2/relu')
    x = conv(x, (64 // groups) * filters, 1, stride=1, scope='3')
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
    x = pad(x, pad_info(kernel_size), name='2/pad')
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
ResNeXt50 = ResNeXt50c32 = resnext50c32
ResNeXt101 = ResNeXt101c32 = resnext101c32
ResNeXt101c64 = resnext101c64
WideResNet50 = wideresnet50
