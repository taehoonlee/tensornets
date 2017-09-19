from __future__ import absolute_import

import tensorflow as tf

from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import max_pool2d

from .ops import *
from .utils import arg_scope
from .utils import collect_outputs
from .utils import get_file
from .utils import init
from .utils import load_keras_weights
from .utils import load_torch_weights
from .utils import var_scope


__keras_url__ = 'https://github.com/fchollet/deep-learning-models/' \
                'releases/download/v0.2/'
__model_url__ = 'https://github.com/taehoonlee/deep-learning-models/' \
                'releases/download/resnet/'
__layers__ = [batch_norm, conv2d, fully_connected, max_pool2d]


def layers_common_args(conv_bias):
    def real_layers_common_args(func):
        def wrapper(*args, **kwargs):
            b_kwargs = {'scale': True,
                        'is_training': kwargs['is_training'],
                        'scope': 'bn'}
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


@var_scope('block1')
def _block1(x, filters, kernel_size=3, stride=1,
            conv_shortcut=True, scope=None):
    if conv_shortcut is True:
        shortcut = conv(x, filters[2], 1, stride=stride, scope='0')
    else:
        shortcut = x
    x = conv(x, filters[0], 1, stride=stride, scope='1')
    x = relu(x, name='1/relu')
    x = conv(x, filters[1], kernel_size, stride=1, padding='SAME', scope='2')
    x = relu(x, name='2/relu')
    x = conv(x, filters[2], 1, stride=1, scope='3')
    x = relu(shortcut + x, name='out')
    return x


@var_scope('block2')
def _block2(x, filters, kernel_size=3, stride=1,
            conv_shortcut=True, scope=None):
    x = batch_norm(x)
    if conv_shortcut is True:
        shortcut = conv2d(x, filters[2], 1, stride=stride, scope='0/conv')
    else:
        shortcut = x
    x = conv(x, filters[0], 1, stride=stride, scope='1')
    x = relu(x, name='1/relu')
    x = conv(x, filters[1], kernel_size, stride=1, padding='SAME', scope='2')
    x = relu(x, name='2/relu')
    x = conv2d(x, filters[2], 1, stride=1, scope='3/conv')
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
    x = lrelu(x, name='1/lrelu')
    groups = []
    channels = int(filters / 32)
    for c in range(32):
        group = conv2d(x[:, :, :, c*channels:(c+1)*channels], channels, 3,
                       stride=stride, padding='SAME',
                       biases_initializer=None, scope="2/%d" % c)
        groups.append(group)
    x = concat(groups, axis=3, name='concat')
    x = batch_norm(x, scope='2/bn')
    x = lrelu(x, name='2/relu')
    x = conv(x, 2 * filters, 1, stride=1, padding='SAME', scope='3')
    x = lrelu(shortcut + x, name='out')
    return x


@var_scope('stack')
def _stack(x, block_fn, filters, blocks, stride1=2, scope=None):
    x = block_fn(x, filters, stride=stride1, scope='block1')
    for i in range(2, blocks+1):
        x = block_fn(x, filters, conv_shortcut=False, scope="block%d" % i)
    return x


def resnet(x, stack_fn, activation_fn, is_training, classes,
           scope=None, reuse=None):
    x = pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], name='pad')
    x = conv(x, 64, 7, stride=2, scope='conv1')
    x = relu(x, name='conv1/relu')
    x = max_pool2d(x, 3, stride=2, scope='pool1')
    x = stack_fn(x)
    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fully_connected(x, classes, scope='logits')
    x = softmax(x, name='probs')
    x.aliases = [tf.get_variable_scope().name]
    return x


@var_scope('resnet50')
@layers_common_args(True)
def resnet50(x, is_training=True, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block1, [64, 64, 256], 3, stride1=1, scope='conv2')
        x = _stack(x, _block1, [128, 128, 512], 4, scope='conv3')
        x = _stack(x, _block1, [256, 256, 1024], 6, scope='conv4')
        x = _stack(x, _block1, [512, 512, 2048], 3, scope='conv5')
        return x
    return resnet(x, stack, relu, is_training, classes, scope, reuse)


@var_scope('resnet50_v2')
@layers_common_args(True)
def resnet50_v2(x, is_training=True, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block2, [64, 64, 256], 3, stride1=1, scope='conv2')
        x = _stack(x, _block2, [128, 128, 512], 4, scope='conv3')
        x = _stack(x, _block2, [256, 256, 1024], 6, scope='conv4')
        x = _stack(x, _block2, [512, 512, 2048], 3, scope='conv5')
        x = batch_norm(x)
        x = relu(x)
        return x
    return resnet(x, stack, relu, is_training, classes, scope, reuse)


@var_scope('resnet101')
@layers_common_args(True)
def resnet101(x, is_training=True, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block1, [64, 64, 256], 3, stride1=1, scope='conv2')
        x = _stack(x, _block1, [128, 128, 512], 4, scope='conv3')
        x = _stack(x, _block1, [256, 256, 1024], 23, scope='conv4')
        x = _stack(x, _block1, [512, 512, 2048], 3, scope='conv5')
        return x
    return resnet(x, stack, relu, is_training, classes, scope, reuse)


@var_scope('resnet101_v2')
@layers_common_args(True)
def resnet101_v2(x, is_training=True, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block2, [64, 64, 256], 3, stride1=1, scope='conv2')
        x = _stack(x, _block2, [128, 128, 512], 4, scope='conv3')
        x = _stack(x, _block2, [256, 256, 1024], 23, scope='conv4')
        x = _stack(x, _block2, [512, 512, 2048], 3, scope='conv5')
        x = batch_norm(x)
        x = relu(x)
        return x
    return resnet(x, stack, relu, is_training, classes, scope, reuse)


@var_scope('resnet152')
@layers_common_args(True)
def resnet152(x, is_training=True, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block1, [64, 64, 256], 3, stride1=1, scope='conv2')
        x = _stack(x, _block1, [128, 128, 512], 8, scope='conv3')
        x = _stack(x, _block1, [256, 256, 1024], 36, scope='conv4')
        x = _stack(x, _block1, [512, 512, 2048], 3, scope='conv5')
        return x
    return resnet(x, stack, relu, is_training, classes, scope, reuse)


@var_scope('resnet152_v2')
@layers_common_args(True)
def resnet152_v2(x, is_training=True, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block2, [64, 64, 256], 3, stride1=1, scope='conv2')
        x = _stack(x, _block2, [128, 128, 512], 8, scope='conv3')
        x = _stack(x, _block2, [256, 256, 1024], 36, scope='conv4')
        x = _stack(x, _block2, [512, 512, 2048], 3, scope='conv5')
        x = batch_norm(x)
        x = relu(x)
        return x
    return resnet(x, stack, relu, is_training, classes, scope, reuse)


@var_scope('resnext50')
@layers_common_args(False)
def resnext50(x, is_training=True, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block3, 128, 3, stride1=1, scope='conv2')
        x = _stack(x, _block3, 256, 4, scope='conv3')
        x = _stack(x, _block3, 512, 6, scope='conv4')
        x = _stack(x, _block3, 1024, 3, scope='conv5')
        return x
    return resnet(x, stack, lrelu, is_training, classes, scope, reuse)


@var_scope('resnext101')
@layers_common_args(False)
def resnext101(x, is_training=True, classes=1000, scope=None, reuse=None):
    def stack(x):
        x = _stack(x, _block3, 128, 3, stride1=1, scope='conv2')
        x = _stack(x, _block3, 256, 4, scope='conv3')
        x = _stack(x, _block3, 512, 23, scope='conv4')
        x = _stack(x, _block3, 1024, 3, scope='conv5')
        return x
    return resnet(x, stack, lrelu, is_training, classes, scope, reuse)


def load_resnet50(scopes):
    filename = 'resnet50.h5'
    weights_path = get_file(
        filename, __model_url__ + filename,
        cache_subdir='models',
        md5_hash='9df0843bdadb58ed24d360564c45b119')
    return load_keras_weights(scopes, weights_path)


def load_resnet101(scopes):
    filename = 'resnet101.h5'
    weights_path = get_file(
        filename, __model_url__ + filename,
        cache_subdir='models',
        md5_hash='e2434bec605870fb4747e1b93f9f0e47')
    return load_keras_weights(scopes, weights_path)


def load_resnet152(scopes):
    filename = 'resnet152.h5'
    weights_path = get_file(
        filename, __model_url__ + filename,
        cache_subdir='models',
        md5_hash='e588285d1f919e538515c1f1b1c07b5b')
    return load_keras_weights(scopes, weights_path)


def load_keras_resnet50(scopes):
    filename = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file(
        filename, __keras_url__ + filename,
        cache_subdir='models',
        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    move_rules = [
        ('bn2a_branch2c', -1),
        ('bn3a_branch2c', -1),
        ('bn4a_branch2c', -1),
        ('bn5a_branch2c', -1),
        ('res2a_branch1', -6),
        ('res3a_branch1', -6),
        ('res4a_branch1', -6),
        ('res5a_branch1', -6),
        ('bn2a_branch1', -6),
        ('bn3a_branch1', -6),
        ('bn4a_branch1', -6),
        ('bn5a_branch1', -6),
    ]
    return load_keras_weights(scopes, weights_path, move_rules)


def load_torch_resnet101(scopes):
    weights_path = '/home/taehoonlee/Data/torch-ResNet/resnet_101_cpu.pth'
    move_rules = []
    for i in range(4, 8):
        move_rules.append(("%d.0.0.1.0.weight" % i, -18))
        move_rules.append(("%d.0.0.1.0.bias" % i, -18))
        move_rules.append(("%d.0.0.1.1.weight" % i, -18))
        move_rules.append(("%d.0.0.1.1.bias" % i, -18))
        move_rules.append(("%d.0.0.1.1.running_mean" % i, -18))
        move_rules.append(("%d.0.0.1.1.running_var" % i, -18))
    return load_torch_weights(scopes, weights_path, move_rules)


def preprocess(x):
    # Copied from keras
    x = x[:, :, :, ::-1]
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    return x


# Simple alias.
ResNet50 = resnet50
ResNet101 = resnet101
ResNet152 = resnet152
ResNet50v2 = resnet50_v2
ResNet101v2 = resnet101_v2
ResNet152v2 = resnet152_v2
ResNeXt50 = resnext50
ResNeXt101 = resnext101
load_resnet50v2 = init  # TODO
load_resnet101v2 = init  # TODO
load_resnet152v2 = init  # TODO
load_resnext50 = init  # TODO
load_resnext101 = init  # TODO
