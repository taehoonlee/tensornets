from __future__ import absolute_import

from .utils import get_file
from .utils import init
from .utils import load_keras_weights
from .utils import load_torch_weights


__keras_url__ = 'https://github.com/fchollet/deep-learning-models/' \
                'releases/download/v0.2/'
__model_url__ = 'https://github.com/taehoonlee/deep-learning-models/' \
                'releases/download/'


def load_inception1(scopes):
    # Converted from the BAIR's Caffe Model Zoo as a Keras format
    # Refer to https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
    filename = 'inception1.h5'
    weights_path = get_file(
        filename, __model_url__ + 'inception/' + filename,
        cache_subdir='models',
        md5_hash='6a212e3cb60b33f49c372906f18ae4a8')
    return load_keras_weights(scopes, weights_path)


def load_inception3(scopes):
    # Copied from keras with a slight modification due to the order of weights
    # Refer to https://github.com/fchollet/keras/tree/master/keras/applications
    filename = 'inception3.h5'
    weights_path = get_file(
        filename, __model_url__ + 'inception/' + filename,
        cache_subdir='models',
        md5_hash='7c4556613c348da3b99b633e1c430fff')
    return load_keras_weights(scopes, weights_path)


def load_resnet50(scopes):
    # Converted from the original Caffe files as a Keras format
    # Refer to https://github.com/KaimingHe/deep-residual-networks
    filename = 'resnet50.h5'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        md5_hash='9df0843bdadb58ed24d360564c45b119')
    return load_keras_weights(scopes, weights_path)


def load_resnet101(scopes):
    # Converted from the original Caffe files as a Keras format
    # Refer to https://github.com/KaimingHe/deep-residual-networks
    filename = 'resnet101.h5'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        md5_hash='e2434bec605870fb4747e1b93f9f0e47')
    return load_keras_weights(scopes, weights_path)


def load_resnet152(scopes):
    # Converted from the original Caffe files as a Keras format
    # Refer to https://github.com/KaimingHe/deep-residual-networks
    filename = 'resnet152.h5'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        md5_hash='e588285d1f919e538515c1f1b1c07b5b')
    return load_keras_weights(scopes, weights_path)


def load_keras_resnet50(scopes):
    # Copied from keras
    # Refer to https://github.com/fchollet/keras/tree/master/keras/applications
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


# Simple alias.
load_inception2 = init  # TODO
load_inception4 = init  # TODO

load_resnet50v2 = init  # TODO
load_resnet101v2 = init  # TODO
load_resnet152v2 = init  # TODO
load_resnext50 = init  # TODO
load_resnext101 = init  # TODO
