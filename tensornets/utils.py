from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from contextlib import contextmanager
from distutils.version import LooseVersion

from tensorflow.contrib.framework import arg_scope
from tensorflow.python.framework import ops

from .imagenet_utils import *
from .keras_utils import *
from .layers import conv2d


__outputs__ = 'outputs'


def print_collection(collection, scope):
    if scope is not None:
        print("Scope: %s" % scope)
    for x in tf.get_collection(collection, scope=scope):
        name = x.name
        if scope is not None:
            name = name[len(scope)+1:]
        print("%s %s" % (name, x.shape))


def parse_scopes(inputs):
    if not isinstance(inputs, list):
        inputs = [inputs]
    outputs = []
    for scope_or_tensor in inputs:
        if isinstance(scope_or_tensor, tf.Tensor):
            outputs.append(scope_or_tensor.aliases[0])
        elif isinstance(scope_or_tensor, str):
            outputs.append(scope_or_tensor)
        else:
            outputs.append(None)
    return outputs


def print_outputs(scopes=None):
    scopes = parse_scopes(scopes)
    for scope in scopes:
        print_collection(__outputs__, scope)


def print_weights(scopes=None):
    scopes = parse_scopes(scopes)
    for scope in scopes:
        print_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)


def print_summary(scopes=None):
    scopes = parse_scopes(scopes)
    for scope in scopes:
        if scope is not None:
            print("Scope: %s" % scope)
        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        names = [w.name for w in weights]
        starts = [n.rfind('/') + 1 for n in names]
        ends = [n.rfind(':') for n in names]

        layers = sum([n[s:e] == 'weights'
                      for (n, s, e) in zip(names, starts, ends)])
        parameters = sum([w.shape.num_elements() for w in weights])
        print("Total layers: %d" % layers)
        print("Total weights: %d" % len(weights))
        print("Total parameters: {:,}".format(parameters))


def get_outputs(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(__outputs__, scope=scope)


def get_weights(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def crop_idx(total_size, crop_size, crop_loc, crop_grid):
    if isinstance(total_size, int):
        total_size = (total_size, total_size)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if crop_loc > -1:
        row_loc = crop_loc // crop_grid[0]
        col_loc = crop_loc % crop_grid[1]
        row_start = row_loc * (total_size[0] - crop_size[0]) // 2
        col_start = col_loc * (total_size[1] - crop_size[1]) // 2
    else:
        row_start = np.random.randint(0, total_size[0] - crop_size[0], 1)[0]
        col_start = np.random.randint(0, total_size[1] - crop_size[1], 1)[0]
    return row_start, col_start


def crop(img, crop_size, crop_loc=4, crop_grid=(3, 3)):
    r, c = crop_idx(img.shape[1:3], crop_size, crop_loc, crop_grid)
    return img[:, r:r+crop_size, c:c+crop_size, :]


def init(scopes):
    sess = tf.get_default_session()
    assert sess is not None, 'The default session should be given.'

    if not isinstance(scopes, list):
        scopes = [scopes]

    for scope in scopes:
        sess.run(tf.variables_initializer(get_weights(scope)))


def var_scope(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            scope = kwargs.get('scope', None)
            reuse = kwargs.get('reuse', None)
            with tf.variable_scope(scope, name, reuse=reuse):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def ops_to_outputs(func):
    def wrapper(*args, **kwargs):
        x = func(*args, **kwargs)
        ops.add_to_collection(__outputs__, x)
        return x
    return wrapper


@contextmanager
def arg_scopes(l):
    for x in l:
        x.__enter__()
    yield


def set_args(layers, largs, conv_bias=True):
    def real_set_args(func):
        def wrapper(*args, **kwargs):
            is_training = kwargs.get('is_training', False)
            layers_args = [arg_scope(x, **y) for (x, y) in largs(is_training)]
            if not conv_bias:
                layers_args += [arg_scope([conv2d], biases_initializer=None)]
            with arg_scope(layers, outputs_collections=__outputs__):
                with arg_scopes(layers_args):
                    return func(*args, **kwargs)
        return wrapper
    return real_set_args


def load_weights(scopes, weights_path):
    sess = tf.get_default_session()
    assert sess is not None, 'The default session should be given.'

    scopes = parse_scopes(scopes)

    data = np.load(weights_path)
    values = data['values']

    if LooseVersion(tf.__version__) > LooseVersion('1.3.0'):
        for (i, name) in enumerate(data['names']):
            if '/beta' in name:
                values[i], values[i+1] = values[i+1], values[i]

    for scope in scopes:
        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(weights) == len(values), 'The sizes of symbolic and ' \
                                            'actual weights do not match.' \

        sess.run([w.assign(v) for (w, v) in zip(weights, values)])


def load_torch_weights(scopes, weights_path, move_rules=None):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        torch = None
    assert torch is not None, '`load_torch_weights` requires `torch`.'

    sess = tf.get_default_session()
    assert sess is not None, 'The default session should be given.'

    scopes = parse_scopes(scopes)

    model = torch.load(weights_path)
    names = model.keys()
    if move_rules is not None:
        if isinstance(move_rules, list):
            for (name, loc) in move_rules:
                idx = names.index(name)
                names.insert(idx + loc, names.pop(idx))

    for (i, name) in enumerate(names):
        if 'running_mean' in name:
            names[i-1], names[i-2] = names[i-2], names[i-1]

    values = []
    for name in names:
        val = model[name].numpy()
        if val.ndim == 4:
            val = np.transpose(val, [2, 3, 1, 0])
        if val.ndim == 2:
            val = np.transpose(val, [1, 0])
        if (val.ndim == 4) and (val.shape[3] // val.shape[2] == 32):
            values += np.split(val, 32, axis=3)
        else:
            values.append(val)

    for scope in scopes:
        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(weights) == len(values), 'The sizes of symbolic and ' \
                                            'actual weights do not match.' \

        sess.run([w.assign(v) for (w, v) in zip(weights, values)])


def remove_utils(module_name, exceptions):
    import sys
    from . import utils
    module = sys.modules[module_name]
    for util in dir(utils):
        if not ((util.startswith('_')) or (util in exceptions)):
            try:
                delattr(module, util)
            except:
                None
    delattr(module, 'keras_utils')
