from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import warnings

from contextlib import contextmanager

from .version_utils import tf_later_than


try:
    import cv2
except ImportError:
    cv2 = None


__middles__ = 'middles'
__outputs__ = 'outputs'


if tf_later_than('1.14'):
    tf = tf.compat.v1


if tf_later_than('2'):
    from .contrib_framework import arg_scope
    from .contrib_layers.utils import collect_named_outputs
else:
    from tensorflow.contrib.framework import arg_scope
    from tensorflow.contrib.layers.python.layers.utils import collect_named_outputs


if tf_later_than('2.1'):
    from tensorflow.python.keras.applications.imagenet_utils \
        import decode_predictions
    from tensorflow.python.keras.utils.data_utils import get_file
elif tf_later_than('1.8.0'):
    from tensorflow.python.keras.applications.imagenet_utils \
        import decode_predictions
    from tensorflow.python.keras.utils import get_file
elif tf_later_than('1.3.0'):
    from tensorflow.python.keras._impl.keras.applications.imagenet_utils \
        import decode_predictions
    from tensorflow.python.keras.utils import get_file
else:
    from tensorflow.contrib.keras.python.keras.applications.imagenet_utils \
        import decode_predictions
    from tensorflow.contrib.keras.python.keras.utils.data_utils \
        import get_file


def print_collection(collection, scope):
    if scope is not None:
        print("Scope: %s" % scope)
    for x in tf.get_collection(collection, scope=scope + '/'):
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


def print_middles(scopes=None):
    scopes = parse_scopes(scopes)
    for scope in scopes:
        print_collection(__middles__, scope)


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
        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope=scope + '/')
        names = [w.name for w in weights]
        starts = [n.rfind('/') + 1 for n in names]
        ends = [n.rfind(':') for n in names]

        layers = sum([n[s:e] == 'weights'
                      for (n, s, e) in zip(names, starts, ends)])
        parameters = sum([w.shape.num_elements() for w in weights])
        print("Total layers: %d" % layers)
        print("Total weights: %d" % len(weights))
        print("Total parameters: {:,}".format(parameters))


def get_bottleneck(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(__middles__, scope=scope + '/')[-1]


def get_middles(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(__middles__, scope=scope + '/')


def get_outputs(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(__outputs__, scope=scope + '/')


def get_weights(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/')


def pad_info(s, symmetry=True):
    pads = [[0, 0], [s // 2, s // 2], [s // 2, s // 2], [0, 0]]
    if not symmetry:
        pads[1][0] -= 1
        pads[2][0] -= 1
    return pads


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
    if isinstance(crop_loc, list):
        imgs = np.zeros((img.shape[0], len(crop_loc), crop_size, crop_size, 3),
                        np.float32)
        for (i, loc) in enumerate(crop_loc):
            r, c = crop_idx(img.shape[1:3], crop_size, loc, crop_grid)
            imgs[:, i] = img[:, r:r+crop_size, c:c+crop_size, :]
        return imgs
    elif crop_loc == np.prod(crop_grid) + 1:
        imgs = np.zeros((img.shape[0], crop_loc, crop_size, crop_size, 3),
                        np.float32)
        r, c = crop_idx(img.shape[1:3], crop_size, 4, crop_grid)
        imgs[:, 0] = img[:, r:r+crop_size, c:c+crop_size, :]
        imgs[:, 1] = img[:, 0:crop_size, 0:crop_size, :]
        imgs[:, 2] = img[:, 0:crop_size, -crop_size:, :]
        imgs[:, 3] = img[:, -crop_size:, 0:crop_size, :]
        imgs[:, 4] = img[:, -crop_size:, -crop_size:, :]
        imgs[:, 5:] = np.flip(imgs[:, :5], axis=3)
        return imgs
    else:
        r, c = crop_idx(img.shape[1:3], crop_size, crop_loc, crop_grid)
        return img[:, r:r+crop_size, c:c+crop_size, :]


def load_img(paths, grayscale=False, target_size=None, crop_size=None,
             interp=None):
    assert cv2 is not None, '`load_img` requires `cv2`.'
    if interp is None:
        interp = cv2.INTER_CUBIC
    if not isinstance(paths, list):
        paths = [paths]
    if len(paths) > 1 and (target_size is None or
                           isinstance(target_size, int)):
        raise ValueError('A tuple `target_size` should be provided '
                         'when loading multiple images.')

    def _load_img(path):
        img = cv2.imread(path)
        if target_size:
            if isinstance(target_size, int):
                hw_tuple = tuple([x * target_size // min(img.shape[:2])
                                  for x in img.shape[1::-1]])
            else:
                hw_tuple = (target_size[1], target_size[0])
            if img.shape[1::-1] != hw_tuple:
                img = cv2.resize(img, hw_tuple, interpolation=interp)
        img = img[:, :, ::-1]
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        return img

    if len(paths) > 1:
        imgs = np.zeros((len(paths),) + target_size + (3,), dtype=np.float32)
        for (i, path) in enumerate(paths):
            imgs[i] = _load_img(path)
    else:
        imgs = np.array([_load_img(paths[0])], dtype=np.float32)

    if crop_size is not None:
        imgs = crop(imgs, crop_size)

    return imgs


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
            stem = kwargs.get('stem', False)
            scope = kwargs.get('scope', name)
            reuse = kwargs.get('reuse', None)
            with tf.variable_scope(scope, reuse=reuse):
                x = func(*args, **kwargs)
                if func.__name__ == 'wrapper':
                    from .middles import direct as p0
                    from .preprocess import direct as p1
                    from .pretrained import direct as p2
                    _scope = tf.get_variable_scope().name
                    if tf_later_than('1.1.0'):
                        _name = tf.get_default_graph().get_name_scope()
                    else:
                        # Note that `get_middles` and `get_outputs`
                        # may NOT work well for TensorFlow == 1.1.0.
                        _name = _scope
                    if tf_later_than('2'):
                        _input_shape = tuple(args[0].shape[1:3])
                    else:
                        _input_shape = tuple([i.value for i in args[0].shape[1:3]])
                    _outs = get_outputs(_name)
                    for i in p0(name)[0]:
                        collect_named_outputs(__middles__, _scope, _outs[i])
                    if stem:
                        x.aliases.insert(0, _scope)
                        x.p = get_middles(_name)[p0(name)[2]]
                    else:
                        x.logits = get_outputs(_name)[-2]
                    setattr(x, 'preprocess', p1(name, _input_shape))
                    setattr(x, 'pretrained', p2(name, x))
                    setattr(x, 'get_bottleneck',
                            lambda: get_bottleneck(_scope))
                    setattr(x, 'get_middles', lambda: get_middles(_name))
                    setattr(x, 'get_outputs', lambda: get_outputs(_name))
                    setattr(x, 'get_weights', lambda: get_weights(_scope))
                    setattr(x, 'middles', lambda: get_middles(_name))
                    setattr(x, 'outputs', lambda: get_outputs(_name))
                    setattr(x, 'weights', lambda: get_weights(_scope))
                    setattr(x, 'summary', lambda: print_summary(_scope))
                    setattr(x, 'print_middles', lambda: print_middles(_name))
                    setattr(x, 'print_outputs', lambda: print_outputs(_name))
                    setattr(x, 'print_weights', lambda: print_weights(_scope))
                    setattr(x, 'print_summary', lambda: print_summary(_scope))
                return x
        return wrapper
    return decorator


def ops_to_outputs(func):
    def wrapper(*args, **kwargs):
        x = func(*args, **kwargs)
        x = collect_named_outputs(__outputs__, tf.get_variable_scope().name, x)
        return x
    return wrapper


@contextmanager
def arg_scopes(l):
    for x in l:
        x.__enter__()
    yield


def set_args(largs, conv_bias=True, weights_regularizer=None):
    from .layers import conv2d
    from .layers import fc
    from .layers import sconv2d

    def real_set_args(func):
        def wrapper(*args, **kwargs):
            is_training = kwargs.get('is_training', False)
            layers = sum([x for (x, y) in largs(is_training)], [])
            layers_args = [arg_scope(x, **y) for (x, y) in largs(is_training)]
            if not conv_bias:
                layers_args += [arg_scope([conv2d], biases_initializer=None)]
            if weights_regularizer is not None:
                layers_args += [arg_scope(
                    [conv2d, fc, sconv2d],
                    weights_regularizer=weights_regularizer)]
            with arg_scope(layers, outputs_collections=__outputs__):
                with arg_scopes(layers_args):
                    x = func(*args, **kwargs)
                    x.model_name = func.__name__
                    return x
        return wrapper
    return real_set_args


def pretrained_initializer(scope, values):
    weights = get_weights(scope)

    if values is None:
        return tf.variables_initializer(weights)

    if len(weights) > len(values):  # excluding weights in Optimizer
        weights = weights[:len(values)]

    if len(weights) != len(values):
        values = values[:len(weights)]
        warnings.warn('The sizes of symbolic and actual weights do not match. '
                      'Never mind if you are trying to load stem layers only.')

    if scope.dtype == tf.float16:
        ops = [weights[0].assign(np.asarray(values[0], dtype=np.float16))]
        for (w, v) in zip(weights[1:-2], values[1:-2]):
            w.load(np.asarray(v, dtype=np.float16))
        if weights[-1].shape != values[-1].shape:
            ops += [w.initializer for w in weights[-2:]]
        else:
            for (w, v) in zip(weights[-2:], values[-2:]):
                w.load(np.asarray(v, dtype=np.float16))
        return ops

    ops = [w.assign(v) for (w, v) in zip(weights[:-2], values[:-2])]
    if weights[-1].shape != values[-1].shape:  # for transfer learning
        ops += [w.initializer for w in weights[-2:]]
    else:
        # The logits layer can be either 1x1 conv or fc. In other words,
        # the weight shape is (1, 1, features, classes) for the former,
        # or (features, classes) the latter.
        if weights[-2].shape != values[-2].shape:
            values[-2] = values[-2].reshape(weights[-2].shape)
            warnings.warn('The weight has been reshaped because 1x1 conv and '
                          'fc layers are interchangeable for a logits layer. '
                          'But, the conversion may affect the precision.')
        ops += [w.assign(v) for (w, v) in zip(weights[-2:], values[-2:])]

    return ops


def parse_weights(weights_path, move_rules=None):
    data = np.load(weights_path, encoding='bytes', allow_pickle=True)
    values = data['values']

    if tf_later_than('1.3.0'):
        for (i, name) in enumerate(data['names']):
            if '/beta' in str(data['names'][i-1]) and '/gamma' in str(name):
                values[i], values[i-1] = values[i-1], values[i]

    return values


def parse_keras_weights(weights_path, move_rules=None):
    try:
        import h5py
    except ImportError:
        h5py = None
    assert h5py is not None, '`get_values_from_keras_file` requires `h5py`.'

    values = []
    with h5py.File(weights_path, mode='r') as f:
        names = [n.decode('utf8')
                 for n in f.attrs['layer_names']
                 if len(f[n.decode('utf8')].attrs['weight_names']) > 0]
        if move_rules is not None:
            if isinstance(move_rules, list):
                for (name, loc) in move_rules:
                    idx = names.index(name)
                    names.insert(idx + loc, names.pop(idx))
            elif move_rules == 'ordered':
                bn_names, conv_names, other_names = [], [], []
                for n in names:
                    if 'batch' in n:
                        bn_names.append(n)
                    elif 'conv' in n:
                        conv_names.append(n)
                    else:
                        other_names.append(n)
                names = []
                for n in range(1, len(conv_names) + 1):
                    names.append("conv2d_%d" % n)
                    names.append("batch_normalization_%d" % n)
                names += other_names

        for name in names:
            g = f[name]
            w = [n.decode('utf8') for n in g.attrs['weight_names']]
            v = [np.asarray(g[n]) for n in w]
            if not tf_later_than('1.3.0'):
                if len(v) == 4:
                    w[0], w[1] = w[1], w[0]
                    v[0], v[1] = v[1], v[0]
            values += v

    return values


def parse_torch_weights(weights_path, move_rules=None):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        torch = None
    assert torch is not None, '`get_values_from_torch_file` requires `torch`.'

    model = torch.load(weights_path)
    names = list(model.keys())
    if move_rules is not None:
        if isinstance(move_rules, list):
            for (name, loc) in move_rules:
                idx = names.index(name)
                names.insert(idx + loc, names.pop(idx))

    if not tf_later_than('1.3.0'):
        for (i, name) in enumerate(names):
            if 'running_mean' in str(name):
                names[i-1], names[i-2] = names[i-2], names[i-1]

    values = []
    for name in names:
        val = model[name].numpy()
        if val.ndim == 4:
            val = np.transpose(val, [2, 3, 1, 0])
        if val.ndim == 2:
            val = np.transpose(val, [1, 0])
        if val.ndim == 4:
            groups = val.shape[3] // val.shape[2]
            if (groups == 32) or (groups == 64):
                values += np.split(val, groups, axis=3)
            else:
                values.append(val)
        else:
            values.append(val)

    return values


def remove_head(original_stem, name):
    _scope = "%s/stem" % tf.get_variable_scope().name
    g = tf.get_default_graph()
    for x in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=_scope + '/')[::-1]:
        if name in x.name:
            break
        g.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES).pop()

    for x in g.get_collection(__outputs__, scope=_scope + '/')[::-1]:
        if name in x.name:
            break
        g.get_collection_ref(__outputs__).pop()
    x.model_name = original_stem.model_name
    return x


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


def remove_commons(module_name, exceptions=[]):
    import sys
    _commons = [
        'absolute_import',
        'division'
        'print_function',
        'remove_commons',
    ]
    module = sys.modules[module_name]
    for _common in _commons:
        if _common not in exceptions:
            try:
                delattr(module, _common)
            except:
                None


remove_commons(__name__, ['remove_commons'])
