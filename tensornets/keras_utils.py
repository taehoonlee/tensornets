# Adapted from keras and tf-keras
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from distutils.version import LooseVersion
if LooseVersion(tf.__version__) > LooseVersion('1.3.0'):
    from tensorflow.python.keras._impl.keras.applications.imagenet_utils \
        import decode_predictions
    from tensorflow.python.keras.utils import get_file
else:
    from tensorflow.contrib.keras.python.keras.applications.imagenet_utils \
        import decode_predictions
    from tensorflow.contrib.keras.python.keras.utils.data_utils \
        import get_file


def load_img(path, grayscale=False, target_size=None, crop_size=None):
    try:
        from PIL import Image
    except ImportError:
        Image = None
    assert Image is not None, '`load_img` requires `PIL.Image`.'

    img = Image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        if isinstance(target_size, int):
            hw_tuple = tuple([x * target_size // min(img.size)
                              for x in img.size])
        else:
            hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple, resample=Image.BICUBIC)
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    if len(img.shape) == 3:
        img = np.expand_dims(img, -1)

    if crop_size is not None:
        from .utils import crop
        img = crop(img, crop_size)

    return img


def load_keras_weights(scopes, weights_path, move_rules=None):
    try:
        import h5py
    except ImportError:
        h5py = None
    assert h5py is not None, '`load_weights` requires `h5py`.'

    sess = tf.get_default_session()
    assert sess is not None, 'The default session should be given.'

    from .utils import parse_scopes
    scopes = parse_scopes(scopes)

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
            if not LooseVersion(tf.__version__) > LooseVersion('1.3.0'):
                if len(v) == 4:
                    w[0], w[1] = w[1], w[0]
                    v[0], v[1] = v[1], v[0]
            values += v

    for scope in scopes:
        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(weights) == len(values), 'The sizes of symbolic and ' \
                                            'actual weights do not match.' \

        sess.run([w.assign(v) for (w, v) in zip(weights, values)])
