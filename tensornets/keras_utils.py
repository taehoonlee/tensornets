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
