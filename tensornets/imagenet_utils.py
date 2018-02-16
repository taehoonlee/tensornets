from __future__ import absolute_import

import numpy as np


def imagenet_val(data_dir, max_rows=None):
    """Reads a \`val.txt\` from
    http://www.image-net.org/challenges/LSVRC/2012/
    """
    files, labels = np.split(
        np.genfromtxt("%s/val.txt" % data_dir,
                      dtype=np.str, max_rows=max_rows),
        [1], axis=1)
    files = files.flatten()
    labels = np.asarray(labels.flatten(), dtype=np.int)
    return files, labels


def imagenet_val_labels(data_dir, max_rows=None):
    _, labels = imagenet_val(data_dir, max_rows=None)
    return labels


def imagenet_val_generator(data_dir, batch_size, resize_wh,
                           crop_locs, crop_wh, total_num=None):
    from os.path import isfile, join
    from .utils import crop, load_img

    files, labels = imagenet_val(data_dir, total_num)
    total_num = len(labels)

    for batch_start in range(0, total_num, batch_size):

        data_spec = [batch_size, 1, crop_wh, crop_wh, 3]
        if isinstance(crop_locs, list):
            data_spec[1] = len(crop_locs)
        elif crop_locs == 10:
            data_spec[1] = 10
        X = np.zeros(data_spec, np.float32)

        for (k, f) in enumerate(files[batch_start:batch_start+batch_size]):
            filename = join("%s/ILSVRC2012_img_val" % data_dir, f)
            if isfile(filename):
                img = load_img(filename, target_size=resize_wh)
                X[k] = crop(img, crop_wh, crop_locs)

        yield X.reshape((-1, crop_wh, crop_wh, 3)), \
            labels[batch_start:batch_start+batch_size]

        del X
