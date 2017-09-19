from __future__ import absolute_import

import numpy as np


def imagenet_val_labels(data_dir):
    _, labels = np.split(np.genfromtxt("%s/val.txt" % data_dir,
                                       dtype=np.str), [1], axis=1)
    labels = np.asarray(labels.flatten(), dtype=np.int)
    return labels


def imagenet_val_generator(data_dir, total_num, batch_size,
                           resize_wh, crop_locs, crop_wh):
    files, _ = np.split(np.genfromtxt("%s/val.txt" % data_dir,
                                      dtype=np.str), [1], axis=1)
    files = files.flatten()

    def load_batch(batch_start, batch_size, resize_wh, crop_locs, crop_wh):
        from PIL import Image
        from os.path import isfile, join
        from .utils import crop, load_img
        X = np.zeros((batch_size * len(crop_locs), crop_wh, crop_wh, 3),
                     dtype=np.float32)
        for (k, f) in enumerate(files[batch_start:batch_start+batch_size]):
            filename = join("%s/ILSVRC2012_img_val" % data_dir, f)
            if isfile(filename):
                img = load_img(filename, target_size=resize_wh)
                for (i, crop_loc) in enumerate(crop_locs):
                    im = crop(img, crop_wh, crop_loc)
                    X[i*batch_size + k] = im
                    # for troubleshooting
                    if im.shape != (1, crop_wh, crop_wh, 3):
                        print("%s %s" % (f, list(im.shape)))
        return X

    return (load_batch(batch_start, batch_size, resize_wh, crop_locs, crop_wh)
            for batch_start in xrange(0, total_num, batch_size))
