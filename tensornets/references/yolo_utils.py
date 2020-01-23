from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import tensorflow as tf

from ..version_utils import tf_later_than

try:
    from .darkflow_utils.get_boxes import yolov3_box
    from .darkflow_utils.get_boxes import yolov2_box
except ImportError:
    yolov3_box = None
    yolov2_box = None

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


if tf_later_than('1.14'):
    tf = tf.compat.v1


with open(os.path.join(os.path.dirname(__file__), 'coco.names'), 'r') as f:
    labels_coco = [line.rstrip() for line in f.readlines()]

with open(os.path.join(os.path.dirname(__file__), 'voc.names'), 'r') as f:
    labels_voc = [line.rstrip() for line in f.readlines()]

bases = dict()
bases['yolov3'] = {'anchors': [10., 13., 16., 30., 33., 23., 30., 61.,
                               62., 45., 59., 119., 116., 90., 156., 198.,
                               373., 326.]}
bases['yolov3coco'] = bases['yolov3']
bases['yolov3voc'] = bases['yolov3']
bases['yolov2'] = {'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                               5.47434, 7.88282, 3.52778, 9.77052, 9.16828]}
bases['yolov2voc'] = {'anchors': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587,
                                  8.09892, 9.47112, 4.84053, 11.2364, 10.0071]}
bases['tinyyolov2voc'] = {'anchors': [1.08, 1.19, 3.42, 4.41, 6.63,
                                      11.38, 9.42, 5.11, 16.62, 10.52]}


def opts(model_name):
    opt = bases[model_name].copy()
    opt.update({'num': len(opt['anchors']) // 2})
    if 'voc' in model_name:
        opt.update({'classes': len(labels_voc), 'labels': labels_voc})
    else:
        opt.update({'classes': len(labels_coco), 'labels': labels_coco})
    return opt


def parse_box(b, t, w, h):
    idx = np.argmax(b.probs)
    score = b.probs[idx]
    if score > t:
        try:
            x1 = int((b.x - b.w / 2) * w)
            y1 = int((b.y - b.h / 2) * h)
            x2 = int((b.x + b.w / 2) * w)
            y2 = int((b.y + b.h / 2) * h)
            if x1 < 0:
                x1 = 0
            if x2 > w - 1:
                x2 = w - 1
            if y1 < 0:
                y1 = 0
            if y2 > h - 1:
                y2 = h - 1
            return idx, (x1, y1, x2, y2, score)
        except:
            return None, None
    else:
        return None, None


def get_v3_boxes(opts, outs, source_size, threshold=0.1):
    h, w = source_size
    boxes = [[] for _ in xrange(opts['classes'])]
    opts['thresh'] = threshold
    results = yolov3_box(opts,
                         np.array(outs[0][0], dtype=np.float32),
                         np.array(outs[1][0], dtype=np.float32),
                         np.array(outs[2][0], dtype=np.float32))
    for b in results:
        idx, box = parse_box(b, threshold, w, h)
        if idx is not None:
            boxes[idx].append(box)
    for i in xrange(opts['classes']):
        boxes[i] = np.asarray(boxes[i], dtype=np.float32)
    return boxes


def get_v2_boxes(opts, outs, source_size, threshold=0.1):
    h, w = source_size
    boxes = [[] for _ in xrange(opts['classes'])]
    opts['thresh'] = threshold
    results = yolov2_box(opts, np.array(outs[0], dtype=np.float32))
    for b in results:
        idx, box = parse_box(b, threshold, w, h)
        if idx is not None:
            boxes[idx].append(box)
    for i in xrange(opts['classes']):
        boxes[i] = np.asarray(boxes[i], dtype=np.float32)
    return boxes


def v2_inputs(out_shape, anchors, classes, dtype):
    sizes = [None, np.prod(out_shape), anchors]
    return [tf.placeholder(dtype, sizes + [classes], name='probs'),
            tf.placeholder(dtype, sizes, name='confs'),
            tf.placeholder(dtype, sizes + [4], name='coord'),
            tf.placeholder(dtype, sizes + [classes], name='proid'),
            tf.placeholder(dtype, sizes, name='areas'),
            tf.placeholder(dtype, sizes + [2], name='upleft'),
            tf.placeholder(dtype, sizes + [2], name='botright')]


def v2_loss(outs, anchorcoords, classes):
    # Refer to the following darkflow loss
    # https://github.com/thtrieu/darkflow/blob/master/darkflow/net/yolov2/train.py
    sprob = 1.
    sconf = 5.
    snoob = 1.
    scoor = 1.
    H = int(outs.shape[1]) if tf_later_than('2') else outs.shape[1].value
    W = int(outs.shape[2]) if tf_later_than('2') else outs.shape[2].value
    cells = H * W
    sizes = np.array([[[[W, H]]]], dtype=np.float32)
    anchors = len(anchorcoords) // 2
    anchorcoords = np.reshape(anchorcoords, [1, 1, anchors, 2])
    _, _probs, _confs, _coord, _proid, _areas, _ul, _br = outs.inputs[:8]

    # Extract the coordinate prediction from net.out
    outs = tf.reshape(outs, [-1, H, W, anchors, (5 + classes)])
    coords = tf.reshape(outs[:, :, :, :, :4], [-1, cells, anchors, 4])
    adj_xy = 1. / (1. + tf.exp(-coords[:, :, :, 0:2]))
    adj_wh = tf.sqrt(tf.exp(coords[:, :, :, 2:4]) * anchorcoords / sizes)
    adj_c = 1. / (1. + tf.exp(-outs[:, :, :, :, 4]))
    adj_c = tf.reshape(adj_c, [-1, cells, anchors, 1])
    adj_prob = tf.reshape(tf.nn.softmax(outs[:, :, :, :, 5:]),
                          [-1, cells, anchors, classes])
    adj_outs = tf.concat([adj_xy, adj_wh, adj_c, adj_prob], 3)

    coords = tf.concat([adj_xy, adj_wh], 3)
    wh = tf.pow(coords[:, :, :, 2:4], 2) * sizes
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2]
    floor = centers - (wh * .5)
    ceil = centers + (wh * .5)

    # calculate the intersection areas
    intersect_upleft = tf.maximum(floor, _ul)
    intersect_botright = tf.minimum(ceil, _br)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(classes * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid], 3)

    loss = tf.pow(adj_outs - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reshape(loss, [-1, cells * anchors * (5 + classes)])
    loss = tf.reduce_sum(loss, 1)
    return .5 * tf.reduce_mean(loss) + tf.losses.get_regularization_loss()
