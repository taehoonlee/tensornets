from __future__ import absolute_import
from __future__ import division

import os
import numpy as np

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
    else:
        return None, None


def get_v3_boxes(opts, outs, source_size, threshold=0.1):
    h, w = source_size
    boxes = [[] for _ in xrange(opts['classes'])]
    opts['thresh'] = threshold
    opts['in_size'] = (416, 416)
    for i in range(3):
        opts['out_size'] = list(outs[i][0].shape)
        opts['anchor_idx'] = 6 - 3 * i
        results = yolov3_box(opts, outs[i][0].copy())
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
    opts['out_size'] = list(outs[0].shape)
    results = yolov2_box(opts, outs[0].copy())
    for b in results:
        idx, box = parse_box(b, threshold, w, h)
        if idx is not None:
            boxes[idx].append(box)
    for i in xrange(opts['classes']):
        boxes[i] = np.asarray(boxes[i], dtype=np.float32)
    return boxes
