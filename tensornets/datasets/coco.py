"""Collection of MS COCO utils

The codes were adapted from [py-faster-rcnn](https://github.com/
rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py).
"""
from __future__ import division

import os
import json
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


metas = {}

with open(os.path.join(os.path.dirname(__file__), 'coco.names'), 'r') as f:
    classnames = [line.rstrip() for line in f.readlines()]


def classidx(classname):
    return dict((k, i) for (i, k) in enumerate(classnames))[classname]


def area(box):
    if box.ndim == 1:
        return (box[2] - box[0] + 1.) * (box[3] - box[1] + 1.)
    else:
        return (box[:, 2] - box[:, 0] + 1.) * (box[:, 3] - box[:, 1] + 1.)


def get_files(data_dir, data_name, total_num=None):
    assert COCO is not None, '`datasets.coco` requires `pycocotools`.'
    if data_name not in metas:
        metas[data_name] = COCO("%s/annotations/instances_%s.json" %
                                (data_dir, data_name))
    images = metas[data_name].imgs
    fileids = images.keys()
    if total_num is not None:
        fileids = fileids[:total_num]
    files = [images[i]['file_name'] for i in fileids]
    return fileids, files


def get_annotations(data_dir, data_name, ids):
    assert COCO is not None, '`datasets.coco` requires `pycocotools`.'
    if data_name not in metas:
        metas[data_name] = COCO("%s/annotations/instances_%s.json" %
                                (data_dir, data_name))
    cmap = dict([(b, a) for (a, b) in enumerate(metas[data_name].getCatIds())])
    annotations = {}
    for i in ids:
        annids = metas[data_name].getAnnIds(imgIds=i, iscrowd=None)
        objs = metas[data_name].loadAnns(annids)
        annotations[i] = [[] for _ in range(80)]
        width = metas[data_name].imgs[i]['width']
        height = metas[data_name].imgs[i]['height']
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj_struct = {'bbox': [x1, y1, x2, y2]}
                cidx = cmap[obj['category_id']]
                annotations[i][cidx].append(obj_struct)
    return annotations


def load(data_dir, data_name, min_shorter_side=None, max_longer_side=1000,
         batch_size=1, total_num=None):
    assert cv2 is not None, '`load` requires `cv2`.'
    _, files = get_files(data_dir, data_name, total_num)
    total_num = len(files)

    for batch_start in range(0, total_num, batch_size):
        x = cv2.imread("%s/%s/%s" % (data_dir, data_name, files[batch_start]))
        if min_shorter_side is not None:
            scale = float(min_shorter_side) / np.min(x.shape[:2])
        else:
            scale = 1.0
        if round(scale * np.max(x.shape[:2])) > max_longer_side:
            scale = float(max_longer_side) / np.max(x.shape[:2])
        x = cv2.resize(x, None, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_LINEAR)
        x = np.array([x], dtype=np.float32)
        scale = np.array([scale], dtype=np.float32)
        yield x, scale
        del x


def evaluate_class(ids, scores, boxes, annotations, files, ovthresh):
    if scores.shape[0] == 0:
        return 0.0, np.zeros(len(ids)), np.zeros(len(ids))

    # extract gt objects for this class
    diff = [np.array([0 for obj in annotations[filename]])
            for filename in files]
    total = sum([sum(x == 0) for x in diff])
    detected = dict(zip(files, [[False] * len(x) for x in diff]))

    # sort by confidence
    sorted_ind = np.argsort(-scores)
    ids = ids[sorted_ind]
    boxes = boxes[sorted_ind, :]

    # go down dets and mark TPs and FPs
    tp_list = []
    fp_list = []
    for d in range(len(ids)):
        actual = np.array([x['bbox'] for x in annotations[ids[d]]])
        difficult = np.array([0 for x in annotations[ids[d]]])

        if actual.size > 0:
            iw = np.maximum(np.minimum(actual[:, 2], boxes[d, 2]) -
                            np.maximum(actual[:, 0], boxes[d, 0]) + 1, 0)
            ih = np.maximum(np.minimum(actual[:, 3], boxes[d, 3]) -
                            np.maximum(actual[:, 1], boxes[d, 1]) + 1, 0)
            inters = iw * ih
            overlaps = inters / (area(actual) + area(boxes[d, :]) - inters)
            jmax = np.argmax(overlaps)
            ovmax = overlaps[jmax]
        else:
            ovmax = -np.inf

        tp = 0.
        fp = 0.
        if ovmax > ovthresh:
            if difficult[jmax] == 0:
                if not detected[ids[d]][jmax]:
                    tp = 1.
                    detected[ids[d]][jmax] = True
                else:
                    fp = 1.
        else:
            fp = 1.
        tp_list.append(tp)
        fp_list.append(fp)

    tp = np.cumsum(tp_list)
    fp = np.cumsum(fp_list)
    recall = tp / float(total)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.mean([0 if np.sum(recall >= t) == 0
                  else np.max(precision[recall >= t])
                  for t in np.linspace(0, 1, 11)])

    return ap, precision, recall


def evaluate(results, data_dir, data_name, ovthresh=0.5, verbose=True):
    fileids, _ = get_files(data_dir, data_name)
    fileids = fileids[:len(results)]
    annotations = get_annotations(data_dir, data_name, fileids)
    aps = []

    for c in range(80):
        ids = []
        scores = []
        boxes = []
        for (i, fileid) in enumerate(fileids):
            pred = results[i][c]
            if pred.shape[0] > 0:
                for k in xrange(pred.shape[0]):
                    ids.append(fileid)
                    scores.append(pred[k, -1])
                    boxes.append(pred[k, :4] + 1)
        ids = np.array(ids)
        scores = np.array(scores)
        boxes = np.array(boxes)
        _annotations = dict((k, v[c]) for (k, v) in annotations.iteritems())
        ap, _, _ = evaluate_class(ids, scores, boxes, _annotations,
                                  fileids, ovthresh)
        aps += [ap]

    strs = ''
    for c in range(80):
        strs += "| %6s " % classnames[c][:6]
    strs += '|\n'

    for ap in aps:
        strs += '|--------'
    strs += '|\n'

    for ap in aps:
        strs += "| %.4f " % ap
    strs += '|\n'

    strs += "Mean = %.4f" % np.mean(aps)
    return strs
