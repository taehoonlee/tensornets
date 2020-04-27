"""Collection of PASCAL VOC utils

The codes were refactored from [py-faster-rcnn](https://github.com/
rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py) and
[darkflow](https://github.com/thtrieu/darkflow/blob/master/darkflow/
net/yolov2/data.py). Especially, each part was from the following:

1. get_annotations: parse_rec in py-faster-rcnn
2. evaluate_class: voc_ap in py-faster-rcnn
3. evaluate: voc_eval in py-faster-rcnn
4. load_train: _batch in darkflow
"""
from __future__ import division

import os
import numpy as np
import xml.etree.ElementTree as ET

try:
    import cv2
except ImportError:
    cv2 = None

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

try:
    reduce
except NameError:
    from functools import reduce


with open(os.path.join(os.path.dirname(__file__), 'voc.names'), 'r') as f:
    classnames = [line.rstrip() for line in f.readlines()]


def classidx(classname):
    return dict((k, i) for (i, k) in enumerate(classnames))[classname]


def area(box):
    if box.ndim == 1:
        return (box[2] - box[0] + 1.) * (box[3] - box[1] + 1.)
    else:
        return (box[:, 2] - box[:, 0] + 1.) * (box[:, 3] - box[:, 1] + 1.)


def get_files(data_dir, data_name, total_num=None):
    with open("%s/ImageSets/Main/%s.txt" % (data_dir, data_name)) as f:
        files = [x.strip() for x in f.readlines()]
    if total_num is not None:
        files = files[:total_num]
    return files


def get_annotations(data_dir, files):
    annotations = {}
    for filename in files:
        tree = ET.parse("%s/Annotations/%s.xml" % (data_dir, filename))
        annotations[filename] = [[] for _ in range(20)]
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            cidx = classidx(obj_struct['name'])
            annotations[filename][cidx].append(obj_struct)
    return annotations


def load(data_dir, data_name, min_shorter_side=None, max_longer_side=1000,
         batch_size=1, total_num=None):
    assert cv2 is not None, '`load` requires `cv2`.'
    files = get_files(data_dir, data_name, total_num)
    total_num = len(files)

    for batch_start in range(0, total_num, batch_size):
        x = cv2.imread("%s/JPEGImages/%s.jpg" % (data_dir, files[batch_start]))
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
    diff = [np.array([obj['difficult'] for obj in annotations[filename]])
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
        difficult = np.array([x['difficult'] for x in annotations[ids[d]]])

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
    files = get_files(data_dir, data_name)
    files = files[:len(results)]
    annotations = get_annotations(data_dir, files)
    aps = []

    for c in range(20):
        ids = []
        scores = []
        boxes = []
        for (i, filename) in enumerate(files):
            pred = results[i][c]
            if pred.shape[0] > 0:
                for k in xrange(pred.shape[0]):
                    ids.append(filename)
                    scores.append(pred[k, -1])
                    boxes.append(pred[k, :4] + 1)
        ids = np.array(ids)
        scores = np.array(scores)
        boxes = np.array(boxes)
        _annotations = dict((k, v[c]) for (k, v) in annotations.items())
        ap, _, _ = evaluate_class(ids, scores, boxes, _annotations,
                                  files, ovthresh)
        aps += [ap]

    strs = ''
    for c in range(20):
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


def load_train(data_dir, data_name,
               batch_size=64, shuffle=True,
               target_size=416, anchors=5, classes=20,
               total_num=None, dtype=np.float32):
    assert cv2 is not None, '`load_train` requires `cv2`.'
    if isinstance(data_dir, list):
        files = []
        annotations = {}
        for d in data_dir:
            files.append(get_files(d, data_name, total_num))
            annotations.update(get_annotations(d, files[-1]))
        dirs = np.concatenate([i * np.ones(len(f), dtype=np.int)
                               for (i, f) in enumerate(files)])
        files = reduce(lambda x, y: x + y, files)
    else:
        files = get_files(data_dir, data_name, total_num)
        annotations = get_annotations(data_dir, files)
        dirs = np.zeros(len(files), dtype=np.int)
        data_dir = [data_dir]  # put in list for consistent further processing

    total_num = len(files)
    for f in files:
        annotations[f] = reduce(lambda x, y: x + y, annotations[f])

    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    feature_size = [x // 32 for x in target_size]
    cells = feature_size[0] * feature_size[1]

    b = 0
    while True:
        if b == 0:
            if shuffle is True:
                idx = np.random.permutation(total_num)
            else:
                idx = np.arange(total_num)
        if b + batch_size > total_num:
            b = 0
            yield None, None
        else:
            batch_num = batch_size

        imgs = np.zeros((batch_num,) + target_size + (3,), dtype=dtype)
        probs = np.zeros((batch_num, cells, anchors, classes), dtype=dtype)
        confs = np.zeros((batch_num, cells, anchors), dtype=dtype)
        coord = np.zeros((batch_num, cells, anchors, 4), dtype=dtype)
        proid = np.zeros((batch_num, cells, anchors, classes), dtype=dtype)
        prear = np.zeros((batch_num, cells, 4), dtype=dtype)
        areas = np.zeros((batch_num, cells, anchors), dtype=dtype)
        upleft = np.zeros((batch_num, cells, anchors, 2), dtype=dtype)
        botright = np.zeros((batch_num, cells, anchors, 2), dtype=dtype)

        for i in range(batch_num):
            d = data_dir[dirs[idx[b + i]]]
            f = files[idx[b + i]]
            x = cv2.imread("%s/JPEGImages/%s.jpg" % (d, f))
            h, w = x.shape[:2]
            cellx = 1. * w / feature_size[1]
            celly = 1. * h / feature_size[0]

            processed_objs = []
            for obj in annotations[f]:
                bbox = obj['bbox']
                centerx = .5 * (bbox[0] + bbox[2])  # xmin, xmax
                centery = .5 * (bbox[1] + bbox[3])  # ymin, ymax
                cx = centerx / cellx
                cy = centery / celly
                if cx >= feature_size[1] or cy >= feature_size[0]:
                    continue
                processed_objs += [[
                    classidx(obj['name']),
                    cx - np.floor(cx),  # centerx
                    cy - np.floor(cy),  # centery
                    np.sqrt(float(bbox[2] - bbox[0]) / w),
                    np.sqrt(float(bbox[3] - bbox[1]) / h),
                    int(np.floor(cy) * feature_size[1] + np.floor(cx))
                ]]

            # Calculate placeholders' values
            for obj in processed_objs:
                probs[i, obj[5], :, :] = [[0.] * classes] * anchors
                probs[i, obj[5], :, obj[0]] = 1.
                proid[i, obj[5], :, :] = [[1.] * classes] * anchors
                coord[i, obj[5], :, :] = [obj[1:5]] * anchors
                prear[i, obj[5], 0] = obj[1] - obj[3]**2 * .5 * feature_size[1]
                prear[i, obj[5], 1] = obj[2] - obj[4]**2 * .5 * feature_size[0]
                prear[i, obj[5], 2] = obj[1] + obj[3]**2 * .5 * feature_size[1]
                prear[i, obj[5], 3] = obj[2] + obj[4]**2 * .5 * feature_size[0]
                confs[i, obj[5], :] = [1.] * anchors

            # Finalise the placeholders' values
            ul = np.expand_dims(prear[i, :, 0:2], 1)
            br = np.expand_dims(prear[i, :, 2:4], 1)
            wh = br - ul
            area = wh[:, :, 0] * wh[:, :, 1]
            upleft[i, :, :, :] = np.concatenate([ul] * anchors, 1)
            botright[i, :, :, :] = np.concatenate([br] * anchors, 1)
            areas[i, :, :] = np.concatenate([area] * anchors, 1)

            imgs[i] = cv2.resize(x, target_size,
                                 interpolation=cv2.INTER_LINEAR)
        yield imgs, [probs, confs, coord, proid, areas, upleft, botright]
        b += batch_size
