"""Collection of PASCAL VOC utils

The codes were refactored from the original py-faster-rcnn
(https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/
datasets/voc_eval.py). Especially, each part was from the following:

1. get_annotations: parse_rec
2. evaluate_class: voc_ap
3. evaluate: voc_eval
"""
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET


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
        _annotations = dict((k, v[c]) for (k, v) in annotations.iteritems())
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
