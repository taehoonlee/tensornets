import os
import numpy as np

try:
    from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor
except ImportError:
    box_constructor = None


with open(os.path.join(os.path.dirname(__file__), 'coco.names'), 'r') as f:
    labels_coco = [line.rstrip() for line in f.readlines()]

with open(os.path.join(os.path.dirname(__file__), 'voc.names'), 'r') as f:
    labels_voc = [line.rstrip() for line in f.readlines()]

bases = dict()
bases['yolov2'] = {'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                               5.47434, 7.88282, 3.52778, 9.77052, 9.16828]}
bases['yolov2voc'] = {'anchors': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587,
                                  8.09892, 9.47112, 4.84053, 11.2364, 10.0071]}
bases['tinyyolov2voc'] = {'anchors': [1.08, 1.19, 3.42, 4.41, 6.63,
                                      11.38, 9.42, 5.11, 16.62, 10.52]}


def opts(model_name, t=0.1):
    opt = bases[model_name].copy()
    if 'voc' in model_name:
        opt.update({'classes': len(labels_voc), 'labels': labels_voc})
    else:
        opt.update({'classes': len(labels_coco), 'labels': labels_coco})
    return opt


def get_boxes(opts, outs, source_size, num=5, threshold=0.1):
    assert box_constructor is not None, '`get_boxes` requires `darkflow`.'
    h, w = source_size
    boxes = [[] for _ in xrange(opts['classes'])]
    opts['num'] = num
    opts['thresh'] = threshold
    opts['out_size'] = list(outs[0].shape)
    results = box_constructor(opts, outs[0].copy())
    for b in results:
        max_idx = np.argmax(b.probs)
        max_score = b.probs[max_idx]
        if max_score > threshold:
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
            boxes[max_idx].append((x1, y1, x2, y2, max_score))
    for i in xrange(opts['classes']):
        boxes[i] = np.asarray(boxes[i], dtype=np.float32)
    return boxes
