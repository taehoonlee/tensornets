import os

with open(os.path.join(os.path.dirname(__file__), 'coco.names'), 'r') as f:
    lines = f.readlines()
    labels_coco = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), 'voc.names'), 'r') as f:
    lines = f.readlines()
    labels_voc = [line.rstrip() for line in lines]

bases = dict()

bases['yolov2'] = {
    'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
    'out_size': [19, 19, 425]}

bases['yolov2voc'] = {
    'anchors': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587,
                8.09892, 9.47112, 4.84053, 11.2364, 10.0071],
    'out_size': [13, 13, 125]}

bases['tinyyolov2voc'] = {
    'anchors': [1.08, 1.19, 3.42, 4.41, 6.63,
                11.38, 9.42, 5.11, 16.62, 10.52],
    'out_size': [13, 13, 125]}


def opts(model_name, t=0.1):
    opt = bases[model_name].copy()
    opt.update({'num': 5, 'thresh': t})
    if 'voc' in model_name:
        opt.update({'classes': len(labels_voc), 'labels': labels_voc})
    else:
        opt.update({'classes': len(labels_coco), 'labels': labels_coco})
    return opt
