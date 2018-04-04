"""Collection of darkflow utils

The codes were copied without modification from the original darkflow
(https://github.com/thtrieu/darkflow), and each module was from the following:

1. nms
 - ${darkflow}/darkflow/cython_utils/nms.pyx
2. get_boxes
 - ${darkflow}/darkflow/cython_utils/cy_yolo2_findboxes.pyx

Additionally, `yolov3_box` was adapted from `yolov2_box` by taehoonlee.
"""
from __future__ import absolute_import

try:
    from . import get_boxes
except ImportError:
    class emptyboxes:
        yolov3_box = None
        yolov2_box = None
    get_boxes = emptyboxes()
