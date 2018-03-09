"""Collection of region proposal related utils

The codes was largely taken from the original py-faster-rcnn
(https://github.com/rbgirshick/py-faster-rcnn), and translated
into TensorFlow. Especially, each part was from the following:

1. _whctrs, _mkanchors, _ratio_enum, _scale_enum, get_anchors
 - ${py-faster-rcnn}/lib/rpn/generate_anchors.py
2. bbox_transform_inv, clip_boxes
 - ${py-faster-rcnn}/lib/fast_rcnn/bbox_transform.py
3. get_shifts, filter_boxes
 - ${py-faster-rcnn}/lib/rpn/proposal_layer.py
4. nms
 - ${py-faster-rcnn}/lib/nms/py_cpu_nms.py
"""
import numpy as np
import tensorflow as tf

try:
    # installation guide:
    # $ git clone git@github.com:deepsense-io/roi-pooling.git
    # $ cd roi-pooling
    # $ vi roi_pooling/Makefile
    # (edit according to https://github.com/tensorflow/tensorflow/
    #                    issues/13607#issuecomment-335530430)
    # $ python setup.py install
    from roi_pooling.roi_pooling_ops import roi_pooling
except ImportError:
    def roi_pooling(x, rois, w, h):
        raise AssertionError('`roi_pooling` requires deepsense-ai\'s package.')


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + (w - 1) / 2
    y_ctr = anchor[1] + (h - 1) / 2
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = (ws - 1) / 2
    hs = (hs - 1) / 2
    return tf.stack([
        x_ctr - ws,
        y_ctr - hs,
        x_ctr + ws,
        y_ctr + hs],
        axis=-1)


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = tf.round(tf.sqrt(size_ratios))
    hs = tf.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def get_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = tf.constant(
        [0, 0, base_size - 1, base_size - 1], dtype=tf.float32)
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = tf.concat(
        [_scale_enum(ratio_anchors[i, :], scales)
         for i in xrange(ratio_anchors.shape[0])],
        axis=0)
    return anchors


def get_shifts(width, height, feat_stride):
    shift_x = tf.range(width) * feat_stride
    shift_y = tf.range(height) * feat_stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.reshape(shift_x, (-1,))
    shift_y = tf.reshape(shift_y, (-1,))
    shifts = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
    shifts = tf.transpose(shifts)
    return tf.cast(shifts, dtype=tf.float32)


def bbox_transform_inv(boxes, deltas):
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, :, 0]
    dy = deltas[:, :, 1]
    dw = deltas[:, :, 2]
    dh = deltas[:, :, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights

    pred_boxes = tf.stack([
        pred_ctr_x - 0.5 * pred_w,  # x1
        pred_ctr_y - 0.5 * pred_h,  # y1
        pred_ctr_x + 0.5 * pred_w,  # x2
        pred_ctr_y + 0.5 * pred_h],  # y2
        axis=-1)
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    pred_boxes = tf.stack([
        tf.maximum(tf.minimum(boxes[:, :, 0], im_shape[1] - 1), 0),
        tf.maximum(tf.minimum(boxes[:, :, 1], im_shape[0] - 1), 0),
        tf.maximum(tf.minimum(boxes[:, :, 2], im_shape[1] - 1), 0),
        tf.maximum(tf.minimum(boxes[:, :, 3], im_shape[0] - 1), 0)],
        axis=-1)
    return pred_boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[0, :, 2] - boxes[0, :, 0] + 1
    hs = boxes[0, :, 3] - boxes[0, :, 1] + 1
    keep = tf.where((ws >= min_size) & (hs >= min_size))[:, 0]
    return keep


def nms(proposals, scores, thresh):
    x1 = proposals[:, 0]
    y1 = proposals[:, 1]
    x2 = proposals[:, 2]
    y2 = proposals[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    num = tf.range(scores.shape[0])

    def body(i, keep, screen):
        xx1 = tf.maximum(x1[i], x1)
        yy1 = tf.maximum(y1[i], y1)
        xx2 = tf.minimum(x2[i], x2)
        yy2 = tf.minimum(y2[i], y2)

        w = tf.maximum(0.0, xx2 - xx1 + 1)
        h = tf.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas - inter)

        bools = (ovr <= thresh) & (num >= i) & (screen)
        i = tf.cond(tf.count_nonzero(bools) > 0,
                    lambda: tf.cast(tf.where(bools)[0, 0], tf.int32),
                    lambda: scores.shape[0])

        return [i, tf.concat([keep, tf.stack([i])], axis=0), bools]

    def condition(i, keep, screen):
        return i < scores.shape[0]

    i = tf.constant(0)
    i, keep, screen = tf.while_loop(
        condition, body, [i, tf.stack([i]), num >= 0],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, ]),
                          tf.TensorShape([None, ])],
        back_prop=False)

    return keep[:-1]
