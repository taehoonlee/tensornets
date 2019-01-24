"""ZF net embedded in Faster RCNN

The reference paper:

 - Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, NIPS 2015
 - Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
 - https://arxiv.org/abs/1506.01497

The reference implementation:

1. Caffe and Python utils
 - https://github.com/rbgirshick/py-faster-rcnn
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .layers import conv2d
from .layers import fc
from .layers import max_pool2d
from .layers import convrelu as conv

from .ops import *
from .utils import pad_info
from .utils import set_args
from .utils import var_scope


def __args__(is_training):
    return [([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'scope': 'conv'}),
            ([fc], {'activation_fn': None, 'scope': 'fc'}),
            ([max_pool2d], {'scope': 'pool'})]


@var_scope('zf')
@set_args(__args__)
def zf(x, is_training=False, classes=1000, stem=False, scope=None, reuse=None):
    x = pad(x, pad_info(7), name='pad1')
    x = conv(x, 96, 7, stride=2, padding='VALID', scope='conv1')
    x = srn(x, depth_radius=3, alpha=0.00005, beta=0.75, name='srn1')
    x = pad(x, pad_info(3, symmetry=False), name='pad2')
    x = max_pool2d(x, 3, stride=2, padding='VALID', scope='pool1')

    x = pad(x, pad_info(5), name='pad3')
    x = conv(x, 256, 5, stride=2, padding='VALID', scope='conv2')
    x = srn(x, depth_radius=3, alpha=0.00005, beta=0.75, name='srn2')
    x = pad(x, pad_info(3, symmetry=False), name='pad4')
    x = max_pool2d(x, 3, stride=2, padding='VALID', scope='pool2')

    x = conv(x, 384, 3, scope='conv3')
    x = conv(x, 384, 3, scope='conv4')
    x = conv(x, 256, 3, scope='conv5')
    if stem: return x

    x = reduce_mean(x, [1, 2], name='avgpool')
    x = fc(x, classes, scope='logits')
    x = softmax(x, name='probs')
    return x


# Simple alias.
ZF = zf
