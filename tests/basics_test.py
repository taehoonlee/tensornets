from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensornets as nets
import os
import pytest
import random

from tensornets.middles import direct

from distutils.version import LooseVersion


pytestmark = pytest.mark.skipif(
    os.environ.get('CORE_CHANGED', 'True') == 'False',
    reason='Runs only when the relevant files have been modified.')


@pytest.mark.parametrize('net,shape,weights,outputs,middles', [
    random.choice([
        (nets.ResNet50, (224, 224, 3), 320, 161, 16),
        (nets.ResNet101, (224, 224, 3), 626, 314, 33),
        (nets.ResNet152, (224, 224, 3), 932, 467, 50),
    ]),
    random.choice([
        (nets.ResNet50v2, (299, 299, 3), 272, 192, 16),
        (nets.ResNet101v2, (299, 299, 3), 544, 379, 33),
        (nets.ResNet152v2, (299, 299, 3), 816, 566, 50),
    ]),
    (nets.ResNet200v2, (224, 224, 3), 1224, 745, 66),
    random.choice([
        (nets.ResNeXt50, (224, 224, 3), 267, 193, 16),
        (nets.ResNeXt101, (224, 224, 3), 522, 380, 33),
        (nets.ResNeXt101c64, (224, 224, 3), 522, 380, 33),
    ]),
    (nets.WideResNet50, (224, 224, 3), 267, 177, 16),
    (nets.Inception1, (224, 224, 3), 116, 143, 11),
    (nets.Inception2, (224, 224, 3), 277, 231, 10),
    (nets.Inception3, (299, 299, 3), 378, 313, 11),
    (nets.Inception4, (299, 299, 3), 598, 494, 17),
    (nets.InceptionResNet2, (299, 299, 3), 898, 744, 43),
    pytest.param(
        nets.NASNetAlarge, (331, 331, 3), 1558, 1029, 20,
        marks=pytest.mark.xfail(
            LooseVersion(tf.__version__) < LooseVersion('1.3.0'),
            reason='NASNetAlarge requires TensorFlow >= 1.3.0')),
    pytest.param(
        nets.NASNetAmobile, (224, 224, 3), 1138, 759, 14,
        marks=pytest.mark.xfail(
            LooseVersion(tf.__version__) < LooseVersion('1.3.0'),
            reason='NASNetAmobile requires TensorFlow >= 1.3.0')),
    pytest.param(
        nets.PNASNetlarge, (331, 331, 3), 1179, 752, 12,
        marks=pytest.mark.xfail(
            LooseVersion(tf.__version__) < LooseVersion('1.3.0'),
            reason='PNASNetlarge requires TensorFlow >= 1.3.0')),
    pytest.param(
        *random.choice([
            (nets.VGG16, (224, 224, 3), 32, 40, 9),
            (nets.VGG19, (224, 224, 3), 38, 46, 12),
        ]),
        marks=pytest.mark.skipif(
            LooseVersion(tf.__version__) == LooseVersion('1.2.0'),
            reason='Deployments of VGGs on local are OK. But there is '
                   'something wrong in those tests on Travis with TF 1.2.0.')),
    random.choice([
        (nets.DenseNet121, (224, 224, 3), 606, 429, 61),
        (nets.DenseNet169, (224, 224, 3), 846, 597, 85),
        (nets.DenseNet201, (224, 224, 3), 1006, 709, 101),
    ]),
    random.choice([
        (nets.MobileNet25, (224, 224, 3), 137, 85, 11),
        (nets.MobileNet50, (224, 224, 3), 137, 85, 11),
        (nets.MobileNet75, (224, 224, 3), 137, 85, 11),
        (nets.MobileNet100, (224, 224, 3), 137, 85, 11),
    ]),
    random.choice([
        (nets.MobileNet35v2, (224, 224, 3), 262, 152, 62),
        (nets.MobileNet50v2, (224, 224, 3), 262, 152, 62),
        (nets.MobileNet75v2, (224, 224, 3), 262, 152, 62),
        (nets.MobileNet100v2, (224, 224, 3), 262, 152, 62),
        (nets.MobileNet130v2, (224, 224, 3), 262, 152, 62),
        (nets.MobileNet140v2, (224, 224, 3), 262, 152, 62),
    ]),
    (nets.SqueezeNet, (224, 224, 3), 52, 65, 10),
], ids=[
    'ResNet',
    'ResNetv2',
    'ResNet200v2',
    'ResNeXt',
    'WideResNet50',
    'Inception1',
    'Inception2',
    'Inception3',
    'Inception4',
    'InceptionResNet2',
    'NASNetAlarge',
    'NASNetAmobile',
    'PNASNetlarge',
    'VGG',
    'DenseNet',
    'MobileNet',
    'MobileNetv2',
    'SqueezeNet',
])
def test_classification_basics(net, shape, weights, outputs, middles):
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, [None] + list(shape))
        model = net(inputs, is_training=False)
        assert isinstance(model, tf.Tensor)

        # Disable tests whether the desired list is returned for TF==1.1.0
        if LooseVersion(tf.__version__) > LooseVersion('1.1.0'):
            assert len(model.get_weights()) == weights
            assert len(model.get_outputs()) == outputs
            assert len(model.get_middles()) == middles

        x = np.random.random((1,) + shape).astype(np.float32) * 255

        with tf.Session() as sess:
            nets.init(model)
            y = model.eval({inputs: model.preprocess(x)})

        for (a, b) in zip(model.get_middles(),
                          direct(model.aliases[0])[1]):
            assert a.name.endswith(b)

        # Disable tests whether the desired list is returned for TF==1.1.0
        if LooseVersion(tf.__version__) > LooseVersion('1.1.0'):
            with tf.name_scope('a'):
                with tf.variable_scope('b'):
                    with tf.name_scope('c'):
                        model = net(inputs, is_training=False)
                        assert len(model.get_weights()) == weights
                        assert len(model.get_outputs()) == outputs
                        assert len(model.get_middles()) == middles

            with tf.variable_scope('d'):
                with tf.name_scope('e'):
                    with tf.variable_scope('f'):
                        model = net(inputs, is_training=False)
                        assert len(model.get_weights()) == weights
                        assert len(model.get_outputs()) == outputs
                        assert len(model.get_middles()) == middles

        assert y.shape == (1, 1000)


@pytest.mark.parametrize('net,shape,stem', [
    (nets.YOLOv2, (416, 416, 3), nets.Darknet19),
    (nets.TinyYOLOv2, (416, 416, 3), nets.TinyDarknet19),
], ids=[
    'YOLOv2',
    'TinyYOLOv2',
])
def test_detection_basics(net, shape, stem):
    # TODO: Once the roi-pooling dependency is removed,
    # FasterRCNN-related tests should be added.
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, [None] + list(shape))
        model = net(inputs, stem, is_training=False)
        assert isinstance(model, tf.Tensor)

        x = np.random.random((1, 733, 490, 3)).astype(np.float32) * 255

        with tf.Session() as sess:
            nets.init(model)
            y = model.eval({inputs: model.preprocess(x)})

        # TODO: Once the get_boxes's are translated from cython,
        # get_boxes tests should be enabled.
        # boxes = model.get_boxes(y, x.shape[1:3])

        # assert len(boxes) == 20
