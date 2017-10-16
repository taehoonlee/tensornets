from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensornets as nets
import pytest


@pytest.mark.parametrize('net,shape', [
    (nets.ResNet50, (224, 224, 3)),
    (nets.ResNet101, (224, 224, 3)),
    (nets.ResNet152, (224, 224, 3)),
    (nets.ResNet50v2, (224, 224, 3)),
    (nets.ResNet101v2, (224, 224, 3)),
    (nets.ResNet152v2, (224, 224, 3)),
    (nets.ResNet200v2, (224, 224, 3)),
    (nets.ResNeXt50, (224, 224, 3)),
    (nets.ResNeXt101, (224, 224, 3)),
    (nets.Inception1, (224, 224, 3)),
    (nets.Inception2, (299, 299, 3)),
    (nets.Inception3, (299, 299, 3)),
    (nets.Inception4, (299, 299, 3)),
    (nets.DenseNet121, (224, 224, 3)),
    (nets.DenseNet169, (224, 224, 3)),
    (nets.DenseNet201, (224, 224, 3)),
    (nets.MobileNet25, (224, 224, 3)),
    (nets.MobileNet50, (224, 224, 3)),
    (nets.MobileNet75, (224, 224, 3)),
    (nets.MobileNet100, (224, 224, 3)),
    (nets.SqueezeNet, (224, 224, 3)),
])
def test_basics(net, shape):
    inputs = tf.placeholder(tf.float32, [None] + list(shape))
    model = net(inputs, is_training=False)
    assert isinstance(model, tf.Tensor)

    x = np.random.random((1,) + shape).astype(np.float32)

    with tf.Session() as sess:
        nets.init(model)
        y = model.eval({inputs: x})

    assert y.shape == (1, 1000)
