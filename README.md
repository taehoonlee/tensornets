# Tensornets

High level network definitions in [TensorFlow](https://github.com/tensorflow/tensorflow).

## Guiding principles

- 1
- 2
- 3

## A quick example

```python
import tensorflow as tf
import tensornets as nets

inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])

with tf.device('cpu:0'):
    model1 = nets.ResNet50(inputs, is_training=False, scope='myresnet')
    model2 = nets.ResNet152(inputs, is_training=False)

assert all(isinstance(m, tf.Tensor) for m in [model1, model2])
```

```python
from tensornets.utils import *

img = load_img('cat.png', target_size=256, crop_size=224)
img = nets.resnets.preprocess(img)

assert img.shape == (1, 224, 224, 3)
```

```python
with tf.Session() as sess:
    nets.load_resnet50(model1)
    nets.load_resnet152(model2)
    preds = sess.run([model1, model2], {inputs: img})
```

```python
for pred in preds:
    print('Predicted:', decode_predictions(pred, top=3)[0])
```

```
('Predicted:', [(u'n02124075', u'Egyptian_cat', 0.27387387), (u'n02127052', u'lynx', 0.11052437), (u'n02123045', u'tabby', 0.074132949)])
('Predicted:', [(u'n02124075', u'Egyptian_cat', 0.13528407), (u'n02123045', u'tabby', 0.094977126), (u'n04033995', u'quilt', 0.070704058)])
```

```python
print_summary(model1)
print_summary(model2)
```

```
Scope: myresnet
Total layers: 54
Total weights: 320
Total parameters: 25,636,712
Scope: resnet152
Total layers: 156
Total weights: 932
Total parameters: 60,419,944
```

```python
print_weights(model1)
```

```
Scope: myresnet
conv1/conv/weights:0 (7, 7, 3, 64)
conv1/conv/biases:0 (64,)
conv1/bn/beta:0 (64,)
conv1/bn/gamma:0 (64,)
conv1/bn/moving_mean:0 (64,)
conv1/bn/moving_variance:0 (64,)
conv2/block1/0/conv/weights:0 (1, 1, 64, 256)
conv2/block1/0/conv/biases:0 (256,)
conv2/block1/0/bn/beta:0 (256,)
conv2/block1/0/bn/gamma:0 (256,)
...
```

```python
print_outputs(model2)
```

```
Scope: resnet152
pad:0 (?, 230, 230, 3)
conv1/conv/BiasAdd:0 (?, 112, 112, 64)
conv1/bn/batchnorm/add_1:0 (?, 112, 112, 64)
conv1/relu:0 (?, 112, 112, 64)
pool1/MaxPool:0 (?, 55, 55, 64)
conv2/block1/0/conv/BiasAdd:0 (?, 55, 55, 256)
conv2/block1/0/bn/batchnorm/add_1:0 (?, 55, 55, 256)
conv2/block1/1/conv/BiasAdd:0 (?, 55, 55, 64)
conv2/block1/1/bn/batchnorm/add_1:0 (?, 55, 55, 64)
conv2/block1/1/relu:0 (?, 55, 55, 64)
...
```

## Performances

- Currently,
- Speed: ms for inferences of 10 images on P100

|             | Top-1 error | Top-5 error | Speed | References                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|-------------|-------------|-------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ResNet50    | 25.076      | 7.884       | 19.54 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) [[keras]](https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py) |
| ResNet101   | 23.574      | 7.208       | 31.17 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-101-deploy.prototxt) |
| ResNet152   | 23.362      | 6.914       | 43.91 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-152-deploy.prototxt) |
| ResNet50v2  |             |             | 20.97 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNet101v2 |             |             | 32.62 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNet152v2 |             |             | 45.52 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNet200v2 | 21.898      | 5.998       | 61.83 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNeXt50   | 22.518      | 6.418       | 26.74 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| ResNeXt101  | 21.426      | 5.928       | 42.79 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| Inception1  | 32.962      | 12.122      | 16.51 | [[paper]](https://arxiv.org/abs/1409.4842) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py) [[caffe-zoo]](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt) |
| Inception2  | 26.420      | 8.450       | 13.43 | [[paper]](https://arxiv.org/abs/1502.03167) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py) |
| Inception3  | 22.092      | 6.220       | 31.46 | [[paper]](https://arxiv.org/abs/1512.00567) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) [[keras]](https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py) |
| Inception4  | 19.854      | 5.032       | 58.21 | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) |
