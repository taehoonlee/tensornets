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
from tensornets import utils

img = utils.load_img('cat.png', target_size=256, crop_size=224)

assert img.shape == (1, 224, 224, 3)
```

```python
img = nets.preprocess('resnet', img)
with tf.Session() as sess:
    nets.pretrained([model1, model2])
    preds = sess.run([model1, model2], {inputs: img})
```

```python
for pred in preds:
    print('Predicted:', utils.decode_predictions(pred, top=3)[0])
```

```
('Predicted:', [(u'n02124075', u'Egyptian_cat', 0.28067607), (u'n02127052', u'lynx', 0.16826589), (u'n02123597', u'Siamese_cat', 0.088474944)])
('Predicted:', [(u'n02124075', u'Egyptian_cat', 0.10482649), (u'n03482405', u'hamper', 0.08210019), (u'n02808304', u'bath_towel', 0.066759109)])
```

```python
utils.print_summary(model1)
utils.print_summary(model2)
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
utils.print_weights(model1)
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
utils.print_outputs(model2)
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
- Speed: ms for inferences of 100 images on P100

|             | Top-1 error | Top-5 error | Speed | References                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|-------------|-------------|-------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ResNet50    | 25.076      | 7.884       | 195.4 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) [[keras]](https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py) |
| ResNet101   | 23.574      | 7.208       | 311.7 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-101-deploy.prototxt) |
| ResNet152   | 23.362      | 6.914       | 439.1 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-152-deploy.prototxt) |
| ResNet50v2  |             |             | 209.7 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNet101v2 |             |             | 326.2 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNet152v2 |             |             | 455.2 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNet200v2 | 21.898      | 5.998       | 618.3 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNeXt50   | 22.518      | 6.418       | 267.4 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| ResNeXt101  | 21.426      | 5.928       | 427.9 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| Inception1  | 32.962      | 12.122      | 165.1 | [[paper]](https://arxiv.org/abs/1409.4842) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py) [[caffe-zoo]](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt) |
| Inception2  | 26.420      | 8.450       | 134.3 | [[paper]](https://arxiv.org/abs/1502.03167) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py) |
| Inception3  | 22.092      | 6.220       | 314.6 | [[paper]](https://arxiv.org/abs/1512.00567) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) [[keras]](https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py) |
| Inception4  | 19.854      | 5.032       | 582.1 | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) |
| DenseNet121  | 25.550      | 8.174       | 202.9 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| DenseNet169  | 24.092      | 7.172       | 219.1 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| DenseNet201  | 22.988      | 6.700       | 272.0 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| MobileNet25  | 48.346      | 24.150      | 29.27 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| MobileNet50  | 35.594      | 14.390      | 42.32 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| MobileNet75  | 31.520      | 11.710      | 57.23 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| MobileNet100 | 29.474      | 10.416      | 70.69 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| SqueezeNet   | 44.656      | 21.432      | 71.43 | [[paper]](https://arxiv.org/abs/1602.07360) [[caffe]](https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/train_val.prototxt) |
