# TensorNets

High level network definitions with pre-trained weights in [TensorFlow](https://github.com/tensorflow/tensorflow) (tested with `>= 1.2.0`).

## Guiding principles

- **Applicability.** Many people already have their own ML workflows, and want to put a new model on their workflows. TensorNets can be easily plugged together because it is designed as simple functional interfaces without custom classes.
- **Manageability.** Models are written in `tf.contrib.layers`, which is lightweight like PyTorch and Keras, and allows for ease of accessibility to every weight and end-point. Also, it is easy to deploy and expand a collection of pre-processing and pre-trained weights.
- **Readability.** With recent TensorFlow APIs, more factoring and less indenting can be possible. For example, all the inception variants are implemented as about 500 lines of code in [TensorNets](tensornets/inceptions.py) while 2000+ lines in [official TensorFlow models](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py).

## A quick example

Each network (see [full list](#performances)) is not a custom class but a function that takes and returns `tf.Tensor` as its input and output. Here is an example of `ResNet50`:

```python
import tensorflow as tf
import tensornets as nets

inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
model = nets.ResNet50(inputs)

assert isinstance(model, tf.Tensor)
```

You can load an example image by using `utils.load_img` returning a `np.ndarray` as the NHWC format:

```python
from tensornets import utils
img = utils.load_img('cat.png', target_size=256, crop_size=224)
assert img.shape == (1, 224, 224, 3)
```

Once your network is created, you can run with regular TensorFlow APIs 😊 because all the networks in TensorNets always return `tf.Tensor`. Using pre-trained weights and pre-processing are as easy as [`pretrained()`](tensornets/pretrained.py) and [`preprocess()`](tensornets/preprocess.py) to reproduce the original results:

```python
with tf.Session() as sess:
    img = model.preprocess(img)  # equivalent to img = nets.preprocess(model, img)
    sess.run(model.pretrained())  # equivalent to nets.pretrained(model)
    preds = sess.run(model, {inputs: img})
```

You can see the most probable classes:

```python
print(utils.decode_predictions(preds, top=2)[0])
[(u'n02124075', u'Egyptian_cat', 0.28067636), (u'n02127052', u'lynx', 0.16826575)]
```

TensorNets enables us to deploy well-known architectures and benchmark those results faster ⚡️. For more information, you can check out the lists of [utilities](#utilities), [examples](#examples), and [architectures](#performances).

## Utilities

An example output of `utils.print_summary(model)`:

```
Scope: resnet50
Total layers: 54
Total weights: 320
Total parameters: 25,636,712
```

An example output of `utils.print_weights(model)`:

```
Scope: resnet50
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

- `utils.get_weights(model)` returns a list of all the `tf.Tensor` weights as shown in the above

An example output of `utils.print_outputs(model)`:

```
Scope: resnet50
conv1/pad:0 (?, 230, 230, 3)
conv1/conv/BiasAdd:0 (?, 112, 112, 64)
conv1/bn/batchnorm/add_1:0 (?, 112, 112, 64)
conv1/relu:0 (?, 112, 112, 64)
pool1/pad:0 (?, 114, 114, 64)
pool1/MaxPool:0 (?, 56, 56, 64)
conv2/block1/0/conv/BiasAdd:0 (?, 56, 56, 256)
conv2/block1/0/bn/batchnorm/add_1:0 (?, 56, 56, 256)
conv2/block1/1/conv/BiasAdd:0 (?, 56, 56, 64)
conv2/block1/1/bn/batchnorm/add_1:0 (?, 56, 56, 64)
conv2/block1/1/relu:0 (?, 56, 56, 64)
...
```

- `utils.get_outputs(model)` returns a list of all the `tf.Tensor` end-points as shown in the above

## Examples

- Comparison of different networks:

```python
inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
models = [
    nets.MobileNet75(inputs),
    nets.MobileNet100(inputs),
    nets.SqueezeNet(inputs),
]

img = utils.load_img('cat.png', target_size=256, crop_size=224)
imgs = nets.preprocess(models, img)

with tf.Session() as sess:
    nets.pretrained(models)
    for (model, img) in zip(models, imgs):
        preds = sess.run(model, {inputs: img})
        print(utils.decode_predictions(preds, top=2)[0])
```

- Transfer learning:

```python
inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
outputs = tf.placeholder(tf.float32, [None, 50])
model = nets.DenseNet169(inputs, is_training=True, classes=50)

loss = tf.losses.softmax_cross_entropy(outputs, model)
train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

with tf.Session() as sess:
    nets.pretrained(model)
    # for (x, y) in your NumPy data (the NHWC and one-hot format):
        sess.run(train, {inputs: x, outputs: y})
```

- Using multi-GPU:

```python
inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
models = []

with tf.device('gpu:0'):
    models.append(nets.ResNeXt50(inputs))

with tf.device('gpu:1'):
    models.append(nets.DenseNet201(inputs))

from tensornets.preprocess import fb_preprocess
img = utils.load_img('cat.png', target_size=256, crop_size=224)
img = fb_preprocess(img)

with tf.Session() as sess:
    nets.pretrained(models)
    preds = sess.run(models, {inputs: img})
    for pred in preds:
        print(utils.decode_predictions(pred, top=2)[0])
```

## Performances

- The top-k errors were obtained with TensorNets and may slightly differ from the original ones. The crop size is 224x224 for all but 331x331 for NASNetAlarge, 299x299 for Inception3,4,ResNet2, and ResNet50-152v2.
  * Top-1: single center crop, top-1 error
  * Top-5: single center crop, top-5 error
  * 10-5: ten crops (1 center + 4 corners and those mirrored ones), top-5 error
  * Size: rounded the number of parameters
- The computation times were measured on NVIDIA Tesla P100 (3584 cores, 16 GB global memory) with cuDNN 6.0 and CUDA 8.0.
  * Speed: milliseconds for inferences of 100 images

|              | Top-1       | Top-5       | 10-5        | Size   | Speed | References                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|--------------|-------------|-------------|-------------|--------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ResNet50](tensornets/resnets.py#L76)             | 25.076      | 7.884       | 6.762       | 26M    | 195.4 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py) |
| [ResNet101](tensornets/resnets.py#L102)           | 23.574      | 7.208       | 6.142       | 45M    | 311.7 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-101-deploy.prototxt) |
| [ResNet152](tensornets/resnets.py#L128)           | 23.362      | 6.914       | 5.956       | 60M    | 439.1 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-152-deploy.prototxt) |
| [ResNet50v2](tensornets/resnets.py#L88)           | 24.442      | 7.174       | 6.036       | 26M    | 209.7 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet101v2](tensornets/resnets.py#L114)         | 23.064      | 6.476       | 5.212       | 45M    | 326.2 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet152v2](tensornets/resnets.py#L140)         | 22.300      | 6.066       | 5.002       | 60M    | 455.2 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet200v2](tensornets/resnets.py#L154)         | 21.898      | 5.998       | 5.046       | 65M    | 618.3 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNeXt50c32](tensornets/resnets.py#L168)        | 22.518      | 6.418       | 5.532       | 25M    | 267.4 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [ResNeXt101c32](tensornets/resnets.py#L180)       | 21.426      | 5.928       | 5.062       | 44M    | 427.9 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [ResNeXt101c64](tensornets/resnets.py#L192)       | 20.696      | 5.598       | 4.762       | 84M    | 877.8 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [WideResNet50](tensornets/resnets.py#L204)        | 22.308      | 6.238       | 5.276       | 69M    | 358.1 | [[paper]](https://arxiv.org/abs/1605.07146) [[torch]](https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/wide-resnet.lua) |
| [Inception1](tensornets/inceptions.py#L66)        | 32.962      | 12.122      | 10.012      | 7.0M   | 165.1 | [[paper]](https://arxiv.org/abs/1409.4842) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py) [[caffe-zoo]](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt) |
| [Inception2](tensornets/inceptions.py#L102)       | 26.420      | 8.450       | 7.360       | 11M    | 134.3 | [[paper]](https://arxiv.org/abs/1502.03167) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py) |
| [Inception3](tensornets/inceptions.py#L138)       | 22.092      | 6.220       | 5.058       | 24M    | 314.6 | [[paper]](https://arxiv.org/abs/1512.00567) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/inception_v3.py) |
| [Inception4](tensornets/inceptions.py#L172)       | 19.854      | 5.032       | 4.238       | 43M    | 582.1 | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) |
| [InceptionResNet2](tensornets/inceptions.py#L254) | 19.660      | 4.806       | 3.968       | 56M    | 656.8 | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py) |
| [NASNetAlarge](tensornets/nasnets.py#L98)         | 17.486      | 3.940       | 3.408       | 94M    | 2081  | [[paper]](https://arxiv.org/abs/1707.07012) [[tf-slim]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) |
| [NASNetAmobile](tensornets/nasnets.py#L105)       | 25.676      | 8.226       | 6.840       | 7.7M   | 165.8 | [[paper]](https://arxiv.org/abs/1707.07012) [[tf-slim]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) |
| [VGG16](tensornets/vggs.py#L60)                   | 29.066      | 10.168      | 9.014       | 138M   | 348.4 | [[paper]](https://arxiv.org/abs/1409.1556) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py) |
| [VGG19](tensornets/vggs.py#L66)                   | 29.132      | 10.232      | 9.030       | 144M   | 399.8 | [[paper]](https://arxiv.org/abs/1409.1556) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py) |
| [DenseNet121](tensornets/densenets.py#L60)        | 25.550      | 8.174       | 7.074       | 8.1M   | 202.9 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [DenseNet169](tensornets/densenets.py#L66)        | 24.092      | 7.172       | 6.350       | 14M    | 219.1 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [DenseNet201](tensornets/densenets.py#L72)        | 22.988      | 6.700       | 5.856       | 20M    | 272.0 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [MobileNet25](tensornets/mobilenets.py#L83)       | 48.346      | 24.150      | 21.020      | 0.48M  | 29.27 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet50](tensornets/mobilenets.py#L89)       | 35.594      | 14.390      | 12.046      | 1.3M   | 42.32 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet75](tensornets/mobilenets.py#L95)       | 31.520      | 11.710      | 9.668       | 2.6M   | 57.23 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet100](tensornets/mobilenets.py#L101)     | 29.474      | 10.416      | 8.642       | 4.3M   | 70.69 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [SqueezeNet](tensornets/squeezenets.py#L47)       | 44.656      | 21.432      | 17.972      | 1.2M   | 71.43 | [[paper]](https://arxiv.org/abs/1602.07360) [[caffe]](https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/train_val.prototxt) |
