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
    nets.pretrained(model)
    img = nets.preprocess(model, img)
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

- The top-k errors were obtained with TensorNets (single center crop 224x224 except Inception3,4,ResNet2 and ResNet50-152v2 299x299) and may slightly differ from the original ones.
- The computation times were measured on NVIDIA Tesla P100 (3584 cores, 16 GB global memory) with cuDNN 6.0 and CUDA 8.0.

|              | Top-1 error | Top-5 error | Speed (ms) | References                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|--------------|-------------|-------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ResNet50     | 25.076      | 7.884       | 195.4 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) [[keras]](https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py) |
| ResNet101    | 23.574      | 7.208       | 311.7 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-101-deploy.prototxt) |
| ResNet152    | 23.362      | 6.914       | 439.1 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-152-deploy.prototxt) |
| ResNet50v2   | 24.442      | 7.174       | 209.7 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNet101v2  | 23.064      | 6.476       | 326.2 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNet152v2  | 22.300      | 6.066       | 455.2 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNet200v2  | 21.898      | 5.998       | 618.3 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| ResNeXt50    | 22.518      | 6.418       | 267.4 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| ResNeXt101   | 21.426      | 5.928       | 427.9 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| WideResNet50 | 22.308      | 6.238       | 358.1 | [[paper]](https://arxiv.org/abs/1605.07146) [[torch]](https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/wide-resnet.lua) |
| Inception1   | 32.962      | 12.122      | 165.1 | [[paper]](https://arxiv.org/abs/1409.4842) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py) [[caffe-zoo]](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt) |
| Inception2   | 26.420      | 8.450       | 134.3 | [[paper]](https://arxiv.org/abs/1502.03167) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py) |
| Inception3   | 22.092      | 6.220       | 314.6 | [[paper]](https://arxiv.org/abs/1512.00567) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) [[keras]](https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py) |
| Inception4   | 19.854      | 5.032       | 582.1 | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) |
| InceptionResNet2 | 19.660  | 4.806       | 656.8 | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py) |
| DenseNet121  | 25.550      | 8.174       | 202.9 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| DenseNet169  | 24.092      | 7.172       | 219.1 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| DenseNet201  | 22.988      | 6.700       | 272.0 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| MobileNet25  | 48.346      | 24.150      | 29.27 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| MobileNet50  | 35.594      | 14.390      | 42.32 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| MobileNet75  | 31.520      | 11.710      | 57.23 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| MobileNet100 | 29.474      | 10.416      | 70.69 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| SqueezeNet   | 44.656      | 21.432      | 71.43 | [[paper]](https://arxiv.org/abs/1602.07360) [[caffe]](https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/train_val.prototxt) |
