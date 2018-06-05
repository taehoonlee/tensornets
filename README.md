# TensorNets

High level network definitions with pre-trained weights in [TensorFlow](https://github.com/tensorflow/tensorflow) (tested with `>= 1.1.0`).

## Guiding principles

- **Applicability.** Many people already have their own ML workflows, and want to put a new model on their workflows. TensorNets can be easily plugged together because it is designed as simple functional interfaces without custom classes.
- **Manageability.** Models are written in `tf.contrib.layers`, which is lightweight like PyTorch and Keras, and allows for ease of accessibility to every weight and end-point. Also, it is easy to deploy and expand a collection of pre-processing and pre-trained weights.
- **Readability.** With recent TensorFlow APIs, more factoring and less indenting can be possible. For example, all the inception variants are implemented as about 500 lines of code in [TensorNets](tensornets/inceptions.py) while 2000+ lines in [official TensorFlow models](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py).
- **Reproducibility.** You can always reproduce the original results with [simple APIs](#utilities) including feature extractions. Furthermore, you don't need to care about a version of TensorFlow beacuse compatibilities with various releases of TensorFlow have been checked with [Travis](https://travis-ci.org/taehoonlee/tensornets/builds).

## Installation

You can install TensorNets from PyPI (`pip install tensornets`) or directly from GitHub (`pip install git+https://github.com/taehoonlee/tensornets.git`).

## A quick example

Each network (see [full list](#image-classification)) is not a custom class but a function that takes and returns `tf.Tensor` as its input and output. Here is an example of `ResNet50`:

```python
import tensorflow as tf
import tensornets as nets

inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
model = nets.ResNet50(inputs)

assert isinstance(model, tf.Tensor)
```

You can load an example image by using `utils.load_img` returning a `np.ndarray` as the NHWC format:

```python
img = nets.utils.load_img('cat.png', target_size=256, crop_size=224)
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
print(nets.utils.decode_predictions(preds, top=2)[0])
[(u'n02124075', u'Egyptian_cat', 0.28067636), (u'n02127052', u'lynx', 0.16826575)]
```

You can also easily obtain values of intermediate layers with `get_middles()` and `get_outputs()`:

```python
with tf.Session() as sess:
    img = model.preprocess(img)
    sess.run(model.pretrained())
    middles = sess.run(model.get_middles(), {inputs: img})
    outputs = sess.run(model.get_outputs(), {inputs: img})

model.print_middles()
assert middles[0].shape == (1, 56, 56, 256)
assert middles[-1].shape == (1, 7, 7, 2048)

model.print_outputs()
assert sum(sum((outputs[-1] - preds) ** 2)) < 1e-8
```

TensorNets enables us to deploy well-known architectures and benchmark those results faster ⚡️. For more information, you can check out the lists of [utilities](#utilities), [examples](#examples), and [architectures](#performances).

## Object detection example

Each object detection model **can be coupled with any network in TensorNets** (see [performances](#object-detection)) and takes two arguments: a placeholder and a function acting as a stem layer. Here is an example of `YOLOv2` for PASCAL VOC:

```python
import tensorflow as tf
import tensornets as nets

inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv2(inputs, nets.Darknet19)

img = nets.utils.load_img('cat.png')

with tf.Session() as sess:
    sess.run(model.pretrained())
    preds = sess.run(model, {inputs: model.preprocess(img)})
    boxes = model.get_boxes(preds, img.shape[1:3])
```

Like other models, a detection model also returns `tf.Tensor` as its output. You can see the bounding box predictions `(x1, y1, x2, y2, score)` by using `model.get_boxes(model_output, original_img_shape)` and visualize the results:

```python
from tensornets.datasets import voc
print("%s: %s" % (voc.classnames[7], boxes[7][0]))  # 7 is cat

import numpy as np
import matplotlib.pyplot as plt
box = boxes[7][0]
plt.imshow(img[0].astype(np.uint8))
plt.gca().add_patch(plt.Rectangle(
    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
    fill=False, edgecolor='r', linewidth=2))
plt.show()
```

More detection examples such as FasterRCNN on VOC2007 are [here](https://github.com/taehoonlee/tensornets-examples/blob/master/test_all_voc_models.ipynb) 😎. Note that:

- APIs of detection models are slightly different:
  * `YOLOv3`: `sess.run(model.preds, {inputs: img})`,
  * `YOLOv2`: `sess.run(model, {inputs: img})`,
  * `FasterRCNN`: `sess.run(model, {inputs: img, model.scales: scale})`,

- `FasterRCNN` requires `roi_pooling`:
  * `git clone https://github.com/deepsense-io/roi-pooling && cd roi-pooling && vi roi_pooling/Makefile` and edit according to [here](https://github.com/tensorflow/tensorflow/issues/13607#issuecomment-335530430),
  * `python setup.py install`.

## Utilities

Besides `pretrained()` and `preprocess()`, the output `tf.Tensor` provides the following useful methods:

- `get_middles()`: returns a list of all the representative `tf.Tensor` end-points,
- `get_outputs()`: returns a list of all the `tf.Tensor` end-points,
- `get_weights()`: returns a list of all the `tf.Tensor` weight matrices,
- `print_middles()`: prints all the representative end-points,
- `print_outputs()`: prints all the end-points,
- `print_weights()`: prints all the weight matrices,
- `print_summary()`: prints the numbers of layers, weight matrices, and parameters.


Example outputs of print methods are:

```
>>> model.print_middles()
Scope: resnet50
conv2/block1/out:0 (?, 56, 56, 256)
conv2/block2/out:0 (?, 56, 56, 256)
conv2/block3/out:0 (?, 56, 56, 256)
conv3/block1/out:0 (?, 28, 28, 512)
conv3/block2/out:0 (?, 28, 28, 512)
conv3/block3/out:0 (?, 28, 28, 512)
conv3/block4/out:0 (?, 28, 28, 512)
conv4/block1/out:0 (?, 14, 14, 1024)
...

>>> model.print_outputs()
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

>>> model.print_weights()
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

>>> model.print_summary()
Scope: resnet50
Total layers: 54
Total weights: 320
Total parameters: 25,636,712
```

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

### Image classification

- The top-k errors were obtained with TensorNets on **ImageNet validation set** and may slightly differ from the original ones. The crop size is 224x224 for all but 331x331 for NASNetAlarge, 299x299 for Inception3,4,ResNet2, and ResNet50-152v2.
  * Top-1: single center crop, top-1 error
  * Top-5: single center crop, top-5 error
  * 10-5: ten crops (1 center + 4 corners and those mirrored ones), top-5 error
  * Size: rounded the number of parameters (w/ fully-connected layers)
  * Stem: rounded the number of parameters (w/o fully-connected layers)
- The computation times were measured on NVIDIA Tesla P100 (3584 cores, 16 GB global memory) with cuDNN 6.0 and CUDA 8.0.
  * Speed: milliseconds for inferences of 100 images

|              | Top-1       | Top-5       | 10-5        | Size   | Stem   | Speed | References                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|--------------|-------------|-------------|-------------|--------|--------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ResNet50](tensornets/resnets.py#L85)             | 25.126      | 7.982       | 6.842       | 25.6M  | 23.6M  | 195.4 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) <br /> [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py) |
| [ResNet101](tensornets/resnets.py#L113)           | 23.580      | 7.214       | 6.092       | 44.7M  | 42.7M  | 311.7 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) <br /> [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-101-deploy.prototxt) |
| [ResNet152](tensornets/resnets.py#L141)           | 23.396      | 6.882       | 5.908       | 60.4M  | 58.4M  | 439.1 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) <br /> [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-152-deploy.prototxt) |
| [ResNet50v2](tensornets/resnets.py#L98)           | 24.526      | 7.252       | 6.012       | 25.6M  | 23.6M  | 209.7 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet101v2](tensornets/resnets.py#L126)         | 23.116      | 6.488       | 5.230       | 44.7M  | 42.6M  | 326.2 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet152v2](tensornets/resnets.py#L154)         | 22.236      | 6.080       | 4.960       | 60.4M  | 58.3M  | 455.2 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet200v2](tensornets/resnets.py#L169)         | 21.714      | 5.848       | 4.830       | 64.9M  | 62.9M  | 618.3 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNeXt50c32](tensornets/resnets.py#L184)        | 22.260      | 6.190       | 5.410       | 25.1M  | 23.0M  | 267.4 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [ResNeXt101c32](tensornets/resnets.py#L200)       | 21.270      | 5.706       | 4.842       | 44.3M  | 42.3M  | 427.9 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [ResNeXt101c64](tensornets/resnets.py#L216)       | 20.506      | 5.408       | 4.564       | 83.7M  | 81.6M  | 877.8 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [WideResNet50](tensornets/resnets.py#L232)        | 21.982      | 6.066       | 5.116       | 69.0M  | 66.9M  | 358.1 | [[paper]](https://arxiv.org/abs/1605.07146) [[torch]](https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/wide-resnet.lua) |
| [Inception1](tensornets/inceptions.py#L67)        | 33.160      | 12.324      | 10.246      | 7.0M   | 6.0M   | 165.1 | [[paper]](https://arxiv.org/abs/1409.4842) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py) [[caffe-zoo]](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt) |
| [Inception2](tensornets/inceptions.py#L105)       | 26.296      | 8.270       | 6.882       | 11.2M  | 10.2M  | 134.3 | [[paper]](https://arxiv.org/abs/1502.03167) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py) |
| [Inception3](tensornets/inceptions.py#L143)       | 22.102      | 6.280       | 5.038       | 23.9M  | 21.8M  | 314.6 | [[paper]](https://arxiv.org/abs/1512.00567) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/inception_v3.py) |
| [Inception4](tensornets/inceptions.py#L179)       | 19.880      | 5.022       | 4.206       | 42.7M  | 41.2M  | 582.1 | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) |
| [InceptionResNet2](tensornets/inceptions.py#L264) | 19.744      | 4.748       | 3.962       | 55.9M  | 54.3M  | 656.8 | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py) |
| [NASNetAlarge](tensornets/nasnets.py#L103)        | 17.502      | 3.996       | 3.412       | 93.5M  | 89.5M  | 2081  | [[paper]](https://arxiv.org/abs/1707.07012) [[tf-slim]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) |
| [NASNetAmobile](tensornets/nasnets.py#L111)       | 25.634      | 8.146       | 6.758       | 7.7M   | 6.7M   | 165.8 | [[paper]](https://arxiv.org/abs/1707.07012) [[tf-slim]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) |
| [PNASNetlarge](tensornets/nasnets.py#L150)        | 17.366      | 3.950       | 3.358       | 86.2M  | 81.9M  | 1978  | [[paper]](https://arxiv.org/abs/1712.00559) [[tf-slim]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) |
| [VGG16](tensornets/vggs.py#L71)                   | 28.732      | 9.950       | 8.834       | 138.4M | 14.7M  | 348.4 | [[paper]](https://arxiv.org/abs/1409.1556) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py) |
| [VGG19](tensornets/vggs.py#L78)                   | 28.744      | 10.012      | 8.774       | 143.7M | 20.0M  | 399.8 | [[paper]](https://arxiv.org/abs/1409.1556) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py) |
| [DenseNet121](tensornets/densenets.py#L63)        | 25.480      | 8.022       | 6.842       | 8.1M   | 7.0M   | 202.9 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [DenseNet169](tensornets/densenets.py#L71)        | 23.926      | 6.892       | 6.140       | 14.3M  | 12.6M  | 219.1 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [DenseNet201](tensornets/densenets.py#L79)        | 22.936      | 6.542       | 5.724       | 20.2M  | 18.3M  | 272.0 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [MobileNet25](tensornets/mobilenets.py#L156)      | 48.418      | 24.208      | 21.196      | 0.5M   | 0.2M   | 34.46 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet50](tensornets/mobilenets.py#L163)      | 35.708      | 14.376      | 12.180      | 1.3M   | 0.8M   | 52.46 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet75](tensornets/mobilenets.py#L170)      | 31.588      | 11.758      | 9.878       | 2.6M   | 1.8M   | 70.11 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet100](tensornets/mobilenets.py#L177)     | 29.576      | 10.496      | 8.774       | 4.3M   | 3.2M   | 83.41 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet35v2](tensornets/mobilenets.py#L184)    | 39.914      | 17.568      | 15.422      | 1.7M   | 0.4M   | 57.04 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNet50v2](tensornets/mobilenets.py#L191)    | 34.806      | 13.938      | 11.976      | 2.0M   | 0.7M   | 64.35 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNet75v2](tensornets/mobilenets.py#L198)    | 30.468      | 10.824      | 9.188       | 2.7M   | 1.4M   | 88.68 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNet100v2](tensornets/mobilenets.py#L205)   | 28.664      | 9.858       | 8.322       | 3.5M   | 2.3M   | 93.82 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNet130v2](tensornets/mobilenets.py#L212)   | 25.320      | 7.878       | 6.728       | 5.4M   | 3.8M   | 130.4 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNet140v2](tensornets/mobilenets.py#L219)   | 24.770      | 7.578       | 6.518       | 6.2M   | 4.4M   | 132.9 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [SqueezeNet](tensornets/squeezenets.py#L45)       | 45.566      | 21.960      | 18.578      | 1.2M   | 0.7M   | 71.43 | [[paper]](https://arxiv.org/abs/1602.07360) [[caffe]](https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/train_val.prototxt) |

### Object detection

- The object detection models can be coupled with any network but mAPs could be measured only for the models with pre-trained weights. Note that:
  * `YOLOv3VOC` was trained by taehoonlee with [this recipe](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-voc.cfg) modified as `max_batches=70000, steps=40000,60000`,
  * `YOLOv2VOC` is equivalent to `YOLOv2(inputs, Darknet19)`,
  * `TinyYOLOv2VOC`: `TinyYOLOv2(inputs, TinyDarknet19)`,
  * `FasterRCNN_ZF_VOC`: `FasterRCNN(inputs, ZF)`,
  * `FasterRCNN_VGG16_VOC`: `FasterRCNN(inputs, VGG16, stem_out='conv5/3')`.
- The mAPs were obtained with TensorNets on **PASCAL VOC2007 test set** and may slightly differ from the original ones.
- The test input sizes were the numbers reported as the best in the papers:
  * `YOLOv3`, `YOLOv2`: 416x416
  * `FasterRCNN`: min\_shorter\_side=600, max\_longer\_side=1000
- The sizes stand for rounded the number of parameters.
- The computation times were measured on NVIDIA Tesla P100 (3584 cores, 16 GB global memory) with cuDNN 6.0 and CUDA 8.0.
  * Speed: milliseconds only for network inferences of a 416x416 single image
  * FPS: 1000 / speed

|                                                                        | mAP    | Size   | Speed |  FPS  | References |
|------------------------------------------------------------------------|--------|--------|-------|-------|------------|
| [YOLOv3VOC](tensornets/references/yolos.py#L175)                       | 0.7423 | 62M    | 24.09 | 41.51 | [[paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[darknet]](https://pjreddie.com/darknet/yolo/) [[darkflow]](https://github.com/thtrieu/darkflow) |
| [YOLOv2VOC](tensornets/references/yolos.py#L195)                       | 0.7320 | 51M    | 14.75 | 67.80 | [[paper]](https://arxiv.org/abs/1612.08242) [[darknet]](https://pjreddie.com/darknet/yolov2/) [[darkflow]](https://github.com/thtrieu/darkflow) |
| [TinyYOLOv2VOC](tensornets/references/yolos.py#L205)                   | 0.5303 | 16M    | 6.534 | 153.0 | [[paper]](https://arxiv.org/abs/1612.08242) [[darknet]](https://pjreddie.com/darknet/yolov2/) [[darkflow]](https://github.com/thtrieu/darkflow) |
| [FasterRCNN\_ZF\_VOC](tensornets/references/rcnns.py#L151)               | 0.4466 | 59M    | 241.4 | 3.325 | [[paper]](https://arxiv.org/abs/1506.01497) [[caffe]](https://github.com/rbgirshick/py-faster-rcnn) [[roi-pooling]](https://github.com/deepsense-ai/roi-pooling) |
| [FasterRCNN\_VGG16\_VOC](tensornets/references/rcnns.py#L187)            | 0.6872 | 137M   | 300.7 | 4.143 | [[paper]](https://arxiv.org/abs/1506.01497) [[caffe]](https://github.com/rbgirshick/py-faster-rcnn) [[roi-pooling]](https://github.com/deepsense-ai/roi-pooling) |

## News 📰

- PNASNetlarge is released, [12 May 2018](https://github.com/taehoonlee/tensornets/commit/e2e0f0f7791731d3b7dfa989cae569c15a22cdd6).
- The six variants of MobileNetv2 are released, [5 May 2018](https://github.com/taehoonlee/tensornets/commit/fb429b6637f943875249dff50f4bc6220d9d50bf).
- YOLOv3 for COCO and VOC are released, [4 April 2018](https://github.com/taehoonlee/tensornets/commit/d8b2d8a54dc4b775a174035da63561028deb6624).
- Generic object detection models for YOLOv2 and FasterRCNN are released, [26 March 2018](https://github.com/taehoonlee/tensornets/commit/67915e659d2097a96c82ba7740b9e43a8c69858d).

## Future work 🔥

- Add training codes.
- Add image classification models (PolyNet).
- Add object detection models (MaskRCNN, SSD).
- Add image segmentation models (FCN, UNet).
- Add image datasets (COCO, OpenImages).
- Add style transfer examples which can be coupled with any network in TensorNets.
- Add speech and language models with representative datasets (WaveNet, ByteNet).
