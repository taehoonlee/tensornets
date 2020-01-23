# TensorNets

High level network definitions with pre-trained weights in [TensorFlow](https://github.com/tensorflow/tensorflow) (tested with `2.1.0 >=` TF `>= 1.4.0`).

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
# import tensorflow.compat.v1 as tf  # for TF 2
import tensornets as nets
# tf.disable_v2_behavior()  # for TF 2

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

You can also easily obtain values of intermediate layers with `middles()` and `outputs()`:

```python
with tf.Session() as sess:
    img = model.preprocess(img)
    sess.run(model.pretrained())
    middles = sess.run(model.middles(), {inputs: img})
    outputs = sess.run(model.outputs(), {inputs: img})

model.print_middles()
assert middles[0].shape == (1, 56, 56, 256)
assert middles[-1].shape == (1, 7, 7, 2048)

model.print_outputs()
assert sum(sum((outputs[-1] - preds) ** 2)) < 1e-8
```

TensorNets enables us to deploy well-known architectures and benchmark those results faster ⚡️. For more information, you can check out the lists of [utilities](#utilities), [examples](#examples), and [architectures](#performance).

## Object detection example

Each object detection model **can be coupled with any network in TensorNets** (see [performance](#object-detection)) and takes two arguments: a placeholder and a function acting as a stem layer. Here is an example of `YOLOv2` for PASCAL VOC:

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

- `logits`: returns the `tf.Tensor` logits (the values before the softmax),
- `middles()` (=`get_middles()`): returns a list of all the representative `tf.Tensor` end-points,
- `outputs()` (=`get_outputs()`): returns a list of all the `tf.Tensor` end-points,
- `weights()` (=`get_weights()`): returns a list of all the `tf.Tensor` weight matrices,
- `summary()` (=`print_summary()`): prints the numbers of layers, weight matrices, and parameters,
- `print_middles()`: prints all the representative end-points,
- `print_outputs()`: prints all the end-points,
- `print_weights()`: prints all the weight matrices.


<details>
<summary>Example outputs of print methods are:</summary>

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

>>> model.summary()
Scope: resnet50
Total layers: 54
Total weights: 320
Total parameters: 25,636,712
```
</details>

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

loss = tf.losses.softmax_cross_entropy(outputs, model.logits)
train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

with tf.Session() as sess:
    nets.pretrained(model)
    for (x, y) in your_NumPy_data:  # the NHWC and one-hot format
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

## Performance

### Image classification

- The top-k accuracies were obtained with TensorNets on **ImageNet validation set** and may slightly differ from the original ones.
  * Input: input size fed into models
  * Top-1: single center crop, top-1 accuracy
  * Top-5: single center crop, top-5 accuracy
  * Size: rounded the number of parameters (w/ fully-connected layers)
  * Stem: rounded the number of parameters (w/o fully-connected layers)
- The computation times were measured on NVIDIA Tesla P100 (3584 cores, 16 GB global memory) with cuDNN 6.0 and CUDA 8.0.
  * Speed: milliseconds for inferences of 100 images

|              | Input | Top-1       | Top-5       | Size   | Stem   | Speed | References                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|--------------|-------|-------------|-------------|--------|--------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ResNet50](tensornets/resnets.py#L85)             |  224  | 74.874      | 92.018      | 25.6M  | 23.6M  | 195.4 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) <br /> [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py) |
| [ResNet101](tensornets/resnets.py#L113)           |  224  | 76.420      | 92.786      | 44.7M  | 42.7M  | 311.7 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) <br /> [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-101-deploy.prototxt) |
| [ResNet152](tensornets/resnets.py#L141)           |  224  | 76.604      | 93.118      | 60.4M  | 58.4M  | 439.1 | [[paper]](https://arxiv.org/abs/1512.03385) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua) <br /> [[caffe]](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-152-deploy.prototxt) |
| [ResNet50v2](tensornets/resnets.py#L98)           |  299  | 75.960      | 93.034      | 25.6M  | 23.6M  | 209.7 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet101v2](tensornets/resnets.py#L126)         |  299  | 77.234      | 93.816      | 44.7M  | 42.6M  | 326.2 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet152v2](tensornets/resnets.py#L154)         |  299  | 78.032      | 94.162      | 60.4M  | 58.3M  | 455.2 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNet200v2](tensornets/resnets.py#L169)         |  224  | 78.286      | 94.152      | 64.9M  | 62.9M  | 618.3 | [[paper]](https://arxiv.org/abs/1603.05027) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) [[torch-fb]](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua) |
| [ResNeXt50c32](tensornets/resnets.py#L184)        |  224  | 77.740      | 93.810      | 25.1M  | 23.0M  | 267.4 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [ResNeXt101c32](tensornets/resnets.py#L200)       |  224  | 78.730      | 94.294      | 44.3M  | 42.3M  | 427.9 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [ResNeXt101c64](tensornets/resnets.py#L216)       |  224  | 79.494      | 94.592      | 83.7M  | 81.6M  | 877.8 | [[paper]](https://arxiv.org/abs/1611.05431) [[torch-fb]](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua) |
| [WideResNet50](tensornets/resnets.py#L232)        |  224  | 78.018      | 93.934      | 69.0M  | 66.9M  | 358.1 | [[paper]](https://arxiv.org/abs/1605.07146) [[torch]](https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/wide-resnet.lua) |
| [Inception1](tensornets/inceptions.py#L62)        |  224  | 66.840      | 87.676      | 7.0M   | 6.0M   | 165.1 | [[paper]](https://arxiv.org/abs/1409.4842) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py) [[caffe-zoo]](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt) |
| [Inception2](tensornets/inceptions.py#L100)       |  224  | 74.680      | 92.156      | 11.2M  | 10.2M  | 134.3 | [[paper]](https://arxiv.org/abs/1502.03167) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py) |
| [Inception3](tensornets/inceptions.py#L138)       |  299  | 77.946      | 93.758      | 23.9M  | 21.8M  | 314.6 | [[paper]](https://arxiv.org/abs/1512.00567) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/inception_v3.py) |
| [Inception4](tensornets/inceptions.py#L174)       |  299  | 80.120      | 94.978      | 42.7M  | 41.2M  | 582.1 | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) |
| [InceptionResNet2](tensornets/inceptions.py#L259) |  299  | 80.256      | 95.252      | 55.9M  | 54.3M  | 656.8 | [[paper]](https://arxiv.org/abs/1602.07261) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py) |
| [NASNetAlarge](tensornets/nasnets.py#L101)        |  331  | 82.498      | 96.004      | 93.5M  | 89.5M  | 2081  | [[paper]](https://arxiv.org/abs/1707.07012) [[tf-slim]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) |
| [NASNetAmobile](tensornets/nasnets.py#L109)       |  224  | 74.366      | 91.854      | 7.7M   | 6.7M   | 165.8 | [[paper]](https://arxiv.org/abs/1707.07012) [[tf-slim]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) |
| [PNASNetlarge](tensornets/nasnets.py#L148)        |  331  | 82.634      | 96.050      | 86.2M  | 81.9M  | 1978  | [[paper]](https://arxiv.org/abs/1712.00559) [[tf-slim]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) |
| [VGG16](tensornets/vggs.py#L69)                   |  224  | 71.268      | 90.050      | 138.4M | 14.7M  | 348.4 | [[paper]](https://arxiv.org/abs/1409.1556) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py) |
| [VGG19](tensornets/vggs.py#L76)                   |  224  | 71.256      | 89.988      | 143.7M | 20.0M  | 399.8 | [[paper]](https://arxiv.org/abs/1409.1556) [[keras]](https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py) |
| [DenseNet121](tensornets/densenets.py#L64)        |  224  | 74.972      | 92.258      | 8.1M   | 7.0M   | 202.9 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [DenseNet169](tensornets/densenets.py#L72)        |  224  | 76.176      | 93.176      | 14.3M  | 12.6M  | 219.1 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [DenseNet201](tensornets/densenets.py#L80)        |  224  | 77.320      | 93.620      | 20.2M  | 18.3M  | 272.0 | [[paper]](https://arxiv.org/abs/1608.06993) [[torch]](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua) |
| [MobileNet25](tensornets/mobilenets.py#L153)      |  224  | 51.582      | 75.792      | 0.5M   | 0.2M   | 34.46 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet50](tensornets/mobilenets.py#L160)      |  224  | 64.292      | 85.624      | 1.3M   | 0.8M   | 52.46 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet75](tensornets/mobilenets.py#L167)      |  224  | 68.412      | 88.242      | 2.6M   | 1.8M   | 70.11 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet100](tensornets/mobilenets.py#L174)     |  224  | 70.424      | 89.504      | 4.3M   | 3.2M   | 83.41 | [[paper]](https://arxiv.org/abs/1704.04861) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) |
| [MobileNet35v2](tensornets/mobilenets.py#L181)    |  224  | 60.086      | 82.432      | 1.7M   | 0.4M   | 57.04 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNet50v2](tensornets/mobilenets.py#L188)    |  224  | 65.194      | 86.062      | 2.0M   | 0.7M   | 64.35 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNet75v2](tensornets/mobilenets.py#L195)    |  224  | 69.532      | 89.176      | 2.7M   | 1.4M   | 88.68 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNet100v2](tensornets/mobilenets.py#L202)   |  224  | 71.336      | 90.142      | 3.5M   | 2.3M   | 93.82 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNet130v2](tensornets/mobilenets.py#L209)   |  224  | 74.680      | 92.122      | 5.4M   | 3.8M   | 130.4 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [MobileNet140v2](tensornets/mobilenets.py#L216)   |  224  | 75.230      | 92.422      | 6.2M   | 4.4M   | 132.9 | [[paper]](https://arxiv.org/abs/1801.04381) [[tf-slim]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) |
| [SqueezeNet](tensornets/squeezenets.py#L44)       |  224  | 54.434      | 78.040      | 1.2M   | 0.7M   | 71.43 | [[paper]](https://arxiv.org/abs/1602.07360) [[caffe]](https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/train_val.prototxt) |

### Object detection

- The object detection models can be coupled with any network but mAPs could be measured only for the models with pre-trained weights. Note that:
  * `YOLOv3VOC` was trained by taehoonlee with [this recipe](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-voc.cfg) modified as `max_batches=70000, steps=40000,60000`,
  * `YOLOv2VOC` is equivalent to `YOLOv2(inputs, Darknet19)`,
  * `TinyYOLOv2VOC`: `TinyYOLOv2(inputs, TinyDarknet19)`,
  * `FasterRCNN_ZF_VOC`: `FasterRCNN(inputs, ZF)`,
  * `FasterRCNN_VGG16_VOC`: `FasterRCNN(inputs, VGG16, stem_out='conv5/3')`.
- The mAPs were obtained with TensorNets and may slightly differ from the original ones. The test input sizes were the numbers reported as the best in the papers:
  * `YOLOv3`, `YOLOv2`: 416x416
  * `FasterRCNN`: min\_shorter\_side=600, max\_longer\_side=1000
- The computation times were measured on NVIDIA Tesla P100 (3584 cores, 16 GB global memory) with cuDNN 6.0 and CUDA 8.0.
  * Size: rounded the number of parameters
  * Speed: milliseconds only for network inferences of a 416x416 or 608x608 single image
  * FPS: 1000 / speed

| PASCAL VOC2007 test                                                    | mAP    | Size   | Speed |  FPS  | References |
|------------------------------------------------------------------------|--------|--------|-------|-------|------------|
| [YOLOv3VOC (416)](tensornets/references/yolos.py#L177)                 | 0.7423 | 62M    | 24.09 | 41.51 | [[paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[darknet]](https://pjreddie.com/darknet/yolo/) [[darkflow]](https://github.com/thtrieu/darkflow) |
| [YOLOv2VOC (416)](tensornets/references/yolos.py#L205)                 | 0.7320 | 51M    | 14.75 | 67.80 | [[paper]](https://arxiv.org/abs/1612.08242) [[darknet]](https://pjreddie.com/darknet/yolov2/) [[darkflow]](https://github.com/thtrieu/darkflow) |
| [TinyYOLOv2VOC (416)](tensornets/references/yolos.py#L241)             | 0.5303 | 16M    | 6.534 | 153.0 | [[paper]](https://arxiv.org/abs/1612.08242) [[darknet]](https://pjreddie.com/darknet/yolov2/) [[darkflow]](https://github.com/thtrieu/darkflow) |
| [FasterRCNN\_ZF\_VOC](tensornets/references/rcnns.py#L150)             | 0.4466 | 59M    | 241.4 | 3.325 | [[paper]](https://arxiv.org/abs/1506.01497) [[caffe]](https://github.com/rbgirshick/py-faster-rcnn) [[roi-pooling]](https://github.com/deepsense-ai/roi-pooling) |
| [FasterRCNN\_VGG16\_VOC](tensornets/references/rcnns.py#L186)          | 0.6872 | 137M   | 300.7 | 4.143 | [[paper]](https://arxiv.org/abs/1506.01497) [[caffe]](https://github.com/rbgirshick/py-faster-rcnn) [[roi-pooling]](https://github.com/deepsense-ai/roi-pooling) |

| MS COCO val2014                                                        | mAP    | Size   | Speed |  FPS  | References |
|------------------------------------------------------------------------|--------|--------|-------|-------|------------|
| [YOLOv3COCO (608)](tensornets/references/yolos.py#L167)                | 0.6016 | 62M    | 60.66 | 16.49 | [[paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[darknet]](https://pjreddie.com/darknet/yolo/) [[darkflow]](https://github.com/thtrieu/darkflow) |
| [YOLOv3COCO (416)](tensornets/references/yolos.py#L167)                | 0.6028 | 62M    | 40.23 | 24.85 | [[paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[darknet]](https://pjreddie.com/darknet/yolo/) [[darkflow]](https://github.com/thtrieu/darkflow) |
| [YOLOv2COCO (608)](tensornets/references/yolos.py#L187)                | 0.5189 | 51M    | 45.88 | 21.80 | [[paper]](https://arxiv.org/abs/1612.08242) [[darknet]](https://pjreddie.com/darknet/yolov2/) [[darkflow]](https://github.com/thtrieu/darkflow) |
| [YOLOv2COCO (416)](tensornets/references/yolos.py#L187)                | 0.4922 | 51M    | 21.66 | 46.17 | [[paper]](https://arxiv.org/abs/1612.08242) [[darknet]](https://pjreddie.com/darknet/yolov2/) [[darkflow]](https://github.com/thtrieu/darkflow) |

## News 📰

- MS COCO utils are released, [9 Jul 2018](https://github.com/taehoonlee/tensornets/commit/4a34243891e6649b72b9c0b7114b8f3d51d1d779).
- PNASNetlarge is released, [12 May 2018](https://github.com/taehoonlee/tensornets/commit/e2e0f0f7791731d3b7dfa989cae569c15a22cdd6).
- The six variants of MobileNetv2 are released, [5 May 2018](https://github.com/taehoonlee/tensornets/commit/fb429b6637f943875249dff50f4bc6220d9d50bf).
- YOLOv3 for COCO and VOC are released, [4 April 2018](https://github.com/taehoonlee/tensornets/commit/d8b2d8a54dc4b775a174035da63561028deb6624).
- Generic object detection models for YOLOv2 and FasterRCNN are released, [26 March 2018](https://github.com/taehoonlee/tensornets/commit/67915e659d2097a96c82ba7740b9e43a8c69858d).

## Future work 🔥

- Add training codes.
- Add image classification models.
  * [PolyNet: A Pursuit of Structural Diversity in Very Deep Networks](https://arxiv.org/abs/1611.05725v2), CVPR 2017, Top-5 4.25%
  * [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507v2), CVPR 2018, Top-5 3.79%
  * [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965), arXiv 2018, Top-5 3.0%
- Add object detection models (MaskRCNN, SSD).
- Add image segmentation models (FCN, UNet).
- Add image datasets (OpenImages).
- Add style transfer examples which can be coupled with any network in TensorNets.
- Add speech and language models with representative datasets (WaveNet, ByteNet).
