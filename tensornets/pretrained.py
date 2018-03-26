"""Collection of pretrained models.

This module provides loading functions for pre-trained weights.
All the weight files are converted as a Keras-like format that
serializes every single tensor from the following repositories:

[1]: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
     "BAIR Caffe Model Zoo"
[2]: https://github.com/tensorflow/models/tree/master/research/slim
     "TF Slim"
[3]: https://github.com/keras-team/keras/tree/master/keras/applications
     "Keras"
[4]: https://github.com/KaimingHe/deep-residual-networks
     "Caffe ResNets"
[5]: https://github.com/facebook/fb.resnet.torch
     "Torch ResNets"
[6]: https://github.com/facebookresearch/ResNeXt
     "Torch ResNeXts"
[7]: https://github.com/liuzhuang13/DenseNet
     "Torch DenseNets"
[8]: https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1
     "Caffe SqueezeNets"
[9]: https://github.com/szagoruyko/wide-residual-networks
     "Torch WideResNets"
[10]: https://pjreddie.com/darknet/yolo/
     "Darknet"
[11]: https://github.com/thtrieu/darkflow
     "TF darkflow"
[12]: https://github.com/rbgirshick/py-faster-rcnn
     "Caffe Faster RCNN"
"""
from __future__ import absolute_import

import tensorflow as tf

from .utils import get_file
from .utils import init
from .utils import parse_scopes
from .utils import parse_weights
from .utils import parse_keras_weights
from .utils import parse_torch_weights
from .utils import pretrained_initializer


__keras_url__ = 'https://github.com/fchollet/deep-learning-models/' \
                'releases/download/v0.2/'
__model_url__ = 'https://github.com/taehoonlee/deep-learning-models/' \
                'releases/download/'


def assign(scopes):
    import warnings
    if not isinstance(scopes, list):
        scopes = [scopes]
    for scope in scopes:
        model_name = parse_scopes(scope)[0]
        try:
            __load_dict__[model_name](scope)
        except KeyError:
            try:
                tf.get_default_session().run(scope.pretrained())
            except:
                found = False
                for (key, fun) in __load_dict__.items():
                    if key in model_name.lower():
                        found = True
                        fun(scope)
                        break
                if not found:
                    warnings.warn('Random initialization will be performed '
                                  'because the pre-trained weights for ' +
                                  model_name + ' are not found.')
                    init(scope)


def direct(model_name, scope):
    if model_name.startswith('gen'):
        fun = load_nothing
        if 'FasterRCNN' in model_name:
            if 'vgg16' in scope.stem_name:
                fun = load_ref_faster_rcnn_vgg16_voc
            elif 'zf' in scope.stem_name:
                fun = load_ref_faster_rcnn_zf_voc
        elif 'TinyYOLOv2' in model_name:
            if 'tinydarknet19' in scope.stem_name:
                fun = load_ref_tiny_yolo_v2_voc
        elif 'YOLOv2' in model_name:
            if 'darknet19' in scope.stem_name:
                fun = load_ref_yolo_v2_voc
    else:
        fun = __load_dict__[model_name]

    def _direct():
        return fun(scope, return_fn=pretrained_initializer)
    return _direct


def _assign(scopes, values):
    sess = tf.get_default_session()
    assert sess is not None, 'The default session should be given.'

    scopes = parse_scopes(scopes)

    for scope in scopes:
        sess.run(pretrained_initializer(scope, values))


def load_nothing(scopes, return_fn=_assign):
    return return_fn(scopes, None)


def load_inception1(scopes, return_fn=_assign):
    """Converted from the [BAIR Caffe Model Zoo][1]."""
    filename = 'inception1.h5'
    weights_path = get_file(
        filename, __model_url__ + 'inception/' + filename,
        cache_subdir='models',
        file_hash='6a212e3cb60b33f49c372906f18ae4a8')
    values = parse_keras_weights(weights_path)
    return return_fn(scopes, values)


def load_inception2(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'inception2.npz'
    weights_path = get_file(
        filename, __model_url__ + 'inception/' + filename,
        cache_subdir='models',
        file_hash='0476b876a5d35a99e2747f98248d856d')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_inception3(scopes, return_fn=_assign):
    """Copied from [keras][3] with modifications on the order of weights."""
    filename = 'inception3.h5'
    weights_path = get_file(
        filename, __model_url__ + 'inception/' + filename,
        cache_subdir='models',
        file_hash='7c4556613c348da3b99b633e1c430fff')
    values = parse_keras_weights(weights_path)
    return return_fn(scopes, values)


def load_inception4(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'inception4.npz'
    weights_path = get_file(
        filename, __model_url__ + 'inception/' + filename,
        cache_subdir='models',
        file_hash='8d5a0e8cb451c85112d5c4e363d77a42')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_inceptionresnet2(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'inception_resnet_v2_2016_08_30.npz'
    weights_path = get_file(
        filename, __model_url__ + 'inception/' + filename,
        cache_subdir='models',
        file_hash='32d685e68e6be6ba1da64e41f939bc49')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_resnet50(scopes, return_fn=_assign):
    """Converted from the original [Caffe ResNets][4]."""
    filename = 'resnet50.h5'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='9df0843bdadb58ed24d360564c45b119')
    values = parse_keras_weights(weights_path)
    return return_fn(scopes, values)


def load_resnet101(scopes, return_fn=_assign):
    """Converted from the original [Caffe ResNets][4]."""
    filename = 'resnet101.h5'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='e2434bec605870fb4747e1b93f9f0e47')
    values = parse_keras_weights(weights_path)
    return return_fn(scopes, values)


def load_resnet152(scopes, return_fn=_assign):
    """Converted from the original [Caffe ResNets][4]."""
    filename = 'resnet152.h5'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='e588285d1f919e538515c1f1b1c07b5b')
    values = parse_keras_weights(weights_path)
    return return_fn(scopes, values)


def load_resnet50v2(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'resnet_v2_50.npz'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='fa2ac006361fd5e79792d163c0130667')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_resnet101v2(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'resnet_v2_101.npz'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='fbc179d55c817e4656992fa582fdc460')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_resnet152v2(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'resnet_v2_152.npz'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='184c9b439e925762f006d288445997a8')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_resnet200v2(scopes, return_fn=_assign):
    """Converted from the [Torch ResNets][5]."""
    filename = 'resnet_v2_200.npz'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='5bff85adbbc499e200c4bf4dc89cde87')
    values = parse_weights(weights_path, move_rules_fb_resnet_torch)
    return return_fn(scopes, values)


def load_resnext50(scopes, return_fn=_assign):
    """Converted from the [Torch ResNeXts][6]."""
    filename = 'resnext_50_32x4d.npz'
    move_rules = [(r, -15) for (r, i) in move_rules_fb_resnet_torch
                  if '1.0.bias' not in r]
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='3eeb3656d95cc9cb7797f638bae56a4e')
    values = parse_weights(weights_path, move_rules)
    return return_fn(scopes, values)


def load_resnext101(scopes, return_fn=_assign):
    """Converted from the [Torch ResNeXts][6]."""
    filename = 'resnext_101_32x4d.npz'
    move_rules = [(r, -15) for (r, i) in move_rules_fb_resnet_torch
                  if '1.0.bias' not in r]
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='04d3037474bc8500f417758d62a8ff2d')
    values = parse_weights(weights_path, move_rules)
    return return_fn(scopes, values)


def load_resnext101c64(scopes, return_fn=_assign):
    """Converted from the [Torch ResNeXts][6]."""
    filename = 'resnext_101_64x4d.npz'
    move_rules = [(r, -15) for (r, i) in move_rules_fb_resnet_torch
                  if '1.0.bias' not in r]
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='2f6e5440d6fcdfe72d4cd2fd1da12894')
    values = parse_weights(weights_path, move_rules)
    return return_fn(scopes, values)


def load_wideresnet50(scopes, return_fn=_assign):
    """Converted from the [Torch WideResNets][9]."""
    filename = 'wrn_50_2.npz'
    move_rules = [(r, -15) for (r, i) in move_rules_fb_resnet_torch
                  if '1.0.bias' not in r]
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='dd8fcad081890a2685638166d2c2d3f9')
    values = parse_weights(weights_path, move_rules)
    return return_fn(scopes, values)


def load_keras_resnet50(scopes, return_fn=_assign):
    """Copied from [keras][3]."""
    filename = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file(
        filename, __keras_url__ + filename,
        cache_subdir='models',
        file_hash='a7b3fe01876f51b976af0dea6bc144eb')
    move_rules = []
    for i in range(2, 6):
        move_rules.append(("bn%da_branch2c" % i, -1))
        move_rules.append(("res%da_branch1" % i, -6))
        move_rules.append(("bn%da_branch1" % i, -6))
    values = parse_keras_weights(weights_path, move_rules)
    return return_fn(scopes, values)


move_rules_fb_resnet_torch = []
for i in range(4, 8):
    move_rules_fb_resnet_torch.append(("%d.0.0.1.0.weight" % i, -18))
    move_rules_fb_resnet_torch.append(("%d.0.0.1.0.bias" % i, -18))
    move_rules_fb_resnet_torch.append(("%d.0.0.1.1.weight" % i, -18))
    move_rules_fb_resnet_torch.append(("%d.0.0.1.1.bias" % i, -18))
    move_rules_fb_resnet_torch.append(("%d.0.0.1.1.running_mean" % i, -18))
    move_rules_fb_resnet_torch.append(("%d.0.0.1.1.running_var" % i, -18))


def load_torch_resnet50(scopes, return_fn=_assign):
    """Converted from the [Torch ResNets][5]."""
    filename = 'resnet_50_cpu.pth'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='5b38c39802c94de00b55596145d304aa')
    values = parse_torch_weights(weights_path, move_rules_fb_resnet_torch)
    return return_fn(scopes, values)


def load_torch_resnet101(scopes, return_fn=_assign):
    """Converted from the [Torch ResNets][5]."""
    filename = 'resnet_101_cpu.pth'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='cb3f0ac4687cb63d5f0861d651da844b')
    values = parse_torch_weights(weights_path, move_rules_fb_resnet_torch)
    return return_fn(scopes, values)


def load_torch_resnet152(scopes, return_fn=_assign):
    """Converted from the [Torch ResNets][5]."""
    filename = 'resnet_152_cpu.pth'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='3339f6aca7f746f8ae7f6ce577efc0c0')
    values = parse_torch_weights(weights_path, move_rules_fb_resnet_torch)
    return return_fn(scopes, values)


def load_torch_resnet200v2(scopes, return_fn=_assign):
    """Converted from the [Torch ResNets][5]."""
    filename = 'resnet_200_cpu.pth'
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='220df3970701d3e0608eed887fb95d82')
    values = parse_torch_weights(weights_path, move_rules_fb_resnet_torch)
    return return_fn(scopes, values)


def load_torch_resnext50(scopes, return_fn=_assign):
    """Converted from the [Torch ResNeXts][6]."""
    filename = 'resnext_50_32x4d_cpu.pth'
    move_rules = [(r, -15) for (r, i) in move_rules_fb_resnet_torch
                  if '1.0.bias' not in r]
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='fdfc372bc47f7bf55313c04aebcef8ca')
    values = parse_torch_weights(weights_path, move_rules)
    return return_fn(scopes, values)


def load_torch_resnext101(scopes, return_fn=_assign):
    """Converted from the [Torch ResNeXts][6]."""
    filename = 'resnext_101_32x4d_cpu.pth'
    move_rules = [(r, -15) for (r, i) in move_rules_fb_resnet_torch
                  if '1.0.bias' not in r]
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='5e97757d9f898aa8174fe8bc6e59bce8')
    values = parse_torch_weights(weights_path, move_rules)
    return return_fn(scopes, values)


def load_torch_resnext101c64(scopes, return_fn=_assign):
    """Converted from the [Torch ResNeXts][6]."""
    filename = 'resnext_101_64x4d_cpu.pth'
    move_rules = [(r, -15) for (r, i) in move_rules_fb_resnet_torch
                  if '1.0.bias' not in r]
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='03c83fe32db97676eace16cc0b577cc2')
    values = parse_torch_weights(weights_path, move_rules)
    return return_fn(scopes, values)


def load_torch_wideresnet50(scopes, return_fn=_assign):
    """Converted from the [Torch WideResNets][9]."""
    filename = 'wrn_50_2_cpu.pth'
    move_rules = [(r, -15) for (r, i) in move_rules_fb_resnet_torch
                  if '1.0.bias' not in r]
    weights_path = get_file(
        filename, __model_url__ + 'resnet/' + filename,
        cache_subdir='models',
        file_hash='7879cd9f3840f92593a87b6be8192206')
    values = parse_torch_weights(weights_path, move_rules)
    return return_fn(scopes, values)


def load_densenet121(scopes, return_fn=_assign):
    """Converted from the [Torch DenseNets][7]."""
    filename = 'densenet121.npz'
    weights_path = get_file(
        filename, __model_url__ + 'densenet/' + filename,
        cache_subdir='models',
        file_hash='c65564d54d4f7d29da6c84865325d7d4')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_densenet169(scopes, return_fn=_assign):
    """Converted from the [Torch DenseNets][7]."""
    filename = 'densenet169.npz'
    weights_path = get_file(
        filename, __model_url__ + 'densenet/' + filename,
        cache_subdir='models',
        file_hash='f17f3aa42e92489a1e3b981061cd3d96')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_densenet201(scopes, return_fn=_assign):
    """Converted from the [Torch DenseNets][7]."""
    filename = 'densenet201.npz'
    weights_path = get_file(
        filename, __model_url__ + 'densenet/' + filename,
        cache_subdir='models',
        file_hash='d2c503199fc0c5537ca2aa3d723cc493')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_torch_densenet121(scopes, return_fn=_assign):
    """Converted from the [Torch DenseNets][7]."""
    filename = 'densenet_121_cpu.pth'
    weights_path = get_file(
        filename, __model_url__ + 'densenet/' + filename,
        cache_subdir='models',
        file_hash='9817430b1d3634645f6b04b8c663c34f')
    values = parse_torch_weights(weights_path)
    return return_fn(scopes, values)


def load_torch_densenet169(scopes, return_fn=_assign):
    """Converted from the [Torch DenseNets][7]."""
    filename = 'densenet_169_cpu.pth'
    weights_path = get_file(
        filename, __model_url__ + 'densenet/' + filename,
        cache_subdir='models',
        file_hash='98c5cac06124192627391adf17d66493')
    values = parse_torch_weights(weights_path)
    return return_fn(scopes, values)


def load_torch_densenet201(scopes, return_fn=_assign):
    """Converted from the [Torch DenseNets][7]."""
    filename = 'densenet_201_cpu.pth'
    weights_path = get_file(
        filename, __model_url__ + 'densenet/' + filename,
        cache_subdir='models',
        file_hash='fa3aa0454be559b81409e92f3bafd155')
    values = parse_torch_weights(weights_path)
    return return_fn(scopes, values)


def load_mobilenet25(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'mobilenet25.npz'
    weights_path = get_file(
        filename, __model_url__ + 'mobilenet/' + filename,
        cache_subdir='models',
        file_hash='aa1f5ccfb8be3d1ef45948a396e04e0a')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_mobilenet50(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'mobilenet50.npz'
    weights_path = get_file(
        filename, __model_url__ + 'mobilenet/' + filename,
        cache_subdir='models',
        file_hash='0c0b667bc9d707e0e5bd4f383c5dece0')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_mobilenet75(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'mobilenet75.npz'
    weights_path = get_file(
        filename, __model_url__ + 'mobilenet/' + filename,
        cache_subdir='models',
        file_hash='d4557a46a44eebfeaf08c82ae33765ed')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_mobilenet100(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'mobilenet100.npz'
    weights_path = get_file(
        filename, __model_url__ + 'mobilenet/' + filename,
        cache_subdir='models',
        file_hash='3d14409e3e119c8881baf7dd1d54e714')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_squeezenet(scopes, return_fn=_assign):
    """Converted from the [Caffe SqueezeNets][8]."""
    filename = 'squeezenet.npz'
    weights_path = get_file(
        filename, __model_url__ + 'squeezenet/' + filename,
        cache_subdir='models',
        file_hash='1d474f6540f7ec34cb56e6440419b5c5')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_nasnetAlarge(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'nasnet-a_large_04_10_2017.npz'
    weights_path = get_file(
        filename, __model_url__ + 'nasnet/' + filename,
        cache_subdir='models',
        file_hash='f14c166457ce43b2c44d4cd3b6325bd6')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_nasnetAmobile(scopes, return_fn=_assign):
    """Converted from the [TF Slim][2]."""
    filename = 'nasnet-a_mobile_04_10_2017.npz'
    weights_path = get_file(
        filename, __model_url__ + 'nasnet/' + filename,
        cache_subdir='models',
        file_hash='7d75cfc284185c0a6db5cbf9f0492c59')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_vgg16(scopes, return_fn=_assign):
    """Copied from [Keras][3]."""
    filename = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file(
        filename, __model_url__ + 'vgg/' + filename,
        cache_subdir='models',
        file_hash='64373286793e3c8b2b4e3219cbf3544b')
    values = parse_keras_weights(weights_path)
    return return_fn(scopes, values)


def load_vgg19(scopes, return_fn=_assign):
    """Copied from [Keras][3]."""
    filename = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file(
        filename, __model_url__ + 'vgg/' + filename,
        cache_subdir='models',
        file_hash='cbe5617147190e668d6c5d5026f83318')
    values = parse_keras_weights(weights_path)
    return return_fn(scopes, values)


def load_ref_yolo_v2(scopes, return_fn=_assign):
    """Converted from the original [Darknet][10] using the [darkflow][11]."""
    filename = 'ref_yolo_v2.npz'
    weights_path = get_file(
        filename, __model_url__ + 'yolo/' + filename,
        cache_subdir='models',
        file_hash='bacf9f08bc229d11287a4fa3736a6bad')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_ref_yolo_v2_voc(scopes, return_fn=_assign):
    """Converted from the original [Darknet][10] using the [darkflow][11]."""
    filename = 'ref_yolo_v2_voc.npz'
    weights_path = get_file(
        filename, __model_url__ + 'yolo/' + filename,
        cache_subdir='models',
        file_hash='5d7e3c739f8876ee5facbdc5ec6e53d5')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_ref_tiny_yolo_v2_voc(scopes, return_fn=_assign):
    """Converted from the original [Darknet][10] using the [darkflow][11]."""
    filename = 'ref_tiny_yolo_v2_voc.npz'
    weights_path = get_file(
        filename, __model_url__ + 'yolo/' + filename,
        cache_subdir='models',
        file_hash='e1ec6a037a217811e08568b105c22c0f')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_ref_faster_rcnn_zf_voc(scopes, return_fn=_assign):
    """Converted from the [Caffe Faster RCNN][12]."""
    filename = 'ref_faster_rcnn_zf_voc.npz'
    weights_path = get_file(
        filename, __model_url__ + 'rcnn/' + filename,
        cache_subdir='models',
        file_hash='825577525217176903ee9b096bb1cb64')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


def load_ref_faster_rcnn_vgg16_voc(scopes, return_fn=_assign):
    """Converted from the [Caffe Faster RCNN][12]."""
    filename = 'ref_faster_rcnn_vgg16_voc.npz'
    weights_path = get_file(
        filename, __model_url__ + 'rcnn/' + filename,
        cache_subdir='models',
        file_hash='2dfa10d64c169cf105b36c209be91316')
    values = parse_weights(weights_path)
    return return_fn(scopes, values)


# Dictionary for loading functions.
__load_dict__ = {
    'inception1': load_inception1,
    'inception2': load_inception2,
    'inception3': load_inception3,
    'inception4': load_inception4,
    'inceptionresnet2_tfslim': load_inceptionresnet2,
    'resnet50': load_resnet50,
    'resnet101': load_resnet101,
    'resnet152': load_resnet152,
    'resnet50v2': load_resnet50v2,
    'resnet101v2': load_resnet101v2,
    'resnet152v2': load_resnet152v2,
    'resnet200v2': load_resnet200v2,
    'resnext50': load_resnext50,
    'resnext101': load_resnext101,
    'resnext50c32': load_resnext50,
    'resnext101c32': load_resnext101,
    'resnext101c64': load_resnext101c64,
    'wideresnet50': load_wideresnet50,
    'nasnetAlarge': load_nasnetAlarge,
    'nasnetAmobile': load_nasnetAmobile,
    'vgg16': load_vgg16,
    'vgg19': load_vgg19,
    'densenet121': load_densenet121,
    'densenet169': load_densenet169,
    'densenet201': load_densenet201,
    'mobilenet25': load_mobilenet25,
    'mobilenet50': load_mobilenet50,
    'mobilenet75': load_mobilenet75,
    'mobilenet100': load_mobilenet100,
    'squeezenet': load_squeezenet,
    'zf': load_nothing,
    'darknet19': load_nothing,
    'tinydarknet19': load_nothing,
    'REFyolov2': load_ref_yolo_v2,
    'REFyolov2voc': load_ref_yolo_v2_voc,
    'REFtinyyolov2voc': load_ref_tiny_yolo_v2_voc,
    'REFfasterrcnnZFvoc': load_ref_faster_rcnn_zf_voc,
    'REFfasterrcnnVGG16voc': load_ref_faster_rcnn_vgg16_voc,
}
