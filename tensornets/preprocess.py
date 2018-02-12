import numpy as np


def preprocess(scopes, inputs):
    import warnings
    from .utils import parse_scopes
    if not isinstance(scopes, list):
        scopes = [scopes]
    outputs = []
    for scope in scopes:
        model_name = parse_scopes(scope)[0]
        try:
            outputs.append(__preprocess_dict__[model_name](inputs))
        except KeyError:
            found = False
            for (key, fun) in __preprocess_dict__.items():
                if key in model_name.lower():
                    found = True
                    outputs.append(fun(inputs))
                    break
            if not found:
                warnings.warn('No pre-processing will be performed '
                              'because the pre-processing for ' +
                              model_name + ' are not found.')
                outputs.append(inputs)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def direct(model_name):
    def _direct(inputs):
        return __preprocess_dict__[model_name](inputs)
    return _direct


def bair_preprocess(x):
    # Refer to the following BAIR Caffe Model Zoo
    # https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt
    x = x.copy()
    x = x[:, :, :, ::-1]
    x[:, :, :, 0] -= 104.
    x[:, :, :, 1] -= 117.
    x[:, :, :, 2] -= 123.
    return x


def tfslim_preprocess(x):
    # Copied from keras (equivalent to the same as in TF Slim)
    x = x.copy()
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def keras_resnet_preprocess(x):
    # Copied from keras
    x = x.copy()
    x = x[:, :, :, ::-1]
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    return x


def fb_preprocess(x):
    # Refer to the following Torch ResNets
    # https://github.com/facebook/fb.resnet.torch/blob/master/pretrained/classify.lua
    x = x.copy()
    x /= 255.
    x[:, :, :, 0] -= 0.485
    x[:, :, :, 1] -= 0.456
    x[:, :, :, 2] -= 0.406
    x[:, :, :, 0] /= 0.229
    x[:, :, :, 1] /= 0.224
    x[:, :, :, 2] /= 0.225
    return x


def wrn_preprocess(x):
    # Refer to the following Torch WideResNets
    # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/main.py
    x = x.copy()
    x /= 255.
    x[:, :, :, 0] -= 0.491
    x[:, :, :, 1] -= 0.482
    x[:, :, :, 2] -= 0.447
    x[:, :, :, 0] /= 0.247
    x[:, :, :, 1] /= 0.244
    x[:, :, :, 2] /= 0.262
    return x


# Dictionary for pre-processing functions.
__preprocess_dict__ = {
    'inception': tfslim_preprocess,
    'inception1': bair_preprocess,
    'inception2': tfslim_preprocess,
    'inception3': tfslim_preprocess,
    'inception4': tfslim_preprocess,
    'inceptionresnet2_tfslim': tfslim_preprocess,
    'resnet': keras_resnet_preprocess,
    'resnet50': keras_resnet_preprocess,
    'resnet101': keras_resnet_preprocess,
    'resnet152': keras_resnet_preprocess,
    'resnetv2': tfslim_preprocess,
    'resnet50v2': tfslim_preprocess,
    'resnet101v2': tfslim_preprocess,
    'resnet152v2': tfslim_preprocess,
    'resnet200v2': fb_preprocess,
    'resnext': fb_preprocess,
    'resnext50': fb_preprocess,
    'resnext101': fb_preprocess,
    'resnext50c32': fb_preprocess,
    'resnext101c32': fb_preprocess,
    'resnext101c64': fb_preprocess,
    'wideresnet50': wrn_preprocess,
    'nasnetAlarge': tfslim_preprocess,
    'nasnetAmobile': tfslim_preprocess,
    'vgg16': keras_resnet_preprocess,
    'vgg19': keras_resnet_preprocess,
    'densenet': fb_preprocess,
    'densenet121': fb_preprocess,
    'densenet169': fb_preprocess,
    'densenet201': fb_preprocess,
    'mobilenet': tfslim_preprocess,
    'mobilenet25': tfslim_preprocess,
    'mobilenet50': tfslim_preprocess,
    'mobilenet75': tfslim_preprocess,
    'mobilenet100': tfslim_preprocess,
    'squeezenet': bair_preprocess,
}
