"""Collection of representative endpoints for each model."""
from __future__ import absolute_import

from .version_utils import tf_equal_to
from .version_utils import tf_later_than


if tf_later_than('1.15'):
    bn_name = 'FusedBatchNormV3:0'
elif tf_later_than('1.4'):
    bn_name = 'FusedBatchNorm:0'
else:
    bn_name = 'batchnorm/add_1:0'


def names_inceptions(k, first_block, omit_first=False,
                     pool_last=False, resnet=False):
    names = []
    postfix = ['concat', 'out' if resnet else 'concat']
    for i in range(3):
        first_char = 98 if omit_first is True and i == 0 else 97
        names += ["block%d%c/%s:0" %
                  (i + first_block, j + first_char, postfix[j > 0])
                  for j in range(k[i])]
        if pool_last is True and i < 2:
            names[-1] = "pool%d/MaxPool:0" % (i + first_block)
    return names


def names_resnets(k, pool_last=False):
    names = []
    for i in range(4):
        names += ["conv%d/block%d/out:0" % (i + 2, j + 1) for j in range(k[i])]
        if pool_last is True and i < 3:
            names += ["pool%d/pool/AvgPool:0" % (i + 2)]
    return names


def names_squeezenet():
    names = ["fire%d/concat:0" % (i + 2) for i in range(8)]
    names.insert(2, 'pool3/MaxPool:0')
    names.insert(5, 'pool5/MaxPool:0')
    return names


def names_nasnets(k):
    names = []
    for i in range(3):
        base = sum(k[:i])
        names += ["normal%d/concat:0" % (base + j + 1) for j in range(k[i])]
        if i < 2:
            names += ["reduction%d/concat:0" % (base + k[i])]
    return names


def names_vggs(k):
    names = []
    for i in range(3):
        names += ["conv%d/%d/Relu:0" % (i + 3, j + 1) for j in range(k)]
    return names


def names_darknets(k):
    names = []
    for i in range(4):
        if k[i] > 1:
            names += ["conv%d/%d/lrelu/Maximum:0" % (i + 3, j + 1)
                      for j in range(k[i])]
        else:
            names += ["conv%d/lrelu/Maximum:0" % (i + 3)]
        if i < 3:
            names += ["pool%d/MaxPool:0" % (i + 3)]
    return names


def tuple_mobilenetv2():
    def baseidx(b):
        return [b, b + 3, b + 5]
    indices = baseidx(2)
    names = ['conv1/Relu6:0', 'sconv1/Relu6:0', 'pconv1/bn/' + bn_name]
    k = 10
    l = 2
    for (i, j) in enumerate([2, 3, 4, 3, 3, 1]):
        indices += baseidx(k)
        names += ["conv%d/conv/Relu6:0" % l,
                  "conv%d/sconv/Relu6:0" % l,
                  "conv%d/pconv/bn/%s" % (l, bn_name)]
        k += 8
        l += 1
        for _ in range(j - 1):
            indices += (baseidx(k) + [k + 6])
            names += ["conv%d/conv/Relu6:0" % l,
                      "conv%d/sconv/Relu6:0" % l,
                      "conv%d/pconv/bn/%s" % (l, bn_name),
                      "conv%d/out:0" % l]
            k += 9
            l += 1
    indices += [k]
    names += ["conv%d/Relu6:0" % l]
    return (indices, names, -16)


def tuple_mobilenetv3(is_large, is_mini):
    if is_mini is True:
        act_name = 'Relu:0'
    else:
        act_name = 'Mul:0'

    names = ["conv1/%s" % act_name]
    indices = [2]
    l = 3

    if is_large is True:
        names += ['conv2/out:0']
        blocks = [2, 3, 4, 2, 3]
        k = 8
    else:
        names += ["conv2/pconv/bn/%s" % bn_name]
        blocks = [2, 3, 2, 3]
        if is_mini is True:
            k = 7
        else:
            k = 13

    indices += [k]

    for (i, j) in enumerate(blocks):
        names += ["conv%d/pconv/bn/%s" % (l, bn_name)]
        l += 1
        k += 8
        if is_mini is False:
            if (is_large is True and i in [1, 3, 4]) or \
               (is_large is False and i in [1, 2, 3, 4]):
                k += 6
        indices += [k]
        for _ in range(j - 1):
            names += ["conv%d/out:0" % l]
            l += 1
            k += 9
            if is_mini is False:
                if (is_large is True and i in [1, 3, 4]) or \
                   (is_large is False and i in [1, 2, 3, 4]):
                    k += 6
            indices += [k]

    indices += [k + 3, k + 4, k + 6]
    names += ["conv%d/%s" % (l, act_name),
              'pool/AvgPool:0',
              "conv%d/out:0" % (l + 1)]
    return (indices, names, -16)


def tuple_efficientnet(index0, index1, blocks, addindices, biases=None):
    indices = [index0] + list(range(index1, index1 + 12 * blocks, 12))
    names = ["block%d/pconv/bn/%s" % (i, bn_name) for i in range(len(indices))]
    for i in addindices:
        for j in range(i, len(indices)):
            indices[j] += 2
        names.insert(i, "block%s/add:0" % names[i - 1].split('/')[0][5:])
        indices.insert(i, indices[i - 1] + 2)
    if biases is not None:
        for b in biases:
            for j in range(b, len(indices)):
                indices[j] -= 3
    return (indices, names, -10)


def direct(model_name):
    try:
        return __middles_dict__[model_name]
    except KeyError:
        return ([-1], ['out:0'])


# Dictionary for lists of representative endpoints.
__middles_dict__ = {
    'inception1': (
        [24, 38, 39, 53] + list(range(67, 110, 14)) + [110, 124, 138],
        names_inceptions([3, 6, 2], 3, pool_last=True),
        -4
    ),
    'inception2': (
        [31, 54] + list(range(71, 164, 23)) + list(range(180, 227, 23)),
        names_inceptions([3, 5, 2], 3),
        -4
    ),
    'inception3': (
        list(range(39, 86, 23)) + list(range(99, 228, 32)) +
        list(range(247, 310, 31)),
        names_inceptions([3, 5, 3], 5, omit_first=True),
        -4
    ),
    'inception4': (
        list(range(37, 130, 23)) + list(range(143, 368, 32)) +
        list(range(387, 490, 34)),
        names_inceptions([5, 8, 4], 5),
        -5
    ),
    'inceptionresnet2_tfslim': (
        [39] + list(range(60, 250, 21)) + [263] + list(range(278, 564, 15)) +
        [586] + list(range(601, 737, 15)),
        names_inceptions([11, 21, 11], 5, omit_first=True, resnet=True),
        -12
    ),
    'resnet50': (
        list(range(16, 35, 9)) + list(range(45, 73, 9)) +
        list(range(83, 129, 9)) + list(range(139, 158, 9)),
        names_resnets([3, 4, 6, 3]),
        -4
    ),
    'resnet101': (
        list(range(16, 35, 9)) + list(range(45, 73, 9)) +
        list(range(83, 282, 9)) + list(range(292, 311, 9)),
        names_resnets([3, 4, 23, 3]),
        -4
    ),
    'resnet152': (
        list(range(16, 35, 9)) + list(range(45, 109, 9)) +
        list(range(119, 435, 9)) + list(range(445, 464, 9)),
        names_resnets([3, 8, 36, 3]),
        -4
    ),
    'resnet50v2': (
        list(range(15, 27, 11)) + [38] + list(range(50, 73, 11)) + [84] +
        list(range(96, 141, 11)) + [152] + list(range(164, 176, 11)) + [186],
        names_resnets([3, 4, 6, 3]),
        -5
    ),
    'resnet101v2': (
        list(range(15, 27, 11)) + [38] + list(range(50, 73, 11)) + [84] +
        list(range(96, 328, 11)) + [339] + list(range(351, 363, 11)) + [373],
        names_resnets([3, 4, 23, 3]),
        -5
    ),
    'resnet152v2': (
        list(range(15, 27, 11)) + [38] + list(range(50, 117, 11)) + [128] +
        list(range(140, 515, 11)) + [526] + list(range(538, 550, 11)) + [560],
        names_resnets([3, 8, 36, 3]),
        -5
    ),
    'resnet200v2': (
        list(range(18, 41, 11)) + list(range(53, 307, 11)) +
        list(range(319, 705, 11)) + list(range(717, 740, 11)),
        names_resnets([3, 24, 36, 3]),
        -4
    ),
    'resnext50': (
        list(range(18, 41, 11)) + list(range(53, 87, 11)) +
        list(range(99, 155, 11)) + list(range(167, 190, 11)),
        names_resnets([3, 4, 6, 3]),
        -4
    ),
    'resnext101': (
        list(range(18, 41, 11)) + list(range(53, 87, 11)) +
        list(range(99, 342, 11)) + list(range(354, 377, 11)),
        names_resnets([3, 4, 23, 3]),
        -4
    ),
    'resnext50c32': (
        list(range(18, 41, 11)) + list(range(53, 87, 11)) +
        list(range(99, 155, 11)) + list(range(167, 190, 11)),
        names_resnets([3, 4, 6, 3]),
        -4
    ),
    'resnext101c32': (
        list(range(18, 41, 11)) + list(range(53, 87, 11)) +
        list(range(99, 342, 11)) + list(range(354, 377, 11)),
        names_resnets([3, 4, 23, 3]),
        -4
    ),
    'resnext101c64': (
        list(range(18, 41, 11)) + list(range(53, 87, 11)) +
        list(range(99, 342, 11)) + list(range(354, 377, 11)),
        names_resnets([3, 4, 23, 3]),
        -4
    ),
    'wideresnet50': (
        list(range(17, 38, 10)) + list(range(49, 80, 10)) +
        list(range(91, 142, 10)) + list(range(153, 174, 10)),
        names_resnets([3, 4, 6, 3]),
        -4
    ),
    'nasnetAlarge': (
        list(range(145, 371, 45)) + [416] + list(range(466, 692, 45)) + [748] +
        list(range(798, 1024, 45)),
        names_nasnets([6, 6, 6]),
        -8
    ),
    'nasnetAmobile': (
        list(range(145, 281, 45)) + [326] + list(range(376, 512, 45)) + [568] +
        list(range(618, 754, 45)),
        names_nasnets([4, 4, 4]),
        -6
    ),
    'pnasnetlarge': (
        list(range(169, 323, 51)) + [376] + list(range(432, 535, 51)) + [588] +
        list(range(644, 747, 51)),
        names_nasnets([4, 3, 3]),
        -5
    ),
    'vgg16': (
        list(range(11, 16, 2)) + list(range(18, 23, 2)) +
        list(range(25, 30, 2)),
        names_vggs(3),
        -1
    ),
    'vgg19': (
        list(range(11, 18, 2)) + list(range(20, 27, 2)) +
        list(range(29, 36, 2)),
        names_vggs(4),
        -1
    ),
    'densenet121': (
        list(range(12, 48, 7)) + [51] + list(range(58, 136, 7)) + [139] +
        list(range(146, 308, 7)) + [311] + list(range(318, 424, 7)),
        names_resnets([6, 12, 24, 16], pool_last=True),
        -18
    ),
    'densenet169': (
        list(range(12, 48, 7)) + [51] + list(range(58, 136, 7)) + [139] +
        list(range(146, 364, 7)) + [367] + list(range(374, 592, 7)),
        names_resnets([6, 12, 32, 32], pool_last=True),
        -34
    ),
    'densenet201': (
        list(range(12, 48, 7)) + [51] + list(range(58, 136, 7)) + [139] +
        list(range(146, 476, 7)) + [479] + list(range(486, 704, 7)),
        names_resnets([6, 12, 48, 32], pool_last=True),
        -34
    ),
    'mobilenet25': (
        list(range(20, 81, 6)),
        ['conv%d/conv/Relu6:0' % (i + 4) for i in range(11)],
        -3
    ),
    'mobilenet50': (
        list(range(20, 81, 6)),
        ['conv%d/conv/Relu6:0' % (i + 4) for i in range(11)],
        -3
    ),
    'mobilenet75': (
        list(range(20, 81, 6)),
        ['conv%d/conv/Relu6:0' % (i + 4) for i in range(11)],
        -3
    ),
    'mobilenet100': (
        list(range(20, 81, 6)),
        ['conv%d/conv/Relu6:0' % (i + 4) for i in range(11)],
        -3
    ),
    'mobilenet35v2': tuple_mobilenetv2(),
    'mobilenet50v2': tuple_mobilenetv2(),
    'mobilenet75v2': tuple_mobilenetv2(),
    'mobilenet100v2': tuple_mobilenetv2(),
    'mobilenet130v2': tuple_mobilenetv2(),
    'mobilenet140v2': tuple_mobilenetv2(),
    'mobilenet75v3large': tuple_mobilenetv3(True, False),
    'mobilenet100v3large': tuple_mobilenetv3(True, False),
    'mobilenet100v3largemini': tuple_mobilenetv3(True, True),
    'mobilenet75v3small': tuple_mobilenetv3(False, False),
    'mobilenet100v3small': tuple_mobilenetv3(False, False),
    'mobilenet100v3smallmini': tuple_mobilenetv3(False, True),
    'efficientnetb0': tuple_efficientnet(
        11, 23, 15,
        [3, 6, 9, 11, 14, 16, 19, 21, 23]),
    'efficientnetb1': tuple_efficientnet(
        11, 20, 22,
        [2, 5, 7, 10, 12, 15, 17, 19, 22, 24, 26, 29, 31, 33, 35, 38]),
    'efficientnetb2': tuple_efficientnet(
        11, 20, 22,
        [2, 5, 7, 10, 12, 15, 17, 19, 22, 24, 26, 29, 31, 33, 35, 38]),
    'efficientnetb3': tuple_efficientnet(
        11, 20, 25,
        [2, 5, 7, 10, 12, 15, 17, 19, 21, 24, 26, 28, 30, 33, 35, 37,
         39, 41, 44]),
    'efficientnetb4': tuple_efficientnet(
        11, 20, 31,
        [2, 5, 7, 9, 12, 14, 16, 19, 21, 23, 25, 27, 30, 32, 34, 36, 38,
         41, 43, 45, 47, 49, 51, 53, 56]),
    'efficientnetb5': tuple_efficientnet(
        11, 20, 38,
        [2, 4, 7, 9, 11, 13, 16, 18, 20, 22, 25, 27, 29, 31, 33, 35, 38,
         40, 42, 44, 46, 48, 51, 53, 55, 57, 59, 61, 63, 65, 68, 70],
        biases=[3]),
    'efficientnetb6': tuple_efficientnet(
        11, 20, 44,
        [2, 4, 7, 9, 11, 13, 15, 18, 20, 22, 24, 26, 29, 31, 33, 35, 37,
         39, 41, 44, 46, 48, 50, 52, 54, 56, 59, 61, 63, 65, 67, 69, 71,
         73, 75, 77, 80, 82],
        biases=[3]),
    'efficientnetb7': tuple_efficientnet(
        11, 20, 54,
        [2, 4, 6, 9, 11, 13, 15, 17, 19, 22, 24, 26, 28, 30, 32, 35, 37,
         39, 41, 43, 45, 47, 49, 51, 54, 56, 58, 60, 62, 64, 66, 68, 70,
         73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 98, 100, 102],
        biases=[3, 5]),
    'squeezenet': (
        [9, 16, 17, 24, 31, 32] + list(range(39, 61, 7)),
        names_squeezenet(),
        -6
    ),
    'darknet19': (
        list(range(13, 22, 4)) + [22] + list(range(26, 35, 4)) + [35] +
        list(range(39, 56, 4)) + [56] + list(range(60, 77, 4)),
        names_darknets([3, 3, 5, 5]),
        -7
    ),
    'tinydarknet19': (
        [13, 14, 18, 19, 23, 24, 28],
        names_darknets([1, 1, 1, 1]),
        -3
    ),
    'REFyolov2': ([-1], ['linear/BiasAdd:0']),
    'REFyolov2voc': ([-1], ['linear/BiasAdd:0']),
    'REFtinyyolov2': ([-1], ['linear/BiasAdd:0']),
    'REFtinyyolov2voc': ([-1], ['linear/BiasAdd:0']),
}
