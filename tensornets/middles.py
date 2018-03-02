"""Collection of representative endpoints for each model."""
from __future__ import absolute_import


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
    names.append('conv10/Relu:0')
    return names


def names_nasnets(k):
    names = ["normal%d/concat:0" % (i + 1) for i in range(k)]
    names.insert(k // 3, "reduction%d/concat:0" % (k // 3))
    names.insert(k // 3 * 2 + 1, "reduction%d/concat:0" % (k // 3 * 2))
    return names


def names_vggs(k):
    names = []
    for i in range(3):
        names += ["conv%d/%d/Relu:0" % (i + 3, j + 1) for j in range(k)]
    return names


def direct(model_name):
    try:
        return __middles_dict__[model_name]
    except KeyError:
        return ([-1], ['out:0'])


# Dictionary for lists of representative endpoints.
__middles_dict__ = {
    'inception1': (
        [24, 38, 39, 53] + list(range(67, 110, 14)) + [110, 124, 138],
        names_inceptions([3, 6, 2], 3, pool_last=True)
    ),
    'inception2': (
        [31, 54] + list(range(71, 164, 23)) + list(range(180, 227, 23)),
        names_inceptions([3, 5, 2], 3)
    ),
    'inception3': (
        list(range(39, 86, 23)) + list(range(99, 228, 32)) +
        list(range(247, 310, 31)),
        names_inceptions([3, 5, 3], 5, omit_first=True)
    ),
    'inception4': (
        list(range(37, 130, 23)) + list(range(143, 368, 32)) +
        list(range(387, 490, 34)),
        names_inceptions([5, 8, 4], 5)
    ),
    'inceptionresnet2_tfslim': (
        [39] + list(range(60, 250, 21)) + [263] + list(range(278, 564, 15)) +
        [586] + list(range(601, 737, 15)),
        names_inceptions([11, 21, 11], 5, omit_first=True, resnet=True)
    ),
    'resnet50': (
        list(range(16, 35, 9)) + list(range(45, 73, 9)) +
        list(range(83, 129, 9)) + list(range(139, 158, 9)),
        names_resnets([3, 4, 6, 3])
    ),
    'resnet101': (
        list(range(16, 35, 9)) + list(range(45, 73, 9)) +
        list(range(83, 282, 9)) + list(range(292, 311, 9)),
        names_resnets([3, 4, 23, 3])
    ),
    'resnet152': (
        list(range(16, 35, 9)) + list(range(45, 109, 9)) +
        list(range(119, 435, 9)) + list(range(445, 464, 9)),
        names_resnets([3, 8, 36, 3])
    ),
    'resnet50v2': (
        list(range(15, 27, 11)) + [38] + list(range(50, 73, 11)) + [84] +
        list(range(96, 141, 11)) + [152] + list(range(164, 176, 11)) + [186],
        names_resnets([3, 4, 6, 3])
    ),
    'resnet101v2': (
        list(range(15, 27, 11)) + [38] + list(range(50, 73, 11)) + [84] +
        list(range(96, 328, 11)) + [339] + list(range(351, 363, 11)) + [373],
        names_resnets([3, 4, 23, 3])
    ),
    'resnet152v2': (
        list(range(15, 27, 11)) + [38] + list(range(50, 117, 11)) + [128] +
        list(range(140, 515, 11)) + [526] + list(range(538, 550, 11)) + [560],
        names_resnets([3, 8, 36, 3])
    ),
    'resnet200v2': (
        list(range(18, 41, 11)) + list(range(53, 307, 11)) +
        list(range(319, 705, 11)) + list(range(717, 740, 11)),
        names_resnets([3, 24, 36, 3])
    ),
    'resnext50': (
        list(range(49, 134, 42)) + list(range(177, 304, 42)) +
        list(range(347, 558, 42)) + list(range(601, 686, 42)),
        names_resnets([3, 4, 6, 3])
    ),
    'resnext101': (
        list(range(49, 134, 42)) + list(range(177, 304, 42)) +
        list(range(347, 1272, 42)) + list(range(1315, 1400, 42)),
        names_resnets([3, 4, 23, 3])
    ),
    'resnext50c32': (
        list(range(49, 134, 42)) + list(range(177, 304, 42)) +
        list(range(347, 558, 42)) + list(range(601, 686, 42)),
        names_resnets([3, 4, 6, 3])
    ),
    'resnext101c32': (
        list(range(49, 134, 42)) + list(range(177, 304, 42)) +
        list(range(347, 1272, 42)) + list(range(1315, 1400, 42)),
        names_resnets([3, 4, 23, 3])
    ),
    'resnext101c64': (
        list(range(81, 230, 74)) + list(range(305, 528, 74)) +
        list(range(603, 2232, 74)) + list(range(2307, 2456, 74)),
        names_resnets([3, 4, 23, 3])
    ),
    'wideresnet50': (
        list(range(17, 38, 10)) + list(range(49, 80, 10)) +
        list(range(91, 142, 10)) + list(range(153, 174, 10)),
        names_resnets([3, 4, 6, 3])
    ),
    'nasnetAlarge': (
        list(range(145, 371, 45)) + [416] + list(range(466, 692, 45)) + [748] +
        list(range(798, 1024, 45)),
        names_nasnets(18),
    ),
    'nasnetAmobile': (
        list(range(145, 281, 45)) + [326] + list(range(376, 512, 45)) + [568] +
        list(range(618, 754, 45)),
        names_nasnets(12),
    ),
    'vgg16': (
        list(range(11, 16, 2)) + list(range(18, 23, 2)) +
        list(range(25, 30, 2)),
        names_vggs(3),
    ),
    'vgg19': (
        list(range(11, 18, 2)) + list(range(20, 27, 2)) +
        list(range(29, 36, 2)),
        names_vggs(4),
    ),
    'densenet121': (
        list(range(12, 48, 7)) + [51] + list(range(58, 136, 7)) + [139] +
        list(range(146, 308, 7)) + [311] + list(range(318, 424, 7)),
        names_resnets([6, 12, 24, 16], pool_last=True)
    ),
    'densenet169': (
        list(range(12, 48, 7)) + [51] + list(range(58, 136, 7)) + [139] +
        list(range(146, 364, 7)) + [367] + list(range(374, 592, 7)),
        names_resnets([6, 12, 32, 32], pool_last=True)
    ),
    'densenet201': (
        list(range(12, 48, 7)) + [51] + list(range(58, 136, 7)) + [139] +
        list(range(146, 476, 7)) + [479] + list(range(486, 704, 7)),
        names_resnets([6, 12, 48, 32], pool_last=True)
    ),
    'mobilenet25': (
        list(range(20, 81, 6)),
        ['conv%d/conv/Relu6:0' % (i + 4) for i in range(11)]
    ),
    'mobilenet50': (
        list(range(20, 81, 6)),
        ['conv%d/conv/Relu6:0' % (i + 4) for i in range(11)]
    ),
    'mobilenet75': (
        list(range(20, 81, 6)),
        ['conv%d/conv/Relu6:0' % (i + 4) for i in range(11)]
    ),
    'mobilenet100': (
        list(range(20, 81, 6)),
        ['conv%d/conv/Relu6:0' % (i + 4) for i in range(11)]
    ),
    'squeezenet': (
        [9, 16, 17, 24, 31, 32] + list(range(39, 61, 7)) + [63],
        names_squeezenet()
    ),
    'REFyolov2': ([-1], ['linear/BiasAdd:0']),
    'REFyolov2voc': ([-1], ['linear/BiasAdd:0']),
    'REFtinyyolov2': ([-1], ['linear/BiasAdd:0']),
    'REFtinyyolov2voc': ([-1], ['linear/BiasAdd:0']),
}
