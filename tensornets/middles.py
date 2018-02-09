"""Collection of representative endpoints for each model."""
from __future__ import absolute_import


def direct(model_name):
    return __middles_dict__[model_name]


# Dictionary for lists of representative endpoints.
__middles_dict__ = {
    'inception1':
        [24, 38, 39, 53] + list(range(67, 110, 14)) + [110, 124, 138],
    'inception2':
        [31, 54] + list(range(71, 164, 23)) + list(range(180, 227, 23)),
    'inception3':
        list(range(39, 86, 23)) + list(range(99, 228, 32)) +
        list(range(247, 310, 31)),
    'inception4':
        list(range(37, 130, 23)) + list(range(143, 368, 32)) +
        list(range(387, 490, 34)),
    'inceptionresnet2_tfslim':
        [39] + list(range(60, 250, 21)) + [263] + list(range(278, 564, 15)) +
        [586] + list(range(601, 737, 15)),
    'resnet50':
        list(range(16, 35, 9)) + list(range(45, 73, 9)) +
        list(range(83, 129, 9)) + list(range(139, 158, 9)),
    'resnet101':
        list(range(16, 35, 9)) + list(range(45, 73, 9)) +
        list(range(83, 282, 9)) + list(range(292, 311, 9)),
    'resnet152':
        list(range(16, 35, 9)) + list(range(45, 109, 9)) +
        list(range(119, 435, 9)) + list(range(445, 464, 9)),
    'resnet50v2':
        list(range(15, 27, 11)) + [38] + list(range(50, 73, 11)) + [84] +
        list(range(96, 141, 11)) + [152] + list(range(164, 175, 11)) + [186],
    'resnet101v2':
        list(range(15, 27, 11)) + [38] + list(range(50, 73, 11)) + [84] +
        list(range(96, 328, 11)) + [339] + list(range(351, 362, 11)) + [373],
    'resnet152v2':
        list(range(15, 27, 11)) + [38] + list(range(50, 117, 11)) + [128] +
        list(range(140, 515, 11)) + [526] + list(range(538, 549, 11)) + [560],
    'resnet200v2':
        list(range(18, 41, 11)) + list(range(53, 307, 11)) +
        list(range(319, 705, 11)) + list(range(717, 740, 11)),
    'resnext50':
        list(range(49, 134, 42)) + list(range(177, 304, 42)) +
        list(range(347, 558, 42)) + list(range(601, 686, 42)),
    'resnext101':
        list(range(49, 134, 42)) + list(range(177, 304, 42)) +
        list(range(347, 1272, 42)) + list(range(1315, 1400, 42)),
    'resnext50c32':
        list(range(49, 134, 42)) + list(range(177, 304, 42)) +
        list(range(347, 558, 42)) + list(range(601, 686, 42)),
    'resnext101c32':
        list(range(49, 134, 42)) + list(range(177, 304, 42)) +
        list(range(347, 1272, 42)) + list(range(1315, 1400, 42)),
    'resnext101c64':
        list(range(81, 230, 74)) + list(range(305, 528, 74)) +
        list(range(603, 2232, 74)) + list(range(2307, 2456, 74)),
    'wideresnet50':
        list(range(17, 38, 10)) + list(range(49, 80, 10)) +
        list(range(91, 142, 10)) + list(range(153, 174, 10)),
    'nasnetAlarge':
        list(range(145, 371, 45)) + [416] + list(range(466, 692, 45)) + [748] +
        list(range(798, 1024, 45)),
    'nasnetAmobile':
        list(range(145, 281, 45)) + [326] + list(range(376, 512, 45)) + [568] +
        list(range(618, 754, 45)),
    'densenet121':
        list(range(12, 48, 7)) + [51] + list(range(58, 136, 7)) + [139] +
        list(range(146, 308, 7)) + [311] + list(range(318, 424, 7)),
    'densenet169':
        list(range(12, 48, 7)) + [51] + list(range(58, 136, 7)) + [139] +
        list(range(146, 364, 7)) + [367] + list(range(374, 592, 7)),
    'densenet201':
        list(range(12, 48, 7)) + [51] + list(range(58, 136, 7)) + [139] +
        list(range(146, 476, 7)) + [479] + list(range(486, 704, 7)),
    'mobilenet25': list(range(20, 81, 6)),
    'mobilenet50': list(range(20, 81, 6)),
    'mobilenet75': list(range(20, 81, 6)),
    'mobilenet100': list(range(20, 81, 6)),
    'squeezenet': [9, 16, 17, 24, 31, 32] + list(range(39, 61, 7)) + [63],
}
