from __future__ import absolute_import

import numpy as np
import tensornets as nets
from tensornets.utils import load_img
import os
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get('CORE_CHANGED', 'True') == 'False',
    reason='Runs only when the relevant files have been modified.')


def test_load_img():
    x = load_img('cat.png')
    assert x.shape == (1, 733, 490, 3)

    x = load_img(['cat.png', 'cat.png'], target_size=(100, 200))
    assert x.shape == (2, 100, 200, 3)

    x = load_img(['cat.png'] * 3, target_size=(100, 200), crop_size=50)
    assert x.shape == (3, 50, 50, 3)

    with pytest.raises(ValueError):
        x = load_img(['cat.png', 'cat.png'])

    with pytest.raises(ValueError):
        x = load_img(['cat.png'] * 3, target_size=100)
