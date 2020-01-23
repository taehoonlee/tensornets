import tensorflow as tf

from distutils.version import LooseVersion


def tf_later_than(v):
    return LooseVersion(tf.__version__) > LooseVersion(v)


def tf_equal_to(v):
   return tf.__version__ == v
