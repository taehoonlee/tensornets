import numpy

from sys import platform
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

ext = 'tensornets.references.darkflow_utils'
ext_modules = [Extension("%s.%s" % (ext, n),
                         sources=["%s/%s.pyx" % (ext.replace('.', '/'), n)],
                         libraries=[] if platform.startswith("win") else ['m'],
                         include_dirs=[numpy.get_include()])
               for n in ['nms', 'get_boxes']]
install_requires = [
        'tensorflow>=1.8.0'
        ]

setup(name='tensornets',
      version='0.4.5',
      description='high level network definitions in tensorflow',
      author='Taehoon Lee',
      author_email='me@taehoonlee.com',
      url='https://github.com/taehoonlee/tensornets',
      download_url='https://github.com/taehoonlee/tensornets/tarball/0.4.5',
      license='MIT',
      packages=['tensornets', 'tensornets.datasets',
                'tensornets.contrib_framework', 'tensornets.contrib_layers',
                'tensornets.references', ext],
      include_package_data=True,
      install_requires=install_requires,
      ext_modules=cythonize(ext_modules))
