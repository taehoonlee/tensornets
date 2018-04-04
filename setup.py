import numpy

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

ext = 'tensornets.references.darkflow_utils'
ext_modules = [Extension("%s.%s" % (ext, n),
                         sources=["%s/%s.pyx" % (ext.replace('.', '/'), n)],
                         libraries=['m'],
                         include_dirs=[numpy.get_include()])
               for n in ['nms', 'get_boxes']]

setup(name='tensornets',
      version='0.3.3',
      description='high level network definitions in tensorflow',
      author='Taehoon Lee',
      author_email='me@taehoonlee.com',
      url='https://github.com/taehoonlee/tensornets',
      download_url='https://github.com/taehoonlee/tensornets/tarball/0.3.3',
      license='MIT',
      install_requires=['tensorflow'],
      packages=['tensornets', 'tensornets.datasets',
                'tensornets.references', ext],
      include_package_data=True,
      ext_modules=cythonize(ext_modules))
