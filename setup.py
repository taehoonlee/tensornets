from setuptools import setup

setup(name='tensornets',
      version='0.3.0',
      description='high level network definitions in tensorflow',
      author='Taehoon Lee',
      author_email='me@taehoonlee.com',
      url='https://github.com/taehoonlee/tensornets',
      download_url='https://github.com/taehoonlee/tensornets/tarball/0.3.0',
      license='MIT',
      install_requires=['tensorflow'],
      packages=['tensornets', 'tensornets.datasets', 'tensornets.references'],
      include_package_data=True)
