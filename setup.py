''' Setup file '''
from os import path
from distutils import extension

from setuptools import setup
import numpy
package_name = 'MNNparcellation'

inc_module = package_name + '.inc'
def readme():
    with open('README.md') as f:
        return f.read()

setup(name=package_name,
      version='0.1',
      description='A tool to parcellate the cortical surface into\
                   regions using tractograms obtained from dMRI',
      url='http://github.com/',
      author='Brahim Belaoucha',
      author_email='brahim.belaoucha@inria.fr',
      packages=[package_name,inc_module],
      scripts=['bin/MNNparcellation'],
      zip_safe=False)
