from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds
from setuptools import setup
import os
import codecs
import re
long_description = 'A Python package implementing carbon box models for tree ring radiocarbon data analysis.'

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# DEPENDENCIES
# 1. What are the required dependencies?
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
# 2. What dependencies required to run the unit tests? (i.e. `pytest --remote-data`)
tests_require = ['pytest', 'pytest-cov', 'pytest-remotedata']

setup(name='ticktack',
      version=find_version("src", "__init__.py"),
      description='Python package to build a carbon box model.',
      long_description=long_description,
      author='Utkarsh Sharma',
      author_email='u.sharma@uqconnect.edu.au',
      url='https://github.com/SharmaLlama/ticktack',
      package_dir={'ticktack': 'src'},
      packages=['ticktack'],
      package_data={'': ['data/*.hd5']},
      include_package_data=True,
      install_requires=install_requires,
      tests_require=tests_require,
      license='MIT',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python"
      ]
      )
