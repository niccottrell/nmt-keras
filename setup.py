"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from os import path


setup(
    name='nmt-keras',
    version='1.0.0',
    description='A project to explore NMT with Keras',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['tensorflow', 'numpy', 'nltk'],
)
