# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import setup

setup(
    name='torchray',
    version=open('torchray/VERSION').readline(),
    packages=[
        'torchray',
        'torchray.attribution',
        'torchray.benchmark'
    ],
    package_data={
        'torchray': ['VERSION'],
        'torchray.benchmark': ['*.txt']
    },
    url='http://pypi.python.org/pypi/torchray/',
    author='Andrea Vedaldi',
    author_email='vedaldi@fb.com',
    license='Creative Commons Attribution-Noncommercial 4.0 International',
    description='TorchRay is a PyTorch library of visualization methods for convnets.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Machine Learning :: Neural Networks',
        'License :: OSI Approved :: Creative Commons Attribution-Noncommercial 4.0 International',
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        'importlib_resources',
        'matplotlib',
        'packaging',
        'pycocotools >= 2.0.0',
        'pymongo',
        'requests',
        'torch >= 1.1',
        'torchvision >= 0.3.0',
    ],
    setup_requires=[
        'cython',
        'numpy',
    ]
)
