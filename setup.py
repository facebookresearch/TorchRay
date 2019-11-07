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
    url='https://github.com/facebookresearch/TorchRay',
    download_url='https://github.com/ruthcfong/TorchRay/archive/v1.0.0.2.tar.gz',
    author='Andrea Vedaldi',
    author_email='vedaldi@fb.com',
    license='Creative Commons Attribution-Noncommercial 4.0 International',
    description='TorchRay is a PyTorch library of visualization methods for convnets.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
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
