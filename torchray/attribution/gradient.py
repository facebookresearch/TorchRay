# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module implements the *gradient* method of [GRAD]_ for visualizing a deep
network. It is a backpropagation method, and in fact the simplest of them all
as it coincides with standard backpropagation. The simplest way to use this
method is via the :func:`gradient` function:

.. literalinclude:: ../examples/gradient.py
    :language: python
    :linenos:

Alternatively, one can do so manually, as follows

.. literalinclude:: ../examples/gradient_manual.py
    :language: python
    :linenos:

Note that in this example, for visualization, the gradient is
convernted into an image by postprocessing by using the function
:func:`torchray.attribution.common.saliency`.

See also :ref:`backprop` for further examples.

References:

    .. [GRAD] Karen Simonyan, Andrea Vedaldi and Andrew Zisserman,
              *Deep Inside Convolutional Networks:
              Visualising Image Classification Models and Saliency Maps,*
              ICLR workshop, 2014,
              `<https://arxiv.org/abs/1312.6034>`__.
"""

__all__ = ["gradient"]

from .common import saliency


def gradient(*args, context_builder=None, **kwargs):
    r"""Gradient method

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the gradient method, and supports the
    same arguments and return values.
    """
    assert context_builder is None
    return saliency(*args, **kwargs)
