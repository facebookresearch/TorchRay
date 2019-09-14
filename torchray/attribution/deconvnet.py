# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module implements the *deconvolution*  method of [DECONV]_ for visualizing
deep networks. The simplest interface is given by the :func:`deconvnet`
function:

.. literalinclude:: ../examples/deconvnet.py
    :language: python
    :linenos:

Alternatively, it is possible to run the method "manually". DeConvNet is a
backpropagation method, and thus works by changing the definition of the
backward functions of some layers. The modified ReLU is implemented by class
:class:`DeConvNetReLU`; however, this is rarely used directly; instead, one
uses the :class:`DeConvNetContext` context instead, as follows:

.. literalinclude:: ../examples/deconvnet_manual.py
    :language: python
    :linenos:

See also :ref:`Backprogation methods <backpropagation>` for further examples
and discussion.

Theory
~~~~~~

The only change is a modified definition of the backward ReLU function:

.. math::
    \operatorname{ReLU}^*(x,p) =
    \begin{cases}
    p, & \mathrm{if}~ p > 0,\\
    0, & \mathrm{otherwise} \\
    \end{cases}

Warning:

    DeConvNets are defined for "standard" networks that use ReLU operations.
    Further modifications may be required for more complex or new networks
    that use other type of non-linearities.

References:

    .. [DECONV] Zeiler and Fergus,
                *Visualizing and Understanding Convolutional Networks*,
                ECCV 2014,
                `<https://doi.org/10.1007/978-3-319-10590-1_53>`__.
"""

__all__ = ["DeConvNetContext", "deconvnet"]

import torch

from .common import ReLUContext, saliency


class DeConvNetReLU(torch.autograd.Function):
    """DeConvNet ReLU autograd function.

    This is an autograd function that redefines the ``relu`` function
    to match the DeConvNet ReLU definition.
    """

    @staticmethod
    def forward(ctx, input):
        """DeConvNet ReLU forward function."""
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """DeConvNet ReLU backward function."""
        return grad_output.clamp(min=0)


class DeConvNetContext(ReLUContext):
    """DeConvNet context.

    This context modifies the computation of gradient to match the DeConvNet
    definition.

    See :mod:`torchray.attribution.deconvnet` for how to use it.
    """

    def __init__(self):
        super(DeConvNetContext, self).__init__(DeConvNetReLU)


def deconvnet(*args, context_builder=DeConvNetContext, **kwargs):
    """DeConvNet method.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the DeConvNet method, and supports the
    same arguments and return values.
    """
    return saliency(*args, context_builder=context_builder, **kwargs)
