# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module provides an implementation of the *linear approximation* method
for saliency visualization. The simplest interface is given by the
:func:`linear_approx` function:

.. literalinclude:: ../examples/linear_approx.py
   :language: python
   :linenos:

Alternatively, it is possible to run the method "manually". Linear
approximation is a variant of the gradient method, applied at an intermediate
layer:

.. literalinclude:: ../examples/linear_approx_manual.py
   :language: python
   :linenos:

Note that the function :func:`gradient_to_linear_approx_saliency` is used to
convert activations and gradients to a saliency map.
"""

__all__ = ['gradient_to_linear_approx_saliency', 'linear_approx']


import torch
from .common import saliency


def gradient_to_linear_approx_saliency(x):
    """Returns the linear approximation of a tensor.

    The tensor :attr:`x` must have a valid gradient ``x.grad``.
    The function then computes the saliency map :math:`s`: given by:

    .. math::

        s_{n1u} = \sum_{c} x_{ncu} \cdot dx_{ncu}

    Args:
        x (:class:`torch.Tensor`): activation tensor with a valid gradient.

    Returns:
        :class:`torch.Tensor`: Saliency map.
    """
    viz = torch.sum(x * x.grad, 1, keepdim=True)
    return viz


def linear_approx(*args,
                  gradient_to_saliency=gradient_to_linear_approx_saliency,
                  **kwargs):
    """Linear approximation.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the linear approximation method, and
    supports the same arguments and return values.
    """
    return saliency(*args,
                    gradient_to_saliency=gradient_to_saliency,
                    **kwargs)
