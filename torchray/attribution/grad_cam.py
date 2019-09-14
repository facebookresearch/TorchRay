# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This module provides an implementation of the *Grad-CAM* method of [GRADCAM]_
for saliency visualization. The simplest interface is given by the
:func:`grad_cam` function:

.. literalinclude:: ../examples/grad_cam.py
   :language: python
   :linenos:

Alternatively, it is possible to run the method "manually". Grad-CAM backprop
is a variant of the gradient method, applied at an intermediate layer:

.. literalinclude:: ../examples/grad_cam_manual.py
   :language: python
   :linenos:

Note that the function :func:`gradient_to_grad_cam_saliency` is used to convert
activations and gradients to a saliency map.

See also :ref:`backprop` for further examples and discussion.

Theory
~~~~~~

Grad-CAM can be seen as a variant of the *gradient* method
(:mod:`torchray.attribution.gradient`) with two differences:

1. The saliency is measured at an intermediate layer of the network, usually at
   the output of the last convolutional layer.

2. Saliency is defined as the clamped product of forward activation and
   backward gradient at that layer.

References:

    .. [GRADCAM] Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam,
                 Michael Cogswell, Devi Parikh and Dhruv Batra,
                 *Visual Explanations from Deep Networks via Gradient-based
                 Localization,*
                 ICCV 2017,
                 `<http://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>`__.
"""

__all__ = ["grad_cam"]

import torch
from .common import saliency


def gradient_to_grad_cam_saliency(x):
    r"""Convert activation and gradient to a Grad-CAM saliency map.

    The tensor :attr:`x` must have a valid gradient ``x.grad``.
    The function then computes the saliency map :math:`s`: given by:

    .. math::

        s_{n1u} = \max\{0, \sum_{c}x_{ncu}\cdot dx_{ncu}\}

    Args:
        x (:class:`torch.Tensor`): activation tensor with a valid gradient.

    Returns:
        :class:`torch.Tensor`: saliency map.
    """
    # Apply global average pooling (GAP) to gradient.
    grad_weight = torch.mean(x.grad, (2, 3), keepdim=True)

    # Linearly combine activations and GAP gradient weights.
    saliency_map = torch.sum(x * grad_weight, 1, keepdim=True)

    # Apply ReLU to visualization.
    saliency_map = torch.clamp(saliency_map, min=0)

    return saliency_map


def grad_cam(*args,
             saliency_layer,
             gradient_to_saliency=gradient_to_grad_cam_saliency,
             **kwargs):
    r"""Grad-CAM method.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the Grad-CAM method, and supports the
    same arguments and return values.
    """
    return saliency(*args,
                    saliency_layer=saliency_layer,
                    gradient_to_saliency=gradient_to_saliency,
                    **kwargs,)
