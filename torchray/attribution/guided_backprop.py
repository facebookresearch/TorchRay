# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module implements *guided backpropagation* method of [GUIDED]_ or
visualizing deep networks. The simplest interface is given by the
:func:`guided_backprop` function:

.. literalinclude:: ../examples/guided_backprop.py
   :language: python
   :linenos:

Alternatively, it is possible to run the method "manually". Guided backprop is
a backpropagation method, and thus works by changing the definition of the
backward functions of some layers.  This can be done using the
:class:`GuidedBackpropContext` context:

.. literalinclude:: ../examples/guided_backprop_manual.py
   :language: python
   :linenos:

See also :ref:`backprop` for further examples.

Theory
~~~~~~

Guided backprop is a backpropagation method, and thus it works by changing the
definition of the backward functions of some layers. The only change is a
modified definition of the backward ReLU function:

.. math::
    \operatorname{ReLU}^*(x,p) =
    \begin{cases}
    p, & \mathrm{if}~p > 0 ~\mathrm{and}~ x > 0,\\
    0, & \mathrm{otherwise} \\
    \end{cases}

The modified ReLU is implemented by class :class:`GuidedBackpropReLU`.

References:

    .. [GUIDED] Springenberg et al.,
               *Striving for simplicity: The all convolutional net*,
               ICLR Workshop 2015,
               `<https://arxiv.org/abs/1412.6806>`__.
"""

__all__ = ['GuidedBackpropContext', 'guided_backprop']

import torch

from .common import ReLUContext, saliency


class GuidedBackpropReLU(torch.autograd.Function):
    """This class implements a ReLU function with the guided backprop rules."""
    @staticmethod
    def forward(ctx, input):
        """Guided backprop ReLU forward function."""
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Guided backprop ReLU backward function."""
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input = grad_input.clamp(min=0)
        return grad_input


class GuidedBackpropContext(ReLUContext):
    r"""GuidedBackprop context.

    This context modifies the computation of gradients
    to match the guided backpropagaton definition.

    See :mod:`torchray.attribution.guided_backprop` for how to use it.
    """

    def __init__(self):
        super(GuidedBackpropContext, self).__init__(GuidedBackpropReLU)


def guided_backprop(*args, context_builder=GuidedBackpropContext, **kwargs):
    r"""Guided backprop.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the guided backprop method, and supports the
    same arguments and return values.
    """
    return saliency(*args,
                    context_builder=context_builder,
                    **kwargs)
