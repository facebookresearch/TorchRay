# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module provides an implementation of the *excitation backpropagation*
method of [EBP]_  for saliency visualization. It is a backpropagation method,
and thus works by changing the definition of the backward functions of some
layers.

In simple cases, the :func:`excitation_backprop` function can be used to obtain
the required visualization, as in the following example:

.. literalinclude:: ../examples/excitation_backprop.py
   :language: python
   :linenos:


Alternatively, you can explicitly use the :class:`ExcitationBackpropContext`,
as follows:

.. literalinclude:: ../examples/excitation_backprop_manual.py
   :language: python
   :linenos:

See also :ref:`backprop` for further examples and discussion.

Contrastive variant
~~~~~~~~~~~~~~~~~~~

The contrastive variant of excitation backprop passes the data twice through
the network. The first pass is used to obtain "contrast" activations at some
intermediate layer ``contrast_layer``. The latter is obtained by flipping the
sign of the last classification layer ``classifier_layer``. The visualization
is then obtained at some earlier ``input_layer``. The function
:func:`contrastive_excitation_backprop` can be used to compute this saliency:

.. literalinclude:: ../examples/contrastive_excitation_backprop.py
   :language: python
   :linenos:

This can also be done "manually", as follows:

.. literalinclude:: ../examples/contrastive_excitation_backprop_manual.py
   :language: python
   :linenos:

Theory
~~~~~~

Excitation backprop modifies the backward version of all linear layers in the
network. For a simple 1D case, let the forward layer be given by:

.. math::
     y_i = \sum_{j=1}^N w_{ij} x_j

where :math:`x \in \mathbb{R}^B` is the input and
:math:`w \in\mathbb{R}^{M \times N}` the weight matrix. On the way back, if
:math:`p \in \mathbb{R}^{N}` is the output pseudo-gradient, the input
pseudo-gradient :math:`p'` is given by:

.. math::
    p'_j =
    \sum_{i=1}^N
        \frac
        {w^+_{ij} x_j}
        {\sum_{k=1}^ Nw^+_{ik} x_k} p_i
    \quad\text{where}\quad
    w^+_{ij} = \max\{0, w_{ij}\}
    :label: linear-back

Note that [EBP]_ assumes that the input activations :math:`x` are always
non-negative. This is often true, as linear layer are preceded by non-negative
activation functions such as ReLU, so there is nothing to be done.

Note also that we can rearrange :eq:`linear-back` as

.. math::
    p'_j =
    x_j
    \sum_{i=1}^N
        w^+_{ij}
        \hat p_i,
    \quad\text{where}\quad
    \hat p_i =
        \frac
        {p_i}
        {\sum_{k=1}^ Nw^+_{ik} x_k}.


Here :math:`\hat p` is a normalized version of the output pseudo-gradient and
the summation is the operation performed by the standard backward function for
the linear layer given as input :math:`\hat p`.

All linear layers, including convolution and deconvolution layers as well as
average pooling layers, can be processed in this manner. In general, let
:math:`y = f(x,w)` be a linear layer. In order to compute :eq:`linear-back`, we
can expand it as:

.. math::
    p' = x \odot f^*(x, w^+, p \oslash f(x, w^+))

where :math:`f^*(x, w, p)` is the standard backward function (vector-Jacobian
product) for the linear layer :math:`f` and :math:`\odot` and :math:`\oslash`
denote element-wise multiplication and division, respectively.

The **contrastive variant** of excitation backprop is similar, but uses the
idea of contrasting the excitation for one class with the ones of all the
others.

In order to obtain the "contrast" signal, excitation backprop is run as before
except for the last *classification* linear layer. In order to backpropagate
the excitation for this layer only, the weight parameter :math:`w` is replaced
with its opposite :math:`-w`. Then, the excitations are backpropagated to an
intermediate *contrast* linear layer as normal.

Once the contrast activations have been obtained, excitation backprop is run
again, this time with the "default" weights even for the last linear layer.
However, during backpropagation, when the contrast layer is reached again, the
contrast is subtracted from the excitations. Then, the excitations are
propagated backward as usual.

References:

    .. [EBP] Jianming Zhang, Zhe Lin, Jonathan Brandt, Xiaohui Shen, Stan
             Sclaroff, *Top-down Neural Attention by Excitation Backprop*,
             ECCV 2016, `<https://arxiv.org/abs/1608.00507>`__.
"""

__all__ = [
    "contrastive_excitation_backprop",
    "eltwise_sum",
    "excitation_backprop",
    "ExcitationBackpropContext",
]

import torch
import torch.nn as nn

from torch.autograd import Function
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

from .common import Patch, Probe
from .common import attach_debug_probes, get_backward_gradient, get_module
from .common import saliency, resize_saliency


class EltwiseSumFunction(Function):
    """
    Implementation of a skip connection (i.e., element-wise sum function)
    as a :class:`torch.autograd.Function`. This is necessary for patching
    the skip connection as a :class:`torch.nn.Module` object (i.e.,
    :class:`EltwiseSum`).
    """
    @staticmethod
    def forward(ctx, *inputs):
        ctx.save_for_backward(*inputs)
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output + inputs[i]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        return (grad_output,) * len(inputs)


# Alias for :class:`EltwiseSumFunction`.
eltwise_sum = EltwiseSumFunction.apply


class EltwiseSum(nn.Module):
    """
    Implemention of skip connection (i.e., element-wise sum) as a
    :class:`torch.nn.Module` (this is necessary to be able to patch the
    skip connection).
    """

    def forward(self, *inputs):
        return eltwise_sum(*inputs)


def update_resnet(model, debug=False):
    """
    Update a ResNet model to use :class:`EltwiseSum` for the skip
    connection.

    Args:
        model (:class:`torchvision.models.resnet.ResNet`): ResNet model.
        debug (bool): If True, print debug statements.

    Returns:
        model (:class:`torchvision.models.resnet.ResNet`): ResNet model
        that uses :class:`EltwiseSum` for the skip connections. The forward
        functions of :class:`torchvision.models.resnet.BasicBlock` and
        :class:`torch.models.resnet.Bottleneck` are modified.
    """
    assert isinstance(model, ResNet)

    def bottleneck_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip(out, identity)
        out = self.relu(out)

        return out

    def basicblock_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip(out, identity)
        out = self.relu(out)

        return out

    for module_name, module in model.named_modules():
        if isinstance(module, Bottleneck):
            module.skip = EltwiseSum()
            module.forward = bottleneck_forward.__get__(module)
        elif isinstance(module, BasicBlock):
            module.skip = EltwiseSum()
            module.forward = basicblock_forward.__get__(module)
        else:
            continue
        if debug:
            print('Adding EltwiseSum as skip connection in {}.'.format(
                module_name))
    return model


class ExcitationBackpropContext(object):
    """Context to use Excitation Backpropagation rules."""
    @staticmethod
    def _patch_conv(target_name, enable, debug):
        """Patch conv functions to use excitation backprop rules.

        Replicated implementation provided in:
        https://github.com/jimmie33/Caffe-ExcitationBP/blob/master/src/caffe/layers/conv_layer.cpp

        Args:
            target_name (str): name of function to patch (i.e.,
                ``'torch.nn.functional.conv_1d'``).
            enable (bool): If True, enable excitation backprop rules.
            debug (bool): If True, print debug statements.

        Returns:
            :class:`.common.Patch`: object that patches the function with
            the name :attr:`target_name` with a new callable that implements
            the excitation backprop rules for the conv function.
        """
        target, attribute = Patch.resolve(target_name)
        conv = getattr(target, attribute)

        def forward(ctx, input, weight, bias=None, *args, **kwargs):
            ctx.save_for_backward(input, weight, bias)
            ctx.args = args
            ctx.kwargs = kwargs
            if debug:
                print("EBP " + target_name)
            return conv(input, weight, bias, *ctx.args, **ctx.kwargs)

        def backward(ctx, grad_output):
            def get(i):
                x = ctx.saved_tensors[i]
                if x is None:
                    return
                x = x.detach()
                x.requires_grad_(ctx.needs_input_grad[i])
                return x

            inputs_ = get(0), get(1), get(2)
            grad_inputs_ = [None, None, None]
            subset = [i for i, g in enumerate(ctx.needs_input_grad) if g]
            inputs_subset = [inputs_[i] for i in subset]

            with torch.enable_grad():
                # EBP changes only the gradients w.r.t. inputs_[0]. We also
                # compute the gradients w.r.t. the parameters if needed,
                # although they are trash. Perhaps it would be better to set
                # them to None?
                #
                # The expectation is that the input to the conv layer is
                # non-negative, which is typical for all but the first layer
                # due to the ReLUs. Some other implementation makes sure by
                # clamping inputs_[0].

                # 1. set weight W+ to be non-negative and disable bias.
                if enable:
                    input = inputs_[0]
                    weight = inputs_[1].clamp(min=0)
                    bias = None
                else:
                    input = inputs_[0]
                    weight = inputs_[1]
                    bias = inputs_[2]

                # 2. do forward pass.
                output_ebp = conv(
                    input,
                    weight,
                    bias,
                    *ctx.args,
                    **ctx.kwargs
                )

                # 3. normalize gradient by the output of the forward pass.
                if enable:
                    grad_output = grad_output / (output_ebp + 1e-20)

                # 4. do backward pass.
                _ = torch.autograd.grad(
                    output_ebp, inputs_subset, grad_output, only_inputs=True)
                for i, j in enumerate(subset):
                    grad_inputs_[j] = _[i]

                # 5. multiply gradient with the layer's input.
                if ctx.needs_input_grad[0] and enable:
                    grad_inputs_[0] *= inputs_[0]

                return (grad_inputs_[0], grad_inputs_[1], grad_inputs_[2],
                        None, None, None, None)

        autograd_conv = type('EBP_' + attribute, (torch.autograd.Function,), {
            'forward': staticmethod(forward),
            'backward': staticmethod(backward),
        })

        return Patch(target_name, autograd_conv.apply)

    @staticmethod
    def _patch_pool(target_name, enable, debug):
        """Patch pool functions to use excitation backprop rules.

        Replicated implementation provided in:
        https://github.com/jimmie33/Caffe-ExcitationBP/blob/master/src/caffe/layers/pooling_layer.cpp

        Args:
            target_name (str): name of function to patch (i.e.,
                ``'torch.nn.functional.avg_pool1d'``).
            enable (bool): If True, enable excitation backprop rules.
            debug (bool): If True, print debug statements.

        Returns:
            :class:`.common.Patch`: object that patches the function with
            the name :attr:`target_name` with a new callable that implements
            the excitation backprop rules for the pool function.
        """
        target, attribute = Patch.resolve(target_name)
        pool = getattr(target, attribute)

        def forward(ctx, input, *args, **kwargs):
            ctx.save_for_backward(input)
            ctx.args = args
            ctx.kwargs = kwargs
            if debug:
                print('EBP ' + target_name)
            return pool(input, *ctx.args, **ctx.kwargs)

        def backward(ctx, grad_output):
            if not ctx.needs_input_grad[0]:
                return None,
            input_ = ctx.saved_tensors[0].detach()
            input_.requires_grad_(True)
            with torch.enable_grad():
                # 1. forward pass.
                output_ebp = pool(input_, *ctx.args, **ctx.kwargs)

                # 2. normalize gradient by the output of the forward pass.
                if enable:
                    grad_output = grad_output / (output_ebp + 1e-20)

                # 3. do backward pass.
                grad_input_ = torch.autograd.grad(
                    output_ebp, input_, grad_output, only_inputs=True)[0]

                # 4. multiply gradient with layer's input.
                if enable:
                    grad_input_ *= input_
                return grad_input_, None, None, None, None, None, None

        autograd_pool = type('EBP_' + attribute, (torch.autograd.Function,), {
            'forward': staticmethod(forward),
            'backward': staticmethod(backward),
        })

        return Patch(target_name, autograd_pool.apply)

    @staticmethod
    def _patch_norm(target_name, enable, debug):
        """
        Patch normalization functions (e.g., batch norm) to use excitation
        backprop rules.

        Args:
            target_name (str): name of function to patch (i.e.,
                ``'torch.nn.functional.avg_pool1d'``).
            enable (bool): If True, enable excitation backprop rules.
            debug (bool): If True, print debug statements.

        Returns:
            :class:`.common.Patch`: object that patches the function with
            the name :attr:`target_name` with a new callable that implements
            the excitation backprop rules for the normalization function.
        """
        target, attribute = Patch.resolve(target_name)
        norm = getattr(target, attribute)

        def forward(ctx, input, *args, **kwargs):
            ctx.save_for_backward(input)
            ctx.args = args
            ctx.kwargs = kwargs
            if debug:
                print('EBP ' + target_name)
            return norm(input, *ctx.args, **ctx.kwargs)

        def backward(ctx, grad_output):
            if enable:
                return grad_output, None, None, None, None, None, None, None

            input_ = ctx.saved_tensors[0].detach()
            input_.requires_grad_(True)
            with torch.enable_grad():
                output_ebp = norm(input_, *ctx.args, **ctx.kwargs)
                grad_input_ = torch.autograd.grad(
                    output_ebp, input_, grad_output, only_inputs=True)[0]

            return grad_input_, None, None, None, None, None, None, None

        autograd_norm = type('EBP_' + attribute, (torch.autograd.Function,), {
            'forward': staticmethod(forward),
            'backward': staticmethod(backward),
        })

        return Patch(target_name, autograd_norm.apply)

    def _patch_eltwise_sum(self, target_name, enable, debug):
        """
        Patch element-wise sum function (e.g., skip connection) to use
        excitation backprop rules.

        Args:
            target_name (str): name of function to patch (i.e.,
                ``'torch.nn.functional.avg_pool1d'``).
            enable (bool): If True, enable excitation backprop rules.
            debug (bool): If True, print debug statements.

        Returns:
            :class:`.common.Patch`: object that patches the function with
            the name :attr:`target_name` with a new callable that implements
            the excitation backprop rules for the element-wise sum function.
        """
        target, attribute = Patch.resolve(target_name)
        eltwise_sum_f = getattr(target, attribute)

        def forward(ctx, *inputs):
            ctx.save_for_backward(*inputs)
            if debug:
                print("EBP " + target_name)
            return eltwise_sum_f(*inputs)

        def backward(ctx, grad_output):
            inputs = ctx.saved_tensors
            if not enable:
                return (grad_output, ) * len(inputs)

            inputs = [inputs[i].detach() for i in range(len(inputs))]
            output = eltwise_sum_f(*inputs)
            grad_outputs = []
            for input in inputs:
                grad_outputs.append(input / output * grad_output)
            return tuple(grad_outputs)

        autograd_eltwise_sum = type('EBP_' + attribute,
                                    (torch.autograd.Function,), {
                                        'forward': staticmethod(forward),
                                        'backward': staticmethod(backward),
                                    })

        return Patch(target_name, autograd_eltwise_sum.apply)

    def __init__(self, enable=True, debug=False):
        self.enable = enable
        self.debug = debug
        self.patches = []

    def __enter__(self):
        # Patch torch functions for convolutional, linear, average pooling,
        # and adaptive average pooling layers. Also patch eltwise_sum function
        # (for skip connection in resnet models).
        self.patches = [
            self._patch_conv('torch.nn.functional.conv1d',
                             self.enable, self.debug),
            self._patch_conv('torch.nn.functional.conv2d',
                             self.enable, self.debug),
            self._patch_conv('torch.nn.functional.conv3d',
                             self.enable, self.debug),
            self._patch_conv('torch.nn.functional.linear',
                             self.enable, self.debug),
            self._patch_pool('torch.nn.functional.avg_pool1d',
                             self.enable, self.debug),
            self._patch_pool('torch.nn.functional.avg_pool2d',
                             self.enable, self.debug),
            self._patch_pool('torch.nn.functional.avg_pool3d',
                             self.enable, self.debug),
            self._patch_pool('torch.nn.functional.adaptive_avg_pool1d',
                             self.enable, self.debug),
            self._patch_pool('torch.nn.functional.adaptive_avg_pool2d',
                             self.enable, self.debug),
            self._patch_pool('torch.nn.functional.adaptive_avg_pool3d',
                             self.enable, self.debug),
            self._patch_norm('torch.nn.functional.batch_norm',
                             self.enable, self.debug),
            self._patch_eltwise_sum(
                'torchray.attribution.excitation_backprop.eltwise_sum',
                self.enable,
                self.debug),
        ]
        return self

    def __exit__(self, type, value, traceback):
        for patch in self.patches:
            patch.remove()
        return False  # re-raise any exception


def _get_classifier_layer(model):
    r"""Get the classifier layer.

    Args:
        model (:class:`torch.nn.Module`): a model.

    Returns:
        (:class:`torch.nn.Module`, str): tuple of the last layer and its name.
    """
    # Get last layer with weight parameters.
    last_layer_name = None
    last_layer = None
    for parameter_name, _ in list(model.named_parameters())[::-1]:
        if '.weight' in parameter_name:
            last_layer_name, _ = parameter_name.split('.weight')
            last_layer = get_module(model, last_layer_name)
            # Check that last layer is convolutional or linear.
            if (isinstance(last_layer, nn.Conv1d)
                or isinstance(last_layer, nn.Conv2d)
                or isinstance(last_layer, nn.Conv3d)
                    or isinstance(last_layer, nn.Linear)):
                break
            else:
                last_layer_name = None
                last_layer = None
    assert last_layer_name is not None
    assert last_layer is not None
    return last_layer, last_layer_name


def gradient_to_excitation_backprop_saliency(x):
    r"""Convert a gradient to an excitation backprop saliency map.

    The tensor :attr:`x` must have a valid gradient ``x.grad``.
    The function then computes the excitation backprop saliency map :math:`s`
    given by:

    .. math::

        s_{n,1,u} = \max(\sum_{0 \leq c < C} dx_{ncu}, 0)

    where :math:`n` is the instance index, :math:`c` the channel
    index and :math:`u` the spatial multi-index (usually of dimension 2 for
    images).

    Args:
        x (:class:`torch.Tensor`): activation with gradient.

    Return:
        :class:`torch.Tensor`: saliency map.
    """
    return torch.sum(x.grad, 1, keepdim=True)


def gradient_to_contrastive_excitation_backprop_saliency(x):
    r"""Convert a gradient to an contrastive excitation backprop saliency map.

    The tensor :attr:`x` must have a valid gradient ``x.grad``.
    The function then computes the excitation backprop saliency map :math:`s`
    given by:

    .. math::

        s_{n,1,u} = \max(\sum_{0 \leq c < C} dx_{ncu}, 0)

    where :math:`n` is the instance index, :math:`c` the channel
    index and :math:`u` the spatial multi-index (usually of dimension 2 for
    images).

    Args:
        x (:class:`torch.Tensor`): activation with gradient.

    Return:
        :class:`torch.Tensor`: saliency map.
    """
    return torch.clamp(torch.sum(x.grad, 1, keepdim=True), min=0)


def excitation_backprop(*args,
                        context_builder=ExcitationBackpropContext,
                        gradient_to_saliency=gradient_to_excitation_backprop_saliency,
                        **kwargs):
    r"""Excitation backprop.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the Excitation backprop method, and supports
    the same arguments and return values.
    """
    assert context_builder is ExcitationBackpropContext
    return saliency(
        *args,
        context_builder=context_builder,
        gradient_to_saliency=gradient_to_saliency,
        **kwargs
    )


def contrastive_excitation_backprop(model,
                                    input,
                                    target,
                                    saliency_layer,
                                    contrast_layer,
                                    classifier_layer=None,
                                    resize=False,
                                    resize_mode='bilinear',
                                    get_backward_gradient=get_backward_gradient,
                                    debug=False):
    """Contrastive excitation backprop.

    Args:
        model (:class:`torch.nn.Module`): a model.
        input (:class:`torch.Tensor`): input tensor.
        target (int or :class:`torch.Tensor`): target label(s).
        saliency_layer (str or :class:`torch.nn.Module`): name of the saliency
            layer (str) or the layer itself (:class:`torch.nn.Module`) in
            the model at which to visualize.
        contrast_layer (str or :class:`torch.nn.Module`): name of the contrast
            layer (str) or the layer itself (:class:`torch.nn.Module`).
        classifier_layer (str or :class:`torch.nn.Module`, optional): name of
            the last classifier layer (str) or the layer itself
            (:class:`torch.nn.Module`). Defaults to ``None``, in which case
            the functions tries to automatically identify the last layer.
            Default: ``None``.
        resize (bool or tuple, optional): If True resizes the saliency map to
            the same size as :attr:`input`. It is also possible to pass a
            (width, height) tuple to specify an arbitrary size. Default:
            ``False``.
        resize_mode (str, optional): Specify the resampling mode.
            Default: ``'bilinear'``.
        get_backward_gradient (function, optional): function that generates
            gradient tensor to backpropagate. Default:
            :func:`.common.get_backward_gradient`.
        debug (bool, optional): If True, also return
            :class:`collections.OrderedDict` of :class:`.common.Probe` objects
            attached to all named modules in the model. Default: ``False``.

    Returns:
        :class:`torch.Tensor` or tuple: If :attr:`debug` is False, returns a
        :class:`torch.Tensor` saliency map at :attr:`saliency_layer`.
        Otherwise, returns a tuple of a :class:`torch.Tensor` saliency map
        at :attr:`saliency_layer` and an :class:`collections.OrderedDict`
        of :class:`Probe` objects for all modules in the model.
    """
    debug_probes = attach_debug_probes(model, debug=debug)

    # Disable gradients for model parameters.
    for param in model.parameters():
        param.requires_grad_(False)

    # Set model to eval mode.
    if model.training:
        model.eval()

    saliency_layer = get_module(model, saliency_layer)
    contrast_layer = get_module(model, contrast_layer)
    if classifier_layer is None:
        classifier_layer, _ = _get_classifier_layer(model)
    classifier_layer = get_module(model, classifier_layer)

    with ExcitationBackpropContext():
        probe_contrast = Probe(contrast_layer, target='output')
        output = model(input)
        gradient = get_backward_gradient(output, target)

        try:
            # Flip the weights of the last layer.
            classifier_layer.weight.data.neg_()
            output.backward(gradient.clone())

            # Save negative gradient and prepare to backpropagated contrastive
            # gradient.
            probe_contrast.contrast = [probe_contrast.data[0].grad]
        finally:
            # Flip back.
            classifier_layer.weight.data.neg_()

        # Forward-backward pass to get positive gradient at the contrastive
        # layer and and backpropagate contrastive gradient to input.
        probe_saliency = Probe(saliency_layer, target='output')
        output = model(input)
        output.backward(gradient)

    saliency_map = gradient_to_contrastive_excitation_backprop_saliency(
        probe_saliency.data[0])

    probe_saliency.remove()
    probe_contrast.remove()

    saliency_map = resize_saliency(input, saliency_map, resize, resize_mode)

    if debug:
        return saliency_map, debug_probes
    else:
        return saliency_map
