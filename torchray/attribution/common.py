# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module defines common code for the backpropagation methods.
"""

import torch
import torch.nn.functional as F
import weakref
from collections import OrderedDict
from packaging import version

from torchray.utils import imsmooth

__all__ = [
    'attach_debug_probes',
    'get_backward_gradient',
    'get_module',
    'get_pointing_gradient',
    'gradient_to_saliency',
    'Probe',
    'Patch',
    'NullContext',
    'ReLUContext',
    'resize_saliency',
    'saliency'
]

# Certain algorithms fail to work properly in earlier versions.
assert version.parse(torch.__version__) >= version.parse('1.1'), \
    'PyTorch 1.1 or above required.'


class Patch(object):
    """Patch a callable in a module."""

    @staticmethod
    def resolve(target):
        """Resolve a target into a module and an attribute.

        The function resolves a string such as ``'this.that.thing'`` into a
        module instance `this.that` (importing the module) and an attribute
        `thing`.

        Args:
            target (str): target string.

        Returns:
            tuple: module, attribute.
        """
        target, attribute = target.rsplit('.', 1)
        components = target.split('.')
        import_path = components.pop(0)
        target = __import__(import_path)
        for comp in components:
            import_path += '.{}'.format(comp)
            __import__(import_path)
            target = getattr(target, comp)
        return target, attribute

    def __init__(self, target, new_callable):
        """Patch a callable in a module.

        Args:
            target (str): path to the callable to patch.
            callable (fun): new callable.
        """
        target, attribute = Patch.resolve(target)
        self.target = target
        self.attribute = attribute
        self.orig_callable = getattr(target, attribute)
        setattr(target, attribute, new_callable)

    def __del__(self):
        self.remove()

    def remove(self):
        """Remove the patch."""
        if self.target is not None:
            setattr(self.target, self.attribute, self.orig_callable)
        self.target = None


class ReLUContext(object):
    """
    A context manager that replaces :func:`torch.relu` with
        :attr:`relu_function`.

    Args:
        relu_func (:class:`torch.autograd.function.FunctionMeta`): class
            definition of a :class:`torch.autograd.Function`.
    """

    def __init__(self, relu_func):
        assert isinstance(relu_func, torch.autograd.function.FunctionMeta)
        self.relu_func = relu_func
        self.patches = []

    def __enter__(self):
        relu = self.relu_func().apply
        self.patches = [
            Patch('torch.relu', relu),
            Patch('torch.relu_', relu),
        ]
        return self

    def __exit__(self, type, value, traceback):
        for p in self.patches:
            p.remove()
        return False  # re-raise any exception


def _wrap_in_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


class _InjectContrast(object):
    def __init__(self, contrast, non_negative):
        self.contrast = contrast
        self.non_negative = non_negative

    def __call__(self, grad):
        assert grad.shape == self.contrast.shape
        delta = grad - self.contrast
        if self.non_negative:
            delta = delta.clamp(min=0)
        return delta


class _Catch(object):
    def __init__(self, probe):
        self.probe = weakref.ref(probe)

    def _process_data(self, data):
        if not self.probe():
            return
        p = self.probe()
        assert isinstance(data, list)
        p.data = data
        for i, x in enumerate(p.data):
            x.requires_grad_(True)
            x.retain_grad()
            if len(p.contrast) > i and p.contrast[i] is not None:
                injector = _InjectContrast(
                    p.contrast[i], p.non_negative_contrast)
                x.register_hook(injector)


class _CatchInputs(_Catch):
    def __call__(self, module, input):
        self._process_data(_wrap_in_list(input))


class _CatchOutputs(_Catch):
    def __call__(self, module, input, output):
        self._process_data(_wrap_in_list(output))


class Probe(object):
    """Probe for a layer.

    A probe attaches to a given :class:`torch.nn.Module` instance.
    While attached, the object records any data produced by the module along
    with the corresponding gradients. Use :func:`remove` to remove the probe.

    Examples:

        .. code:: python

            module = torch.nn.ReLU
            probe = Probe(module)
            x = torch.randn(1, 10)
            y = module(x)
            z = y.sum()
            z.backward()
            print(probe.data[0].shape)
            print(probe.data[0].grad.shape)
    """

    def __init__(self, module, target='input'):
        """Create a probe attached to the specified module.

        The probe intercepts calls to the module on the way forward, capturing
        by default all the input activation tensor with their gradients.

        The activation tensors are stored as a sequence :attr:`data`.

        Args:
            module (torch.nn.Module): Module to attach.
            target (str): Choose from ``'input'`` or ``'output'``. Use
                ``'output'`` to intercept the outputs of a module
                instead of the inputs into the module. Default: ``'input'``.

        .. Warning:

            PyTorch module interface (at least until 1.1.0) is partially
            broken. In particular, the hook functionality used by the probe
            work properly only for atomic module, not for containers such as
            sequences or for complex module that run several functions
            internally.
        """
        self.module = module
        self.data = []
        self.target = target
        self.hook = None
        self.contrast = []
        self.non_negative_contrast = False
        if hasattr(self.module, "inplace"):
            self.inplace = self.module.inplace
            self.module.inplace = False
        if self.target == 'input':
            self.hook = module.register_forward_pre_hook(_CatchInputs(self))
        elif self.target == 'output':
            self.hook = module.register_forward_hook(_CatchOutputs(self))
        else:
            assert False

    def __del__(self):
        self.remove()

    def remove(self):
        """Remove the probe."""
        if self.module is not None:
            if hasattr(self.module, "inplace"):
                self.module.inplace = self.inplace
            self.hook.remove()
            self.module = None


class NullContext(object):
    def __init__(self):
        r"""Null context.

        This context does nothing.
        """

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False


def get_pointing_gradient(pred_y, y, normalize=True):
    """Returns a gradient tensor for the pointing game.

    Args:
        pred_y (:class:`torch.Tensor`): 4D tensor that the model outputs.
        y (int): target label.
        normalize (bool): If True, normalize the gradient tensor s.t. it
            sums to 1. Default: ``True``.

    Returns:
        :class:`torch.Tensor`: gradient tensor with the same shape as
        :attr:`pred_y`.
    """
    assert isinstance(pred_y, torch.Tensor)
    assert len(pred_y.shape) == 4 or len(pred_y.shape) == 2
    assert pred_y.shape[0] == 1
    assert isinstance(y, int)
    backward_gradient = torch.zeros_like(pred_y)
    backward_gradient[0, y] = torch.exp(pred_y[0, y])
    if normalize:
        backward_gradient[0, y] /= backward_gradient[0, y].sum()
    return backward_gradient


def get_backward_gradient(pred_y, y):
    r"""
    Returns a gradient tensor that is either equal to :attr:`y` (if y is a
    tensor with the same shape as pred_y) or a one-hot encoding in the channels
    dimension.

    :attr:`y` can be either an ``int``, an array-like list of integers,
    or a tensor. If :attr:`y` is a tensor with the same shape as
    :attr:`pred_y`, the function returns :attr:`y` unchanged.

    Otherwise, :attr:`y` is interpreted as a list of class indices. These
    are first unfolded/expanded to one index per batch element in
    :attr:`pred_y` (i.e. along the first dimension). Then, this list
    is further expanded to all spatial dimensions of :attr:`pred_y`.
    (i.e. all but the first two dimensions of :attr:`pred_y`).
    Finally, the function return a "gradient" tensor that is a one-hot
    indicator tensor for these classes.

    Args:
        pred_y (:class:`torch.Tensor`): model output tensor.
        y (int, :class:`torch.Tensor`, list, or :class:`np.ndarray`): target
            label(s) that can be cast to :class:`torch.long`.

    Returns:
        :class:`torch.Tensor`: gradient tensor with the same shape as
            :attr:`pred_y`.
    """

    assert isinstance(pred_y, torch.Tensor)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long, device=pred_y.device)
    assert isinstance(y, torch.Tensor)

    if y.shape == pred_y.shape:
        return y
    assert y.dtype == torch.long

    nspatial = len(pred_y.shape) - 2
    grad = torch.zeros_like(pred_y)
    y = y.reshape(-1, 1, *((1,) * nspatial)).expand_as(grad)
    grad.scatter_(1, y, 1.)
    return grad


def get_module(model, module):
    r"""Returns a specific layer in a model based.

    :attr:`module` is either the name of a module (as given by the
    :func:`named_modules` function for :class:`torch.nn.Module` objects) or
    a :class:`torch.nn.Module` object. If :attr:`module` is a
    :class:`torch.nn.Module` object, then :attr:`module` is returned unchanged.
    If :attr:`module` is a str, the function searches for a module with the
    name :attr:`module` and returns a :class:`torch.nn.Module` if found;
    otherwise, ``None`` is returned.

    Args:
        model (:class:`torch.nn.Module`): model in which to search for layer.
        module (str or :class:`torch.nn.Module`): name of layer (str) or the
            layer itself (:class:`torch.nn.Module`).

    Returns:
        :class:`torch.nn.Module`: specific PyTorch layer (``None`` if the layer
            isn't found).
    """
    if isinstance(module, torch.nn.Module):
        return module

    assert isinstance(module, str)
    if module == '':
        return model

    for name, curr_module in model.named_modules():
        if name == module:
            return curr_module

    return None


def gradient_to_saliency(x):
    r"""Convert a gradient to a saliency map.

    The tensor :attr:`x` must have a valid gradient ``x.grad``.
    The function then computes the saliency map :math:`s` given by:

    .. math::

        s_{n,1,u} = \max_{0 \leq c < C} |dx_{ncu}|

    where :math:`n` is the instance index, :math:`c` the channel
    index and :math:`u` the spatial multi-index (usually of dimension 2 for
    images).

    Args:
        x (Tensor): activation with gradient.

    Return:
        Tensor: saliency
    """
    return x.grad.abs().max(dim=1, keepdim=True)[0]


def resize_saliency(tensor, saliency, size, mode):
    """Resize a saliency map.

    Args:
        tensor (:class:`torch.Tensor`): reference tensor.
        saliency (:class:`torch.Tensor`): saliency map.
        size (bool or tuple of int): if a tuple (i.e., (width, height),
            resize :attr:`saliency` to :attr:`size`. If True, resize
            :attr:`saliency: to the shape of :attr:`tensor`; otherwise,
            return :attr:`saliency` unchanged.
        mode (str): mode for :func:`torch.nn.functional.interpolate`.

    Returns:
        :class:`torch.Tensor`: Resized saliency map.
    """
    if size is not False:
        if size is True:
            size = tensor.shape[2:]
        elif isinstance(size, tuple) or isinstance(size, list):
            # width, height -> height, width
            size = size[::-1]
        else:
            assert False, "resize must be True, False or a tuple."
        saliency = F.interpolate(
            saliency, size, mode=mode, align_corners=False)
    return saliency


def attach_debug_probes(model, debug=False):
    r"""
    Returns an :class:`collections.OrderedDict` of :class:`Probe` objects for
    all modules in the model if :attr:`debug` is ``True``; otherwise, returns
    ``None``.

    Args:
        model (:class:`torch.nn.Module`): a model.
        debug (bool, optional): if True, return an OrderedDict of Probe objects
            for all modules in the model; otherwise returns ``None``.
            Default: ``False``.

    Returns:
        :class:`collections.OrderedDict`: dict of :class:`Probe` objects for
            all modules in the model.
    """
    if not debug:
        return None

    debug_probes = OrderedDict()
    for module_name, module in model.named_modules():
        debug_probe_target = "input" if module_name == "" else "output"
        debug_probes[module_name] = Probe(
            module, target=debug_probe_target)
    return debug_probes


def saliency(model,
             input,
             target,
             saliency_layer='',
             resize=False,
             resize_mode='bilinear',
             smooth=0,
             context_builder=NullContext,
             gradient_to_saliency=gradient_to_saliency,
             get_backward_gradient=get_backward_gradient,
             debug=False):
    """Apply a backprop-based attribution method to an image.

    The saliency method is specified by a suitable context factory
    :attr:`context_builder`. This context is used to modify the backpropagation
    algorithm to match a given visualization method. This:

    1. Attaches a probe to the output tensor of :attr:`saliency_layer`,
       which must be a layer in :attr:`model`. If no such layer is specified,
       it selects the input tensor to :attr:`model`.

    2. Uses the function :attr:`get_backward_gradient` to obtain a gradient
       for the output tensor of the model. This function is passed
       as input the output tensor as well as the parameter :attr:`target`.
       By default, the :func:`get_backward_gradient` function is used.
       The latter generates as gradient a one-hot vector selecting
       :attr:`target`, usually the index of the class predicted by
       :attr:`model`.

    3. Evaluates :attr:`model` on :attr:`input` and then computes the
       pseudo-gradient of the model with respect the selected tensor. This
       calculation is controlled by :attr:`context_builder`.

    4. Extract the pseudo-gradient at the selected tensor as a raw saliency
       map.

    5. Call :attr:`gradient_to_saliency` to obtain an actual saliency map.
       This defaults to :func:`gradient_to_saliency` that takes the maximum
       absolute value along the channel dimension of the pseudo-gradient
       tensor.

    6. Optionally resizes the saliency map thus obtained. By default,
       this uses bilinear interpolation and resizes the saliency to the same
       spatial dimension of :attr:`input`.

    7. Optionally applies a Gaussian filter to the resized saliency map.
       The standard deviation :attr:`sigma` of this filter is measured
       as a fraction of the maxmum spatial dimension of the resized
       saliency map.

    8. Removes the probe.

    9. Returns the saliency map or optionally a tuple with the saliency map
       and a OrderedDict of Probe objects for all modules in the model, which
       can be used for debugging.

    Args:
        model (:class:`torch.nn.Module`): a model.
        input (:class:`torch.Tensor`): input tensor.
        target (int or :class:`torch.Tensor`): target label(s).
        saliency_layer (str or :class:`torch.nn.Module`, optional): name of the
            saliency layer (str) or the layer itself (:class:`torch.nn.Module`)
            in the model at which to visualize. Default: ``''`` (visualize
            at input).
        resize (bool or tuple, optional): if True, upsample saliency map to the
            same size as :attr:`input`. It is also possible to specify a pair
            (width, height) for a different size. Default: ``False``.
        resize_mode (str, optional): upsampling method to use. Default:
            ``'bilinear'``.
        smooth (float, optional): amount of Gaussian smoothing to apply to the
            saliency map. Default: ``0``.
        context_builder (type, optional): type of context to use. Default:
            :class:`NullContext`.
        gradient_to_saliency (function, optional): function that converts the
            pseudo-gradient signal to a saliency map. Default:
            :func:`gradient_to_saliency`.
        get_backward_gradient (function, optional): function that generates
            gradient tensor to backpropagate. Default:
            :func:`get_backward_gradient`.
        debug (bool, optional): if True, also return an
            :class:`collections.OrderedDict` of :class:`Probe` objects for
            all modules in the model. Default: ``False``.

    Returns:
        :class:`torch.Tensor` or tuple: If :attr:`debug` is False, returns a
        :class:`torch.Tensor` saliency map at :attr:`saliency_layer`.
        Otherwise, returns a tuple of a :class:`torch.Tensor` saliency map
        at :attr:`saliency_layer` and an :class:`collections.OrderedDict`
        of :class:`Probe` objects for all modules in the model.
    """

    # Clear any existing gradient.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradients for model parameters.
    orig_requires_grad = {}
    for name, param in model.named_parameters():
        orig_requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Set model to eval mode.
    if model.training:
        orig_is_training = True
        model.eval()
    else:
        orig_is_training = False

    # Attach debug probes to every module.
    debug_probes = attach_debug_probes(model, debug=debug)

    # Attach a probe to the saliency layer.
    probe_target = 'input' if saliency_layer == '' else 'output'
    saliency_layer = get_module(model, saliency_layer)
    assert saliency_layer is not None, 'We could not find the saliency layer'
    probe = Probe(saliency_layer, target=probe_target)

    # Do a forward and backward pass.
    with context_builder():
        output = model(input)
        backward_gradient = get_backward_gradient(output, target)
        output.backward(backward_gradient)

    # Get saliency map from gradient.
    saliency_map = gradient_to_saliency(probe.data[0])

    # Resize saliency map.
    saliency_map = resize_saliency(input,
                                   saliency_map,
                                   resize,
                                   mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        saliency_map = imsmooth(
            saliency_map,
            sigma=smooth * max(saliency_map.shape[2:]),
            padding_mode='replicate'
        )

    # Remove probe.
    probe.remove()

    # Restore gradient saving for model parameters.
    for name, param in model.named_parameters():
        param.requires_grad_(orig_requires_grad[name])

    # Restore model's original mode.
    if orig_is_training:
        model.train()

    if debug:
        return saliency_map, debug_probes
    else:
        return saliency_map
