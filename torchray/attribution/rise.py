# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module provides an implementation of the *RISE* method of [RISE]_ for
saliency visualization. This is given by the :func:`rise` function, which
can be used as follows:

.. literalinclude:: ../examples/rise.py
    :language: python
    :linenos:

References:

    .. [RISE] V. Petsiuk, A. Das and K. Saenko
              *RISE: Randomized Input Sampling for Explanation of Black-box
              Models,*
              BMVC 2018,
              `<https://arxiv.org/pdf/1806.07421.pdf>`__.
"""

__all__ = ['rise', 'rise_class']

import numpy as np

import torch
import torch.nn.functional as F
from .common import resize_saliency


def _upsample_reflect(x, size, interpolate_mode="bilinear"):
    r"""Upsample 4D :class:`torch.Tensor` with reflection padding.

    Args:
        x (:class:`torch.Tensor`): 4D tensor to interpolate.
        size (int or list or tuple of ints): target size
        interpolate_mode (str): mode to pass to
            :function:`torch.nn.functional.interpolate` function call
            (default: "bilinear").

    Returns:
        :class:`torch.Tensor`: upsampled tensor.
    """
    # Check and get input size.
    assert len(x.shape) == 4
    orig_size = x.shape[2:]

    # Check target size.
    if not isinstance(size, tuple) and not isinstance(size, list):
        assert isinstance(size, int)
        size = (size, size)
    assert len(size) == 2

    # Ensure upsampling.
    for i, o_s in enumerate(orig_size):
        assert o_s <= size[i]

    # Get size of input cell when interpolated.
    cell_size = [int(np.ceil(s / orig_size[i])) for i, s in enumerate(size)]

    # Get size of interpolated input with padding.
    pad_size = [int(cell_size[i] * (orig_size[i] + 2))
                for i in range(len(orig_size))]

    # Pad input with reflection padding.
    x_padded = F.pad(x, (1, 1, 1, 1), mode="reflect")

    # Interpolated padded input.
    x_up = F.interpolate(x_padded,
                         pad_size,
                         mode=interpolate_mode,
                         align_corners=False)

    # Slice interpolated input to size.
    x_new = x_up[:,
                 :,
                 cell_size[0]:cell_size[0] + size[0],
                 cell_size[1]:cell_size[1] + size[1]]

    return x_new


def rise_class(*args, target, **kwargs):
    r"""Class-specific RISE.

    This function has the all the arguments of :func:`rise` with the following
    additional argument and returns a class-specific saliency map for the
    given :attr:`target` class(es).

    Args:
        target (int, :class:`torch.Tensor`, list, or :class:`np.ndarray`):
            target label(s) that can be cast to :class:`torch.long`.
    """
    saliency = rise(*args, **kwargs)
    assert len(saliency.shape) == 4
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.long, device=saliency.device)
    assert isinstance(target, torch.Tensor)
    assert target.dtype == torch.long
    assert len(target) == len(saliency)

    class_saliency = torch.cat([saliency[i, t].unsqueeze(0).unsqueeze(1)
                                for i, t in enumerate(target)], dim=0)
    output_shape = list(saliency.shape)
    output_shape[1] = 1
    assert list(class_saliency.shape) == output_shape

    return class_saliency


def rise(model,
         input,
         target=None,
         seed=0,
         num_masks=8000,
         num_cells=7,
         filter_masks=None,
         batch_size=32,
         p=0.5,
         resize=False,
         resize_mode='bilinear'):
    r"""RISE.

    Args:
        model (:class:`torch.nn.Module`): a model.
        input (:class:`torch.Tensor`): input tensor.
        seed (int, optional): manual seed used to generate random numbers.
            Default: ``0``.
        num_masks (int, optional): number of RISE random masks to use.
            Default: ``8000``.
        num_cells (int, optional): number of cells for one spatial dimension
            in low-res RISE random mask. Default: ``7``.
        filter_masks (:class:`torch.Tensor`, optional): If given, use the
            provided pre-computed filter masks. Default: ``None``.
        batch_size (int, optional): batch size to use. Default: ``128``.
        p (float, optional): with prob p, a low-res cell is set to 0;
            otherwise, it's 1. Default: ``0.5``.
        resize (bool or tuple of ints, optional): If True, resize saliency map
            to size of :attr:`input`. If False, don't resize. If (width,
            height) tuple, resize to (width, height). Default: ``False``.
        resize_mode (str, optional): If resize is not None, use this mode for
            the resize function. Default: ``'bilinear'``.

    Returns:
        :class:`torch.Tensor`: RISE saliency map.
    """
    with torch.no_grad():
        # Get device of input (i.e., GPU).
        dev = input.device

        # Initialize saliency mask and mask normalization term.
        input_shape = input.shape
        saliency_shape = list(input_shape)

        height = input_shape[2]
        width = input_shape[3]

        out = model(input)
        num_classes = out.shape[1]

        saliency_shape[1] = num_classes
        saliency = torch.zeros(saliency_shape, device=dev)

        # Number of spatial dimensions.
        nsd = len(input.shape) - 2
        assert nsd == 2

        # Spatial size of low-res grid cell.
        cell_size = tuple([int(np.ceil(s / num_cells))
                           for s in input_shape[2:]])

        # Spatial size of upsampled mask with buffer (input size + cell size).
        up_size = tuple([input_shape[2 + i] + cell_size[i]
                         for i in range(nsd)])

        # Save current random number generator state.
        state = torch.get_rng_state()

        # Set seed.
        torch.manual_seed(seed)

        if filter_masks is not None:
            assert len(filter_masks) == num_masks

        num_chunks = (num_masks + batch_size - 1) // batch_size
        for chunk in range(num_chunks):
            # Generate RISE random masks on the fly.
            mask_bs = min(num_masks - batch_size * chunk, batch_size)

            if filter_masks is None:
                # Generate low-res, random binary masks.
                grid = (torch.rand(mask_bs, 1, *((num_cells,) * nsd),
                                   device=dev) < p).float()

                # Upsample low-res masks to input shape + buffer.
                masks_up = _upsample_reflect(grid, up_size)

                # Save final RISE masks with random shift.
                masks = torch.empty(mask_bs, 1, *input_shape[2:], device=dev)
                shift_x = torch.randint(0,
                                        cell_size[0],
                                        (mask_bs,),
                                        device='cpu')
                shift_y = torch.randint(0,
                                        cell_size[1],
                                        (mask_bs,),
                                        device='cpu')
                for i in range(mask_bs):
                    masks[i] = masks_up[i,
                                        :,
                                        shift_x[i]:shift_x[i] + height,
                                        shift_y[i]:shift_y[i] + width]
            else:
                masks = filter_masks[
                    chunk * batch_size:chunk * batch_size + mask_bs]

            # Accumulate saliency mask.
            for i, inp in enumerate(input):
                out = torch.sigmoid(model(inp.unsqueeze(0) * masks))
                if len(out.shape) == 4:
                    # TODO: Consider handling FC outputs more flexibly.
                    assert out.shape[2] == 1
                    assert out.shape[3] == 1
                    out = out[:, :, 0, 0]
                sal = torch.matmul(out.data.transpose(0, 1),
                                   masks.view(mask_bs, height * width))
                sal = sal.view((num_classes, height, width))
                saliency[i] = saliency[i] + sal

        # Normalize saliency mask.
        saliency /= num_masks

        # Restore original random number generator state.
        torch.set_rng_state(state)

        # Resize saliency mask if needed.
        saliency = resize_saliency(input,
                                   saliency,
                                   resize,
                                   mode=resize_mode)
        return saliency
