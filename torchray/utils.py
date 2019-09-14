# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""Utility functions."""

import json
import math
import os
from urllib.parse import urlparse
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

EPSILON_DOUBLE = torch.tensor(2.220446049250313e-16, dtype=torch.float64)
EPSILON_SINGLE = torch.tensor(1.19209290E-07, dtype=torch.float32)
SQRT_TWO_DOUBLE = torch.tensor(math.sqrt(2), dtype=torch.float32)
SQRT_TWO_SINGLE = SQRT_TWO_DOUBLE.to(torch.float32)

_DEFAULT_CONFIG = {
    'mongo': {
        'server': 'mongod',
        'hostname': 'localhost',
        'port': 27017,
        'database': './data/db'
    },
    'benchmark': {
        'voc_dir': './data/datasets/voc',
        'coco_dir': './data/datasets/coco',
        'coco_anno_dir': './data/datasets/coco/annotations',
        'imagenet_dir': './data/datasets/imagenet',
        'models_dir': './data/models',
        'experiments_dir': './data'
    }
}

_config_read = False


def get_config():
    """Read the TorchRay config file.

    Read the config file from the current directory or the user's home
    directory and return the configuration.

    Returns:
        dict: configuration.
    """
    global _config_read
    config = _DEFAULT_CONFIG
    if _config_read:
        return config

    def _update(source, delta):
        if isinstance(source, dict):
            assert isinstance(delta, dict)
            for k in source.keys():
                if k in delta:
                    source[k] = _update(source[k], delta[k])
            for k in delta.keys():
                # Catch name errors in config file.
                assert k in source
        else:
            source = delta
        return source

    config = _DEFAULT_CONFIG
    for curr_dir in os.curdir, os.path.expanduser('~'):
        path = os.path.join(curr_dir, '.torchrayrc')
        if os.path.exists(path):
            with open(path, 'r') as file:
                config_ = json.load(file)
                _update(config, config_)
                break

    _config_read = True
    return config


def get_device(gpu=0):
    r"""Get the :class`torch.device` to use; specify device with :attr:`gpu`.

    Args:
        gpu (int, optional): Index of the GPU device; specify ``None`` to
            force CPU. Default: ``0``.

    Returns:
        :class:`torch.device`: device to use.
    """
    device = torch.device(
        f'cuda:{gpu}'
        if torch.cuda.is_available() and gpu is not None
        else 'cpu')
    return device


def xmkdir(path):
    r"""Create a directory path recursively.

    The function creates :attr:`path` if the directory does not exist.

    Args::
        path (str): path to create.
    """
    if path is not None and not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            # Race condition in multi-processing.
            pass


def is_url(obj):
    r"""Check if an object is an URL.

    Args:
        obj (object): object to test.

    Returns:
        bool: ``True`` if :attr:`x` is an URL string; otherwise ``False``.
    """
    try:
        result = urlparse(obj)
        return all([result.scheme, result.netloc, result.path])
    except Exception:
        return False


def tensor_to_im(tensor):
    r"""Reshape a tensor as a grayscale image stack.

    The function reshapes the tensor :attr:`x` of size
    :math:`N\times K\times H\times W`
    to have shape :math:`(NK)\times 1\times H\times W`.

    Args:
        tensor (:class:`torch.Tensor`): tensor to rearrange.

    Returns:
        :class:`torch.Tensor`: Reshaped tensor.
    """
    return tensor.reshape(-1, *tensor.shape[2:])[:, None, :, :]


def pil_to_tensor(pil_image):
    r"""Convert a PIL image to a tensor.

    Args:
        pil_image (:class:`PIL.Image`): PIL image.

    Returns:
        :class:`torch.Tensor`: the image as a :math:`3\times H\times W` tensor
        in the [0, 1] range.
    """
    pil_image = np.array(pil_image)
    if len(pil_image.shape) == 2:
        pil_image = pil_image[:, :, None]
    return torch.tensor(pil_image, dtype=torch.float32).permute(2, 0, 1) / 255


def im_to_numpy(tensor):
    r"""Convert a tensor image to a NumPy image.

    The function converts the :math:`K\times H\times W` tensor :attr:`tensor`
    to a corresponding :math:`H\times W\times K` NumPy array.

    Args:
        tensor (:class:`torch.Tensor`): input tensor.

    Returns:
        :class:`numpy.ndarray`: NumPy array.
    """
    tensor_reshaped = tensor.expand(3, *tensor.shape[1:]).permute(1, 2, 0)
    return tensor_reshaped.detach().cpu().numpy()


def imread(file, as_pil=False, resize=None, to_rgb=False):
    r"""
    Read an image as a tensor.

    The function reads the image :attr:`file` as a PyTorch tensor.
    `file` can also be an URL.

    To reshape the image use the option :attr:`reshape`, passing the desired
    shape ``(W, H)`` as tuple. Passing an integer sets the shortest side to
    that length while preserving the aspect ratio.

    Args:
        file (str): Path or ULR to the image.
        resize (float, int, tuple or list): Resize the image to this size.
        as_pil (bool): If ``True``, returns the PIL image instead of converting
            to a tensor.
        to_rgb (optional, bool): If `True`, convert the PIL image to RGB.
            Default: ``False``.

    Returns:
        :class:`torch.Tensor`:
            The image read as a :math:`3\times H\times W` tensor in
            the [0, 1] range.
    """
    # Read an example image as a numpy array.
    if is_url(file):
        hdr = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 '
                          '(KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.'
                          '11',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*'
                      '/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive'
        }
        req = urllib.request.Request(file, headers=hdr)
        file = urllib.request.urlopen(req)

    img = Image.open(file)
    if to_rgb:
        img = img.convert('RGB')
    if resize is not None:
        if not isinstance(resize, tuple) and not isinstance(resize, list):
            scale = float(resize) / float(min(img.size[0], img.size[1]))
            resize = [round(scale * h) for h in img.size]
        if resize != img.size:
            img = img.resize(resize, Image.ANTIALIAS)
    if as_pil:
        return img
    return pil_to_tensor(img)


def imsc(img, *args, quiet=False, lim=None, interpolation='lanczos', **kwargs):
    r"""Rescale and displays an image represented as a img.

    The function scales the img :attr:`im` to the [0 ,1] range.
    The img is assumed to have shape :math:`3\times H\times W` (RGB)
    :math:`1\times H\times W` (grayscale).

    Args:
        img (:class:`torch.Tensor` or :class:`PIL.Image`): image.
        quiet (bool, optional): if False, do not display image.
            Default: ``False``.
        lim (list, optional): maximum and minimum intensity value for
            rescaling. Default: ``None``.
        interpolation (str, optional): The interpolation mode to use with
            :func:`matplotlib.pyplot.imshow` (e.g. ``'lanczos'`` or
            ``'nearest'``). Default: ``'lanczos'``.

    Returns:
        :class:`torch.Tensor`: Rescaled image img.
    """
    if isinstance(img, Image.Image):
        img = pil_to_tensor(img)
    handle = None
    with torch.no_grad():
        if not lim:
            lim = [img.min(), img.max()]
        img = img - lim[0]  # also makes a copy
        img.mul_(1 / (lim[1] - lim[0]))
        img = torch.clamp(img, min=0, max=1)
        if not quiet:
            bitmap = img.expand(3,
                                *img.shape[1:]).permute(1, 2, 0).cpu().numpy()
            handle = plt.imshow(
                bitmap, *args, interpolation=interpolation, **kwargs)
            curr_ax = plt.gca()
            curr_ax.axis('off')
    return img, handle


def resample(source, target_size, transform):
    r"""Spatially resample a tensor.

    The function resamples the :attr:`source` tensor generating a
    :attr:`target` tensors of size :attr:`target_size`. Resampling uses the
    transform :attr:`transform`, specified as a :math:`2\times 2` matrix in the
    form

    .. math::
        \begin{bmatrix}
        s_u & t_u\\
        s_v & t_v
        \end{bmatrix}

    where :math:`s_u` is the scaling factor along the horizontal spatial
    direction, :math:`t_u` the horizontal offset, and :math:`s_v, t_v` the
    corresponding quantity for the vertical direction.

    Internally, the function uses :func:`torch.nn.functional.grid_sample` with
    bilinear interpolation and zero padding.

    The transformation defines the forward
    mapping, so that a pixel :math:`(u,v)` in the source tensro is mapped to
    pixel :math:`u' = s_u  u + t_u, v' = s_v v + tv`.

    The reference frames are defined as follows. Pixels are unit squares, so
    that a :math:`H\times W` tensor maps to the rectangle :math:`[0, W) \times
    [0, H)`. Hence element :math:`x_{ncij}` of a tensor :math:`x` maps
    to a unit square whose center is :math:`(u,v) = (i + 1/2, j+1/2)`.

    Example:
        In order to stretch an :math:`H \times W` source tensor to a target
        :math:`H' \times W'` tensor, one would use the transformation matrix

        .. math::
            \begin{bmatrix}
            W'/W & 0\\
            H'/H & 0\\
            \end{bmatrix}

    Args:
        source (:class:`torch.Tensor`): :math:`N\times C\times H\times W`
            tensor.
        target_size (tuple of int): target size.
        transform (:class:`torch.Tensor`): :math:`2\times 2` transformation
            tensor.

    Returns:
        :class:`torch.Tensor`: resampled tensor.

    """
    dtype = source.dtype
    dev = source.device

    height_, width_ = target_size
    ur_ = torch.arange(width_, dtype=dtype, device=dev) + 0.5
    vr_ = torch.arange(height_, dtype=dtype, device=dev) + 0.5

    height, weight = source.shape[2:]
    ur = 2 * ((ur_ + transform[0, 1]) / transform[0, 0]) / weight - 1
    vr = 2 * ((vr_ + transform[1, 1]) / transform[1, 0]) / height - 1

    v, u = torch.meshgrid(vr, ur)
    v = v.unsqueeze(2)
    u = u.unsqueeze(2)

    grid = torch.cat((u, v), dim=2)
    grid = grid.unsqueeze(0).expand(len(source), -1, -1, -1)

    return torch.nn.functional.grid_sample(source, grid)


def imsmooth(tensor,
             sigma,
             stride=1,
             padding=0,
             padding_mode='constant',
             padding_value=0):
    r"""Apply a 2D Gaussian filter to a tensor.

    The 2D filter itself is implementing by separating the 2D convolution in
    two 1D convolutions, first along the vertical direction and then along
    the horizontal one. Each 1D Gaussian kernel is given by:

    .. math::
        f_i \propto \exp\left(-\frac{1}{2} \frac{i^2}{\sigma^2} \right),
            ~~~ i \in \{-W,\dots,W\},
            ~~~ W = \lceil 4\sigma \rceil.

    This kernel is normalized to sum to one exactly. Given the latter, the
    function calls `torch.nn.functional.conv2d`
    to perform the actual convolution. Various padding parameters and the
    stride are passed to the latter.

    Args:
        tensor (:class:`torch.Tensor`): :math:`N\times C\times H\times W`
            image tensor.
        sigma (float): standard deviation of the Gaussian kernel.
        stride (int, optional): subsampling factor. Default: ``1``.
        padding (int, optional): extra padding. Default: ``0``.
        padding_mode (str, optional): ``'constant'``, ``'reflect'`` or
            ``'replicate'``. Default: ``'constant'``.
        padding_value (float, optional): constant value for the `constant`
            padding mode. Default: ``0``.

    Returns:
        :class:`torch.Tensor`: :math:`N\times C\times H\times W` tensor with
        the smoothed images.
    """
    assert sigma >= 0
    width = math.ceil(4 * sigma)
    filt = (torch.arange(-width,
                         width + 1,
                         dtype=torch.float32,
                         device=tensor.device) /
            (SQRT_TWO_SINGLE * sigma + EPSILON_SINGLE))
    filt = torch.exp(-filt * filt)
    filt /= torch.sum(filt)
    num_channels = tensor.shape[1]
    width = width + padding
    if padding_mode == 'constant' and padding_value == 0:
        other_padding = width
        x = tensor
    else:
        # pad: (before, after) pairs starting from last dimension backward
        x = F.pad(tensor,
                  (width, width, width, width),
                  mode=padding_mode,
                  value=padding_value)
        other_padding = 0
        padding = 0
    x = F.conv2d(x,
                 filt.reshape((1, 1, -1, 1)).expand(num_channels, -1, -1, -1),
                 padding=(other_padding, padding),
                 stride=(stride, 1),
                 groups=num_channels)
    x = F.conv2d(x,
                 filt.reshape((1, 1, 1, -1)).expand(num_channels, -1, -1, -1),
                 padding=(padding, other_padding),
                 stride=(1, stride),
                 groups=num_channels)
    return x


def imarraysc(tiles,
              spacing=0,
              quiet=False,
              lim=None,
              interpolation='lanczos'):
    r"""Display or arrange an image or tensor batch as a mosaic.

    The function displays the tensor `tiles` as a set of tiles. `tiles` has
    shape :math:`K\times C\times H\times W` and the generated mosaic
    is a *new* tensor with shape :math:`C\times (MH) \times (NW)` where
    :math:`MN \geq K`.

    Missing tiles are filled with zeros.

    The range of each tile is individually scaled to the range [0, 1].

    Args:
        tiles (:class:`torch.Tensor`): tensor to display or rearrange.
        spacing (int, optional): thickness of the border (infilled
            with zeros) around each tile. Default: ``0``.
        quiet (bool, optional): If False, do not display the mosaic.
            Default: ``False``.
        lim (list, optional): maximum and minimum intensity value for
            rescaling. Default: ``None``.
        interpolation (str, optional): interpolation to use with
            :func:`matplotlib.pyplot.imshow`. Default: ``'lanczos'``.

    Returns:
        class:`torch.Tensor`: The rearranged tensor.
    """
    num = tiles.shape[0]
    num_cols = math.ceil(math.sqrt(num))
    num_rows = (num + num_cols - 1) // num_cols
    num_channels = tiles.shape[1]
    height = tiles.shape[2]
    width = tiles.shape[3]
    mosaic = torch.zeros(num_channels,
                         height * num_rows + spacing * (num_rows - 1),
                         width * num_cols + spacing * (num_cols - 1))
    for t in range(num):
        u = t % num_cols
        v = t // num_cols
        mosaic[0:num_channels,
               v*(height+spacing):v*(height+spacing)+height,
               u*(width+spacing):u*(width+spacing)+width] = imsc(tiles[t],
                                                                 quiet=True,
                                                                 lim=lim)[0]
    return imsc(mosaic, quiet=quiet, interpolation=interpolation)
