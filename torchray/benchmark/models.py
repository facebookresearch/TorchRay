# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module allows obtaining standard models for benchmarking attribution
methods. The models can be obtained via the function :func:`get_model`.

The function can edit models slightly to make them compatible with benchmarks.
Optional modifications include

1. Converting a model to fully-convolutional (by replacing linear layers
   with equivalent convolutional layers.)
2. Adding a Global Average Pooling (GAP) layer at the end, so that
   a fully-convolutional model can still work as an image classifier.

For the *pointing game*, we support the VGG16 and ResNet50 models
fine-tuned on the PASCAL VOC 2017 and COCO 2014 classification tasks
from the paper [EBP]_ that introduced this test. These models are converted
from the original Caffe implementation and reproduce the results in [EBP]_.
"""

import os
import copy
import types
import re

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models, transforms


__all__ = ['get_model', 'get_transform', 'replace_module']


base_model_url = 'https://dl.fbaipublicfiles.com/torchray/'

model_urls = {
    'vgg16': {
        'coco': os.path.join(base_model_url, 'vgg16_coco-d82c8150.pth.tar'),
        'voc': os.path.join(base_model_url, 'vgg16_voc-b050e8c3.pth.tar'),
    },
    'resnet50': {
        'coco': os.path.join(base_model_url, 'resnet50_coco-e99468c5.pth.tar'),
        'voc': os.path.join(base_model_url, 'resnet50_voc-9ca920d6.pth.tar'),
    },
}


def _fix_caffe_maxpool(model):
    for module in model.modules():
        if isinstance(module, torch.nn.MaxPool2d):
            module.ceil_mode = True


def _load_caffe_vgg16(model, checkpoint):
    def filt(key, value):
        # Rename some parameters to allow for the dropout layers,
        # which are not in the original checkpointed data.
        remap = {
            'classifier.0.weight': 'classifier.0.weight',
            'classifier.0.bias': 'classifier.0.bias',
            'classifier.2.weight': 'classifier.3.weight',
            'classifier.2.bias': 'classifier.3.bias',
            'classifier.4.weight': 'classifier.6.weight',
            'classifier.4.bias': 'classifier.6.bias',
        }
        key = remap.get(key, key)

        # Reshape the classifier weights.
        if key == 'features.0.weight':
            # BGR -> RGB
            value = value[:, [2, 1, 0], :, :]
        elif 'classifier' in key and 'weight' in key:
            value = value.reshape(value.shape[0], -1)
        return key, value

    checkpoint = {k: v for k, v in [
        filt(k, v) for k, v in checkpoint.items()]}

    model.load_state_dict(checkpoint)
    _fix_caffe_maxpool(model)


def _caffe_vgg16_to_fc(model):
    # Make shallow copy.
    model_ = copy.copy(model)

    # Transform the fully-connected layers into convolutional ones.
    for i, layer in enumerate(model.classifier.children()):
        if isinstance(layer, nn.Linear):
            out_ch, in_ch = layer.weight.shape
            k_size = 1
            if i == 0:
                in_ch = 512
                k_size = 7
            conv = nn.Conv2d(in_ch, out_ch, (k_size, k_size))
            conv.weight.data.copy_(layer.weight.view(conv.weight.shape))
            conv.bias.data.copy_(layer.bias.view(conv.bias.shape))
            model_.classifier[i] = conv

    def forward(self, x):
        # PyTorch uses a 7x7 adaptive pooling layer to feed the first
        # FC layer; here we skip it for fully-conv operation.
        x = self.features(x)
        x = self.classifier(x)
        return x

    model_.forward = types.MethodType(forward, model_)
    return model_


def _load_caffe_resnet50(model, checkpoint, make_bn_positive=False):
    # Patch the torchvision model to match the Caffe definition.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=True)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
                                 ceil_mode=True)
    for i in range(2, 5):
        getattr(model, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(model, 'layer%d' % i)[0].conv2.stride = (1, 1)

    # Patch the checkpoint dict and load it.
    def rename(name):
        name = re.sub(r'bn(\d).(0|1).(.*)', r'bn\1.\3', name)
        name = re.sub(r'downsample.(\d).(0|1).(.*)', r'downsample.\1.\3', name)
        return name

    checkpoint = {rename(k): v for k, v in checkpoint.items()}

    # Convert from BGR to RGB.
    checkpoint['conv1.weight'] = checkpoint['conv1.weight'][:, [2, 1, 0], :, :]

    model.load_state_dict(checkpoint)

    # For EBP: the signs of the linear BN weights should be positive.
    # In practice there is only a tiny fraction of neg weights
    # and this does not seem to affect the results much.
    if make_bn_positive:
        conv = None
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                sign = module.weight.sign()
                module.weight.data *= sign
                module.running_mean.data *= sign
                conv.weight.data *= sign.view(-1, 1, 1, 1)
                if conv.bias is not None:
                    conv.bias.data *= sign
            conv = module

    _fix_caffe_maxpool(model)


def _caffe_resnet50_to_fc(model):
    # Shallow copy.
    model_ = copy.copy(model)

    # Patch the last layer: fc -> conv.
    out_ch, in_ch = model.fc.weight.shape
    conv = nn.Conv2d(in_ch, out_ch, (1, 1))
    conv.weight.data.copy_(model.fc.weight.view(conv.weight.shape))
    conv.bias.data.copy_(model.fc.bias)
    model_.fc = conv

    # Patch average pooling.
    # model_.avgpool = nn.AvgPool2d((7, 7), stride=1, ceil_mode=True)

    def forward(self, x):
        # Same as original, but skip flatten layer.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    model_.forward = types.MethodType(forward, model_)
    return model_


def replace_module(model, module_name, new_module):
    r"""Replace a :class:`torch.nn.Module` with another one in a model.

    Args:
        model (:class:`torch.nn.Module`): model in which to find and replace
            the module with the name :attr:`module_name` with
            :attr:`new_module`.
        module_name (str): path of module to replace in the model as a string,
            with ``'.'`` denoting membership in another module. For example,
            ``'features.11'`` in AlexNet (given by
            :func:`torchvision.models.alexnet.alexnet`) refers to the 11th
            module in the ``'features'`` module, that is, the
            :class:`torch.nn.ReLU` module after the last conv layer in
            AlexNet.
        new_module (:class:`torch.nn.Module`): replacement module.
    """
    return _replace_module(model, module_name.split('.'), new_module)


def _replace_module(curr_module, module_path, new_module):
    r"""Recursive helper function used in :func:`replace_module`.

    Args:
        curr_module (:class:`torch.nn.Module`): current module in which
            to search for the module with the relative path given by
            ``module_path``.
        module_path (list of str): path of module to replace in the model as
            a list, where each element is a member of the previous element's
            module. For example, ``'features.11'`` in AlexNet (given by
            :func:`torchvision.models.alexnet.alexnet`) refers to the 11th
            module in the ``'features'`` module, that is, the
            :class:`torch.nn.ReLU` module after the last conv layer in
            AlexNet.
        new_module (:class:`torch.nn.Module`): replacement module.
    """

    # TODO(ruthfong): Extend support to nn.ModuleList and nn.ModuleDict.
    if isinstance(curr_module, nn.Sequential):
        module_dict = OrderedDict(curr_module.named_children())
        assert module_path[0] in module_dict
        if len(module_path) == 1:
            submodule = new_module
        else:
            submodule = _replace_module(
                module_dict[module_path[0]],
                module_path[1:], new_module)
        if module_dict[module_path[0]] is not submodule:
            module_dict[module_path[0]] = submodule
            curr_module = nn.Sequential(module_dict)
    else:
        assert hasattr(curr_module, module_path[0])
        if len(module_path) == 1:
            submodule = new_module
        else:
            submodule = _replace_module(
                getattr(curr_module, module_path[0]),
                module_path[1:], new_module)
        setattr(curr_module, module_path[0], submodule)

    return curr_module


def get_model(arch='vgg16',
              dataset='voc',
              convert_to_fully_convolutional=False):
    r"""
    Return a reference model for the specified architecture and dataset.

    The model is returned in evaluation mode.

    Args:
        arch (str, optional): name of architecture. If :attr:`dataset`
            contains ``"imagenet"``, all :mod:`torchvision.models`
            architectures are supported; otherwise, only "vgg16" and
            "resnet50" are currently supported). Default: ``'vgg16'``.
        dataset (str, optional): name of dataset, should contain
            ``'imagenet'``, ``'voc'``, or ``'coco'``. Default: ``'voc'``.
        convert_to_fully_convolutional (bool, optional): If True, convert the
            model to be fully convolutional. Default: False.

    Returns:
        :class:`torch.nn.Module`: model.
    """

    # Set number of classes in dataset.
    if 'voc' in dataset:
        num_classes = 20
    elif 'coco' in dataset:
        num_classes = 80
    elif 'imagenet' in dataset:
        num_classes = 1000
    else:
        assert False, 'Unknown dataset {}'.format(dataset)

    # Get/load the model from torchvision.
    model = models.__dict__[arch](pretrained='imagenet' in dataset)

    if arch == 'inception_v3':
        model.aux_logits = False

    if 'imagenet' not in dataset:
        # The torchvision models terminate in a classifier for ImageNet.
        # Replace that classifier if we target a different dataset.
        last_name, last_module = list(model.named_modules())[-1]

        # Construct new last layer.
        assert isinstance(last_module, nn.Linear)
        in_features = last_module.in_features
        bias = last_module.bias is not None
        new_layer_module = nn.Linear(in_features, num_classes, bias=bias)

        # Replace the last layer.
        model = replace_module(model, last_name, new_layer_module)

        # Load the model state dict from url.
        if 'voc' in dataset:
            k = 'voc'
        elif 'coco' in dataset:
            k = 'coco'
        else:
            assert False

        checkpoint = torch.hub.load_state_dict_from_url(model_urls[arch][k])

        # Apply the state dict and patch the torchvision models. the
        if arch == 'vgg16':
            _load_caffe_vgg16(model, checkpoint)
            if convert_to_fully_convolutional:
                model = _caffe_vgg16_to_fc(model)

        elif arch == 'resnet50':
            _load_caffe_resnet50(model, checkpoint)
            if convert_to_fully_convolutional:
                model = _caffe_resnet50_to_fc(model)

        else:
            assert False

    else:
        # We don't know how to convert generic models.
        assert not convert_to_fully_convolutional

    # Set model to evaluation mode.
    model.eval()

    return model


def get_transform(dataset='imagenet', size=224):
    r"""
    Returns a composition of standard pre-processing transformations for
    feeding models. For non-ImageNet datasets, the transforms are
    for models converted from Caffe (i.e., Caffe pre-processing).

    Args:
        dataset (str): name of dataset, should contain either ``'imagenet'``,
            ``'coco'`` or ``'voc'`` (default: ``'imagenet'``).
        size (sequence or int): desired output size (see
            :class:`torchvision.transforms.Resize` for more details).

    Returns:
        :class:`torchvision.Transform`: transform.
    """
    # Get the data loader transforms.
    if "imagenet" in dataset:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    else:
        # Normalization for Caffe networks.
        bgr_mean = [103.939, 116.779, 123.68]

        # TODO(vedaldi): This legacy code will be removed in the future.
        # # This is exactly the same as pycaffe.
        # from skimage.transform import resize
        # import numpy as np
        # def transform(pil_image):
        #     x = np.array(pil_image).astype(np.float)
        #     h, w = x.shape[:2]
        #     if w < h:
        #         ow = size
        #         oh = int(size * h / w)
        #     else:
        #         oh = size
        #         ow = int(size * w / h)
        #     # This seems unnecessary:
        #     # mn, mx = x.min(), x.max()
        #     # x = (x - mn) / (mx - mn)
        #     # x = resize(x, (oh, ow),
        #     #            order=1,
        #     #            mode='constant',
        #     #            anti_aliasing=False)
        #     # x = x.astype(np.float32) * (mx - mn) + mn
        #     # x = (x - mn) / (mx - mn)
        #     x = resize(x, (oh, ow),
        #                order=1,
        #                mode='constant',
        #                anti_aliasing=False)
        #     x = torch.tensor(x, dtype=torch.float32)
        #     x -= torch.tensor(list(reversed(bgr_mean)), dtype=torch.float32)
        #     x = x.permute([2, 0, 1])
        #     return x

        import torch.nn.functional as F
        mean = [m / 255. for m in reversed(bgr_mean)]
        std = [1 / 255.] * 3

        # Note: image should always be downsampled. If image is being
        # upsampled, then this resize function will not match the behavior
        # of skimage.transform.resize in "constant" mode
        # (torch.nn.functional.interpolate uses "edge" padding).
        def resize(x):
            if not isinstance(size, int):
                orig_height, orig_width = size
            else:
                height, width = x.shape[1:3]
                if width < height:
                    orig_width = size
                    orig_height = int(size * height / width)
                else:
                    orig_height = size
                    orig_width = int(size * width / height)
            with torch.no_grad():
                x = F.interpolate(x.unsqueeze(0), (orig_height, orig_width),
                                  mode='bilinear', align_corners=False)
                x = x.squeeze(0)
            return x

        transform = transforms.Compose([
            transforms.ToTensor(),
            resize,
            transforms.Normalize(mean=mean, std=std),
        ])

    return transform
