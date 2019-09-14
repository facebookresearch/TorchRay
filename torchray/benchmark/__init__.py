r"""This script provides a few functions for getting and plotting example data.
"""
import os
import torchvision
from matplotlib import pyplot as plt

from .datasets import *  # noqa
from .models import *  # noqa


def get_example_data(arch='vgg16', shape=224):
    """Get example data to demonstrate visualization techniques.

    Args:
        arch (str, optional): name of torchvision.models architecture.
            Default: ``'vgg16'``.
        shape (int or tuple of int, optional): shape to resize input image to.
            Default: ``224``.

    Returns:
        (:class:`torch.nn.Module`, :class:`torch.Tensor`, int, int): a tuple
        containing

            - a convolutional neural network model in evaluation mode.
            - a sample input tensor image.
            - the ImageNet category id of an object in the image.
            - the ImageNet category id of another object in the image.

    """

    # Get a network pre-trained on ImageNet.
    model = torchvision.models.__dict__[arch](pretrained=True)

    # Switch to eval mode to make the visualization deterministic.
    model.eval()

    # We do not need grads for the parameters.
    for param in model.parameters():
        param.requires_grad_(False)

    # Download an example image from wikimedia.
    import requests
    from io import BytesIO
    from PIL import Image

    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Arthur_Heyer_-_Dog_and_Cats.jpg/592px-Arthur_Heyer_-_Dog_and_Cats.jpg'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Pre-process the image and convert into a tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(shape),
        torchvision.transforms.CenterCrop(shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    x = transform(img).unsqueeze(0)

    # bulldog category id.
    category_id_1 = 245

    # persian cat category id.
    category_id_2 = 285

    # Move model and input to device.
    from torchray.utils import get_device
    dev = get_device()
    model = model.to(dev)
    x = x.to(dev)

    return model, x, category_id_1, category_id_2


def plot_example(input,
                 saliency,
                 method,
                 category_id,
                 show_plot=False,
                 save_path=None):
    """Plot an example.

    Args:
        input (:class:`torch.Tensor`): 4D tensor containing input images.
        saliency (:class:`torch.Tensor`): 4D tensor containing saliency maps.
        method (str): name of saliency method.
        category_id (int): ID of ImageNet category.
        show_plot (bool, optional): If True, show plot. Default: ``False``.
        save_path (str, optional): Path to save figure to. Default: ``None``.
    """
    from torchray.utils import imsc
    from torchray.benchmark.datasets import IMAGENET_CLASSES

    if isinstance(category_id, int):
        category_id = [category_id]

    batch_size = len(input)

    plt.clf()
    for i in range(batch_size):
        class_i = category_id[i % len(category_id)]

        plt.subplot(batch_size, 2, 1 + 2 * i)
        imsc(input[i])
        plt.title('input image', fontsize=8)

        plt.subplot(batch_size, 2, 2 + 2 * i)
        imsc(saliency[i], interpolation='none')
        plt.title('{} for category {} ({})'.format(
            method, IMAGENET_CLASSES[class_i], class_i), fontsize=8)

    # Save figure if path is specified.
    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        # Create directory if necessary.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ext = os.path.splitext(save_path)[1].strip('.')
        plt.savefig(save_path, format=ext, bbox_inches='tight')

    # Show plot if desired.
    if show_plot:
        plt.show()
