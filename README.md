# TorchRay

The *TorchRay* package implements several visualization methods for deep
convolutional neural networks using PyTorch. In this release, TorchRay focuses
on *attribution*, namely the problem of determining which part of the input,
usually an image, is responsible for the value computed by a neural network.

*TorchRay* is research oriented: in addition to implementing well known
techniques form the literature, it provides code for reproducing results that
appear in several papers, in order to support *reproducible research*.

*TorchRay* was initially developed to support the paper:

* *Understanding deep networks via extremal perturbations and smooth masks.*
  Fong, Patrick, Vedaldi.
  Proceedings of the International Conference on Computer Vision (ICCV), 2019.

## Examples

The package contains several usage examples in the
[`examples`](https://github.com/facebookresearch/TorchRay/tree/master/examples)
subdirectory.

Here is a complete example for using GradCAM:

```python
from torchray.attribution.grad_cam import grad_cam
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Grad-CAM backprop.
saliency = grad_cam(model, x, category_id, saliency_layer='features.29')

# Plots.
plot_example(x, saliency, 'grad-cam backprop', category_id)
```

## Requirements

TorchRay requires:

* Python 3.4 or greater
* pytorch 1.1.0 or greater
* matplotlib

For benchmarking, it also requires:

* torchvision 0.3.0 or greater
* pycocotools
* mongodb (suggested)
* pymongod (suggested)

On Linux/macOS, using conda you can install

```bash
while read requirement; do conda install \
-c defaults -c pytorch -c conda-forge --yes $requirement; done <<EOF
pytorch>=1.1.0
pycocotools
torchvision>=0.3.0
mongodb
pymongo
EOF
```

## Installing TorchRay

Using `pip`:

```shell
pip install torchray
```

From source:

```shell
python setup.py install
```

or

```shell
pip install .
```

## Full documentation

The full documentation can be found
[here](https://facebookresearch.github.io/TorchRay).

## Changes

See the [CHANGELOG](CHANGELOG.md).

## Join the TorchRay community

* Website: https://github.com/facebookresearch/TorchRay

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## The team

TorchRay has been primarily developed by Ruth C. Fong and Andrea Vedaldi.

## License

TorchRay is CC-BY-NC licensed, as found in the [LICENSE](LICENSE) file.
