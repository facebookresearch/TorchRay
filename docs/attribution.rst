.. _backprop:

Attribution
===========

*Attribution* is the problem of determining which part of the input,
e.g. an image, is responsible for the value computed by a predictor
such as a neural network.

Formally, let :math:`\mathbf{x}` be the input to a convolutional neural
network, e.g., a :math:`N \times C \times H \times W` real tensor. The neural
network is a function :math:`\Phi` mapping :math:`\mathbf{x}` to a scalar
output :math:`z \in \mathbb{R}`. Thus the goal is to find which of the
elements of :math:`\mathbf{x}` are "most responsible" for the outcome
:math:`z`.

Some attribution methods are "black box" approaches, in the sense that they
ignore the nature of the function :math:`\Phi` (however, most assume that it is
at least possible to compute the gradient of :math:`\Phi` efficiently). Most
attribution methods, however, are "white box" approaches, in the sense that
they exploit the knowledge of the structure of :math:`\Phi`.

:ref:`Backpropagation methods <backpropagation>` are "white box" visualization
approaches that build on backpropagation, thus leveraging the functionality
already implemented in standard deep learning packages toolboxes such as
PyTorch.

:ref:`Perturbation methods <perturbation>` are "black box" visualization
approaches that generate attribution visualizations by perturbing the input
and observing the changes in a model's output.

TorchRay implements the following methods:

* Backpropagation methods

  * Deconvolution (:mod:`.deconvnet`)
  * Excitation backpropagation (:mod:`.excitation_backprop`)
  * Gradient [1]_ (:mod:`.gradient`)
  * Grad-CAM (:mod:`.grad_cam`)
  * Guided backpropagation (:mod:`.guided_backprop`)
  * Linear approximation (:mod:`.linear_approx`)

* Perturbation methods

  * Extremal perturbation [1]_ (:mod:`.extremal_perturbation`)
  * RISE (:mod:`.rise`)

.. rubric:: Footnotes

.. [1] The :mod:`.gradient` and :mod:`.extremal_perturbation` methods actually
       straddle the boundaries between white and black box methods, as they
       only require the ability to compute the gradient of the predictor,
       which does not necessarily require to know the predictor internals.
       However, in TorchRay both are implemented using backpropagation.

.. _backpropagation:

Backpropagation methods
-----------------------

Backpropagation methods work by tweaking the backpropagation algorithm that, on
its own, computes the gradient of tensor functions. Formally, a neural network
:math:`\Phi` is a collection :math:`\Phi_1,\dots,\Phi_n` of :math:`n` layers.
Each layer is in itself a "smaller" function inputting and outputting tensors,
called *activations* (for simplicity, we call activations the network input and
parameter tensors as well). Layers are interconnected in a *Directed Acyclic
Graph* (DAG). The DAG is bipartite with some nodes representing the activation
tensors and the other nodes representing the layers, with interconnections
between layers and input/output tensors in the obvious way. The DAG sources are
the network's input and parameter tensors and the DAG sinks are the network's
output tensors.

The main goal of a deep neural network toolbox such as PyTorch is to evaluate
the function :math:`\Phi` implemented by the DAG as well as its gradients with
respect to various tensors (usually the model parameters). The calculation of
the gradients, which uses backpropagation, associates to the forward DAG a
backward DAG, obtained as follows:

*   Activation tensors :math:`\mathbf{x}_j` become gradient tensors
    :math:`d\mathbf{x}_j` (preserving their shape).
*   Forward layers :math:`\Phi_i` become backward layers :math:`\Phi_i^*`.
*   All arrows are reversed.
*   Additional arrows connecting the activation tensors :math:`\mathbf{x}_i`
    as inputs to the corresponding backward function :math:`\Phi_i^*` are added
    as well.

Backpropagation methods modify the backward graph in order to generate a
visualization of the network forward pass. Additionally, inputs as well as
intermediate activations can be inspected to obtain different visualizations.
These two concepts are explained next.

Changing the backward propagation rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Changing the backward propagation rules amounts to redefining the functions
:math:`\Phi_i^*`. After doing so, the "gradients" computed by backpropagation
change their meaning into something useful for visualization. We call these
modified gradients *pseudo-gradients*.

TorchRay provides a number of context managers that enable patching PyTorch
functions on the fly in order to change the backward propagation rules for
a segment of code. For example, let ``x`` be an input tensor and ``model``
a deep classification network.  Furthermore, let ``category_id`` be the
index of the class for which we want to attribute input regions. The following
code uses :mod:`.guided_backprop` to compute and store the pseudo gradient in
``x.grad``.

.. code-block:: python

      from torchray.attribution.guided_backprop import GuidedBackpropContext

      x.requires_grad_(True)

      with GuidedBackpropContext():
            y = model(x)
            z = y[0, category_id]
            z.backward()

At this point, ``x.grad`` contains the "guided gradient" computed by this
method. This gradient is usually flattened along the channel dimension to
produce a saliency map for visualization:

.. code-block:: python

      from torchray.attribution.common import gradient_to_saliency

      saliency = gradient_to_saliency(x)

TorchRay contains also some wrapper code, such as
:func:`.guided_backprop.guided_backprop`, that combine these steps in a way
that would work for common networks.

Probing intermediate activations and gradients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most visualization methods are based on inspecting the activations when the
network is evaluated and the pseudo-gradients during backpropagation. This is
generally easy for input tensors. For intermediate tensors, when using PyTorch
functional interface, this is also easy: simply use ``retain_grad_(True)`` in
order to retain the gradient of an intermediate tensor:

.. code-block:: python

      from torch.nn.functional import relu, conv2d
      from torchray.attribution import GuidedBackpropContext

      with GuidedBackpropContext():
            y = conv2d(x, weight)
            y.requires_grad_(True)
            y.retain_grad_(True)
            z = relu(y)[0, class_index]
            z.backward()

      # Now y and y.grad contain the activation and guided gradient,
      # respectively.

However, in PyTorch most network components are implemented as
:class:`torch.nn.Module` objects. In this case, is not obvious how to access a
specific layer's information. In order to simplify this process, the library
provides the :class:`Probe` class:

.. code-block:: python

      from torch.nn.functional import relu, conv2d
      from torchray.attribution.guided_backprop import GuidedBackpropContext
      import torchray.attribution.Probe

      # Attach a probe to the last conv layer.
      probe = Probe(alexnet.features[11])

      with GuidedBackpropContext():
            y = alexnet(x)
            z = y[0, class_index]
            z.backward()

      # Now probe.data[0] and probe.data[0].grad contain
      # the activations and guided gradients.

The probe automatically applies :func:`torch.Tensor.requires_grad_` and
:func:`torch.Tensor.retain_grad_` as needed. You can use ``probe.remove()`` to
remove the probe from the network once you are done.

Limitations
^^^^^^^^^^^

Except for the gradient method, backpropagation methods require modifying
the backward function of each layer. TorchRay implements the rules
necessary to do so as originally defined by each authors' method.
However, as new neural network layers are introduced, it is possible
that the default behavior, which is to not change backpropagation, may
be inappropriate or suboptimal for them.

.. _perturbation:

Perturbation methods
--------------------

Perturbation methods work by changing the input to the neural network in a
controlled manner, observing the outcome on the output generated by the
network. Attribution can be achieved by occluding (setting to zero) specific
parts of the image and checking whether this has a strong effect on the output.
This can be thought of as a form of sensitivity analysis which is still
specific to a given input, but is not differential as for the gradient method.


DeConvNet
---------

.. automodule:: torchray.attribution.deconvnet
    :members:
    :show-inheritance:


Excitation backprop
-------------------

.. automodule:: torchray.attribution.excitation_backprop
    :members:
    :show-inheritance:

Extremal perturbation
---------------------

.. automodule:: torchray.attribution.extremal_perturbation
    :members:
    :show-inheritance:

Gradient
--------

.. automodule:: torchray.attribution.gradient
      :members:
      :show-inheritance:

Grad-CAM
--------

.. automodule:: torchray.attribution.grad_cam
      :members:
      :show-inheritance:

Guided backprop
---------------

.. automodule:: torchray.attribution.guided_backprop
      :members:
      :show-inheritance:

Linear approximation
--------------------

.. automodule:: torchray.attribution.linear_approx
      :members:
      :show-inheritance:

RISE
----

.. automodule:: torchray.attribution.rise
      :members:
      :show-inheritance:

Common code
-----------

.. automodule:: torchray.attribution.common
      :members:
      :show-inheritance: