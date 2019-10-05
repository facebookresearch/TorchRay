.. _benchmark:

Benchmarking
============

This module contains code for benchmarking attribution methods, including
reproducing several published results. In addition to implementations of
benchmarking protocols (:mod:`.pointing_game`), the module also provides
implementations of *reference datasets* and *reference models* used in prior
research work, properly converted to PyTorch. Overall, this implementations
closely reproduces prior results, notably the ones in the [EBP]_ paper.

A standard benchmarking suite is included in this library as
:mod:`examples.standard_suite`. For slow methods, a computer cluster may be
required for evaluation (we do not include explicit support for clusters, but
it is easy to add on top of this example code).

It is also recommended to turn on logging (see
:mod:`torchray.benchmark.logging`), which allows the driver to
uses MongoDB to store partial benchmarking results as it goes.
Computations can then be cached and reused to resume the calculations
after a crash or other issue. In order to start the logging server, use

.. code:: shell

      $ python -m torchray.benchmark.server

The server parameters (address, port, etc) can be configured by writing
a ``.torchrayrc`` file in your current or home directory. The package
contains an example configuration file. The server creates a regular
MongoDB database (by default in ``./data/db``) which can be manually
explored by means of the MongoDB shell.

By default, the driver writes data in the ``./data/`` subfolder.
You can change that via the configuration file, or, possibly more easily,
add a symbolic link to where you want to store the data.

The data include the *datasets* (PASCAL VOC, COCO, ImageNet; see
:mod:`torchray.benchmark.datasets`).  These must be downloaded manually and
stored in ``./data/datasets/{voc,coco,imagenet}`` unless this is changed via
the configuration file. Note that these datasets can be very large (many GBs).

The data also include *reference models* (see
:mod:`torchray.benchmark.models`).

.. automodule:: torchray.benchmark
      :members:
      :show-inheritance:

Pointing Game
-------------

The *Pointing Game* [EBP]_ assesses the quality of an attribution method by
testing how well it can extract from a predictor a response correlated with the
presence of known object categories in the image.

Given an input image :math:`x` containing an object of category :math:`c`, the
attribution method is applied to the predictor in order to find the part of the
images responsible for predicting :math:`c`. The attribution method usually
returns a saliency heatmap. The latter must then be converted in a single point
:math:`(u,v)` that is "most likely" to be contained by an object of that class.
The specific way the point is obtained is method-dependent.

The attribution method then scores a hit if the point is within a *tolerance*
:math:`\tau` (set to 15 pixels by default) to the image region :math:`\Omega`
containing that object:

    .. math::
        \operatorname{hit}(u,v|\Omega)
        = [ \exists (u',v') \in \Omega : \|(u,v) - (u',v')\| \leq \tau].

The point coordinates :math:`(u,v)` are also indices :math:`x_{ncvu}` in the
input image tensor :math:`x`.

RISE [RISE]_ and Extremal Perturbation [EP]_ results are averaged over 3 runs.

.. csv-table:: Pointing game results
   :widths: auto
   :header-rows: 2
   :stub-columns: 1
   :file: pointing.csv


.. automodule:: torchray.benchmark.pointing_game
      :members:
      :show-inheritance:

Datasets
--------

.. automodule:: torchray.benchmark.datasets
      :members:
      :show-inheritance:

      .. autodata:: IMAGENET_CLASSES
         :annotation:

      .. autodata:: VOC_CLASSES
         :annotation:

      .. autodata:: COCO_CLASSES
         :annotation:

Reference models
----------------

.. automodule:: torchray.benchmark.models
      :members:
      :show-inheritance:

Logging with MongoDB
--------------------

.. automodule:: torchray.benchmark.logging
        :members:
        :show-inheritance:

