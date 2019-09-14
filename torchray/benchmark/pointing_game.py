# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
The :mod:`pointing_game` modules implements the pointing game benchmark.
The basic benchmark is implemented by the :class:`PointingGame` class. However,
for benchmarking purposes it is recommended to use the wrapper class
:class:`PointingGameBenchmark` instead. This class supports *PASCAL VOC 2007
test* and *COCO 2014 val* with the modifications used in [EBP]_, including the
ability to run on their "difficult" subsets as defined in the original paper.

The class can be used as follows:

1. Obtain a dataset (usually COCO or PASCAL VOC) and choose a subset.
2. Initialize an instance of :class:`PointingGameBenchmark`.
3. For each image in the dataset:

   1. For each class in the image:

      1. Run the attribution method, usually resulting in a saliency map for
         class :math:`c`.
      2. Convert the result to a point, usually by finding the maximizer of the
         saliency map.
      3. Use the :func:`PointingGameBenchmark.evaluate` function to run the
         test and accumulate the statistics.
4. Extract the :attr:`PointingGame.hits` and :attr:`PointingGame.misses` or
   ``print`` the instance to display the results.
"""

import torch
from torchvision import datasets as ds

from . import datasets as ads


class PointingGame:
    r"""Pointing game.

    Args:
        num_classes (int): number of classes in the dataset.
        tolerance (int, optional): tolerance (in pixels) of margin around
            ground truth annotation. Default: 15.

    Attributes:
        hits (:class:`torch.Tensor`): :attr:`num_classes`-dimensional vector of
            hits counts.
        misses (:class:`torch.Tensor`): :attr:`num_classes`-dimensional vector
            of misses counts.
    """

    def __init__(self, num_classes, tolerance=15):
        assert isinstance(num_classes, int)
        assert isinstance(tolerance, int)
        self.num_classes = num_classes
        self.tolerance = tolerance
        self.hits = torch.zeros((num_classes,), dtype=torch.float64)
        self.misses = torch.zeros((num_classes,), dtype=torch.float64)

    def evaluate(self, mask, point):
        r"""Evaluate a point prediction.

        The function tests whether the prediction :attr:`point` is within a
        certain tolerance of the object ground-truth region :attr:`mask`
        expressed as a boolean occupancy map.

        Use the :func:`reset` method to clear all counters.

        Args:
            mask (:class:`torch.Tensor`): :math:`\{0,1\}^{H\times W}`.
            point (tuple of ints): predicted point :math:`(u,v)`.

        Returns:
            int: +1 if the point hits the object; otherwise -1.
        """
        # Get an acceptance region around the point. There is a hit whenever
        # the acceptance region collides with the class mask.
        v, u = torch.meshgrid((
            (torch.arange(mask.shape[0],
                          dtype=torch.float32) - point[1])**2,
            (torch.arange(mask.shape[1],
                          dtype=torch.float32) - point[0])**2,
        ))
        accept = (v + u) < self.tolerance**2

        # Test for a hit with the corresponding class.
        hit = (mask & accept).view(-1).any()

        return +1 if hit else -1

    def aggregate(self, hit, class_id):
        """Add pointing result from one example."""
        if hit == 0:
            return
        if hit == 1:
            self.hits[class_id] += 1
        elif hit == -1:
            self.misses[class_id] += 1
        else:
            assert False

    def reset(self):
        """Reset hits and misses."""
        self.hits = torch.zeros_like(self.hits)
        self.misses = torch.zeros_like(self.misses)

    @property
    def class_accuracies(self):
        """
        (:class:`torch.Tensor`): :attr:`num_classes`-dimensional vector
            containing per-class accuracy.
        """
        return self.hits / (self.hits + self.misses).clamp(min=1)

    @property
    def accuracy(self):
        """
        (:class:`torch.Tensor`): mean accuracy, computed by averaging
            :attr:`class_accuracies`.
        """
        return self.class_accuracies.mean()

    def __str__(self):
        class_accuracies = self.class_accuracies
        return '{:4.1f}% ['.format(100 * class_accuracies.mean()) + " ".join([
            '{}:{:4.1f}%'.format(c, 100 * a)
            for c, a in enumerate(class_accuracies)
        ]) + ']'


class PointingGameBenchmark(PointingGame):
    """Pointing game benchmark on standard datasets.

    The pointing game should be initialized with a dataset, set to either:

    * (:class:`torchvision.VOCDetection`) VOC 2007 *test* subset.
    * (:class:`torchvision.CocoDetection`) COCO *val2014* subset.

    Args:
        dataset (:class:`torchvision.VisionDataset`): The dataset.
        tolerance (int): the tolerance for the pointing game. Default: ``15``.
        difficult (bool): whether to use the difficult subset.
            Default: ``False``.
    """

    def __init__(self, dataset, tolerance=15, difficult=False):
        if isinstance(dataset, ds.VOCDetection):
            num_classes = 20
        elif isinstance(dataset, ds.CocoDetection):
            num_classes = 80
        else:
            assert False, 'Only VOCDetection and CocoDetection are supported.'

        super(PointingGameBenchmark, self).__init__(
            num_classes=num_classes, tolerance=tolerance)
        self.dataset = dataset
        self.difficult = difficult

        if difficult:
            def load_flags(name):
                try:
                    import importlib.resources as res
                except ImportError:
                    import importlib_resources as res
                with res.open_text('torchray.benchmark', name) as file:
                    rows = [[x for x in row.split('\t')] for row in file]
                    return {
                        row[0]: [bool(int(x)) for x in row[1:]]
                        for row in rows
                    }
            if isinstance(self.dataset, ds.VOCDetection):
                self.difficult_flags = load_flags(
                    'pointing_game_ebp_voc07_difficult.txt')
            elif isinstance(self.dataset, ds.CocoDetection):
                self.difficult_flags = load_flags(
                    'pointing_game_ebp_coco_difficult.txt')

    def evaluate(self, label, class_id, point):
        """Evaluate an label-class-point triplet.

        Args:
            label (dict): a label in VOC or Coco detection format.
            class_id (int): a class id.
            point (iterable): a point specified as a pair of u, v coordinates.

        Returns:
            int: +1 if the point hits the object, -1 if the point misses the
                object, and 0 if the point is skipped during evaluation.
        """

        # Skip if testing on the EBP difficult subset and the image/class pair
        # is an easy one.
        if self.difficult:
            if isinstance(self.dataset, ds.VOCDetection):
                image_name = label['annotation']['filename'].split('.')[0]

            elif isinstance(self.dataset, ds.CocoDetection):
                image_id = label[0]['image_id']
                image = self.dataset.coco.loadImgs(image_id)[0]
                image_name = image['file_name'].split('.')[0]

            if image_name in self.difficult_flags:
                if not self.difficult_flags[image_name][class_id]:
                    return 0

        # Get the mask for all occurrences of class_id.
        if isinstance(self.dataset, ds.VOCDetection):
            # Skip if all boxes for class_id are PASCAL difficult.
            objs = label['annotation']['object']
            if not isinstance(objs, list):
                objs = [objs]
            objs = [obj for obj in objs if
                    ads.VOC_CLASSES.index(obj['name']) == class_id
                    ]
            if all([bool(int(obj['difficult'])) for obj in objs]):
                return 0
            mask = ads.voc_as_mask(label, class_id)

        elif isinstance(self.dataset, ds.CocoDetection):
            mask = ads.coco_as_mask(self.dataset, label, class_id)

        assert mask is not None
        return super(PointingGameBenchmark, self).evaluate(mask, point)
