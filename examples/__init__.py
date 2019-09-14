"""
Define :func:`run_all_examples` to run all examples of saliency methods
(excluding :mod:`examples.standard_suite`).
"""

__all__ = ['run_all_examples']

from matplotlib import pyplot as plt


def run_all_examples():
    """Run all examples."""

    plt.figure()
    import examples.extremal_perturbation
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.deconvnet_manual
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.deconvnet
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.grad_cam_manual
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.grad_cam
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.contrastive_excitation_backprop_manual
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.contrastive_excitation_backprop
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.excitation_backprop_manual
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.excitation_backprop
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.guided_backprop_manual
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.guided_backprop
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.gradient_manual
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.gradient
    plt.draw()
    plt.pause(0.001)

    plt.figure()
    import examples.rise
    plt.draw()
    plt.pause(0.001)
