from torchray.attribution.common import Probe, get_module
from torchray.attribution.excitation_backprop import ExcitationBackpropContext
from torchray.attribution.excitation_backprop import gradient_to_excitation_backprop_saliency
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Contrastive excitation backprop.
saliency_layer = get_module(model, 'features.9')
saliency_probe = Probe(saliency_layer, target='output')

with ExcitationBackpropContext():
    y = model(x)
    z = y[0, category_id]
    z.backward()

saliency = gradient_to_excitation_backprop_saliency(saliency_probe.data[0])

saliency_probe.remove()

# Plots.
plot_example(x, saliency, 'excitation backprop', category_id)
