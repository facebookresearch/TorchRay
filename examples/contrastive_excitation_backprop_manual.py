from torchray.attribution.common import Probe, get_module
from torchray.attribution.excitation_backprop import ExcitationBackpropContext
from torchray.attribution.excitation_backprop import gradient_to_contrastive_excitation_backprop_saliency
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Contrastive excitation backprop.
input_layer = get_module(model, 'features.9')
contrast_layer = get_module(model, 'features.30')
classifier_layer = get_module(model, 'classifier.6')

input_probe = Probe(input_layer, target='output')
contrast_probe = Probe(contrast_layer, target='output')

with ExcitationBackpropContext():
    y = model(x)
    z = y[0, category_id]
    classifier_layer.weight.data.neg_()
    z.backward()

    classifier_layer.weight.data.neg_()

    contrast_probe.contrast = [contrast_probe.data[0].grad]

    y = model(x)
    z = y[0, category_id]
    z.backward()

saliency = gradient_to_contrastive_excitation_backprop_saliency(input_probe.data[0])

input_probe.remove()
contrast_probe.remove()

# Plots.
plot_example(x, saliency, 'contrastive excitation backprop', category_id)
