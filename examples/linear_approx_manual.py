from torchray.attribution.common import Probe, get_module
from torchray.attribution.linear_approx import gradient_to_linear_approx_saliency
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Linear approximation.
saliency_layer = get_module(model, 'features.29')

probe = Probe(saliency_layer, target='output')

y = model(x)
z = y[0, category_id]
z.backward()

saliency = gradient_to_linear_approx_saliency(probe.data[0])

# Plots.
plot_example(x, saliency, 'linear approx', category_id)
