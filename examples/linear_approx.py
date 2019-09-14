from torchray.attribution.linear_approx import linear_approx
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Linear approximation backprop.
saliency = linear_approx(model, x, category_id, saliency_layer='features.29')

# Plots.
plot_example(x, saliency, 'linear approx', category_id)
