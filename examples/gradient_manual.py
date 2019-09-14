from torchray.attribution.common import gradient_to_saliency
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Gradient method.

x.requires_grad_(True)
y = model(x)
z = y[0, category_id]
z.backward()

saliency = gradient_to_saliency(x)

# Plots.
plot_example(x, saliency, 'gradient', category_id)
