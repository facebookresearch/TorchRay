from torchray.attribution.gradient import gradient
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Gradient method.
saliency = gradient(model, x, category_id)

# Plots.
plot_example(x, saliency, 'gradient', category_id)
