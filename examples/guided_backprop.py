from torchray.attribution.guided_backprop import guided_backprop
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Guided backprop.
saliency = guided_backprop(model, x, category_id)

# Plots.
plot_example(x, saliency, 'guided backprop', category_id)
