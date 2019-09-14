from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Contrastive excitation backprop.
saliency = excitation_backprop(
    model,
    x,
    category_id,
    saliency_layer='features.9',
)

# Plots.
plot_example(x, saliency, 'excitation backprop', category_id)
