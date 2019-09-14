from torchray.attribution.excitation_backprop import contrastive_excitation_backprop
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Contrastive excitation backprop.
saliency = contrastive_excitation_backprop(
    model,
    x,
    category_id,
    saliency_layer='features.9',
    contrast_layer='features.30',
    classifier_layer='classifier.6',
)

# Plots.
plot_example(x, saliency, 'contrastive excitation backprop', category_id)
