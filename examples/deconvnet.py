from torchray.attribution.deconvnet import deconvnet
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# DeConvNet method.
saliency = deconvnet(model, x, category_id)

# Plots.
plot_example(x, saliency, 'deconvnet', category_id)
