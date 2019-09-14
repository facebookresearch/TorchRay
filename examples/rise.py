from torchray.attribution.rise import rise
from torchray.benchmark import get_example_data, plot_example
from torchray.utils import get_device

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Run on GPU if available.
device = get_device()
model.to(device)
x = x.to(device)

# RISE method.
saliency = rise(model, x)
saliency = saliency[:, category_id].unsqueeze(0)

# Plots.
plot_example(x, saliency, 'RISE', category_id)
