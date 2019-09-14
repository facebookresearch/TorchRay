from torchray.attribution.grad_cam import grad_cam
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# Grad-CAM backprop.
saliency = grad_cam(model, x, category_id, saliency_layer='features.29')

# Plots.
plot_example(x, saliency, 'grad-cam backprop', category_id)
