from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.benchmark import get_example_data, plot_example
from torchray.utils import get_device

# Obtain example data.
model, x, category_id_1, category_id_2 = get_example_data()

# Run on GPU if available.
device = get_device()
model.to(device)
x = x.to(device)

# Extremal perturbation backprop.
masks_1, _ = extremal_perturbation(
    model, x, category_id_1,
    reward_func=contrastive_reward,
    debug=True,
    areas=[0.12],
)

masks_2, _ = extremal_perturbation(
    model, x, category_id_2,
    reward_func=contrastive_reward,
    debug=True,
    areas=[0.05],
)

# Plots.
plot_example(x, masks_1, 'extremal perturbation', category_id_1)
plot_example(x, masks_2, 'extremal perturbation', category_id_2)
