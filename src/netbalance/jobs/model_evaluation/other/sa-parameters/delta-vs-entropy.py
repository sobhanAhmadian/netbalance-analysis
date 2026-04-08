import os

import matplotlib.pyplot as plt
import numpy as np

from netbalance.configs.common import RESULTS_DIR
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

plt.rcParams.update(
    {
        "font.weight": "normal",  # options: 'normal', 'light', 'regular'
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelweight": "regular",
        "axes.titleweight": "regular",
    }
)

dataset = "luodti"

save_dir = os.path.join(
    RESULTS_DIR, "numeric", "other", f"dataset-{dataset}-delta-comparison"
)
os.makedirs(save_dir, exist_ok=True)

figs_folder = f"{RESULTS_DIR}/figs/other/sa-parameters"
os.makedirs(figs_folder, exist_ok=True)

seeds = [0, 1, 2]

delta_list = [
    0.0,
    0.0001,
    0.0002,
    0.0003,
    0.0004,
    0.0005,
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
]
x = np.arange(len(delta_list))

entropies = np.load(os.path.join(save_dir, "delta_entropies.npy"))
dataset_sizes = np.load(os.path.join(save_dir, "delta_dataset_sizes.npy"))

mean_entropies = np.mean(entropies, axis=0)
mean_dataset_sizes = np.mean(dataset_sizes, axis=0)

std_entropies = np.std(entropies, axis=0)
std_dataset_sizes = np.std(dataset_sizes, axis=0)


# Entropy vs Delta
fig, ax = plt.subplots(1, 1, figsize=(4, 2.3))

ax.plot(x, mean_entropies, color="#bf812d", linestyle="-", lw=1.1)
ax.fill_between(
    x,
    mean_entropies - std_entropies,
    mean_entropies + std_entropies,
    color="#bf812d",
    alpha=0.2,
)
ax.set_xticks(x)
ax.set_xticklabels(delta_list, rotation=90)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
file_name = f"{figs_folder}/delta_vs_entropy.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")

# Dataset Size vs Delta
fig, ax = plt.subplots(1, 1, figsize=(4, 2.3))

ax.plot(x, mean_dataset_sizes, color="#9970ab", linestyle="-", lw=1.1)
ax.fill_between(
    x,
    mean_dataset_sizes - std_dataset_sizes,
    mean_dataset_sizes + std_dataset_sizes,
    color="#9970ab",
    alpha=0.2,
)
ax.set_xticks(x)
ax.set_xticklabels(delta_list, rotation=90)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
file_name = f"{figs_folder}/delta_vs_dataset_sizes.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
