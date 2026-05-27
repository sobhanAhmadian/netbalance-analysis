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
    RESULTS_DIR, "numeric", "other", f"dataset-{dataset}-cooling-rate-comparison"
)
os.makedirs(save_dir, exist_ok=True)

figs_folder = f"{RESULTS_DIR}/figs/other/sa-parameters"
os.makedirs(figs_folder, exist_ok=True)

seeds = [0, 1, 2]

cooling_rate_list = [0.9, 0.95, 0.99, 0.995, 0.999]
x = np.arange(len(cooling_rate_list))

entropies = np.load(os.path.join(save_dir, "cooling_rate_entropies.npy"))

mean_entropies = np.mean(entropies, axis=0)
std_entropies = np.std(entropies, axis=0)

# Entropy vs Cooling Rate
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.0))

ax.plot(x, mean_entropies, color="#66c2a5d3", linestyle="-", lw=1.1)
ax.fill_between(
    x,
    mean_entropies - std_entropies,
    mean_entropies + std_entropies,
    color="#66c2a5d3",
    alpha=0.2,
)
ax.set_xticks(x)
ax.set_xticklabels([str(cr) for cr in cooling_rate_list])
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_ylim(0.949, 1.01)
ax.set_yticks([])

ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.05)

file_name = f"{figs_folder}/cooling_rate_vs_entropy.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
