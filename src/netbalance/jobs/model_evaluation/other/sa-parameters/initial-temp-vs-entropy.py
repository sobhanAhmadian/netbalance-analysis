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
    RESULTS_DIR, "numeric", "other", f"dataset-{dataset}-initial-temp-comparison"
)
os.makedirs(save_dir, exist_ok=True)

figs_folder = f"{RESULTS_DIR}/figs/other/sa-parameters"
os.makedirs(figs_folder, exist_ok=True)

seeds = [0, 1, 2]

initial_temp_list = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
x = np.arange(len(initial_temp_list))

entropies = np.load(os.path.join(save_dir, "initial_temp_entropies.npy"))

mean_entropies = np.mean(entropies, axis=0)
std_entropies = np.std(entropies, axis=0)

# Entropy vs Initial Temperature
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.0))

ax.plot(x, mean_entropies, color="#9970ab", linestyle="-", lw=1.1)
ax.fill_between(
    x,
    mean_entropies - std_entropies,
    mean_entropies + std_entropies,
    color="#9970ab",
    alpha=0.2,
)
ax.set_xticks(x)
ax.set_xticklabels([str(int(temp)) for temp in initial_temp_list])
ax.set_ylim(0.949, 1.01)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

file_name = f"{figs_folder}/initial_temp_vs_entropy.svg"
plt.savefig(file_name, transparent=True)
print(f"\nFigure Saved: {file_name}")
