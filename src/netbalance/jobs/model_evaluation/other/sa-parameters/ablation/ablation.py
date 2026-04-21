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
    RESULTS_DIR, "numeric", "other", f"dataset-{dataset}-sa-parameters-ablation"
)

figs_folder = f"{RESULTS_DIR}/figs/other/sa-parameters"
os.makedirs(figs_folder, exist_ok=True)

seeds = list(range(30))

file_name = os.path.join(save_dir, "beta_entropies.npy")
beta_entropies = np.load(file_name)

file_name = os.path.join(save_dir, "heuristic_entropies.npy")
heuristic_entropies = np.load(file_name)

file_name = os.path.join(save_dir, "both_entropies.npy")
both_entropies = np.load(file_name)

file_name = os.path.join(save_dir, "sa_entropies.npy")
sa_entropies = np.load(file_name)

average_entropies = [
    np.mean(beta_entropies),
    np.mean(heuristic_entropies),
    np.mean(sa_entropies),
    np.mean(both_entropies),
]
std_entropies = [
    np.std(beta_entropies),
    np.std(heuristic_entropies),
    np.std(sa_entropies),
    np.std(both_entropies),
]

fig = plt.figure(figsize=(1.5, 3.0))
ax = fig.add_subplot(111)

x = [0, 1, 2, 3]
bar_width = 0.5

ax.bar(
    x,
    average_entropies,
    bar_width,
    capsize=1.0,
    yerr=std_entropies,
    ecolor="black",
    error_kw=dict(lw=1, alpha=0.7),
    color=("#fdae61", "#4393c3", "#bf812d", "#d53e4f"),
    alpha=1.0,
)

ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
ax.set_ylim(
    0.4,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xticks([])

fig.subplots_adjust(left=0, right=1, top=1, bottom=0.1)

file_name = f"{figs_folder}/ablation.svg"
fig.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
print(f"\nFigure Saved: {file_name}")
