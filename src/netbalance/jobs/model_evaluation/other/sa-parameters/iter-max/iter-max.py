import os

import matplotlib.pyplot as plt
import numpy as np

from netbalance.configs.common import RESULTS_DIR
from netbalance.utils import prj_logger
from matplotlib.ticker import FuncFormatter

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

colors = ["#4393c3", "#d6604d", "#bf812d", "#66c2a5d3", "#9970ab"]

figs_folder = f"{RESULTS_DIR}/figs/other/sa-parameters"
os.makedirs(figs_folder, exist_ok=True)

save_dir = os.path.join(RESULTS_DIR, "numeric", "other", f"sa-parameters")
os.makedirs(save_dir, exist_ok=True)

max_iter = 60000
x = list(range(max_iter + 1))


def get_entropy_track(seed, with_gamma=True):
    if with_gamma:
        entropy_track_path = os.path.join(save_dir, f"rho_entropy_track_{seed}.txt")
    else:
        entropy_track_path = os.path.join(
            save_dir, f"rho_without_gamma_entropy_track_{seed}.txt"
        )
    track = np.loadtxt(entropy_track_path)[: max_iter + 1]
    return track


with_gamma_tracks = [get_entropy_track(seed) for seed in [0, 1, 2, 3, 4]]
mean_with_gamma = np.mean(with_gamma_tracks, axis=0)
std_with_gamma = np.std(with_gamma_tracks, axis=0)

without_gamma_tracks = [
    get_entropy_track(seed, with_gamma=False) for seed in [0, 1, 2, 3, 4]
]
mean_without_gamma = np.mean(without_gamma_tracks, axis=0)
std_without_gamma = np.std(without_gamma_tracks, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.0))

ax.plot(x, mean_with_gamma, color="#d53e4f", lw=1.2)
ax.fill_between(
    x,
    mean_with_gamma - std_with_gamma,
    mean_with_gamma + std_with_gamma,
    color="#d53e4f",
    alpha=0.2,
)

ax.plot(x, mean_without_gamma, color="#bf812d", lw=1.2)
ax.fill_between(
    x,
    mean_without_gamma - std_without_gamma,
    mean_without_gamma + std_without_gamma,
    color="#bf812d",
    alpha=0.2,
)

ax.set_ylim(0.35, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([0, 20000, 40000, 60000])
ax.set_xticklabels(["0", "20K", "40K", "60K"])

os.makedirs(figs_folder, exist_ok=True)

fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

file_name = f"{figs_folder}/entropy-track.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
