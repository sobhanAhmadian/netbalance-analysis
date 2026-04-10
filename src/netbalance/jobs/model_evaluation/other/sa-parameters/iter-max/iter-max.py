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

max_iter = 100000
x = list(range(max_iter + 1))


def get_entropy_track(seed, with_gamma=True):
    if with_gamma:
        entropy_track_path = os.path.join(save_dir, f"rho_entropy_track_{seed}.txt")
    else:
        entropy_track_path = os.path.join(
            save_dir, f"rho_without_gamma_entropy_track_{seed}.txt"
        )
    track = np.loadtxt(entropy_track_path)
    return track


with_gamma_tracks = [get_entropy_track(seed) for seed in [0, 1, 2, 3, 4]]
without_gamma_tracks = [
    get_entropy_track(seed, with_gamma=False) for seed in [0, 1, 2, 3, 4]
]

fig, ax = plt.subplots(1, 1, figsize=(4, 2.3))

for idx, track in enumerate(with_gamma_tracks):
    ax.plot(x, track, color="#9970ab", alpha=0.7, lw=0.4)

for idx, track in enumerate(without_gamma_tracks):
    ax.plot(x, track, color="#4393c3", alpha=0.7, lw=0.4)

ax.set_ylim(0.4, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", rotation=10)

os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/entropy-track.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
