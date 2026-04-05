import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

color1 = "#fdae61"
color2 = "#8c6bb1"
color3 = "#d53e4f"

dataset = "luodti"

save_dir = os.path.join(
    RESULTS_DIR, "numeric", "other", f"dataset-{dataset}-train-pairwise-similarities"
)
rho_file = os.path.join(save_dir, "irho_pairwise_similarities.npy")
beta_file = os.path.join(save_dir, "ibeta_pairwise_similarities.npy")

figs_folder = f"{RESULTS_DIR}/figs/other/sa-parameters/dataset-similarities-{dataset}"
os.makedirs(figs_folder, exist_ok=True)


rho_similarities = np.load(rho_file)
beta_similarities = np.load(beta_file)

np.fill_diagonal(rho_similarities, np.nan)
np.fill_diagonal(beta_similarities, np.nan)

fig1 = plt.figure(figsize=(2.0, 2.0))
fig2 = plt.figure(figsize=(2.0, 2.0))
fig3 = plt.figure(figsize=(0.1, 2.0))
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
cax = fig3.add_subplot(111)

vmin = min(np.nanmin(rho_similarities), np.nanmin(beta_similarities))
vmax = max(np.nanmax(rho_similarities), np.nanmax(beta_similarities))

cmap = LinearSegmentedColormap.from_list(
    "purple_hot",
    [
        "#ceebf9",
        "#92c5de",
        "#4393c3",
        "#8073ac",
    ],
)
cmap.set_bad(color="white")

im1 = ax1.imshow(rho_similarities, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
im2 = ax2.imshow(beta_similarities, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])

fig3.colorbar(sm, cax=cax)

fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig3.subplots_adjust(left=0, right=1, top=1, bottom=0)

file_name = f"{figs_folder}/rho_pairwise_similarities.svg"
fig1.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
print(f"\nFigure Saved: {file_name}")

file_name = f"{figs_folder}/beta_pairwise_similarities.svg"
fig2.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
print(f"\nFigure Saved: {file_name}")

file_name = f"{figs_folder}/pairwise_similarities_colorbar.svg"
fig3.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
print(f"\nFigure Saved: {file_name}")


#####################################
# Boxplots of pairwise similarities #
#####################################

fig4 = plt.figure(figsize=(2.0, 3.0))
ax4 = fig4.add_subplot(111)

# Prepare data for boxplot
rho_flat = rho_similarities[~np.isnan(rho_similarities)]
beta_flat = beta_similarities[~np.isnan(beta_similarities)]

ax4.boxplot([rho_flat, beta_flat], tick_labels=["", ""], patch_artist=True)

ax4.set_xticks([])
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

fig4.tight_layout()
file_name = f"{figs_folder}/pairwise_similarities_boxplot.svg"
fig4.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
print(f"\nFigure Saved: {file_name}")


#############################
# Barplots of Dataset Sizes #
#############################

rho_file = os.path.join(save_dir, "irho_dataset_sizes.npy")
beta_file = os.path.join(save_dir, "ibeta_dataset_sizes.npy")

rho_sizes = np.load(rho_file)
beta_sizes = np.load(beta_file)

fig5 = plt.figure(figsize=(2.0, 3.0))
ax5 = fig5.add_subplot(111)

x = [0, 1]
bar_width = 0.4
average_sizes = [np.mean(rho_sizes), np.mean(beta_sizes)]
std_sizes = [np.std(rho_sizes), np.std(beta_sizes)]

ax5.bar(
    x,
    average_sizes,
    bar_width,
    capsize=1.0,
    yerr=std_sizes,
    ecolor="black",
    error_kw=dict(lw=1, alpha=0.7),
    color=(color3, color1),
    alpha=1.0,
)

ax5.set_xlim(x[0] - 0.5, x[-1] + 0.5)
ax5.set_ylim(1000, max(average_sizes) + 200)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

ax5.set_xticks([])

fig5.tight_layout()

file_name = f"{figs_folder}/dataset_sizes_barplot.svg"
fig5.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
print(f"\nFigure Saved: {file_name}")
