import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from netbalance.configs.bmlpphi import BMLPPHI_RESULTS_DIR as RESULTS_DIR
from netbalance.configs.ecoli import ECOLI_PROCESSED_DATA_DIR
from netbalance.features.ecoli import EcoliDataset as Dataset
from netbalance.utils import prj_logger

plt.rcParams.update(
    {
        "font.weight": "normal",  # options: 'normal', 'light', 'regular'
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelweight": "regular",
        "axes.titleweight": "regular",
    }
)

logger = prj_logger.getLogger(__name__)

dataset = "ecoli"

save_dir = os.path.join(
    RESULTS_DIR,
    "numeric",
    "other",
    "feature_importance",
    f"dataset-{dataset}",
    "train_neg_samp-irho",
)
os.makedirs(save_dir, exist_ok=True)

figs_folder = f"{RESULTS_DIR}/figs/other/feature_importance/ecoli/bmlpphi/irho"
os.makedirs(figs_folder, exist_ok=True)

ds = Dataset()
feature_names = ds.get_strain_feature_names() + ds.get_phage_feature_names()

mean_abs_shap = np.load(os.path.join(save_dir, "mean-abs-shap-values.npy"))

strain_file = Path(f"{ECOLI_PROCESSED_DATA_DIR}/strains-features.csv")
phage_file = Path(f"{ECOLI_PROCESSED_DATA_DIR}/phages-features.csv")

strain_features = np.loadtxt(strain_file, delimiter=",").astype(np.float32).mean(axis=0)
phage_features = np.loadtxt(phage_file, delimiter=",").astype(np.float32).mean(axis=0)

feature_abundance = np.concatenate([strain_features, phage_features])

num_strain_features = len(ds.get_strain_feature_names())
num_phage_features = len(ds.get_phage_feature_names())

print(
    np.percentile(mean_abs_shap, [99.75, 99.50, 99.25])
)

min_shap_value = 0.02110365
top_mask = mean_abs_shap >= min_shap_value

top_idx = np.where(top_mask)[0]
strain_top = top_idx[top_idx < num_strain_features]
phage_top = top_idx[top_idx >= num_strain_features]

STRAIN_COLOR = "#4393c3"
PHAGE_COLOR = "#d6604d"

# Layout: 3x3 grid — top-left=top hist, top-right=empty, bottom-left=scatter, bottom-right=right hist
fig = plt.figure(figsize=(4, 4))
gs = gridspec.GridSpec(
    2,
    2,
    width_ratios=[3, 1],
    height_ratios=[1, 3],
    hspace=0.05,
    wspace=0.05,
)

ax_scatter = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
ax_corner = fig.add_subplot(gs[0, 1])
ax_corner.set_visible(False)

hist_bins_x = 30
hist_bins_y = 30

# --- Top histogram: feature abundance of top features ---
ax_top.hist(
    feature_abundance[strain_top],
    bins=hist_bins_x,
    color=STRAIN_COLOR,
    alpha=0.6,
    label="Strain",
)
ax_top.hist(
    feature_abundance[phage_top],
    bins=hist_bins_x,
    color=PHAGE_COLOR,
    alpha=0.6,
    label="Phage",
)
ax_top.tick_params(labelbottom=False, bottom=False)
ax_top.set_ylim(0, 40)
for spine in ["top", "right"]:
    ax_top.spines[spine].set_visible(False)

# --- Right histogram: SHAP values of top features ---
if len(strain_top) > 0:
    ax_right.hist(
        mean_abs_shap[strain_top],
        bins=hist_bins_y,
        color=STRAIN_COLOR,
        alpha=0.6,
        orientation="horizontal",
    )
if len(phage_top) > 0:
    ax_right.hist(
        mean_abs_shap[phage_top],
        bins=hist_bins_y,
        color=PHAGE_COLOR,
        alpha=0.6,
        orientation="horizontal",
    )
ax_right.tick_params(labelleft=False, left=False)
ax_right.set_xlim(0, 45)
for spine in ["top", "right"]:
    ax_right.spines[spine].set_visible(False)

# --- Main scatter ---
ax_scatter.scatter(
    feature_abundance[~top_mask],
    mean_abs_shap[~top_mask],
    c="lightgrey",
    s=16,
    alpha=0.5,
    zorder=1,
    rasterized=True,
)
if len(strain_top) > 0:
    ax_scatter.scatter(
        feature_abundance[strain_top],
        mean_abs_shap[strain_top],
        c=STRAIN_COLOR,
        s=24,
        alpha=0.3,
        zorder=2,
        marker="o",
    )
if len(phage_top) > 0:
    ax_scatter.scatter(
        feature_abundance[phage_top],
        mean_abs_shap[phage_top],
        c=PHAGE_COLOR,
        s=28,
        alpha=0.3,
        zorder=3,
        marker="o",
    )

ax_scatter.set_ylim(-0.005, 0.103)
for spine in ["top", "right"]:
    ax_scatter.spines[spine].set_visible(False)

fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)

file_name = f"{figs_folder}/shap_vs_abundance_{min_shap_value}.svg"
plt.savefig(file_name, transparent=True)
print(f"\nFigure Saved: {file_name}")
