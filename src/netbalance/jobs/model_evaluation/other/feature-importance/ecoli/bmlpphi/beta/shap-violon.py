import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from netbalance.configs.bmlpphi import BMLPPHI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.features.ecoli import EcoliDataset as Dataset  # Parameter
from netbalance.utils import prj_logger

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

logger = prj_logger.getLogger(__name__)

dataset = "ecoli"  # Parameter

save_dir = os.path.join(
    RESULTS_DIR,
    "numeric",
    "other",
    f"feature_importance",
    f"dataset-{dataset}",
    "train_neg_samp-beta",
)

figs_folder = f"{RESULTS_DIR}/figs/other/feature_importance/ecoli/bmlpphi/beta"
os.makedirs(figs_folder, exist_ok=True)

ds = Dataset()
feature_names = ds.get_strain_feature_names() + ds.get_phage_feature_names()

all_shape_values = []
all_features = []
for fold in range(5):

    shap_path = os.path.join(save_dir, f"shap_values_fold_{fold + 1}.npy")
    shap_values = np.load(shap_path).squeeze()  # Shape: (num_samples, num_features)
    all_shape_values.append(shap_values)

    featues_path = os.path.join(save_dir, f"features_fold_{fold + 1}.npy")
    features = np.load(featues_path)  # Shape: (num_samples, num_features)
    all_features.append(features)

stacked_shap_values = np.concatenate(
    all_shape_values, axis=0
)  # Shape: (num_samples, num_features)
stacked_features = np.concatenate(
    all_features, axis=0
)  # Shape: (num_samples, num_features)

mean_abs_shap = np.abs(stacked_shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1][:20][::-1]  # Top 20, most important on top

n_features = len(sorted_idx)

fig = plt.figure(figsize=(4, n_features * 0.25 + 1))
ax = fig.add_axes([0.15, 0.2, 0.8, 0.75])

color_0 = "#d53e50d5"  # feature = 0
color_1 = "#66c2a5d3"  # feature = 1

for i, feat_idx in enumerate(sorted_idx):
    shap_col = stacked_shap_values[:, feat_idx]
    feat_col = stacked_features[:, feat_idx].astype(int)

    shap_0 = shap_col[feat_col == 0]
    shap_1 = shap_col[feat_col == 1]

    for shap_group, color, side in [
        (shap_0, color_0, "low"),
        (shap_1, color_1, "high"),
    ]:
        if len(shap_group) < 2:
            continue
        parts = ax.violinplot(
            shap_group,
            positions=[i],
            vert=False,
            widths=0.7,
            showmedians=True,
            showextrema=False,
        )
        body = parts["bodies"][0]
        # Clip to show only top or bottom half (split violin)
        verts = body.get_paths()[0].vertices
        if side == "low":
            verts[:, 1] = np.clip(verts[:, 1], -np.inf, i)  # bottom half
        else:
            verts[:, 1] = np.clip(verts[:, 1], i, np.inf)  # top half
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.9)

        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(0.5)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_yticks(range(n_features))
ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

fig_path = os.path.join(figs_folder, "shap_violin.svg")
fig.savefig(fig_path, dpi=150, bbox_inches="tight", transparent=True)
print(f"Saved SHAP violin plot to: {fig_path}")
