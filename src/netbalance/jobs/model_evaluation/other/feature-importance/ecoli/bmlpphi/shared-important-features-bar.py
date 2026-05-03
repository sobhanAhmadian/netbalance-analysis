import os

import matplotlib.pyplot as plt
import numpy as np

from netbalance.configs.bmlpphi import BMLPPHI_RESULTS_DIR as RESULTS_DIR
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

beta_save_dir = os.path.join(
    RESULTS_DIR,
    "numeric",
    "other",
    "feature_importance",
    f"dataset-{dataset}",
    "train_neg_samp-beta",
)
irho_save_dir = os.path.join(
    RESULTS_DIR,
    "numeric",
    "other",
    "feature_importance",
    f"dataset-{dataset}",
    "train_neg_samp-irho",
)

figs_folder = f"{RESULTS_DIR}/figs/other/feature_importance/ecoli/bmlpphi"
os.makedirs(figs_folder, exist_ok=True)

ds = Dataset()
feature_names = ds.get_strain_feature_names() + ds.get_phage_feature_names()

beta_mean_abs_shap = np.load(os.path.join(beta_save_dir, "mean-abs-shap-values.npy"))
irho_mean_abs_shap = np.load(os.path.join(irho_save_dir, "mean-abs-shap-values.npy"))

num_strain_features = len(ds.get_strain_feature_names())
num_phage_features = len(ds.get_phage_feature_names())

print(num_phage_features, num_strain_features)

STRAIN_COLOR = "#d9f0d3"
PHAGE_COLOR = "#9970ab"

shared_percentage = []
phage_shared_percentage = []
bacterial_shared_percentage = []

for i in range(num_strain_features + num_phage_features):
    threshold = np.sort(irho_mean_abs_shap)[-i - 1]
    beta_top_mask = beta_mean_abs_shap >= threshold
    irho_top_mask = irho_mean_abs_shap >= threshold
    shared_count = np.sum(beta_top_mask & irho_top_mask)
    shared_percentage.append(shared_count / (i + 1))

    phage_beta_top_mask = beta_top_mask[num_strain_features:]
    phage_irho_top_mask = irho_top_mask[num_strain_features:]
    num_phage_top = np.sum(irho_top_mask[num_strain_features:])
    phage_shared_count = np.sum(phage_beta_top_mask & phage_irho_top_mask)
    phage_shared_percentage.append(
        phage_shared_count / num_phage_top if num_phage_top > 0 else 0
    )

    bacterial_beta_top_mask = beta_top_mask[:num_strain_features]
    bacterial_irho_top_mask = irho_top_mask[:num_strain_features]
    num_bacterial_top = np.sum(irho_top_mask[:num_strain_features])
    bacterial_shared_count = np.sum(bacterial_beta_top_mask & bacterial_irho_top_mask)
    bacterial_shared_percentage.append(
        bacterial_shared_count / num_bacterial_top if num_bacterial_top > 0 else 0
    )

fig = plt.figure(figsize=(8, 2.5))
ax = fig.add_subplot(1, 1, 1)

ax.plot(
    range(1, num_strain_features + num_phage_features + 1),
    shared_percentage,
    color="#9970ab",
    linestyle="-",
    lw=1.5,
    alpha=0.7,
)
ax.plot(
    range(1, num_strain_features + num_phage_features + 1),
    phage_shared_percentage,
    color="#d6604d",
    linestyle="-",
    lw=1.5,
    alpha=0.7,
)
ax.plot(
    range(1, num_strain_features + num_phage_features + 1),
    bacterial_shared_percentage,
    color="#4393c3",
    linestyle="-",
    lw=1.5,
    alpha=0.7,
)

fig.tight_layout()
file_name = f"{figs_folder}/shared-important-features-bar.svg"
plt.savefig(file_name, transparent=True)
print(f"\nFigure Saved: {file_name}")
