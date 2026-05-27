import os

from matplotlib_venn import venn2, venn2_circles
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

# Select the first n most important features for each model.
n_top_features = 1000
beta_top_idx = np.argsort(beta_mean_abs_shap)[::-1][:n_top_features]
irho_top_idx = np.argsort(irho_mean_abs_shap)[::-1][:n_top_features]

beta_top_mask = np.zeros_like(beta_mean_abs_shap, dtype=bool)
irho_top_mask = np.zeros_like(irho_mean_abs_shap, dtype=bool)
beta_top_mask[beta_top_idx] = True
irho_top_mask[irho_top_idx] = True

beta_color = "#d9f0d3"
irho_color = "#9970ab"

fig = plt.figure(figsize=(2.5, 2.0))

beta_only = beta_top_mask & ~irho_top_mask
irho_only = irho_top_mask & ~beta_top_mask
shared = beta_top_mask & irho_top_mask

n_beta_only = beta_only.sum()
n_irho_only = irho_only.sum()
n_shared = shared.sum()


ax = fig.add_subplot(111)
v = venn2(
    subsets=(n_beta_only, n_irho_only, n_shared),
    set_labels=("", ""),
    set_colors=(beta_color, irho_color),
    alpha=0.6,
    ax=ax,
)

# Style the counts
for lbl in ["10", "01", "11"]:
    if v.get_label_by_id(lbl):
        v.get_label_by_id(lbl).set_fontsize(10)

# Outline circles
c = venn2_circles(
    subsets=(n_beta_only, n_irho_only, n_shared),
    linestyle="solid",
    linewidth=0.8,
    color="grey",
    ax=ax,
)

fig.tight_layout()
file_name = f"{figs_folder}/ven-top-{n_top_features}.svg"
plt.savefig(file_name, transparent=True)
print(f"\nFigure Saved: {file_name}")


# Strain features only
beta_top_mask = np.zeros_like(beta_mean_abs_shap, dtype=bool)
irho_top_mask = np.zeros_like(irho_mean_abs_shap, dtype=bool)
beta_top_mask[beta_top_idx] = True
irho_top_mask[irho_top_idx] = True

beta_strain_mask = np.arange(len(feature_names)) < num_strain_features
irho_strain_mask = np.arange(len(feature_names)) < num_strain_features

beta_top_mask = beta_top_mask & beta_strain_mask
irho_top_mask = irho_top_mask & irho_strain_mask

fig = plt.figure(figsize=(2.5, 2.0))

beta_only = beta_top_mask & ~irho_top_mask
irho_only = irho_top_mask & ~beta_top_mask
shared = beta_top_mask & irho_top_mask

n_beta_only = beta_only.sum()
n_irho_only = irho_only.sum()
n_shared = shared.sum()


ax = fig.add_subplot(111)
v = venn2(
    subsets=(n_beta_only, n_irho_only, n_shared),
    set_labels=("", ""),
    set_colors=(beta_color, irho_color),
    alpha=0.6,
    ax=ax,
)

# Style the counts
for lbl in ["10", "01", "11"]:
    if v.get_label_by_id(lbl):
        v.get_label_by_id(lbl).set_fontsize(10)

# Outline circles
c = venn2_circles(
    subsets=(n_beta_only, n_irho_only, n_shared),
    linestyle="solid",
    linewidth=0.8,
    color="grey",
    ax=ax,
)

fig.tight_layout()
file_name = f"{figs_folder}/ven-top-{n_top_features}-strain.svg"
plt.savefig(file_name, transparent=True)
print(f"\nFigure Saved: {file_name}")


# Phage features only
beta_top_mask = np.zeros_like(beta_mean_abs_shap, dtype=bool)
irho_top_mask = np.zeros_like(irho_mean_abs_shap, dtype=bool)
beta_top_mask[beta_top_idx] = True
irho_top_mask[irho_top_idx] = True

beta_strain_mask = np.arange(len(feature_names)) >= num_strain_features
irho_strain_mask = np.arange(len(feature_names)) >= num_strain_features

beta_top_mask = beta_top_mask & beta_strain_mask
irho_top_mask = irho_top_mask & irho_strain_mask

fig = plt.figure(figsize=(2.5, 2.0))

beta_only = beta_top_mask & ~irho_top_mask
irho_only = irho_top_mask & ~beta_top_mask
shared = beta_top_mask & irho_top_mask

n_beta_only = beta_only.sum()
n_irho_only = irho_only.sum()
n_shared = shared.sum()


ax = fig.add_subplot(111)
v = venn2(
    subsets=(n_beta_only, n_irho_only, n_shared),
    set_labels=("", ""),
    set_colors=(beta_color, irho_color),
    alpha=0.6,
    ax=ax,
)

# Style the counts
for lbl in ["10", "01", "11"]:
    if v.get_label_by_id(lbl):
        v.get_label_by_id(lbl).set_fontsize(10)

# Outline circles
c = venn2_circles(
    subsets=(n_beta_only, n_irho_only, n_shared),
    linestyle="solid",
    linewidth=0.8,
    color="grey",
    ax=ax,
)

fig.tight_layout()
file_name = f"{figs_folder}/ven-top-{n_top_features}-phage.svg"
plt.savefig(file_name, transparent=True)
print(f"\nFigure Saved: {file_name}")
