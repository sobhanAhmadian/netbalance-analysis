import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from netbalance.configs.ecoli import (
    ECOLI_PHAGE_FEATURES_ANNOTATIONS_FILE,
    ECOLI_STRAIN_FEATURES_ANNOTATIONS_FILE,
)
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

pfa = pd.read_csv(ECOLI_PHAGE_FEATURES_ANNOTATIONS_FILE)
sfa = pd.read_csv(ECOLI_STRAIN_FEATURES_ANNOTATIONS_FILE)
fa = pd.concat([sfa, pfa], ignore_index=True)

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
fa["mean_abs_shap"] = mean_abs_shap
fa["is_strain"] = fa.index < len(ds.get_strain_feature_names())
fa["is_phage"] = ~fa["is_strain"]

fa_sorted = fa.sort_values("mean_abs_shap", ascending=False)

# save fa_sorted to csv
fa_sorted_path = os.path.join(save_dir, "feature_annotations_sorted_by_shap.csv")
fa_sorted.to_csv(fa_sorted_path, index=False)
print(f"Saved feature annotations sorted by mean absolute SHAP to: {fa_sorted_path}")
