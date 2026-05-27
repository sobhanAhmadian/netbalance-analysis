import os

import numpy as np

from netbalance.configs.bmlpphi import BMLPPHI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.features.ecoli import EcoliDataset as Dataset  # Parameter
from netbalance.utils import prj_logger

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

ds = Dataset()
feature_names = ds.get_strain_feature_names() + ds.get_phage_feature_names()

all_shape_values = []
for fold in range(5):

    shap_path = os.path.join(save_dir, f"shap_values_fold_{fold + 1}.npy")
    shap_values = np.load(shap_path).squeeze()  # Shape: (num_samples, num_features)
    all_shape_values.append(shap_values)

stacked_shap_values = np.concatenate(
    all_shape_values, axis=0
)  # Shape: (num_samples, num_features)

mean_abs_shap = np.abs(stacked_shap_values).mean(axis=0)

save_path = os.path.join(save_dir, f"mean-abs-shap-values.npy")
np.save(save_path, mean_abs_shap)
print(f"Mean Absolute SHAP values saved to: {save_path}")
