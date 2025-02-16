import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from netbalance.configs import cold_color1, cold_color2, warm_color1, warm_color2
from netbalance.configs.a_degree_ratio import A_DEGREE_RATIO_RESULTS_DIR
from netbalance.configs.b_degree_ratio import B_DEGREE_RATIO_RESULTS_DIR
from netbalance.configs.blindti import BLINDTI_RESULTS_DIR
from netbalance.configs.bmlpdti import BMLPDTI_RESULTS_DIR
from netbalance.configs.brandom import BRANDOM_RESULTS_DIR
from netbalance.configs.brfdti import BRFDTI_RESULTS_DIR
from netbalance.configs.bxgbdti import BXGBDTI_RESULTS_DIR
from netbalance.configs.common import RESULTS_DIR
from netbalance.configs.fmidti import FMIDTI_RESULTS_DIR
from netbalance.configs.midti import MIDTI_RESULTS_DIR
from netbalance.configs.weighted_mean_degree_ratio import (
    WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR,
)
from netbalance.utils.result import get_auc_of_cv_folds

dataset = "luodti"
train_neg_samp_method = "beta"

figs_folder = (
    f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other/compare_rho_hit_k"
)

model_names = [
    "A Degree Ratio",
    "B Degree Ratio",
    "W Degree Ratio",
    "BRANDOM",
    "BLINDTI",
    "BMLPDTI",
    "BRFDTI",
    "BXGBDTI",
    "FMIDTI",
    "MIDTI",
]

model_result_dirs = [
    A_DEGREE_RATIO_RESULTS_DIR,
    B_DEGREE_RATIO_RESULTS_DIR,
    WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR,
    BRANDOM_RESULTS_DIR,
    BLINDTI_RESULTS_DIR,
    BMLPDTI_RESULTS_DIR,
    BRFDTI_RESULTS_DIR,
    BXGBDTI_RESULTS_DIR,
    FMIDTI_RESULTS_DIR,
    MIDTI_RESULTS_DIR,
]

test_balance_method_beta = "beta"
test_balance_kwargs_beta = {}

per_model_rho_hit_k_accuracies = []
for r in model_result_dirs:
    save_rho_hit_k_file = os.path.join(
        r,
        f"rho_hit_k",
        f"dataset_{dataset}",
        f"train_neg_samp_{train_neg_samp_method}",
        "rho_hit_k",
    )
    temp = np.loadtxt(save_rho_hit_k_file, delimiter=",")
    per_model_rho_hit_k_accuracies.append(temp)

#########################
model_names.append("BMLPDTI-IRHO")
save_rho_hit_k_file = os.path.join(
    BMLPDTI_RESULTS_DIR,
    f"rho_hit_k",
    f"dataset_{dataset}",
    f"train_neg_samp_irho",
    "rho_hit_k",
)
temp = np.loadtxt(save_rho_hit_k_file, delimiter=",")
per_model_rho_hit_k_accuracies.append(temp)
#########################

x_common = np.linspace(0, 1, len(per_model_rho_hit_k_accuracies[0]))

fig, axe = plt.subplots(figsize=(20, 12))

x_ticks = x_common

for temp in per_model_rho_hit_k_accuracies:
    axe.plot(x_common, temp)
    

axe.set_xticks(x_ticks)
axe.set_xticklabels([round(i, 2) for i in x_ticks])
axe.set_ylabel("")
axe.set_title("")
axe.set_ylim(0, 1)
axe.set_xlim(-0.05, 1.05)
axe.grid(axis="y", linestyle="--", alpha=0.4)
axe.legend(model_names, loc="upper right")

os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/compare_rho_hit_k.pdf"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
