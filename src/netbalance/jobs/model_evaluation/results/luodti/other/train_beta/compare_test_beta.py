import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from netbalance.configs import cold_color1, cold_color2, warm_color1, warm_color2
from netbalance.configs.a_degree_ratio import A_DEGREE_RATIO_RESULTS_DIR
from netbalance.configs.b_degree_ratio import B_DEGREE_RATIO_RESULTS_DIR
from netbalance.configs.blindti import BLINDTI_RESULTS_DIR
from netbalance.configs.bmlpdti import BMLPDTI_RESULTS_DIR
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
train_balance_method = "beta"

figs_folder = f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other/train_beta"

model_names = [
    "A Degree Ratio",
    "B Degree Ratio",
    "W Degree Ratio",
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
    BLINDTI_RESULTS_DIR,
    BMLPDTI_RESULTS_DIR,
    BRFDTI_RESULTS_DIR,
    BXGBDTI_RESULTS_DIR,
    FMIDTI_RESULTS_DIR,
    MIDTI_RESULTS_DIR,
]

test_balance_method_beta = "beta"
test_balance_kwargs_beta = {}

beta_aucs = []
for r in model_result_dirs:
    aucs = get_auc_of_cv_folds(
        r,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_beta,
        test_balance_kwargs=test_balance_kwargs_beta,
    )
    beta_aucs.append(aucs.mean())

fig, axe = plt.subplots(figsize=(12, 8))

x_ticks = [i for i in range(len(model_names))]

axe.bar(
    x_ticks,
    beta_aucs,
    width=0.6,
    color=cold_color1,
)

axe.set_xticks(x_ticks)
axe.set_xticklabels(model_names, rotation=45, ha="right")
axe.set_ylabel("Mean AUC")
axe.set_title("AUC Comparison in Beta Evaluation Frameworks")
axe.set_ylim(0.0, 1)
axe.grid(axis="y", linestyle="--", alpha=0.4)

os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/compare_aucs_beta.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")


for i in range(len(model_names)):
    print(f"{model_names[i]}: {beta_aucs[i]}")
