import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

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
from netbalance.utils.result import get_mean_precision_of_cv_folds

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

dataset = "luodti"

figs_folder = f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other"

# (Model Result Dir, Train Method, Display Name)
path_dict = [
    (FMIDTI_RESULTS_DIR, "beta", "#3288bd", "MIDTI"),
    (BLINDTI_RESULTS_DIR, "beta", "#66c2a5", "Leaner"),
    (BXGBDTI_RESULTS_DIR, "beta", "#abdda4", "XGBoost"),
    (BRFDTI_RESULTS_DIR, "beta", "#5e4fa2", "RF"),
    # (BMLPDTI_RESULTS_DIR, "beta", "#e6f598", "BMLPDTI"),
    # (BMLPDTI_RESULTS_DIR, "ibeta", "#abdda4", "BMLPDTI-I"),
    (BMLPDTI_RESULTS_DIR, "irho", "#9e0142", "UnbiasNet"),
    (A_DEGREE_RATIO_RESULTS_DIR, "beta", "#fee08b", "Drug"),
    (B_DEGREE_RATIO_RESULTS_DIR, "beta", "#fdae61", "Target"),
    (WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR, "beta", "#f46d43", "Both"),
    (BRANDOM_RESULTS_DIR, "beta", "gray", "Random"),
]
model_names = [r[-1] for r in path_dict]
model_colors = [r[-2] for r in path_dict]

test_balance_method_beta = "beta"
test_balance_kwargs_beta = {}

test_balance_method_eta = "eta"
test_balance_kwargs_eta = {}

test_balance_method_rho = "rho"
test_balance_kwargs_rho = {
    "max_iter": 100000,
    "delta": 0.1,
    "cooling_rate": 0.99,
    "initial_temp": 40.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}

beta_precisions = []
eta_precisions = []
rho_precisions = []

for model_dir, train_balance_method, _, _ in path_dict:

    values = get_mean_precision_of_cv_folds(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_beta,
        test_balance_kwargs=test_balance_kwargs_beta,
    )

    beta_precisions.append(values)

    values = get_mean_precision_of_cv_folds(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_eta,
        test_balance_kwargs=test_balance_kwargs_eta,
    )
    eta_precisions.append(values)

    values = get_mean_precision_of_cv_folds(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_rho,
        test_balance_kwargs=test_balance_kwargs_rho,
    )
    rho_precisions.append(values)

recalls = np.linspace(0, 1, 100)

precisions_list = rho_precisions

fig, ax = plt.subplots(figsize=(3, 2.8))

for idx, precision in enumerate(precisions_list):
    ax.plot(
        recalls[:-1],
        precision[:-1],
        color=model_colors[idx],
        lw=1.1,
        label=model_names[idx],
    )

ax.set_ylabel("Precision")
ax.set_xlabel("Recall")
ax.set_ylim(0.3, 1.02)

os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/compare_prs.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
