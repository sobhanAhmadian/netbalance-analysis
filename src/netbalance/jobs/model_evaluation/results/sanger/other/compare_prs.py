import os

import matplotlib.pyplot as plt
import numpy as np

from netbalance.configs.a_degree_ratio import A_DEGREE_RATIO_RESULTS_DIR
from netbalance.configs.blinsyn import BLINSYN_RESULTS_DIR
from netbalance.configs.bmlpsyn import BMLPSYN_RESULTS_DIR
from netbalance.configs.brandom import BRANDOM_RESULTS_DIR
from netbalance.configs.brfsyn import BRFSYN_RESULTS_DIR
from netbalance.configs.bxgbsyn import BXGBSYN_RESULTS_DIR
from netbalance.configs.c_degree_ratio import C_DEGREE_RATIO_RESULTS_DIR
from netbalance.configs.ccsynergy import CCSYNERGY_RESULTS_DIR
from netbalance.configs.common import RESULTS_DIR
from netbalance.configs.weighted_mean_degree_ratio import (
    WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR,
)
from netbalance.utils.result import get_mean_precision_of_cv_folds

dataset = "sanger"

figs_folder = f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other"

# (Model Result Dir, Train Method, Color, Display Name)
path_dict = [
    (CCSYNERGY_RESULTS_DIR, "beta", "#9e0142", "CCSYNERGY"),
    (BLINSYN_RESULTS_DIR, "beta", "#d53e4f", "BLINSYN"),
    (BXGBSYN_RESULTS_DIR, "beta", "#f46d43", "BXGBSYN"),
    (BRFSYN_RESULTS_DIR, "beta", "#fdae61", "BRFSYN"),
    (BMLPSYN_RESULTS_DIR, "beta", "#fee08b", "BMLPSYN"),
    (BMLPSYN_RESULTS_DIR, "ibeta", "#e6f598", "BMLPSYN-I"),
    (BMLPSYN_RESULTS_DIR, "irho", "#abdda4", "BMLPSYN-II"),
    (A_DEGREE_RATIO_RESULTS_DIR, "beta", "#66c2a5", "DDRC"),
    (C_DEGREE_RATIO_RESULTS_DIR, "beta", "#3288bd", "CDRC"),
    (WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR, "beta", "#5e4fa2", "WDRC"),
    (BRANDOM_RESULTS_DIR, "beta", "#bf812d", "BRANDOM"),
]
model_names = [r[-1] for r in path_dict]
model_colors = [r[-2] for r in path_dict]

test_balance_method_beta = "beta"
test_balance_kwargs_beta = {}

test_balance_method_eta = "eta"
test_balance_kwargs_eta = {}

test_balance_method_rho = "rho"
test_balance_kwargs_rho = {
    "max_iter": 20000,
    "delta": 0.1,
    "cooling_rate": 0.99,
    "initial_temp": 40.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}  # Parameter

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

fig, axe = plt.subplots(3, 1, figsize=(5, 14), sharey=True)

################################# Beta

for idx, precision in enumerate(beta_precisions):
    axe[0].plot(
        recalls, precision, color=model_colors[idx], lw=2, label=model_names[idx]
    )
axe[0].set_title("Balanced PR Curve")
axe[0].set_xlabel("")
axe[0].set_ylabel("Precision")
axe[0].grid(True)

################################# Eta

for idx, precision in enumerate(eta_precisions):
    axe[1].plot(
        recalls, precision, color=model_colors[idx], lw=2, label=model_names[idx]
    )
axe[1].set_title("Full Test PR Curve")
axe[1].set_xlabel("")
axe[1].set_ylabel("Precision")
axe[1].grid(True)

################################# Rho

for idx, precision in enumerate(rho_precisions):
    axe[2].plot(
        recalls, precision, color=model_colors[idx], lw=2, label=model_names[idx]
    )
axe[2].set_title("Entity-Balanced PR Curve")
axe[2].set_xlabel("Recall")
axe[2].set_ylabel("Precision")
axe[2].grid(True)

#################################

# Shared legend
handles, labels = axe[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    fontsize="small",
)

os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout(rect=[0, 0.05, 1, 1])
file_name = f"{figs_folder}/compare_prs.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
