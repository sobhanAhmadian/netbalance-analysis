import os

import matplotlib.patches as mpatches
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
from netbalance.utils.result import get_hit_k_of_cv_folds

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

beta_measures = []
eta_measures = []
rho_measures = []

for model_dir, train_balance_method, _, _ in path_dict:

    values = get_hit_k_of_cv_folds(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_beta,
        test_balance_kwargs=test_balance_kwargs_beta,
    )

    beta_measures.append(values)

    values = get_hit_k_of_cv_folds(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_eta,
        test_balance_kwargs=test_balance_kwargs_eta,
    )
    eta_measures.append(values)

    values = get_hit_k_of_cv_folds(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_rho,
        test_balance_kwargs=test_balance_kwargs_rho,
    )
    rho_measures.append(values)


fig, axes = plt.subplots(3, 1, figsize=(6, 5 * 3 + 1))

x_common = np.linspace(0, 1, 30)

f_list = [beta_measures, eta_measures, rho_measures]
f_names = ["Balanced", "Full Test", "Entity-Balanced"]
for i, axe in enumerate(axes):
    for idx, m_list in enumerate(f_list[i]):
        axe.plot(
            x_common,
            m_list,
            color=model_colors[idx],
            lw=2,
            label=model_names[idx],
        )

    axe.set_ylabel("Hit@K Accuracy")
    axe.set_xlabel("Normalized Hit@K")
    axe.set_title(f"Hit@K in {f_names[i]} Framework")
    axe.set_ylim(-0.1, 1.1)

# Add legend
handles = [mpatches.Patch(color=a[-2], label=a[-1]) for a in path_dict]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize="small")

os.makedirs(figs_folder, exist_ok=True)
fig.tight_layout(rect=[0, 0.05, 1, 1])
file_name = f"{figs_folder}/compare_hit_k.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
