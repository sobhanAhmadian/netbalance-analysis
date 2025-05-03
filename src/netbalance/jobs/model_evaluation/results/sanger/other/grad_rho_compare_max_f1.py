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
from netbalance.utils.result import (
    get_auc_of_cv_folds,
    get_aupr_of_cv_folds,
    get_max_f1_of_cv_folds,
)

measure = "max_f1"  # max_f1, auc, aupr

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

test_balance_method_rho = "rho"
test_balance_kwargs_rho = {
    "max_iter": 20000,
    "delta": 0.1,
    "cooling_rate": 0.99,
    "initial_temp": 40.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}  # Parameter
desired_ent_list = np.arange(0.0, 1.05, 0.1).tolist()

rho_measures = []

f = None
if measure == "auc":
    f = get_auc_of_cv_folds
elif measure == "max_f1":
    f = get_max_f1_of_cv_folds
elif measure == "aupr":
    f = get_aupr_of_cv_folds

for model_dir, train_balance_method, _, _ in path_dict:
    model_rho_measures = []
    for ent in desired_ent_list:
        test_balance_kwargs_rho["ent_desired"] = ent
        values = f(
            model_dir,
            dataset=dataset,
            train_balance_method=train_balance_method,
            test_balance_method=test_balance_method_rho,
            test_balance_kwargs=test_balance_kwargs_rho,
        )
        model_rho_measures.append(np.mean(values).item())

    rho_measures.append(model_rho_measures)

fig, axe = plt.subplots(figsize=(8, 6))

for idx, m_list in enumerate(rho_measures):
    axe.plot(
        desired_ent_list, m_list, color=model_colors[idx], lw=2, label=model_names[idx]
    )


axe.set_ylabel(measure.upper())
axe.set_xlabel("Entropy")
axe.set_title(
    f"{measure.upper().replace("_", " ")} Comparison in Different Entitiy-Balanced States"
)
axe.set_ylim(0.4, 1)

# Add legend
handles = [mpatches.Patch(color=a[-2], label=a[-1]) for a in path_dict]
axe.legend(handles=handles, loc="lower right", ncol=6, fontsize="small")

os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/grad_rho_compare_{measure}s.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
