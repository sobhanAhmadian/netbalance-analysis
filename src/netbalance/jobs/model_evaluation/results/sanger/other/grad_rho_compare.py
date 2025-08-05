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

measure = "max_f1"  # max_f1, auc, aupr

dataset = "sanger"

figs_folder = f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other"

# (Model Result Dir, Train Method, Color, Display Name)
path_dict = [
    (CCSYNERGY_RESULTS_DIR, "beta", "#3288bd", "CCSynergy"),
    (BLINSYN_RESULTS_DIR, "beta", "#66c2a5", "Linear"),
    (BXGBSYN_RESULTS_DIR, "beta", "#abdda4", "XGBoost"),
    (BRFSYN_RESULTS_DIR, "beta", "#5e4fa2", "RF"),
    # (BMLPSYN_RESULTS_DIR, "beta", "#fee08b", "BMLPSYN"),
    # (BMLPSYN_RESULTS_DIR, "ibeta", "#e6f598", "BMLPSYN-I"),
    (BMLPSYN_RESULTS_DIR, "irho", "#9e0142", "UnbiasNet"),
    (A_DEGREE_RATIO_RESULTS_DIR, "beta", "#fee08b", "Drug"),
    (C_DEGREE_RATIO_RESULTS_DIR, "beta", "#fdae61", "Cell Line"),
    (WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR, "beta", "#f46d43", "Both"),
    (BRANDOM_RESULTS_DIR, "beta", "gray", "BRANDOM"),
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
    model_rho_means = []
    model_rho_stds = []

    for ent in desired_ent_list:
        test_balance_kwargs_rho["ent_desired"] = ent
        values = f(
            model_dir,
            dataset=dataset,
            train_balance_method=train_balance_method,
            test_balance_method=test_balance_method_rho,
            test_balance_kwargs=test_balance_kwargs_rho,
        )
        model_rho_means.append(np.mean(values).item())
        model_rho_stds.append(np.std(values).item())

    rho_measures.append((model_rho_means, model_rho_stds))

fig, axe = plt.subplots(figsize=(3, 2.8))

for idx, (means, stds) in enumerate(rho_measures):
    axe.plot(
        desired_ent_list,
        means,
        color=model_colors[idx],
        linestyle="-",
        lw=1.1,
        label=model_names[idx],
    )
    axe.fill_between(
        desired_ent_list,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        color=model_colors[idx],
        alpha=0.2,
    )


axe.set_ylabel(measure.upper())
axe.set_xlabel("Entropy")

axe.spines["top"].set_visible(False)
axe.spines["right"].set_visible(False)

os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/grad_rho_compare_{measure}s.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
