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

dataset = "luodti"

figs_folder = f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other"

# (Model Result Dir, Train Method, Color, Display Name)
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

test_balance_method_rho = "rho"
test_balance_kwargs_rho = {
    "max_iter": 100000,
    "delta": 0.1,
    "cooling_rate": 0.99,
    "initial_temp": 40.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}
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
