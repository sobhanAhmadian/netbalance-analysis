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

color11 = "#fdae61"
color21 = "#8c6bb1"
color31 = "#d53e4f"

with_eta = True

measure = "aupr"  # max_f1, auc, aupr

dataset = "luodti"

figs_folder = f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other"


# (Model Result Dir, Train Method, Display Name)
path_dict = [
    (BMLPDTI_RESULTS_DIR, "irho", "UnbiasNet"),
    (BRFDTI_RESULTS_DIR, "beta", "RF"),
    (BXGBDTI_RESULTS_DIR, "beta", "XGBoost"),
    # (MIDTI_RESULTS_DIR, "beta", "MIDTI"),
    (FMIDTI_RESULTS_DIR, "beta", "MIDTI"),
    (BLINDTI_RESULTS_DIR, "beta", "Linear"),
    # (BMLPDTI_RESULTS_DIR, "beta", "MLP"),
    # (BMLPDTI_RESULTS_DIR, "ibeta", "MLP-I"),
    (WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR, "beta", "Both"),
    (B_DEGREE_RATIO_RESULTS_DIR, "beta", "Target"),
    (A_DEGREE_RATIO_RESULTS_DIR, "beta", "Drug"),
    (BRANDOM_RESULTS_DIR, "beta", "Random"),
]
model_names = [r[-1] for r in path_dict]

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

beta_measures = []
eta_measures = []
rho_measures = []

f = None
if measure == "auc":
    f = get_auc_of_cv_folds
elif measure == "max_f1":
    f = get_max_f1_of_cv_folds
elif measure == "aupr":
    f = get_aupr_of_cv_folds

for model_dir, train_balance_method, _ in path_dict:

    values = f(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_beta,
        test_balance_kwargs=test_balance_kwargs_beta,
    )

    beta_measures.append(values)

    values = f(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_eta,
        test_balance_kwargs=test_balance_kwargs_eta,
    )
    eta_measures.append(values)

    values = f(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_rho,
        test_balance_kwargs=test_balance_kwargs_rho,
    )
    rho_measures.append(values)


# Compute mean and std
def get_mean_std(data_list):
    means = [np.mean(x) for x in data_list]
    stds = [np.std(x) for x in data_list]
    return means, stds


beta_means_vals, beta_stds = get_mean_std(beta_measures)
eta_means_vals, eta_stds = get_mean_std(eta_measures)
rho_means_vals, rho_stds = get_mean_std(rho_measures)

# X-axis setup
x = np.arange(len(model_names))  # [0, 1, ..., N-1]
bar_width = 0.2 if with_eta else 0.3

fig, axe = plt.subplots(figsize=(len(path_dict) * 0.75, 2.1))

# Plot bars
bars_beta = axe.bar(
    x - (bar_width if with_eta else bar_width / 2),
    beta_means_vals,
    bar_width,
    yerr=beta_stds,
    capsize=1.0,
    ecolor="black",
    error_kw=dict(lw=1, alpha=0.7),
    label="Balanced",
    color=color11,
    alpha=1.0,
)

if with_eta:
    bars_eta = axe.bar(
        x,
        eta_means_vals,
        bar_width,
        yerr=eta_stds,
        capsize=1.0,
        ecolor="black",
        error_kw=dict(lw=1, alpha=0.7),
        label="Full Test",
        color=color21,
        alpha=0.9,
    )

bars_rho = axe.bar(
    x + (bar_width if with_eta else bar_width / 2),
    rho_means_vals,
    bar_width,
    yerr=rho_stds,
    capsize=1.0,
    ecolor="black",
    error_kw=dict(lw=1, alpha=0.7),
    label="Entity-Balanced",
    color=color31,
    alpha=0.9,
)

# line at 0.5
axe.axhline(
    0.5,
    color="#d53e4f",
    linestyle="--",
    linewidth=0.8,
    label="Random",
)

# Labels and grid
axe.set_xticks(x)
axe.set_xticklabels(model_names)
axe.set_ylabel(measure.upper())
axe.set_ylim((0.0 if with_eta else 0.4), 1)

# Remove top and right borders
axe.spines["top"].set_visible(False)
axe.spines["right"].set_visible(False)

# Legend
# axe.legend(loc="lower right")

os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/compare_{measure}s.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
