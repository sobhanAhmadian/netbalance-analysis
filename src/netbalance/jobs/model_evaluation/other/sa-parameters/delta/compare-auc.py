import os

import matplotlib.pyplot as plt
import numpy as np

from netbalance.configs.bmlpdti import BMLPDTI_RESULTS_DIR
from netbalance.configs.common import RESULTS_DIR
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

measure = "auc"  # max_f1, auc, aupr

dataset = "luodti"

figs_folder = f"{RESULTS_DIR}/figs/other/sa-parameters"
os.makedirs(figs_folder, exist_ok=True)


# (Model Result Dir, Train Method, Display Name)
path_dict = [
    (BMLPDTI_RESULTS_DIR, "irho-d0", "delta=0.0"),
    (BMLPDTI_RESULTS_DIR, "irho", "delta=0.1"),
    (BMLPDTI_RESULTS_DIR, "irho-d2", "delta=2.0"),
]
model_names = [r[-1] for r in path_dict]

test_balance_method_beta = "beta"
test_balance_kwargs_beta = {}

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
rho_measures = []

f = None
if measure == "auc":
    f = get_auc_of_cv_folds
elif measure == "max_f1":
    f = get_max_f1_of_cv_folds
elif measure == "aupr":
    f = get_aupr_of_cv_folds

k = 5

for model_dir, train_balance_method, _ in path_dict:

    values = f(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_beta,
        test_balance_kwargs=test_balance_kwargs_beta,
    )
    values = values[:k]

    beta_measures.append(values)

    values = f(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_rho,
        test_balance_kwargs=test_balance_kwargs_rho,
    )
    values = values[:k]
    rho_measures.append(values)


# Compute mean and std
def get_mean_std(data_list):
    means = [np.mean(x) for x in data_list]
    stds = [np.std(x) for x in data_list]
    return means, stds


beta_means_vals, beta_stds = get_mean_std(beta_measures)
rho_means_vals, rho_stds = get_mean_std(rho_measures)

# X-axis setup
x = np.arange(len(model_names))  # [0, 1, ..., N-1]
bar_width = 0.2

fig, ax = plt.subplots(1, 1, figsize=(3.8, 2.2))

# Plot bars
bars_beta = ax.bar(
    x - bar_width / 2 - 0.02,
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

bars_rho = ax.bar(
    x + bar_width / 2 + 0.02,
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

# Labels and grid
ax.set_xticks(x)
ax.set_xticklabels([])
ax.set_ylim(0.5, 1.0)

# Remove top and right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.25)
file_name = f"{figs_folder}/compare_{measure}s.svg"
plt.savefig(file_name, transparent=True)
print(f"\nFigure Saved: {file_name}")
