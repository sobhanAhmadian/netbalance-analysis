import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from netbalance.configs.bmlpsyn import BMLPSYN_RESULTS_DIR
from netbalance.configs.common import RESULTS_DIR
from netbalance.utils.result import (
    get_auc_of_cv_folds,
    get_aupr_of_cv_folds,
    get_max_f1_of_cv_folds,
)

color11 = "#fdae61"
color12 = "#fdae61"
color31 = "#d53e4f"
color32 = "#d53e4f"

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

measure = "auc"  # max_f1, auc, aupr
dataset = "sanger"
figs_folder = f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other"

# (Model Result Dir, Train Method, Display Name)
path_dict = [
    (BMLPSYN_RESULTS_DIR, "beta", "One balanced train dataset"),
    (BMLPSYN_RESULTS_DIR, "ibeta", "30 balanced train datasets"),
    (BMLPSYN_RESULTS_DIR, "rho", "One entity-balanced train datasets"),
    (BMLPSYN_RESULTS_DIR, "irho", "30 entity-balanced train datasets"),
]
model_names = [r[-1] for r in path_dict]

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

i = 1
x_list_beta = []
x_list_rho = []
x_ticks = []
vertical_lines = []
gap = 2.5
for j in range(len(path_dict)):
    x_list_beta.append(i)
    i += 1

    x_ticks.append(i)
    
    i += 1

    x_list_rho.append(i)
    i += gap


fig, axe = plt.subplots(figsize=(3, len(path_dict) * 0.7))

################################# Beta

violins = axe.violinplot(
    beta_measures,
    positions=x_list_beta,
    widths=1,
    showmeans=True,
    showextrema=True,
    showmedians=False,
    orientation="horizontal",
)

# Customize violin plot colors
for pc in violins["bodies"]:
    pc.set_facecolor(color11)
    pc.set_edgecolor(color12)
    pc.set_alpha(0.4)

violins["cbars"].set_color(color12)
violins["cmins"].set_color(color12)
violins["cmaxes"].set_color(color12)
violins["cmeans"].set_color(color12)

################################# Rho

violins = axe.violinplot(
    rho_measures,
    positions=x_list_rho,
    widths=1,
    showmeans=True,
    showextrema=True,
    showmedians=False,
    orientation="horizontal",
)

# Customize violin plot colors
for pc in violins["bodies"]:
    pc.set_facecolor(color31)
    pc.set_edgecolor(color32)
    pc.set_alpha(0.4)

violins["cbars"].set_color(color32)
violins["cmins"].set_color(color32)
violins["cmaxes"].set_color(color32)
violins["cmeans"].set_color(color32)

axe.set_yticks(x_ticks)
axe.set_yticklabels(["" for _ in range(len(path_dict))])
axe.set_xlabel(measure.upper())

# Remove top and right borders
axe.spines["top"].set_visible(False)
axe.spines["right"].set_visible(False)

axe.set_xlim(0.69, 0.93)


os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/compare_{measure}s_ablation.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
