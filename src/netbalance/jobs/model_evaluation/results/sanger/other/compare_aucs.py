import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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

color11 = "#3288bd"
color12 = "#3288bd"
color21 = "#f46d43"
color22 = "#f46d43"
color31 = "#5e4fa2"
color32 = "#5e4fa2"

measure = "auc"  # max_f1, auc, aupr

dataset = "sanger"

figs_folder = f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other"


# (Model Result Dir, Train Method, Display Name)
path_dict = [
    (CCSYNERGY_RESULTS_DIR, "beta", "CCSYNERGY"),
    (BLINSYN_RESULTS_DIR, "beta", "BLINSYN"),
    (BXGBSYN_RESULTS_DIR, "beta", "BXGBSYN"),
    (BRFSYN_RESULTS_DIR, "beta", "BRFSYN"),
    (BMLPSYN_RESULTS_DIR, "beta", "BMLPSYN"),
    (BMLPSYN_RESULTS_DIR, "ibeta", "BMLPSYN-I"),
    (BMLPSYN_RESULTS_DIR, "irho", "BMLPSYN-II"),
    (A_DEGREE_RATIO_RESULTS_DIR, "beta", "DDRC"),
    (C_DEGREE_RATIO_RESULTS_DIR, "beta", "CDRC"),
    (WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR, "beta", "WDRC"),
    (BRANDOM_RESULTS_DIR, "beta", "BRANDOM"),
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
x_list_eta = []
x_list_rho = []
x_ticks = []
vertical_lines = []
gap = 3
for j in range(len(path_dict)):
    x_list_beta.append(i)
    i += 2

    x_ticks.append(i)

    x_list_eta.append(i)
    i += 2

    x_list_rho.append(i)
    i += gap
    vertical_lines.append(i)
    i += gap

del vertical_lines[-1]

fig, axe = plt.subplots(figsize=(12, 6))

################################# Beta

violins = axe.violinplot(
    beta_measures,
    positions=x_list_beta,
    widths=1.2,
    showmeans=True,
    showextrema=True,
    showmedians=False,
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

################################# Eta

violins = axe.violinplot(
    eta_measures,
    positions=x_list_eta,
    widths=1.2,
    showmeans=True,
    showextrema=True,
    showmedians=False,
)

# Customize violin plot colors
for pc in violins["bodies"]:
    pc.set_facecolor(color21)
    pc.set_edgecolor(color22)
    pc.set_alpha(0.4)

violins["cbars"].set_color(color22)
violins["cmins"].set_color(color22)
violins["cmaxes"].set_color(color22)
violins["cmeans"].set_color(color22)

################################# Rho

violins = axe.violinplot(
    rho_measures,
    positions=x_list_rho,
    widths=1.2,
    showmeans=True,
    showextrema=True,
    showmedians=False,
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

axe.set_xticks(x_ticks)
axe.set_xticklabels(model_names, rotation=45, ha="right")
axe.set_ylabel(measure.upper())
axe.set_title(
    f"{measure.upper().replace("_", " ")} Comparison in Different Evaluation Frameworks"
)
axe.set_ylim(0.0, 1)

# Add vertical lines
for v in vertical_lines:
    axe.axvline(v, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

axe.grid(axis="y", linestyle="--", alpha=0.4)

# Add legend
beta_patch = mpatches.Patch(color=color11, label="Balanced")
eta_patch = mpatches.Patch(color=color21, label="Full Test")
rho_patch = mpatches.Patch(color=color31, label="Entity-Balanced")
axe.legend(handles=[beta_patch, eta_patch, rho_patch], loc="lower right")

os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/compare_{measure}s.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
