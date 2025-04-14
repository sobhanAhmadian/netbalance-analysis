import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from netbalance.configs import cold_color1, cold_color2, warm_color1, warm_color2
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
from netbalance.utils.result import get_auc_of_cv_folds

dataset = "luodti"
train_balance_method = "beta"

figs_folder = f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other/train_beta"


# (Model Result Dir, Train Method, Display Name)
path_dict = [
    (A_DEGREE_RATIO_RESULTS_DIR, "beta", "DDRC"),
    (B_DEGREE_RATIO_RESULTS_DIR, "beta", "TDRC"),
    (WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR, "beta", "WDRC"),
    (MIDTI_RESULTS_DIR, "beta", "MIDTI"),
    (FMIDTI_RESULTS_DIR, "beta", "FMIDTI"),
    (BLINDTI_RESULTS_DIR, "beta", "BLINDTI"),
    (BXGBDTI_RESULTS_DIR, "beta", "BXGBDTI"),
    (BRFDTI_RESULTS_DIR, "beta", "BRFDTI"),
    (BMLPDTI_RESULTS_DIR, "beta", "BMLPDTI"),
    (BRANDOM_RESULTS_DIR, "beta", "BRANDDTI"),
    (BMLPDTI_RESULTS_DIR, "ibeta", "BMLPDTI-I"),
    (BMLPDTI_RESULTS_DIR, "irho", "BMLPDTI-II"),
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

beta_aucs = []
eta_aucs = []
rho_aucs = []
for model_dir, train_balance_method, _ in path_dict:
    aucs = get_auc_of_cv_folds(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_beta,
        test_balance_kwargs=test_balance_kwargs_beta,
    )
    beta_aucs.append(aucs)

    aucs = get_auc_of_cv_folds(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_eta,
        test_balance_kwargs=test_balance_kwargs_eta,
    )
    eta_aucs.append(aucs)

    aucs = get_auc_of_cv_folds(
        model_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method_rho,
        test_balance_kwargs=test_balance_kwargs_rho,
    )
    rho_aucs.append(aucs)

i = 1
x_list_beta = []
x_list_eta = []
x_list_rho = []
x_ticks = []
vertical_lines = []
gap = 2
for j in range(len(path_dict)):
    x_list_beta.append(i)
    i += 1
    x_ticks.append(i)
    i += 1
    
    x_list_eta.append(i)
    i += 1
    x_ticks.append(i)
    i += 1

    x_list_rho.append(i)
    i += gap
    vertical_lines.append(i)
    i += gap

del vertical_lines[-1]

fig, axe = plt.subplots(figsize=(12, 8))

################################# Beta

violins = axe.violinplot(
    beta_aucs,
    positions=x_list_beta,
    widths=1.2,
    showmeans=True,
    showextrema=True,
    showmedians=False,
)

# Customize violin plot colors
for pc in violins["bodies"]:
    pc.set_facecolor(warm_color1)
    pc.set_edgecolor(warm_color2)
    pc.set_alpha(0.4)

violins["cbars"].set_color(warm_color2)
violins["cmins"].set_color(warm_color2)
violins["cmaxes"].set_color(warm_color2)
violins["cmeans"].set_color(warm_color2)

################################# Eta

violins = axe.violinplot(
    eta_aucs,
    positions=x_list_eta,
    widths=1.2,
    showmeans=True,
    showextrema=True,
    showmedians=False,
)

# Customize violin plot colors
for pc in violins["bodies"]:
    pc.set_facecolor(warm_color1)
    pc.set_edgecolor(warm_color2)
    pc.set_alpha(0.4)

violins["cbars"].set_color(warm_color2)
violins["cmins"].set_color(warm_color2)
violins["cmaxes"].set_color(warm_color2)
violins["cmeans"].set_color(warm_color2)

################################# Rho

violins = axe.violinplot(
    rho_aucs,
    positions=x_list_rho,
    widths=1.2,
    showmeans=True,
    showextrema=True,
    showmedians=False,
)

# Customize violin plot colors
for pc in violins["bodies"]:
    pc.set_facecolor(cold_color1)
    pc.set_edgecolor(cold_color2)
    pc.set_alpha(0.4)

violins["cbars"].set_color(cold_color2)
violins["cmins"].set_color(cold_color2)
violins["cmaxes"].set_color(cold_color2)
violins["cmeans"].set_color(cold_color2)

axe.set_xticks(x_ticks)
axe.set_xticklabels(model_names, rotation=45, ha="right")
axe.set_ylabel("AUC")
axe.set_title("AUC Comparison of Beta vs Rho Evaluation Frameworks")
axe.set_ylim(0.4, 1)

# Add vertical lines
for v in vertical_lines:
    axe.axvline(v, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

axe.grid(axis="y", linestyle="--", alpha=0.4)

# Add legend
beta_patch = mpatches.Patch(color=warm_color1, label="Beta")
rho_patch = mpatches.Patch(color=cold_color1, label="Rho")
axe.legend(handles=[beta_patch, rho_patch], loc="upper right")

os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/compare_aucs_beta_vs_rho.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
