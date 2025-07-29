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
from netbalance.utils.result import get_mean_tpr_of_cv_folds

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

dataset = "luodti"

figs_folder = f"{RESULTS_DIR}/figs/model_evaluation/results/{dataset}/other"

c1 = "#9e0142"
c2 = "#3288bd"
c3 = "#f46d43"

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

beta_tprs = []
eta_tprs = []
rho_tprs = []

fmidti_beta_tprs = get_mean_tpr_of_cv_folds(
    FMIDTI_RESULTS_DIR,
    dataset=dataset,
    train_balance_method="beta",
    test_balance_method=test_balance_method_beta,
    test_balance_kwargs=test_balance_kwargs_beta,
)
fmidti_rho_tprs = get_mean_tpr_of_cv_folds(
    FMIDTI_RESULTS_DIR,
    dataset=dataset,
    train_balance_method="beta",
    test_balance_method=test_balance_method_rho,
    test_balance_kwargs=test_balance_kwargs_rho,
)

wdrc_beta_tprs = get_mean_tpr_of_cv_folds(
    WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR,
    dataset=dataset,
    train_balance_method="beta",
    test_balance_method=test_balance_method_beta,
    test_balance_kwargs=test_balance_kwargs_beta,
)
wdrc_rho_tprs = get_mean_tpr_of_cv_folds(
    WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR,
    dataset=dataset,
    train_balance_method="beta",
    test_balance_method=test_balance_method_rho,
    test_balance_kwargs=test_balance_kwargs_rho,
)


mlp2_beta_tprs = get_mean_tpr_of_cv_folds(
    BMLPDTI_RESULTS_DIR,
    dataset=dataset,
    train_balance_method="irho",
    test_balance_method=test_balance_method_beta,
    test_balance_kwargs=test_balance_kwargs_beta,
)
mlp2_rho_tprs = get_mean_tpr_of_cv_folds(
    BMLPDTI_RESULTS_DIR,
    dataset=dataset,
    train_balance_method="irho",
    test_balance_method=test_balance_method_rho,
    test_balance_kwargs=test_balance_kwargs_rho,
)


fprs = np.linspace(0, 1, 100)

fig, axe = plt.subplots(1, 1, figsize=(3, 2.8))


# axe.plot(fprs, mlp2_beta_tprs, color=c1, lw=1.1, label="MLP-II (beta)")
axe.plot(fprs, mlp2_rho_tprs, color=c1, lw=1.1, linestyle="--", label="MLP-II (rho)")
# axe.plot(fprs, fmidti_beta_tprs, color=c2, lw=1.1, label="FMI-DTI (beta)")
axe.plot(fprs, fmidti_rho_tprs, color=c2, lw=1.1, linestyle="--", label="FMI-DTI (rho)")
# axe.plot(fprs, wdrc_beta_tprs, color=c3, lw=1.1, label="WDRC (beta)")
axe.plot(fprs, wdrc_rho_tprs, color=c3, lw=1.1, linestyle="--", label="WDRC (rho)")

axe.set_xlabel("False Positive Rate")
axe.set_ylabel("True Positive Rate")


os.makedirs(figs_folder, exist_ok=True)

fig.tight_layout()
file_name = f"{figs_folder}/compare_rocs.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
