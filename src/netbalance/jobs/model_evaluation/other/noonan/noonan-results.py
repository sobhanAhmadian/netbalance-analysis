import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from scipy.stats import linregress, pearsonr

from netbalance.configs.a_degree_ratio import A_DEGREE_RATIO_RESULTS_DIR
from netbalance.configs.b_degree_ratio import B_DEGREE_RATIO_RESULTS_DIR
from netbalance.configs.common import RESULTS_DIR
from netbalance.configs.weighted_mean_degree_ratio import (
    WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR,
)
from netbalance.features import (
    EcoliDataset,
    Klebsiella1Dataset,
    Klebsiella2Dataset,
    PseudomonasDataset,
    VibrioDataset,
)
from netbalance.utils.result import get_auc_of_cv_folds

plt.rcParams.update(
    {
        "font.weight": "normal",  # options: 'normal', 'light', 'regular'
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelweight": "regular",
        "axes.titleweight": "regular",
    }
)

color_bacteria = "#4393c3"
color_phage = "#d6604d"
color_both = "#bf812d"
color_pos = "#66c2a5d3"
color_neg = "#d53e50d5"
color_noonan = "#9970ab"

with_eta = True

measure = "auc"  # max_f1, auc, aupr
datasets = {
    "pseudomonas": PseudomonasDataset(),
    "klebsiella1": Klebsiella1Dataset(),
    "klebsiella2": Klebsiella2Dataset(),
    "ecoli": EcoliDataset(),
    "vibrio": VibrioDataset(),
}

figs_folder = f"{RESULTS_DIR}/figs/other/noonan"
os.makedirs(figs_folder, exist_ok=True)

# (Model Result Dir, Train Method, Display Name)
path_dict = [
    (A_DEGREE_RATIO_RESULTS_DIR, "beta", "bacteria"),
    (B_DEGREE_RATIO_RESULTS_DIR, "beta", "phage"),
    (WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR, "beta", "both"),
]
model_names = [r[-1] for r in path_dict]

test_balance_methods = {
    "beta": {},
    "rho": {
        "vibrio": {
            "max_iter": 50000,
            "delta": 0.1,
            "cooling_rate": 0.99,
            "initial_temp": 20.0,
            "ent_desired": 1.0,
            "shrinkage": 1.0,
        },
        "ecoli": {
            "max_iter": 50000,
            "delta": 0.1,
            "cooling_rate": 0.99,
            "initial_temp": 20.0,
            "ent_desired": 1.0,
            "shrinkage": 1.0,
        },
        "klebsiella2": {
            "max_iter": 10000,
            "delta": 0.1,
            "cooling_rate": 0.99,
            "initial_temp": 10.0,
            "ent_desired": 1.0,
            "shrinkage": 1.0,
            "gamma_penalty": 0.2,
        },
        "klebsiella1": {
            "max_iter": 10000,
            "delta": 0.1,
            "cooling_rate": 0.99,
            "initial_temp": 10.0,
            "ent_desired": 1.0,
            "shrinkage": 1.0,
            "gamma_penalty": 0.2,
        },
        "pseudomonas": {
            "max_iter": 10000,
            "delta": 0.1,
            "cooling_rate": 0.99,
            "initial_temp": 10.0,
            "ent_desired": 1.0,
            "shrinkage": 1.0,
            "gamma_penalty": 0.2,
        },
    },
}

data = {
    "auc": {"beta": {}, "rho": {}},
    "entropy": {"beta": {}, "rho": {}},
    "numbers": {},
}

for test_balance_method, test_balance_kwargs in test_balance_methods.items():
    for dataset in datasets.keys():
        for model_dir, train_balance_method, display_name in path_dict:
            # Get AUC
            values = get_auc_of_cv_folds(
                model_dir,
                dataset=dataset,
                train_balance_method=train_balance_method,
                test_balance_method=test_balance_method,
                test_balance_kwargs=test_balance_kwargs.get(dataset, {}),
            )
            if dataset not in data["auc"][test_balance_method]:
                data["auc"][test_balance_method][dataset] = {}
            data["auc"][test_balance_method][dataset][display_name] = values

            # Get Entropy
            entropy_dir = (
                f"{RESULTS_DIR}/numeric/data_analysis/{dataset}/{test_balance_method}"
            )
            name = (
                f"{display_name}_entropies" if display_name != "both" else "entropies"
            )
            entropy_file = f"{entropy_dir}/{name}.txt"
            entopies = np.loadtxt(entropy_file, delimiter=",")
            if dataset not in data["entropy"][test_balance_method]:
                data["entropy"][test_balance_method][dataset] = {}
            data["entropy"][test_balance_method][dataset][display_name] = entopies

# Get Numbers
for dataset, dataset_obj in datasets.items():
    associations = dataset_obj.get_associations()
    num_pos = int(np.sum(associations[:, -1]))
    num = associations.shape[0]
    num_neg = num - num_pos

    bacteria, phages = dataset_obj.get_node_names()

    data["numbers"][dataset] = {
        "num": num,
        "num_pos": num_pos,
        "num_neg": num_neg,
        "num_bacteria": len(bacteria),
        "num_phages": len(phages),
    }


# Compute mean and std
def get_mean_std(data_list):
    means = [np.mean(x) for x in data_list]
    stds = [np.std(x) for x in data_list]
    return means, stds


# Unseen Bacteria Part

## Barplot
fig = plt.figure(figsize=(len(datasets) * 0.9, 2))
axe = fig.add_axes([0.15, 0.2, 0.8, 0.75])
x = np.arange(len(datasets))  # [0, 1, ..., N-1]
bar_width = 0.3
average_phage_aucs = [
    np.mean(data["auc"]["beta"][dataset]["phage"]) for dataset in datasets.keys()
]
noonan_bacteria_aucs = [0.745, 0.674, 0.878, 0.869, 0.941]

axe.bar(
    x - bar_width / 2 - 0.02,
    average_phage_aucs,
    bar_width,
    capsize=1.0,
    ecolor="black",
    error_kw=dict(lw=1, alpha=0.7),
    color=color_phage,
    alpha=1.0,
)
axe.bar(
    x + bar_width / 2 + 0.02,
    noonan_bacteria_aucs,
    bar_width,
    capsize=1.0,
    ecolor="black",
    error_kw=dict(lw=1, alpha=0.7),
    color=color_noonan,
    alpha=1.0,
)

axe.set_xlim(x[0] - 0.5, x[-1] + 0.5)
axe.set_ylim(0.5, 1.0)
axe.set_xticks([])

axe.spines["top"].set_visible(False)
axe.spines["right"].set_visible(False)

file_name = f"{figs_folder}/compare_noonan_bacteria_aucs_vs_ours.svg"
fig.savefig(file_name, transparent=True, bbox_inches="tight", pad_inches=0.0)
print(f"\nFigure Saved: {file_name}")

## Scatter plot of phage-baseline AUCs vs Noonan Unseen Bacteria AUCs
fig = plt.figure(figsize=(2.2, 2.0))
axe = fig.add_axes([0.15, 0.2, 0.8, 0.75])

axe.scatter(
    average_phage_aucs, noonan_bacteria_aucs, color=color_phage, s=50, alpha=0.7
)

slope, intercept, r_value, p_value, std_err = linregress(
    average_phage_aucs, noonan_bacteria_aucs
)
line_x = np.array([min(average_phage_aucs), max(average_phage_aucs)])
line_y = slope * line_x + intercept
axe.plot(line_x, line_y, color="black", linestyle="--", linewidth=1)

corr, corr_pvalue = pearsonr(average_phage_aucs, noonan_bacteria_aucs)

axe.text(
    0.05,
    0.95,
    f"Corr = {corr:.3f}",
    transform=axe.transAxes,
    verticalalignment="bottom",
    fontsize=8,
)

axe.spines["top"].set_visible(False)
axe.spines["right"].set_visible(False)
# axe.set_yticks([])
axe.set_ylim(0.5, 1.0)

file_name = f"{figs_folder}/scatter_noonan_bacteria_vs_ours_phage.svg"
fig.savefig(file_name, transparent=True, bbox_inches="tight", pad_inches=0.0)
print(f"\nFigure Saved: {file_name}")
print(f"R²: {r_value**2:.3f}")
print(f"Correlation: {corr:.3f}")

# Unseen Phage Part

## Barplot
fig = plt.figure(figsize=(len(datasets) * 0.9, 2))
axe = fig.add_axes([0.15, 0.2, 0.8, 0.75])
x = np.arange(len(datasets))  # [0, 1, ..., N-1]

average_bacteria_aucs = [
    np.mean(data["auc"]["beta"][dataset]["bacteria"]) for dataset in datasets.keys()
]

noonan_phage_aucs = [0.925, 0.687, 0.814, 0.87, 0.924]
axe.bar(
    x - bar_width / 2 - 0.02,
    average_bacteria_aucs,
    bar_width,
    capsize=1.0,
    ecolor="black",
    error_kw=dict(lw=1, alpha=0.7),
    color=color_bacteria,
    alpha=1.0,
)
axe.bar(
    x + bar_width / 2 + 0.02,
    noonan_phage_aucs,
    bar_width,
    capsize=1.0,
    ecolor="black",
    error_kw=dict(lw=1, alpha=0.7),
    color=color_noonan,
    alpha=1.0,
)

axe.set_xlim(x[0] - 0.5, x[-1] + 0.5)
axe.set_ylim(0.5, 1.0)
axe.set_xticks([])

axe.spines["top"].set_visible(False)
axe.spines["right"].set_visible(False)

file_name = f"{figs_folder}/compare_noonan_phage_aucs_vs_ours.svg"
fig.savefig(file_name, transparent=True, bbox_inches="tight", pad_inches=0.0)
print(f"\nFigure Saved: {file_name}")

## Scatter plot of bacteria-baseline AUCs vs Noonan Unseen Phage AUCs

fig = plt.figure(figsize=(2.2, 2.0))
axe = fig.add_axes([0.15, 0.2, 0.8, 0.75])

axe.scatter(
    average_bacteria_aucs, noonan_phage_aucs, color=color_bacteria, s=50, alpha=0.7
)

slope, intercept, r_value, p_value, std_err = linregress(
    average_bacteria_aucs, noonan_phage_aucs
)
line_x = np.array([min(average_bacteria_aucs), max(average_bacteria_aucs)])
line_y = slope * line_x + intercept
axe.plot(line_x, line_y, color="black", linestyle="--", linewidth=1)

corr, corr_pvalue = pearsonr(average_bacteria_aucs, noonan_phage_aucs)

axe.text(
    0.05,
    0.95,
    f"Corr = {corr:.3f}",
    transform=axe.transAxes,
    verticalalignment="bottom",
    fontsize=8,
)

axe.spines["top"].set_visible(False)
axe.spines["right"].set_visible(False)
# axe.set_yticks([])
axe.set_ylim(0.5, 1.0)

file_name = f"{figs_folder}/scatter_noonan_phage_vs_ours_bacteria.svg"
fig.savefig(file_name, transparent=True, bbox_inches="tight", pad_inches=0.0)
print(f"\nFigure Saved: {file_name}")
print(f"R²: {r_value**2:.3f}")
print(f"Correlation: {corr:.3f}")
