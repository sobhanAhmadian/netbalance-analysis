import os

import matplotlib.pyplot as plt
import numpy as np

from netbalance.configs.common import RESULTS_DIR

color1 = "#fdae61"
color2 = "#d53e4f"

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
dataset = "luodti"

ratios_folder = f"{RESULTS_DIR}/numeric/data_analysis/{dataset}/"

beta_drug_ratios = np.loadtxt(f"{ratios_folder}/beta/ratios_drug.txt", delimiter=",")
beta_drug_num = np.loadtxt(f"{ratios_folder}/beta/num_drug.txt", delimiter=",")

beta_drug_num_zero_pos = beta_drug_num[beta_drug_ratios == 0]
beta_drug_ratios = beta_drug_ratios[~np.isnan(beta_drug_ratios)]
beta_drug_ratios = beta_drug_ratios[beta_drug_ratios > 0]


rho_drug_ratios = np.loadtxt(f"{ratios_folder}/rho/ratios_drug.txt", delimiter=",")
rho_drug_num = np.loadtxt(f"{ratios_folder}/rho/num_drug.txt", delimiter=",")

rho_drug_num_zero_pos = rho_drug_num[rho_drug_ratios == 0]
rho_drug_ratios = rho_drug_ratios[~np.isnan(rho_drug_ratios)]
rho_drug_ratios = rho_drug_ratios[rho_drug_ratios > 0]


fig, ax = plt.subplots(figsize=(3, 2.5))

ax.hist(
    [beta_drug_ratios, rho_drug_ratios],
    bins=40,
    alpha=0.9,
    color=[color1, color2],
    density=True,
)

# Remove top and right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

figs_folder = f"{RESULTS_DIR}/figs/data_analysis/{dataset}"
os.makedirs(figs_folder, exist_ok=True)
fig.tight_layout()
file_name = f"{figs_folder}/drug_ratios.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")

# Plotting the number of negatives for zero ratios

fig, ax = plt.subplots(figsize=(3, 2.5))

ax.hist(
    [beta_drug_num_zero_pos, rho_drug_num_zero_pos],
    bins=40,
    alpha=0.9,
    color=[color1, color2],
    density=True,
)

# Remove top and right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

figs_folder = f"{RESULTS_DIR}/figs/data_analysis/{dataset}"
os.makedirs(figs_folder, exist_ok=True)
fig.tight_layout()
file_name = f"{figs_folder}/drug_hist_num_neg_for_zero_pos.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
