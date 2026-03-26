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

dataset = "picard"  # Parameter

ratios_folder = f"{RESULTS_DIR}/numeric/data_analysis/{dataset}/"

beta_bacteria_ratios = np.loadtxt(f"{ratios_folder}/beta/ratios_bacteria.txt", delimiter=",")
beta_bacteria_num = np.loadtxt(f"{ratios_folder}/beta/num_bacteria.txt", delimiter=",")

beta_bacteria_num_zero_pos = beta_bacteria_num[beta_bacteria_ratios == 0]
beta_bacteria_ratios = beta_bacteria_ratios[~np.isnan(beta_bacteria_ratios)]
beta_bacteria_ratios = beta_bacteria_ratios[beta_bacteria_ratios > 0]


rho_bacteria_ratios = np.loadtxt(f"{ratios_folder}/rho/ratios_bacteria.txt", delimiter=",")
rho_bacteria_num = np.loadtxt(f"{ratios_folder}/rho/num_bacteria.txt", delimiter=",")

rho_bacteria_num_zero_pos = rho_bacteria_num[rho_bacteria_ratios == 0]
rho_bacteria_ratios = rho_bacteria_ratios[~np.isnan(rho_bacteria_ratios)]
rho_bacteria_ratios = rho_bacteria_ratios[rho_bacteria_ratios > 0]

beta_phage_ratios = np.loadtxt(
    f"{ratios_folder}/beta/ratios_phage.txt", delimiter=","
)
beta_phage_num = np.loadtxt(f"{ratios_folder}/beta/num_phage.txt", delimiter=",")

beta_phage_num_zero_pos = beta_phage_num[beta_phage_ratios == 0]
beta_phage_ratios = beta_phage_ratios[~np.isnan(beta_phage_ratios)]
beta_phage_ratios = beta_phage_ratios[beta_phage_ratios > 0]

rho_phage_ratios = np.loadtxt(
    f"{ratios_folder}/rho/ratios_phage.txt", delimiter=","
)
rho_phage_num = np.loadtxt(f"{ratios_folder}/rho/num_phage.txt", delimiter=",")

rho_phage_num_zero_pos = rho_phage_num[rho_phage_ratios == 0]
rho_phage_ratios = rho_phage_ratios[~np.isnan(rho_phage_ratios)]
rho_phage_ratios = rho_phage_ratios[rho_phage_ratios > 0]

######### Parameters ##########
ratios = [beta_bacteria_ratios, rho_bacteria_ratios]
num_zero_pos = [beta_bacteria_num_zero_pos, rho_bacteria_num_zero_pos]
# ratios = [beta_phage_ratios, rho_phage_ratios]
# num_zero_pos = [beta_phage_num_zero_pos, rho_phage_num_zero_pos]
###############################

fig, ax = plt.subplots(figsize=(3.2, 2.8))

ax.hist(
    ratios,
    bins=40,
    alpha=0.9,
    color=[color1, color2],
    density=True,
)

ax.set_xlabel("Degree Ratio")
ax.set_ylabel("Frequency")

# Remove top and right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

figs_folder = f"{RESULTS_DIR}/figs/data_analysis/{dataset}"
os.makedirs(figs_folder, exist_ok=True)
fig.tight_layout()
file_name = f"{figs_folder}/entity_ratios.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")

# Plotting the number of negatives for zero ratios

fig, ax = plt.subplots(figsize=(3.2, 2.8))

ax.hist(
    num_zero_pos,
    bins=40,
    alpha=0.9,
    color=[color1, color2],
    density=True,
)

ax.set_xlabel("Number of Negatives")
ax.set_ylabel("Frequency")

# Remove top and right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

figs_folder = f"{RESULTS_DIR}/figs/data_analysis/{dataset}"
os.makedirs(figs_folder, exist_ok=True)
fig.tight_layout()
file_name = f"{figs_folder}/entity_hist_num_neg_for_zero_pos.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
