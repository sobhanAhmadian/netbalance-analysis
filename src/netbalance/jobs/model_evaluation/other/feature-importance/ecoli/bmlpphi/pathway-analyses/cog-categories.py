import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from netbalance.configs.bmlpphi import BMLPPHI_RESULTS_DIR as RESULTS_DIR
from netbalance.utils import prj_logger
import requests

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

logger = prj_logger.getLogger(__name__)


dataset = "ecoli"  # Parameter
figs_folder = f"{RESULTS_DIR}/figs/other/feature_importance/{dataset}/bmlpphi"
save_dir = os.path.join(
    RESULTS_DIR,
    "numeric",
    "other",
    f"feature_importance",
    f"dataset-{dataset}",
    "pathway-analyses",
)

n = 500
spe = "bacteria"
category_col = "COG_Category"

beta_file_name = f"{save_dir}/{spe}-{category_col}-top-{n}-beta.csv"
irho_file_name = f"{save_dir}/{spe}-{category_col}-top-{n}-irho.csv"
beta_results = pd.read_csv(beta_file_name)
irho_results = pd.read_csv(irho_file_name)

merged = pd.merge(
    beta_results[["category", "fg_count", "enriched"]],
    irho_results[["category", "fg_count", "enriched"]],
    on="category",
    suffixes=("_beta", "_irho"),
)
merged = merged.sort_values("fg_count_irho", ascending=True)

beta_num_unannotated = merged[merged["category"] == "-"]["fg_count_beta"].values[0]
irho_num_unannotated = merged[merged["category"] == "-"]["fg_count_irho"].values[0]
beta_num_s = merged[merged["category"] == "S"]["fg_count_beta"].values[0]
irho_num_s = merged[merged["category"] == "S"]["fg_count_irho"].values[0]
beta_num_other = merged["fg_count_beta"].sum() - beta_num_unannotated - beta_num_s
irho_num_other = merged["fg_count_irho"].sum() - irho_num_unannotated - irho_num_s

merged = merged[merged["category"] != "-"]
merged = merged[merged["category"] != "S"]
merged = merged[merged["fg_count_beta"] + merged["fg_count_irho"] > 0]

categories = merged["category"]
y = np.arange(len(categories))


# Compare absolute numbers
fig, ax = plt.subplots(
    figsize=(len(categories) * 0.25 + 1, 2.5)
)  # Adjust height based on number of categories
bar_height = 0.4

beta_color = "#a6dba0"  # green
irho_color = "#9970ab"  # red

ax.bar(
    y + bar_height / 2,
    merged["fg_count_beta"],
    width=bar_height,
    color=beta_color,
    label="Beta",
)
ax.bar(
    y - bar_height / 2,
    merged["fg_count_irho"],
    width=bar_height,
    color=irho_color,
    label="IRho",
)

ax.set_xticks(y)
ax.set_xticklabels([f"{cat}" for cat in categories])
# ax.set_xlabel(f"Count in top {n} features")
ax.set_xlim(-1, y[-1] + 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()
file_name = f"{figs_folder}/{spe}-{category_col}-top-{n}.svg"
plt.savefig(file_name, transparent=True)
print(f"\nFigure Saved: {file_name}")

merged.to_csv(f"test.csv", index=False)

#######
# Beta Pie Chart
#######

fig_pie, ax_pie = plt.subplots(figsize=(1.2, 1.2))

labels = ["Unannotated or Function unknown", "Other"]
sizes = [beta_num_unannotated + beta_num_s, beta_num_other]
colors = ["#dfdfdf", "#fdb863"]

ax_pie.pie(
    sizes,
    colors=colors,
    autopct=lambda p: f"{int(round(p * sum(sizes) / 100))}",
    textprops={"fontsize": 6},
)

fig_pie.tight_layout()
pie_file = f"{figs_folder}/{spe}-{category_col}-top-{n}-beta-pie.svg"
plt.savefig(pie_file, transparent=True, bbox_inches="tight")
print(f"\nPie Figure Saved: {pie_file}")

#######
# IRho Pie Chart
#######

fig_pie, ax_pie = plt.subplots(figsize=(1.2, 1.2))

labels = ["Unannotated or Function unknown", "Other"]
sizes = [irho_num_unannotated + irho_num_s, irho_num_other]

ax_pie.pie(
    sizes,
    colors=colors,
    autopct=lambda p: f"{int(round(p * sum(sizes) / 100))}",
    textprops={"fontsize": 6},
)

fig_pie.tight_layout()
pie_file = f"{figs_folder}/{spe}-{category_col}-top-{n}-irho-pie.svg"
plt.savefig(pie_file, transparent=True, bbox_inches="tight")
print(f"\nPie Figure Saved: {pie_file}")
