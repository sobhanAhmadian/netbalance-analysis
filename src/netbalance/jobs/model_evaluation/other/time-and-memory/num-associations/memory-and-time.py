import os

import matplotlib.pyplot as plt
import numpy as np

from netbalance.configs.common import RESULTS_DIR

save_dir = os.path.join(
    RESULTS_DIR, "numeric", "other", "time-and-memory", "num-associations"
)
os.makedirs(save_dir, exist_ok=True)

figs_folder = f"{RESULTS_DIR}/figs/other/time-and-memory/num-associations"
os.makedirs(figs_folder, exist_ok=True)

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

num_associations_list = [
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
]
seeds = [
    0,
    1,
    2,
]

file_path = f"{save_dir}/time_usage_both.npy"
time_usages_both = np.load(file_path)
mean_time_usages_both = np.mean(time_usages_both, axis=1)
std_time_usages_both = np.std(time_usages_both, axis=1)

file_path = f"{save_dir}/time_usage_heuristic.npy"
time_usages_heuristic = np.load(file_path)
mean_time_usages_heuristic = np.mean(time_usages_heuristic, axis=1)
std_time_usages_heuristic = np.std(time_usages_heuristic, axis=1)

file_path = f"{save_dir}/time_usage_sa.npy"
time_usages_sa = np.load(file_path)
mean_time_usages_sa = np.mean(time_usages_sa, axis=1)
std_time_usages_sa = np.std(time_usages_sa, axis=1)


file_path = f"{save_dir}/memory_usage_both.npy"
memory_usages_both = np.load(file_path)
mean_memory_usages_both = np.mean(memory_usages_both, axis=1)
std_memory_usages_both = np.std(memory_usages_both, axis=1)

file_path = f"{save_dir}/memory_usage_heuristic.npy"
memory_usages_heuristic = np.load(file_path)
mean_memory_usages_heuristic = np.mean(memory_usages_heuristic, axis=1)
std_memory_usages_heuristic = np.std(memory_usages_heuristic, axis=1)

file_path = f"{save_dir}/memory_usage_sa.npy"
memory_usages_sa = np.load(file_path)
mean_memory_usages_sa = np.mean(memory_usages_sa, axis=1)
std_memory_usages_sa = np.std(memory_usages_sa, axis=1)


# Time vs Number of Associations
fig, ax = plt.subplots(1, 1, figsize=(4, 2.3))

ax.plot(
    num_associations_list, mean_time_usages_both, color="#ff248e", linestyle="-", lw=1.1
)
ax.fill_between(
    num_associations_list,
    mean_time_usages_both - std_time_usages_both,
    mean_time_usages_both + std_time_usages_both,
    color="#ff248e",
    alpha=0.2,
)
ax.plot(
    num_associations_list,
    mean_time_usages_heuristic,
    color="#80b1d3",
    linestyle="-",
    lw=1.1,
)
ax.fill_between(
    num_associations_list,
    mean_time_usages_heuristic - std_time_usages_heuristic,
    mean_time_usages_heuristic + std_time_usages_heuristic,
    color="#80b1d3",
    alpha=0.2,
)
ax.plot(
    num_associations_list, mean_time_usages_sa, color="#fdb462", linestyle="-", lw=1.1
)
ax.fill_between(
    num_associations_list,
    mean_time_usages_sa - std_time_usages_sa,
    mean_time_usages_sa + std_time_usages_sa,
    color="#fdb462",
    alpha=0.2,
)

ax.set_xticks(num_associations_list)
ax.set_xticklabels(num_associations_list, rotation=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
file_name = f"{figs_folder}/time_usage_vs_num_associations.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")


# Memory vs Number of Associations
fig, ax = plt.subplots(1, 1, figsize=(4, 2.3))

ax.plot(
    num_associations_list,
    mean_memory_usages_both,
    color="#ff248e",
    linestyle="-",
    lw=1.1,
    alpha=0.8,
)
ax.fill_between(
    num_associations_list,
    mean_memory_usages_both - std_memory_usages_both,
    mean_memory_usages_both + std_memory_usages_both,
    color="#ff248e",
    alpha=0.2,
)
ax.plot(
    num_associations_list,
    mean_memory_usages_heuristic,
    color="#80b1d3",
    linestyle="-",
    lw=1.1,
    alpha=0.8,
)
ax.fill_between(
    num_associations_list,
    mean_memory_usages_heuristic - std_memory_usages_heuristic,
    mean_memory_usages_heuristic + std_memory_usages_heuristic,
    color="#80b1d3",
    alpha=0.2,
)
ax.plot(
    num_associations_list, mean_memory_usages_sa, color="#fdb462", linestyle="-", lw=1.1
)
ax.fill_between(
    num_associations_list,
    mean_memory_usages_sa - std_memory_usages_sa,
    mean_memory_usages_sa + std_memory_usages_sa,
    color="#fdb462",
    alpha=0.2,
)
ax.set_xticks(num_associations_list)
ax.set_xticklabels(num_associations_list, rotation=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
file_name = f"{figs_folder}/memory_usage_vs_num_associations.svg"
plt.savefig(file_name)
print(f"\nFigure Saved: {file_name}")
