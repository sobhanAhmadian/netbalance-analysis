import os
import time

import numpy as np
from tqdm import tqdm
from netbalance.methods.simulation import generate_samples
from netbalance.configs.common import RESULTS_DIR

save_dir = os.path.join(
    RESULTS_DIR, "numeric", "other", "time-and-memory", "num-associations"
)
os.makedirs(save_dir, exist_ok=True)


def task(
    num_nodes,
    seed,
    with_gamma,
    max_iter,
    num_associations,
    positive_ratio,
    alpha,
):
    num_nodes_a = num_nodes // 2
    num_nodes_b = num_nodes - num_nodes_a

    data = generate_samples(
        num_nodes_a=num_nodes_a,
        num_nodes_b=num_nodes_b,
        num_associations=num_associations,
        positive_ratio=positive_ratio,
        alpha=alpha,
        seed=seed,
    )
    start_time = time.perf_counter()
    data.balance_data(
        balance_method="rho",
        negative_ratio=1.0,
        seed=seed,
        ent_desired=1.0,
        shrinkage=1.0,
        max_iter=max_iter,
        delta=0.1,
        cooling_rate=0.99,
        initial_temp=10.0,
        with_gamma=with_gamma,
    )
    elapsed_time = time.perf_counter() - start_time

    return elapsed_time


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
seeds = list(range(10))
time_usages_both = np.zeros((len(num_associations_list), len(seeds)))
time_usages_heuristic = np.zeros((len(num_associations_list), len(seeds)))
time_usages_sa = np.zeros((len(num_associations_list), len(seeds)))

with tqdm(
    total=len(num_associations_list) * len(seeds) * 3, desc="Evaluating time"
) as pbar:
    for i, num_associations in enumerate(num_associations_list):
        for j, seed in enumerate(seeds):
            time_usages_both[i, j] = task(
                200,
                seed,
                with_gamma=True,
                max_iter=10000,
                num_associations=num_associations,
                positive_ratio=0.2,
                alpha=10.0,
            )
            pbar.update(1)
            time_usages_heuristic[i, j] = task(
                200,
                seed,
                with_gamma=True,
                max_iter=1,
                num_associations=num_associations,
                positive_ratio=0.2,
                alpha=10.0,
            )
            pbar.update(1)
            time_usages_sa[i, j] = task(
                200,
                seed,
                with_gamma=False,
                max_iter=10000,
                num_associations=num_associations,
                positive_ratio=0.2,
                alpha=10.0,
            )
            pbar.update(1)


file_path = f"{save_dir}/time_usage_both.npy"
np.save(file_path, time_usages_both)
print(f"Saved time usages for both to {file_path}")

file_path = f"{save_dir}/time_usage_heuristic.npy"
np.save(file_path, time_usages_heuristic)
print(f"Saved time usages for heuristic to {file_path}")

file_path = f"{save_dir}/time_usage_sa.npy"
np.save(file_path, time_usages_sa)
print(f"Saved time usages for SA to {file_path}")
