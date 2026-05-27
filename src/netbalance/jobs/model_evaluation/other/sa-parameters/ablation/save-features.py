import copy
import os

import numpy as np

from netbalance.configs.common import RESULTS_DIR
from netbalance.data.association_data import BGData, BGTrainTestSpliter
from netbalance.features.luodti import LuoDTIDataset as Dataset
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


dataset = "luodti"

save_dir = os.path.join(
    RESULTS_DIR, "numeric", "other", f"dataset-{dataset}-sa-parameters-ablation"
)
os.makedirs(save_dir, exist_ok=True)

ds = Dataset()

associations = ds.get_associations(with_negatives=True)
bg_data = BGData(
    associations=ds.get_associations(with_negatives=True),
    cluster_a_node_names=ds.get_cluster_a_node_names(),
    cluster_b_node_names=ds.get_cluster_b_node_names(),
)

spliter = BGTrainTestSpliter(data=bg_data, seed=0, k=5)

train_data, test_data = spliter.split(0)


def get_entropies(
    input_data,
    seed,
    max_iter,
    with_gamma=True,
    delta=0.1,
    cooling_rate=0.99,
    initial_temp=40.0,
):
    save_name = f"sa_parameters_irho_bmlpdti_seed{seed}_iter{max_iter}_delta{delta}_cooling{cooling_rate}_temp{initial_temp}"
    if not with_gamma:
        save_name += "_without_gamma"

    input_data.balance_data(
        balance_method="rho",
        negative_ratio=1.0,
        seed=seed,
        save_name=save_name,
        ent_desired=1.0,
        shrinkage=1.0,
        max_iter=max_iter,
        delta=delta,
        cooling_rate=cooling_rate,
        initial_temp=initial_temp,
        with_gamma=with_gamma,
    )

    return input_data.get_stats()["ent"]


seeds = list(range(30))

beta_entorpies = np.zeros((len(seeds)))
for seed in seeds:
    temp_data = copy.deepcopy(train_data)
    train_data.balance_data(
        balance_method="beta",
        negative_ratio=1.0,
        seed=seed,
    )
    beta_entorpies[seed] = train_data.get_stats()["ent"]

heuristic_entropies = np.zeros((len(seeds)))
for seed in seeds:
    heuristic_entropies[seed] = get_entropies(
        input_data=copy.deepcopy(train_data),
        seed=seed,
        max_iter=0,
    )

both_entropies = np.zeros((len(seeds)))
for seed in seeds:
    both_entropies[seed] = get_entropies(
        input_data=copy.deepcopy(train_data),
        seed=seed,
        max_iter=40000,
    )

sa_entropies = np.zeros((len(seeds)))
for seed in seeds:
    sa_entropies[seed] = get_entropies(
        input_data=copy.deepcopy(train_data),
        seed=seed,
        max_iter=40000,
        with_gamma=False,
    )

file_name = os.path.join(save_dir, "beta_entropies.npy")
np.save(file_name, beta_entorpies)
print(f"Beta entropies saved to {file_name}")

file_name = os.path.join(save_dir, "heuristic_entropies.npy")
np.save(file_name, heuristic_entropies)
print(f"Heuristic entropies saved to {file_name}")

file_name = os.path.join(save_dir, "both_entropies.npy")
np.save(file_name, both_entropies)
print(f"Both entropies saved to {file_name}")

file_name = os.path.join(save_dir, "sa_entropies.npy")
np.save(file_name, sa_entropies)
print(f"SA entropies saved to {file_name}")
