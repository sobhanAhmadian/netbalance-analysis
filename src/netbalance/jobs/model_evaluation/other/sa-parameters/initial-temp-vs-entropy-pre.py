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
    RESULTS_DIR, "numeric", "other", f"dataset-{dataset}-initial-temp-comparison"
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


def get_entropy(
    input_data,
    seed,
    max_iter,
    delta,
    cooling_rate,
    initial_temp,
):
    save_name = f"sa_parameters_irho_bmlpdti_seed{seed}_iter{max_iter}_delta{delta}_cooling{cooling_rate}_temp{initial_temp}"
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
    )
    return input_data.get_stats()["ent"]


tasks = []

seeds = [0, 1, 2]

initial_temp_list = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

entropies = np.zeros((len(seeds), len(initial_temp_list)))

for i, initial_temp in enumerate(initial_temp_list):
    for seed in seeds:

        temp_data = copy.deepcopy(train_data)
        entropies[seed, i] = get_entropy(
            input_data=temp_data,
            seed=seed,
            max_iter=40000,
            delta=0.1,
            cooling_rate=0.99,
            initial_temp=initial_temp,
        )

file_name = os.path.join(save_dir, "initial_temp_entropies.npy")
np.save(file_name, entropies)
print(f"\nEntropies saved: {file_name}")
