import copy
import hashlib
import os

import dask
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from netbalance.configs.common import RESULTS_DIR
from netbalance.data.association_data import BGData, BGTrainTestSpliter
from netbalance.features.luodti import LuoDTIDataset as Dataset
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


dataset = "luodti"

ds = Dataset()

associations = ds.get_associations(with_negatives=True)
bg_data = BGData(
    associations=ds.get_associations(with_negatives=True),
    cluster_a_node_names=ds.get_cluster_a_node_names(),
    cluster_b_node_names=ds.get_cluster_b_node_names(),
)

spliter = BGTrainTestSpliter(data=bg_data, seed=0, k=5)

train_data, test_data = spliter.split(0)


def task(
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


tasks = []

seeds = list(range(30))

for seed in seeds:
    temp_data = copy.deepcopy(train_data)
    tasks.append(
        dask.delayed(task)(
            input_data=temp_data,
            seed=seed,
            max_iter=0,
        )
    )

for seed in seeds:
    temp_data = copy.deepcopy(train_data)
    tasks.append(
        dask.delayed(task)(
            input_data=temp_data,
            seed=seed,
            max_iter=40000,
        )
    )

for seed in seeds:
    temp_data = copy.deepcopy(train_data)
    tasks.append(
        dask.delayed(task)(
            input_data=temp_data,
            seed=seed,
            max_iter=40000,
            with_gamma=False,
        )
    )

if __name__ == "__main__":
    local_cluster = LocalCluster(
        n_workers=min(int(os.getenv("NUM_WORKERS")), len(tasks)),
        threads_per_worker=int(os.getenv("THREADS_PER_WORKER")),
    )
    with (
        Client(local_cluster) as client,
        tqdm(total=len(tasks), desc="Calc Result of RCV") as pbar,
    ):
        futures = client.compute(tasks)

        for future in dask.distributed.as_completed(futures):
            pbar.update(1)

        local_cluster.close()
