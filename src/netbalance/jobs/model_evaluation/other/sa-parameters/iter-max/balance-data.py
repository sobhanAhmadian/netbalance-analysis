import copy
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

save_dir = os.path.join(RESULTS_DIR, "numeric", "other", f"sa-parameters")
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


def task(input_data, seed, with_gamma=True):
    train_kwargs = {
        "max_iter": 100000,
        "delta": 0.1,
        "cooling_rate": 0.99,
        "initial_temp": 40.0,
        "ent_desired": 1.0,
        "shrinkage": 1.0,
    }

    if with_gamma:
        entropy_track_path = os.path.join(save_dir, f"rho_entropy_track_{seed}.txt")

        input_data.balance_data(
            balance_method="rho",
            seed=seed,
            entropy_track_path=entropy_track_path,
            **train_kwargs,
        )
    else:
        entropy_track_path = os.path.join(
            save_dir, f"rho_without_gamma_entropy_track_{seed}.txt"
        )
        input_data.balance_data(
            balance_method="rho",
            seed=seed,
            entropy_track_path=entropy_track_path,
            with_gamma=False,
            **train_kwargs,
        )


tasks = []
for seed in [0, 1, 2, 3, 4]:
    input_data = copy.deepcopy(train_data)
    tasks.append(dask.delayed(task)(input_data=input_data, seed=seed))
    tasks.append(dask.delayed(task)(input_data=input_data, seed=seed, with_gamma=False))

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
