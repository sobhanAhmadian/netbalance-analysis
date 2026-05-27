import copy
import hashlib
import os

import numpy as np
import pandas as pd

from netbalance.configs.common import RESULTS_DIR
from netbalance.data.association_data import BGData, BGTrainTestSpliter
from netbalance.features.luodti import LuoDTIDataset as Dataset
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


dataset = "luodti"

save_dir = os.path.join(
    RESULTS_DIR, "numeric", "other", f"dataset-{dataset}-train-pairwise-similarities"
)

ds = Dataset()


def hash_numpy_array(arr):
    """Hashes a NumPy array using SHA256."""
    arr_bytes = arr.tobytes()  # Convert to bytes
    hash_obj = hashlib.sha256(arr_bytes)  # Compute hash
    return hash_obj.hexdigest()  # Return as hexadecimal string


associations = ds.get_associations(with_negatives=True)
bg_data = BGData(
    associations=ds.get_associations(with_negatives=True),
    cluster_a_node_names=ds.get_cluster_a_node_names(),
    cluster_b_node_names=ds.get_cluster_b_node_names(),
)

spliter = BGTrainTestSpliter(data=bg_data, seed=0, k=5)

train_data, test_data = spliter.split(0)

rho_train_kwargs = {
    "max_iter": 40000,
    "delta": 0.1,
    "cooling_rate": 0.99,
    "initial_temp": 40.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}
beta_train_kwargs = {}

hash_associations = hash_numpy_array(train_data.associations)

rho_associations_list = []
beta_associations_list = []
for i in range(30):
    temp_data = copy.deepcopy(train_data)

    save_name = f"irho_bmlpdti_{hash_associations}_{i}"

    temp_data.balance_data(
        balance_method="rho",
        seed=0,
        save_name=save_name,
        **rho_train_kwargs,
    )

    rho_associations_list.append(temp_data.associations)

    temp_data = copy.deepcopy(train_data)

    save_name = f"ibeta_bmlpdti_{hash_associations}_{i}"

    temp_data.balance_data(
        balance_method="beta",
        seed=0,
        save_name=save_name,
        **beta_train_kwargs,
    )

    beta_associations_list.append(temp_data.associations)

################
# Similarities #
################


def compute_pairwise_similarities(associations_list):
    """Compute pairwise Jaccard similarities between association sets.

    Args:
        associations_list (list of np.ndarray): List of association arrays.

    Returns:
        np.ndarray: Pairwise similarity matrix.
    """
    pairwise_similarities = np.zeros((len(associations_list), len(associations_list)))
    for i in range(len(associations_list)):
        for j in range(i + 1, len(associations_list)):
            a = set(map(tuple, associations_list[i]))
            b = set(map(tuple, associations_list[j]))
            sim = len(a & b) / len(a | b) if len(a | b) > 0 else 0
            pairwise_similarities[i, j] = sim
            pairwise_similarities[j, i] = sim
    return pairwise_similarities


beta_union = set()
for assocs in beta_associations_list:
    beta_union.update(map(tuple, assocs))
beta_union_size = len(beta_union)

rho_union = set()
for assocs in rho_associations_list:
    rho_union.update(map(tuple, assocs))
rho_union_size = len(rho_union)
size_df = {
    "method": ["rho", "beta", "original"],
    "union_size": [rho_union_size, beta_union_size, len(train_data.associations)],
}

size_df = pd.DataFrame(size_df)
size_csv_file = os.path.join(save_dir, "dataset_union_sizes.csv")
size_df.to_csv(size_csv_file, index=False)
print("Dataset union sizes saved to:", size_csv_file)

rho_similarities = compute_pairwise_similarities(rho_associations_list)
beta_similarities = compute_pairwise_similarities(beta_associations_list)

os.makedirs(save_dir, exist_ok=True)
rho_file = os.path.join(save_dir, "irho_pairwise_similarities.npy")
beta_file = os.path.join(save_dir, "ibeta_pairwise_similarities.npy")
np.save(rho_file, rho_similarities)
np.save(beta_file, beta_similarities)
print("Rho pairwise similarities saved to:", rho_file)
print("Beta pairwise similarities saved to:", beta_file)

#################
# Dataset Sizes #
#################

rho_file = os.path.join(save_dir, "irho_dataset_sizes.npy")
beta_file = os.path.join(save_dir, "ibeta_dataset_sizes.npy")

rho_sizes = np.array([len(assocs) for assocs in rho_associations_list])
beta_sizes = np.array([len(assocs) for assocs in beta_associations_list])

np.save(rho_file, rho_sizes)
np.save(beta_file, beta_sizes)
print("Rho dataset sizes saved to:", rho_file)
print("Beta dataset sizes saved to:", beta_file)
