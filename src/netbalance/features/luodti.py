import os

import numpy as np
import pandas as pd

from netbalance.configs.luodti import (
    LUODTI_DATASET_FILE,
    LUODTI_DRUG_NAMES_FILE,
    LUODTI_PROTEIN_NAMES_FILE,
    LUODTI_PROCESSED_DATA_DIR,
    LUODTI_RAW_DATASET_FILE,
    LUODTI_RAW_DRUG_NAMES_FILE,
    LUODTI_RAW_PROTEIN_NAMES_FILE,
)
from netbalance.utils import data as data_utils
from netbalance.utils import prj_logger
from netbalance.utils.io import json_load

logger = prj_logger.getLogger(__name__)

from .bipartite_graph_dataset import BGDataset


def process():
    # READ Drug Names (the first row is drug name)
    drug_names = pd.read_csv(LUODTI_RAW_DRUG_NAMES_FILE, header=None)
    # READ Protein Names
    protein_names = pd.read_csv(LUODTI_RAW_PROTEIN_NAMES_FILE, header=None)
    # READ Dataset
    dataset = np.loadtxt(LUODTI_RAW_DATASET_FILE, dtype=float, delimiter=" ")

    # Create Directory
    os.makedirs(LUODTI_PROCESSED_DATA_DIR, exist_ok=True)

    # Save Drug Names
    drug_names.to_csv(LUODTI_DRUG_NAMES_FILE, index=False)
    # Save Protein Names
    protein_names.to_csv(LUODTI_PROTEIN_NAMES_FILE, index=False)
    # Save Dataset
    np.save(LUODTI_DATASET_FILE, dataset)


class LuoDTIDataset(BGDataset):

    def __init__(self) -> None:
        super().__init__("drug", "protein")

    def get_cluster_a_node_names(self):
        names = pd.read_csv(LUODTI_DRUG_NAMES_FILE)
        return list(names.iloc[:, 0])

    def get_cluster_b_node_names(self):
        names = pd.read_csv(LUODTI_PROTEIN_NAMES_FILE)
        return list(names.iloc[:, 0])

    def get_dataset_dir(self):
        return LUODTI_PROCESSED_DATA_DIR

    def get_dataset_file_path(self):
        return LUODTI_DATASET_FILE
