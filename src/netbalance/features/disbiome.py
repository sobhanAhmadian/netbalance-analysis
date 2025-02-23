import os

import numpy as np
import pandas as pd

from netbalance.configs.disbiome import (
    DISBIOME_DATASET_FILE,
    DISBIOME_DISEASE_NAMES_FILE,
    DISBIOME_MICROBE_NAMES_FILE,
    DISBIOME_RAW_DATA_DIR,
)
from netbalance.utils import prj_logger
from netbalance.utils.io import json_load

logger = prj_logger.getLogger(__name__)

from netbalance.features.bipartite_graph_dataset import ADataset


def _get_table(table_name):
    return json_load(os.path.join(DISBIOME_RAW_DATA_DIR, f"{table_name}.json"))


def get_raw_diseases():
    return _get_table("disease")


def get_raw_experiments():
    return _get_table("experiment")


def get_raw_methods():
    return _get_table("method")


def get_raw_organisms():
    return _get_table("organism")


def get_raw_publications():
    return _get_table("publication")


def get_raw_samples():
    return _get_table("sample")


def _process_disease_names(df_associations):
    dis_df = pd.DataFrame(get_raw_diseases())
    dis_df = dis_df.loc[
        dis_df["disease_id"].isin(df_associations["disease_id"].tolist())
    ]  # Select diseases for which exist at list one association
    dis_id_list = dis_df["disease_id"].tolist()
    dis_df["name"].to_csv(DISBIOME_DISEASE_NAMES_FILE)
    logger.info(f"Disease names saved at {DISBIOME_DISEASE_NAMES_FILE}")
    return dis_id_list


def _process_microbe_names(df_associations):
    mic_df = pd.DataFrame(get_raw_organisms())
    mic_df = mic_df.loc[
        mic_df["organism_id"].isin(df_associations["organism_id"].tolist())
    ]  # Select microbes for which exist at list one association
    mic_id_list = mic_df["organism_id"].tolist()
    mic_df["name"].to_csv(DISBIOME_MICROBE_NAMES_FILE)
    logger.info(f"Microbe names saved at {DISBIOME_MICROBE_NAMES_FILE}")
    return mic_id_list


def process():
    df_associations = pd.DataFrame(get_raw_experiments())
    df_associations = df_associations[["organism_id", "disease_id"]]

    dis_id_list = _process_disease_names(df_associations)
    mic_id_list = _process_microbe_names(df_associations)

    logger.info("Calculating associations matrix.")
    associations = np.zeros((len(mic_id_list), len(dis_id_list)))
    for i in range(df_associations.shape[0]):
        mic_id = df_associations.iloc[i, 0]
        dis_id = df_associations.iloc[i, 1]
        mic_index = mic_id_list.index(mic_id)
        dis_index = dis_id_list.index(dis_id)
        associations[mic_index, dis_index] = 1
    logger.info(f"associations shape: {associations.shape}")
    logger.info(f"associations sum of elements: {associations.sum()}")

    np.save(DISBIOME_DATASET_FILE, associations)
    logger.info(f"Associations saved at {DISBIOME_DATASET_FILE}")


class DisbiomDataset(ADataset):

    def __init__(self) -> None:
        super().__init__(["microbe", "disease"])

    def get_node_names(self):
        return [self.get_cluster_a_node_names(), self.get_cluster_b_node_names()]

    def get_cluster_a_node_names(self):
        names = pd.read_csv(DISBIOME_MICROBE_NAMES_FILE)
        return list(names.iloc[:, 1])

    def get_cluster_b_node_names(self):
        names = pd.read_csv(DISBIOME_DISEASE_NAMES_FILE)
        return list(names.iloc[:, 1])

    def get_dataset_file_path(self):
        return DISBIOME_DATASET_FILE
