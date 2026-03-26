import pandas as pd

from netbalance.configs.picard import (
    PICARD_BACTERIA_NAMES_FILE,
    PICARD_PHAGE_NAMES_FILE,
    PICARD_DATASET_FILE,
)
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

from netbalance.features.bipartite_graph_dataset import ADataset


class PicardDataset(ADataset):

    def __init__(self) -> None:
        super().__init__(["bacteria", "phage"])

    def get_node_names(self):
        return [
            self.get_cluster_a_node_names(),
            self.get_cluster_b_node_names(),
        ]

    def get_cluster_a_node_names(self):
        names = pd.read_csv(PICARD_BACTERIA_NAMES_FILE, index_col=0)
        return list(names.iloc[:, 0])

    def get_cluster_b_node_names(self):
        names = pd.read_csv(PICARD_PHAGE_NAMES_FILE, index_col=0)
        return list(names.iloc[:, 0])

    def get_dataset_file_path(self):
        return PICARD_DATASET_FILE
