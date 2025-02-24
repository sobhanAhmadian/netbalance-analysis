import pandas as pd

from netbalance.configs.sanger import (
    SANGER_CELL_LINE_NAMES_FILE,
    SANGER_DATASET_FILE,
    SANGER_DRUG_NAMES_FILE,
)
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

from netbalance.features.bipartite_graph_dataset import ADataset


class SangerDataset(ADataset):

    def __init__(self) -> None:
        super().__init__(["drug1", "drug2", "cell-line"])

    def get_node_names(self):
        return [
            self.get_cluster_a_node_names(),
            self.get_cluster_b_node_names(),
            self.get_cluster_c_node_names(),
        ]

    def get_cluster_a_node_names(self):
        names = pd.read_csv(SANGER_DRUG_NAMES_FILE, index_col=0)
        return list(names.iloc[:, 0])

    def get_cluster_b_node_names(self):
        names = pd.read_csv(SANGER_DRUG_NAMES_FILE, index_col=0)
        return list(names.iloc[:, 0])

    def get_cluster_c_node_names(self):
        names = pd.read_csv(SANGER_CELL_LINE_NAMES_FILE, index_col=0)
        return list(names.iloc[:, 0])

    def get_dataset_file_path(self):
        return SANGER_DATASET_FILE
