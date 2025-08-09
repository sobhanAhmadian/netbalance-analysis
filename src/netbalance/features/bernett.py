import pandas as pd

from netbalance.configs.bernett import (
    BERNETT_INTRA_0_FILE,
    BERNETT_INTRA_1_FILE,
    BERNETT_INTRA_1N_FILE,
    BERNETT_INTRA_2_FILE,
    BERNETT_PROTEIN_NAMES_FILE,
)
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

from netbalance.features.bipartite_graph_dataset import ADataset


class BernettDataset(ADataset):

    def __init__(self, type="intra1") -> None:
        super().__init__(["protein1", "protein2"])

        if type == "intra0":
            self._dataset_file = BERNETT_INTRA_0_FILE
        elif type == "intra1":
            self._dataset_file = BERNETT_INTRA_1_FILE
        elif type == "intra2":
            self._dataset_file = BERNETT_INTRA_2_FILE
        elif type == "intra1n":
            self._dataset_file = BERNETT_INTRA_1N_FILE
        else:
            raise ValueError(f"Unknown type: {type}")

    def get_node_names(self):
        return [
            self.get_cluster_a_node_names(),
            self.get_cluster_b_node_names(),
        ]

    def get_cluster_a_node_names(self):
        names = pd.read_csv(BERNETT_PROTEIN_NAMES_FILE, index_col=0)
        return list(names.iloc[:, 0])

    def get_cluster_b_node_names(self):
        names = pd.read_csv(BERNETT_PROTEIN_NAMES_FILE, index_col=0)
        return list(names.iloc[:, 0])

    def get_dataset_file_path(self):
        return self._dataset_file
