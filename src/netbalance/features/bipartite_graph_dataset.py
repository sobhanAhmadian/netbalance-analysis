import random
from abc import ABC, abstractmethod

import numpy as np
from netbalance.utils.logger import logging as prj_logger

logger = prj_logger.getLogger(__name__)


class BGDataset(ABC):

    def __init__(self, cluster_a_name, cluster_b_name) -> None:
        super().__init__()
        self.cluster_a_name = cluster_a_name
        self.cluster_b_name = cluster_b_name

    @abstractmethod
    def get_cluster_a_node_names(self):
        raise NotImplementedError

    @abstractmethod
    def get_cluster_b_node_names(self):
        raise NotImplementedError

    @abstractmethod
    def get_dataset_dir(self):
        raise NotImplementedError

    @abstractmethod
    def get_dataset_file_path(self):
        raise NotImplementedError

    def get_associations(
        self,
        with_negatives=False,
    ):
        """
        Get all associations between cluster A nodes and cluster B nodes.

        Args:
            with_negatives (bool, optional): Whether to include negative associations. Defaults to False.

        Returns:
            numpy.ndarray: Array of associations (n, 3), where columns are:
                [Cluster A node index, Cluster B node index, 1 for positive/0 for negative].
        """

        rng = random.Random(0)

        dataset_file_path = self.get_dataset_file_path()
        associations = np.load(dataset_file_path)
        row_num, col_num = associations.shape
        logger.info(f"Cluster A and Cluster B sizes: {row_num}, {col_num}")
        logger.info(f"Positive associations count: {np.sum(associations)}")

        # Extract positive samples
        positive_samples = np.argwhere(associations == 1).tolist()
        positive_samples = [[i, j, 1] for i, j in positive_samples]

        if not with_negatives:
            return np.array(positive_samples, dtype=np.int32)

        negative_samples = [
            [i, j, 0]
            for i in range(row_num)
            for j in range(col_num)
            if associations[i, j] == 0
        ]

        samples = positive_samples + negative_samples
        rng.shuffle(samples)
        logger.info(f"Total samples generated: {len(samples)}")
        return np.array(samples, dtype=np.int32)
