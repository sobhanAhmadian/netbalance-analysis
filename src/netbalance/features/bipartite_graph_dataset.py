import os
import random
from abc import ABC, abstractmethod
from itertools import product

import numpy as np

from netbalance.configs.common import RESULTS_DIR
from netbalance.utils.logger import logging as prj_logger

logger = prj_logger.getLogger(__name__)

cache_dir = os.path.join(RESULTS_DIR, ".cache")


class ADataset(ABC):

    def __init__(self, cluster_names) -> None:
        super().__init__()
        self.cluster_names = cluster_names

    @abstractmethod
    def get_node_names(self) -> list[list[str]]:
        raise NotImplementedError

    @abstractmethod
    def get_dataset_file_path(self) -> str:
        raise NotImplementedError

    def get_associations(
        self,
        with_negatives=False,
    ):
        """
        Get all associations between clusters.

        Args:
            with_negatives (bool, optional): Whether to generate all possible negative samples. Default to False.

        Returns:
            numpy.ndarray: Array of associations (n, num_clusters + 1).
        """

        # Check cache first
        cache_file = os.path.join(
            cache_dir,
            f"dataset-{'-'.join(self.get_dataset_file_path().split('/')[-2:])}.cache",
        )
        if os.path.exists(cache_file) and with_negatives:
            logger.info(f"Loading from cache: {cache_file}")
            return np.load(cache_file, allow_pickle=True).astype(np.int32)

        os.makedirs(cache_dir, exist_ok=True)

        rng = random.Random(0)

        dataset_file_path = self.get_dataset_file_path()
        associations = np.loadtxt(
            dataset_file_path, delimiter=",", dtype=np.int32
        ).tolist()

        # Extract positive samples
        positive_samples = [samp for samp in associations if samp[-1] == 1]
        negative_samples = [samp for samp in associations if samp[-1] == 0]

        if with_negatives:
            ranges = [
                range(len(cluster_node_names))
                for cluster_node_names in self.get_node_names()
            ]
            negative_samples = [list(x) + [0] for x in product(*ranges)]
            for samp in positive_samples:
                negative_samples.remove(samp[:-1] + [0])

        samples = positive_samples + negative_samples
        rng.shuffle(samples)
        logger.info(f"Total samples generated: {len(samples)}")

        result = np.array(samples, dtype=np.int32)

        if with_negatives:
            np.save(cache_file, result)
            logger.info(f"Saved to cache: {cache_file}")
        return result
