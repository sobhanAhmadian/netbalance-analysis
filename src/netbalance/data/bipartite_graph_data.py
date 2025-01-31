import math
import os
import random
from typing import Union

import numpy as np

from netbalance.configs.common import RESULTS_DIR
from netbalance.utils import get_header_format, prj_logger

from .general import Data, TrainTestSplitter

logger = prj_logger.getLogger(__name__)


class BGData(Data):

    def __init__(
        self,
        associations: np.ndarray,
        cluster_a_node_names,
        cluster_b_node_names,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.associations = associations
        self.cluster_a_node_names = cluster_a_node_names
        self.cluster_b_node_names = cluster_b_node_names

    def __len__(self):
        return self.associations.shape[0]

    def balance_data(
        self,
        balance_method: Union[str, None] = None,
        negative_ratio: float = 1.0,
        seed: int = 42,
        save_name: Union[str, None] = None,
        force_calculation: bool = False,
        **kwargs,
    ):
        """
        Balance associations based on node degrees.

        Args:
            balance_method (str, optional): Balance method: 'beta', 'gamma', 'rho' or None.
                Defaults to None.
            negative_ratio (float, optional): Ratio of negative to positive samples. Defaults to 1.0.
            seed (int, optional): Random seed. Defaults to 42.
            save_name (str, optional): If provided, the result will be saved to the specified file.
            force_calculation (bool, optional): If True, the balance method will be applied regardless of the cache.
            kwargs (dict): Additional keyword arguments for specific balance methods.
        """

        logger.info(
            get_header_format(f"Getting Associations between Cluster A and Cluster B")
        )
        logger.info(f"Balance Method: {balance_method}")

        if save_name is not None and not force_calculation:
            if os.path.exists(f"{RESULTS_DIR}/.cache/{save_name}.csv"):
                self.load_associations(f"{RESULTS_DIR}/.cache/{save_name}.csv")
                return

        rng = np.random.default_rng(seed)

        pos_associations = [[i, j, 1] for i, j, k in self.associations if k == 1]
        neg_associations = [[i, j, 0] for i, j, k in self.associations if k == 0]

        if balance_method is None:
            return np.array(pos_associations, dtype=np.int32)

        # Select negative balance method
        samples = []
        if balance_method == "beta":
            logger.info("Balancing Data using Beta Method")
            samples = self._beta_neg_sampling(
                pos_associations=pos_associations,
                neg_associations=neg_associations,
                negative_ratio=negative_ratio,
                rng=rng,
                **kwargs,
            )
        elif balance_method == "rho":
            logger.info("Balancing Data using Rho Method")
            samples = self._rho_neg_sampling(
                pos_associations=pos_associations,
                neg_associations=neg_associations,
                negative_ratio=negative_ratio,
                rng=rng,
                **kwargs,
            )
        elif balance_method == "gamma":
            logger.info("Balancing Data using Gamma Method")
            samples = self._gamma_neg_sampling(
                pos_associations=pos_associations,
                neg_associations=neg_associations,
                negative_ratio=negative_ratio,
                rng=rng,
            )

        # Combine and shuffle
        rng.shuffle(samples)
        self.associations = np.array(samples, dtype=np.int32)

        if save_name is not None:
            os.makedirs(f"{RESULTS_DIR}/.cache", exist_ok=True)
            file = f"{RESULTS_DIR}/.cache/{save_name}.csv"
            self.save_associations(file)

    def save_associations(self, file: str):
        """
        Save associations to a file.

        Args:
            file (str): Path to the file.
        """
        with open(file, "w") as f:
            f.write("Node A,Node B,Association\n")
            for i in range(len(self.associations)):
                f.write(
                    f"{self.associations[i, 0]},{self.associations[i, 1]},{self.associations[i, 2]}\n"
                )
            logger.info(f"Associations saved to {file}")

    def load_associations(self, file: str):
        """
        Load associations from a file.

        Args:
            file (str): Path to the file.
        """
        with open(file, "r") as f:
            lines = f.readlines()[1:]
            associations = []
            for line in lines:
                associations.append([int(x) for x in line.strip().split(",")])
            self.associations = np.array(associations, dtype=np.int32)
            logger.info(f"Associations loaded from {file}")

    def _beta_neg_sampling(
        self,
        pos_associations,
        neg_associations,
        negative_ratio,
        rng,
        **kwargs,
    ):
        """Uniformly sample negative edges."""
        num_negative = int(len(pos_associations) * negative_ratio)

        selected = rng.choice(len(neg_associations), size=num_negative, replace=False)
        neg_samples = [neg_associations[i] for i in selected]

        return neg_samples + pos_associations

    def _rho_neg_sampling(
        self,
        pos_associations,
        neg_associations,
        negative_ratio,
        rng,
        max_iter=1000,
        initial_temp=10.0,
        cooling_rate=0.99,
        delta=1,
        shrinkage=0.5,
        ent_desired=1,
    ):
        """
        Perform rho-based negative sampling using Simulated Annealing.

        Args:
            pos_associations (np.ndarray): Positive associations.
            neg_associations (np.ndarray): Negative associations.
            negative_ratio (float): Ratio of negative to positive samples.
            rng (numpy.random.Generator): Random number generator.
            seed (int): Random seed for reproducibility.
            max_iter (int, optional): Maximum number of iterations for Simulated Annealing. Defaults to 1000.
            initial_temp (float, optional): Initial temperature for Simulated Annealing. Defaults to 10.0.
            cooling_rate (float, optional): Cooling rate for temperature reduction. Defaults to 0.99.
            delta (float, optional): Parameter for controlling the remove of positive samples
            shrinkage (float, optional): Shrinkage factor for the initial graph. Defaults to 0.5.
            ent_desired (float, optional): Desired entropy value. Defaults to 1.
        """

        num_negative = int(len(pos_associations) * negative_ratio)
        initial_graph_len = num_negative + len(pos_associations)

        current_graph = self._gamma_neg_sampling(
            pos_associations=pos_associations,
            neg_associations=neg_associations,
            negative_ratio=negative_ratio,
            rng=rng,
        )
        current_ent_score, current_len_score = self._calculate_graph_score(
            current_graph, initial_graph_len
        )
        current_score = self._combine_scores(
            current_ent_score, current_len_score, delta, ent_desired
        )
        logger.info(f"Initial entropy score: {current_ent_score}")

        best_graph = current_graph[:]
        best_ent_score = current_ent_score
        best_len_score = current_len_score
        best_score = current_score

        temperature = initial_temp

        for k in range(max_iter):

            pos_edges = [edge for edge in current_graph if edge[2] == 1]
            neg_edges = [edge for edge in current_graph if edge[2] == 0]

            temp = rng.random()
            if temp < 0.5:  # Add one positive and one negative edge

                if len(current_graph) <= initial_graph_len - 2:

                    while True:
                        new_pos_edge = random.choice(pos_associations)
                        if new_pos_edge not in pos_edges:
                            current_graph.append([new_pos_edge[0], new_pos_edge[1], 1])
                            break

                    while True:
                        new_neg_edge = random.choice(neg_associations)
                        if new_neg_edge not in neg_edges:
                            current_graph.append([new_neg_edge[0], new_neg_edge[1], 0])
                            break

            else:  # Remove one positive and one negative edge

                # Remove positive edge
                if len(pos_edges) > 0:
                    edge_to_remove = rng.choice(pos_edges).tolist()
                    current_graph.remove(edge_to_remove)

                # Remove negative edge
                if len(neg_edges) > 0:
                    edge_to_remove = rng.choice(neg_edges).tolist()
                    current_graph.remove(edge_to_remove)

            # Calculate new score
            new_ent_score, new_len_score = self._calculate_graph_score(
                current_graph, initial_graph_len
            )
            new_score = self._combine_scores(
                new_ent_score, new_len_score, delta, ent_desired
            )
            score_difference = new_score - current_score

            # Accept new state with a probability based on the temperature
            if score_difference > 0 or rng.random() < math.exp(
                score_difference / temperature
            ):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_ent_score = new_ent_score
                    best_len_score = new_len_score
                    best_graph = current_graph[:]
            else:
                # Revert the change
                current_graph = best_graph[:]

            # Update temperature
            temperature *= cooling_rate

        logger.info(f"Best entropy score achieved: {best_ent_score}")
        logger.info(f"Best length score achieved: {best_len_score}")
        logger.info(f"Best score achieved: {best_score}")
        logger.info(f"Graph size: {len(best_graph)}")

        return best_graph

    def _gamma_neg_sampling(
        self,
        pos_associations,
        neg_associations,
        negative_ratio,
        rng,
    ):
        """
        Weighted negative sampling with caching for a specific seed.
        Samples are selected sequentially, and the weight matrix is updated after each selection.

        Args:
            pos_associations (np.ndarray): Positive associations.
            neg_associations (np.ndarray): Negative associations.
            negative_ratio (float): Ratio of negative to positive samples.
            rng (numpy.random.Generator): Random number generator.

        Returns:
            list: List of negative samples as [i, j, 0].
        """
        num_negative = int(len(pos_associations) * negative_ratio)

        row_num, col_num = len(self.cluster_a_node_names), len(
            self.cluster_b_node_names
        )

        # Initialize the weight matrix
        weights = np.zeros((row_num, col_num), dtype=float)
        for i, j, _ in neg_associations:
            for k, l, _ in pos_associations:
                if i == k or j == l:
                    weights[i, j] += 1.0

        neg_samples = []
        for _ in range(num_negative):
            # Normalize weights
            weights_sum = weights.sum()
            if weights_sum == 0:
                logger.warning("No more valid negative samples to select.")
                break
            normalized_weights = weights / weights_sum

            # Select a negative sample based on weights
            indices = np.argwhere(weights > 0)
            probabilities = normalized_weights[weights > 0]
            selected_idx = rng.choice(len(indices), size=1, p=probabilities)[0]
            i, j = indices[selected_idx]
            neg_samples.append([i, j, 0])

            # Update the weight matrix
            weights[i, j] = 0  # Set the selected edge weight to 0
            weights[i, :] -= (weights[i, :] > 0).astype(float)  # Penalize row i
            weights[:, j] -= (weights[:, j] > 0).astype(float)  # Penalize column j

        logger.info(f"Number of negative samples generated: {len(neg_samples)}")

        return neg_samples + pos_associations

    def _calculate_graph_score(self, associations, initial_graph_len):
        """Calculate the score for the bipartite graph."""
        graph_len = len(associations)
        interaction = self._generate_interaction_matrix(associations)
        per_a_ent = self._calculate_cluster_score(interaction, axis=1)
        per_b_ent = self._calculate_cluster_score(interaction, axis=0)
        ent_score = (per_a_ent + per_b_ent) / 2
        len_score = graph_len / initial_graph_len

        return ent_score, len_score

    def _calculate_cluster_score(self, interaction: np.ndarray, axis: int):
        """Calculate cluster score."""
        num_neg = np.sum(interaction == 0, axis=axis)
        num_pos = np.sum(interaction == 1, axis=axis)
        total = num_neg + num_pos
        return self.get_entropy(total + 1e-5, num_neg, num_pos)

    def _combine_scores(self, ent_score, len_score, delta, ent_desired):
        """Combine entropy and length scores."""
        return (1 - abs(ent_score - ent_desired)) + delta * len_score

    def get_stats(
        self,
    ) -> dict:
        """
        Calculate statistics of the bipartite graph dataset.

        Returns:
            dict: A nested dictionary of statistics for the dataset.
        """
        stats = self._initialize_stats()

        interaction = self._generate_interaction_matrix(self.associations)
        self._calculate_node_stats(interaction, stats["a"], axis=1)
        self._calculate_node_stats(interaction, stats["b"], axis=0)

        per_a_ent = stats["a"]["ent"].item()
        per_b_ent = stats["b"]["ent"].item()
        stats["ent"] = (per_a_ent + per_b_ent) / 2

        return stats

    def _initialize_stats(self) -> dict:
        """Initialize the statistics structure for the dataset."""

        def init_node_stats(size: int):
            return {
                "num_neg": np.zeros(size),
                "num_pos": np.zeros(size),
                "num": np.zeros(size),
                "r": np.zeros(size),
                "ent": np.zeros(1),
            }

        cluster_a_size = len(self.cluster_a_node_names)
        cluster_b_size = len(self.cluster_b_node_names)

        stats = {
            "ent": np.zeros(1),
            "a": init_node_stats(cluster_a_size),
            "b": init_node_stats(cluster_b_size),
        }
        return stats

    def _generate_interaction_matrix(self, associations: np.ndarray) -> np.ndarray:
        """Generate an interaction matrix from associations."""
        interaction = np.full(
            (
                len(self.cluster_a_node_names),
                len(self.cluster_b_node_names),
            ),
            np.nan,
        )
        for a, b, val in associations:
            interaction[a, b] = val
        return interaction

    def _calculate_node_stats(self, interaction: np.ndarray, stats: dict, axis: int):
        """Calculate node-specific statistics."""
        num_neg = np.sum(interaction == 0, axis=axis)
        num_pos = np.sum(interaction == 1, axis=axis)
        total = num_neg + num_pos
        r = (num_pos + 1e-5) / (total + 1e-5)
        ent = self.get_entropy(total + 1e-5, num_neg, num_pos)

        stats["num_neg"] += num_neg
        stats["num_pos"] += num_pos
        stats["num"] += total
        stats["r"] += r
        stats["ent"] += np.array([ent])

    def get_entropy(
        self, total: np.ndarray, num_neg: np.ndarray, num_pos: np.ndarray
    ) -> float:
        """Calculate the entropy of a node group."""
        p_neg = num_neg / (total + 1e-5)
        p_pos = num_pos / (total + 1e-5)

        # Handle the case where both p_neg and p_pos are 0
        entropy = np.where(
            (p_neg == 0) & (p_pos == 0),
            1,  # Set entropy to 1 for these cases
            -p_neg * np.log2(p_neg + 1e-10) - p_pos * np.log2(p_pos + 1e-10),
        )

        weights = total / total.sum()
        return np.dot(entropy, weights).item()


class BGTrainTestSpliter(TrainTestSplitter):

    def __init__(
        self,
        k: int,
        data: BGData,
        seed=42,
        train_balance=False,
        train_balance_kwargs={},
    ):
        super().__init__(k, data, seed)
        self.data_size = len(data)
        self.train_balance = train_balance
        self.train_balance_kwargs = train_balance_kwargs

    def get_subsets(self):
        """Stratified K-Fold cross-validation."""
        subsets = dict()

        subset_size = int(self.data_size / self.k)
        subset_pos_size = int(self.data.associations[:, 2].sum() / self.k)
        subset_neg_size = subset_size - subset_pos_size
        remain_positive = [
            i for i in range(self.data_size) if self.data.associations[i, 2] == 1
        ]
        remain_negative = [
            i for i in range(self.data_size) if self.data.associations[i, 2] == 0
        ]

        for i in range(self.k - 1):
            subsets[i] = self.rng.sample(
                remain_positive, subset_pos_size
            ) + self.rng.sample(remain_negative, subset_neg_size)
            remain_positive = list(set(remain_positive).difference(subsets[i]))
            remain_negative = list(set(remain_negative).difference(subsets[i]))
        subsets[self.k - 1] = remain_positive + remain_negative

        return subsets

    def split(self, i):
        indices = set(range(0, self.data_size))
        test_indices = list(self.subsets[i])
        train_indices = list(indices.difference(self.subsets[i]))

        train_data = BGData(
            self.data.associations[train_indices],
            self.data.cluster_a_node_names,
            self.data.cluster_b_node_names,
        )
        test_data = BGData(
            self.data.associations[test_indices],
            self.data.cluster_a_node_names,
            self.data.cluster_b_node_names,
        )

        if self.train_balance:
            train_data.balance_data(**self.train_balance_kwargs)
            logger.info(
                f"Train Data has been balanced using {self.train_balance_kwargs}"
            )
            logger.info(f"Train Data Shape: {train_data.associations.shape}")
            logger.info(f"Dataset Entropy: {train_data.get_stats()["ent"]}")

        return train_data, test_data

    def get_data_size(self):
        return len(self.data)


class AClusterCVTrainTestSpliter(TrainTestSplitter):

    def __init__(self, k: int, data: BGData, seed=42):
        super().__init__(k, data, seed)
        self.microbe_list = self.get_microbe_list(data)

    def get_microbe_list(self, data):
        microbe_list = set()
        for i in range(data.associations.shape[0]):
            microbe_list.add(data.associations[i, 0])
        microbe_list = list(microbe_list)
        logger.info(f"List of Ids of Unique Microbes: {microbe_list[:5]} ...")
        return microbe_list

    def split(self, i):
        indices = set(range(0, self.data_size))
        test_indices = list(self.subsets[i])
        train_indices = list(indices.difference(self.subsets[i]))

        test_microbes = [
            x for i, x in enumerate(self.microbe_list) if i in test_indices
        ]
        train_microbes = [
            x for i, x in enumerate(self.microbe_list) if i in train_indices
        ]
        logger.info(f"Test Microbes: {test_microbes[:5]} ...")
        logger.info(f"Train Microbes: {train_microbes[:5]} ...")

        train_data = BGData(
            self.data.associations[
                np.isin(self.data.associations[:, 0], train_microbes)
            ]
        )
        test_data = BGData(
            self.data.associations[np.isin(self.data.associations[:, 0], test_microbes)]
        )

        logger.info(f"Test Data: {test_data.associations[:5].tolist()} ...")
        logger.info(f"Train Data: {train_data.associations[:5].tolist()} ...")

        return train_data, test_data

    def get_data_size(self):
        return len(self.get_microbe_list(self.data))


class BClusterCVTrainTestSpliter(TrainTestSplitter):

    def __init__(self, k: int, data: BGData, seed=42):
        super().__init__(k, data, seed)
        self.disease_list = self.get_disease_list(data)
        self.data_size = len(self.disease_list)

    def get_disease_list(self, data):
        disease_list = set()
        for i in range(data.associations.shape[0]):
            disease_list.add(data.associations[i, 1])
        disease_list = list(disease_list)
        logger.info(f"List of Ids of Unique Diseases:: {disease_list[:5]} ...")
        return disease_list

    def split(self, i):
        indices = set(range(0, self.data_size))
        test_indices = list(self.subsets[i])
        train_indices = list(indices.difference(self.subsets[i]))

        test_diseases = [
            x for i, x in enumerate(self.disease_list) if i in test_indices
        ]
        train_diseases = [
            x for i, x in enumerate(self.disease_list) if i in train_indices
        ]
        logger.info(f"Test Diseases: {test_diseases[:5]} ...")
        logger.info(f"Train Diseases : {train_diseases[:5]} ...")

        train_data = BGData(
            self.data.associations[
                np.isin(self.data.associations[:, 1], train_diseases)
            ]
        )
        test_data = BGData(
            self.data.associations[np.isin(self.data.associations[:, 1], test_diseases)]
        )

        logger.info(f"Test Data: {train_data.associations[:5].tolist()} ...")
        logger.info(f"Train Data: {test_data.associations[:5].tolist()} ...")

        return train_data, test_data

    def get_data_size(self):
        return len(self.get_disease_list(self.data))


class AClusterLOOCVTrainTestSpliter(AClusterCVTrainTestSpliter):

    def __init__(self, data, seed=42):
        super().__init__(k=len(self.get_microbe_list(data)), data=data, seed=seed)


class BClusterLOOCVTrainTestSpliter(BClusterCVTrainTestSpliter):

    def __init__(self, data, seed=42):
        super().__init__(k=len(self.get_disease_list(data)), data=data, seed=seed)
