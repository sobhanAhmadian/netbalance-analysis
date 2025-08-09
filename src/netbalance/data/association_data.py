import math
import os
from typing import List, Union

import numpy as np

from netbalance.configs.common import RESULTS_DIR
from netbalance.data.general import Data, TrainTestSplitter
from netbalance.utils import get_header_format, prj_logger

logger = prj_logger.getLogger(__name__)


class AData(Data):

    def __init__(
        self,
        associations: np.ndarray,
        node_names: List[List[str]],
        **kwargs,
    ) -> None:
        """Initial AData

        Args:
            associations (np.ndarray): associations matrix with the shape num_associations x (num_clusters + 1)
            node_names (List[List[str]]): list of list of node names. For each cluster there should be a list
                containing node names in that cluster

        Raises:
            ValueError: associations.shape[1] - 1 should be equal to len(node_names)
        """
        super().__init__(**kwargs)
        self.associations = associations
        self.node_names = node_names

        num_clusters = associations.shape[1] - 1
        if num_clusters != len(node_names):
            raise ValueError(
                f"associations.shape[1] - 1 ({num_clusters}) should be equal to len(node_names) ({len(node_names)})"
            )

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
            balance_method (str, optional): Balance method: 'beta' or 'balanced', 'gamma' or 'entity-balanced', 'rho' or full-test or None.
                Defaults to None.
            negative_ratio (float, optional): Ratio of negative to positive samples. Defaults to 1.0.
            seed (int, optional): Random seed. Defaults to 42.
            save_name (str, optional): If provided, the result will be saved to the specified file.
            force_calculation (bool, optional): If True, the balance method will be applied regardless of the cache.
            kwargs (dict): Additional keyword arguments for specific balance methods.
        """

        logger.info(get_header_format(f"Balancing associations data ..."))
        logger.info(f"Balance Method: {balance_method}")

        if save_name is not None and not force_calculation:
            if os.path.exists(f"{RESULTS_DIR}/.cache/{save_name}.txt"):
                self.load_associations(f"{RESULTS_DIR}/.cache/{save_name}.txt")
                return

        rng = np.random.default_rng(seed)

        pos_associations = [
            asso.tolist() for asso in self.associations if asso[-1] == 1
        ]
        neg_associations = [
            asso.tolist() for asso in self.associations if asso[-1] == 0
        ]

        if balance_method is None:
            return

        # Select negative balance method
        samples = []
        if balance_method == "beta" or balance_method.lower() == "balanced":
            logger.info("Balancing Data using Beta Method")
            samples = self._beta_neg_sampling(
                pos_associations=pos_associations,
                neg_associations=neg_associations,
                negative_ratio=negative_ratio,
                rng=rng,
                **kwargs,
            )
        elif balance_method == "rho" or balance_method.lower() == "entity-balanced":
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
        elif balance_method == "eta" or balance_method.lower() == "full-test":
            samples = pos_associations + neg_associations

        # Combine and shuffle
        rng.shuffle(samples)
        self.associations = np.array(samples, dtype=np.int32)

        if save_name is not None:
            os.makedirs(f"{RESULTS_DIR}/.cache", exist_ok=True)
            file = f"{RESULTS_DIR}/.cache/{save_name}.txt"
            self.save_associations(file)

    def save_associations(self, file: str):
        """
        Save associations to a file.

        Args:
            file (str): Path to the file.
        """
        a_array = np.array(self.associations)
        np.savetxt(file, a_array, delimiter=",", fmt="%d")
        logger.info(f"Associations saved to {file}")

    def load_associations(self, file: str):
        """
        Load associations from a file.

        Args:
            file (str): Path to the file.
        """
        a_array = np.loadtxt(file, delimiter=",", dtype=np.int32)
        self.associations = a_array
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

            pos_edges = [edge for edge in current_graph if edge[-1] == 1]
            neg_edges = [edge for edge in current_graph if edge[-1] == 0]

            temp = rng.random()
            if temp < 0.5:  # Add one positive and one negative edge

                if len(current_graph) <= initial_graph_len - 2:

                    while True:
                        new_pos_edge = rng.choice(pos_associations).tolist()
                        if new_pos_edge not in pos_edges:
                            current_graph.append(new_pos_edge)
                            break

                    while True:
                        new_neg_edge = rng.choice(neg_associations).tolist()
                        if new_neg_edge not in neg_edges:
                            current_graph.append(new_neg_edge)
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

        dims = tuple([len(cluster) for cluster in self.node_names])

        # Initialize the weight matrix
        weights = np.zeros(dims, dtype=float)

        for neg_asso in neg_associations:
            for pos_asso in pos_associations:
                for i in range(len(dims)):
                    if neg_asso[i] == pos_asso[i]:
                        weights[tuple(neg_asso[:-1])] += 1.0
                        # break  # TODO remove break

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
            s = indices[selected_idx].tolist() + [0]
            neg_samples.append(s)

            # Update the weight matrix
            weights[tuple(s[:-1])] = 0  # Set the selected edge weight to 0

            for r in range(len(dims)):
                # penelize s[r] in all other dimensions
                temp_range = tuple(
                    [s[r] if i == r else slice(None) for i in range(len(dims))]
                )
                weights[temp_range] -= (weights[temp_range] > 0).astype(float)

        logger.info(f"Number of negative samples generated: {len(neg_samples)}")

        return neg_samples + pos_associations

    def _calculate_graph_score(self, associations, initial_graph_len):
        """Calculate the score for the bipartite graph."""
        graph_len = len(associations)
        per_cluster_entorpies = np.zeros(len(self.node_names))
        for cluster_id in range(len(self.node_names)):
            per_cluster_entorpies[cluster_id] = self._calculate_cluster_score(
                associations, cluster_id
            )
        ent_score = np.mean(per_cluster_entorpies).item()
        len_score = graph_len / initial_graph_len

        return ent_score, len_score

    def _calculate_cluster_score(self, associations, axis):

        if len(associations) == 0:
            return 1.0

        per_node_num_neg = np.zeros(len(self.node_names[axis]))
        per_node_num_pos = np.zeros(len(self.node_names[axis]))

        arr = np.array(associations)[:, axis]
        tar = np.array(associations)[:, -1]

        for node_id in range(len(self.node_names[axis])):
            per_node_num_neg[node_id] = np.sum((arr == node_id) & (tar == 0))
            per_node_num_pos[node_id] = np.sum((arr == node_id) & (tar == 1))

        return self.get_entropy(
            per_node_num_neg + per_node_num_pos + 1e-5,
            per_node_num_neg,
            per_node_num_pos,
        )

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

        for i in range(len(self.node_names)):
            self._calculate_node_stats(self.associations, stats[chr(i + 97)], i)

        ents = np.array(
            [stats[chr(i + 97)]["ent"].item() for i in range(len(self.node_names))]
        )
        stats["ent"] = np.mean(ents).item()

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

        stats = {
            "ent": np.zeros(1),
        }

        for i, cluster_node_names in enumerate(self.node_names):
            # convert i to ith alphabetic letter for example 1 -> a, 2 -> b
            cluster_name = chr(i + 97)
            stats[cluster_name] = init_node_stats(len(cluster_node_names))

        return stats

    def _calculate_node_stats(
        self, associations: list[list[int]], stats: dict, axis: int
    ):

        if len(associations) == 0:
            raise ValueError("Associations is empty")

        per_node_num_neg = np.zeros(len(self.node_names[axis]))
        per_node_num_pos = np.zeros(len(self.node_names[axis]))

        arr = np.array(associations)[:, axis]
        tar = np.array(associations)[:, -1]

        for node_id in range(len(self.node_names[axis])):
            per_node_num_neg[node_id] = np.sum((arr == node_id) & (tar == 0))
            per_node_num_pos[node_id] = np.sum((arr == node_id) & (tar == 1))

        per_node_total = per_node_num_neg + per_node_num_pos
        per_node_r = (per_node_num_pos + 1e-5) / (per_node_total + 1e-5)
        ent = self.get_entropy(
            per_node_total + 1e-5, per_node_num_neg, per_node_num_pos
        )

        stats["num_neg"] += per_node_num_neg
        stats["num_pos"] += per_node_num_pos
        stats["num"] += per_node_total
        stats["r"] += per_node_r
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


class BGData(AData):

    def __init__(
        self,
        associations: np.ndarray,
        cluster_a_node_names,
        cluster_b_node_names,
        **kwargs,
    ) -> None:
        super().__init__(
            associations=associations,
            node_names=[cluster_a_node_names, cluster_b_node_names],
            **kwargs,
        )
        self.cluster_a_node_names = cluster_a_node_names
        self.cluster_b_node_names = cluster_b_node_names


class TGData(AData):

    def __init__(
        self,
        associations: np.ndarray,
        cluster_a_node_names,
        cluster_b_node_names,
        cluster_c_node_names,
        **kwargs,
    ) -> None:
        super().__init__(
            associations=associations,
            node_names=[
                cluster_a_node_names,
                cluster_b_node_names,
                cluster_c_node_names,
            ],
            **kwargs,
        )
        self.cluster_a_node_names = cluster_a_node_names
        self.cluster_b_node_names = cluster_b_node_names
        self.cluster_c_node_names = cluster_c_node_names


class ATrainTestSpliter(TrainTestSplitter):

    def __init__(
        self,
        k: int,
        data: AData,
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
        subset_pos_size = int(self.data.associations[:, -1].sum() / self.k)
        subset_neg_size = subset_size - subset_pos_size
        remain_positive = [
            i for i in range(self.data_size) if self.data.associations[i, -1] == 1
        ]
        remain_negative = [
            i for i in range(self.data_size) if self.data.associations[i, -1] == 0
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

        train_data = AData(
            self.data.associations[train_indices],
            self.data.node_names,
        )
        test_data = AData(
            self.data.associations[test_indices],
            self.data.node_names,
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


class BGTrainTestSpliter(ATrainTestSpliter):

    def split(self, i):
        a_train_data, a_test_data = super().split(i)

        train_data = BGData(
            a_train_data.associations,
            a_train_data.node_names[0],
            a_train_data.node_names[1],
        )
        test_data = BGData(
            a_test_data.associations,
            a_test_data.node_names[0],
            a_test_data.node_names[1],
        )

        return train_data, test_data


class TGTrainTestSpliter(ATrainTestSpliter):

    def split(self, i):
        a_train_data, a_test_data = super().split(i)

        train_data = TGData(
            a_train_data.associations,
            a_train_data.node_names[0],
            a_train_data.node_names[1],
            a_train_data.node_names[2],
        )
        test_data = TGData(
            a_test_data.associations,
            a_test_data.node_names[0],
            a_test_data.node_names[1],
            a_test_data.node_names[2],
        )

        return train_data, test_data


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
