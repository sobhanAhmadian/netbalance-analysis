import copy
import os
from itertools import combinations
from typing import Callable, List

import numpy as np
from tqdm import tqdm

from netbalance.data.association_graph_data import BGData, BGTrainTestSpliter
from netbalance.features.bipartite_graph_dataset import BGDataset
from netbalance.visualization import plot_per_group_associations

from .logger import logging as prj_logger

logger = prj_logger.getLogger(__name__)


def analyse_datasest(
    dataset: BGDataset,
    dataset_name: str,
    figs_folder: str,
    num_cross_validation: int,
    num_negative_sampling: int,
    k: int,
    test_balance_method: str = "beta",
    test_balance_kwargs: dict = {},
    test_balance_negative_ratio: float = 1.0,
    c_pos: str = "#a2d2ff",
    c_neg: str = "#ffafcc",
    summary_size: int = 40,
) -> None:
    """Analyse the dataset.
    This function will print and plot the statistics of the dataset including entropy, pairwise average similarity, and per node stats.

    Args:
        dataset (BGDataset): The dataset to be analysed.
        dataset_name (str): The name of the dataset.
        figs_folder (str): The path to the folder where the figures will be saved.
        num_cross_validation (int): The number of cross validations.
        num_negative_sampling (int): The number of negative samplings.
        k (int): The number of folds.
        test_balance_method (str, optional): The negative sampling method. Defaults to "beta".
        test_balance_kwargs (dict, optional): The negative sampling method arguments. Defaults to {}.
        test_balance_negative_ratio (float, optional): The negative ratio. Defaults to 1.0.
        c_pos (str, optional): Color for positive edges. Defaults to "#a2d2ff".
        c_neg (str, optional): Color for negative edges. Defaults to "#ffafcc".
        summary_size (int, optional): The number of nodes to show in the summary plot. Defaults to 40.
    """
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder, exist_ok=True)

    def get_data():
        return BGData(
            associations=dataset.get_associations(with_negatives=True),
            cluster_a_node_names=dataset.get_cluster_a_node_names(),
            cluster_b_node_names=dataset.get_cluster_b_node_names(),
        )

    data_list = get_balanced_test_data_list(
        get_data,
        dataset_name,
        num_cross_validation,
        k,
        num_negative_sampling,
        test_balance_method,
        test_balance_kwargs,
        test_balance_negative_ratio,
    )

    test_stats = get_ave_stats(
        data_list=data_list,
        cluster_a_node_names=dataset.get_cluster_a_node_names(),
        cluster_b_node_names=dataset.get_cluster_b_node_names(),
    )

    print("\n>> Entropy")
    print(f"Cluster B ({dataset.cluster_b_name}) Entropy: {test_stats["b"]["ent"]}")
    print(f"Cluster A ({dataset.cluster_a_name}) Entropy: {test_stats["a"]["ent"]}")
    print(f"Mean of Two Entropies: {test_stats["ent"]}")

    print("\n>> Pairwise Average Similarity")
    print(f"Pairwise Average Similarity: {test_stats["pas"]}")
    print(f"Pairwise Average Similarity (Pos Edges): {test_stats["pas_pos"]}")
    print(f"Pairwise Average Similarity (Neg Edges): {test_stats["pas_neg"]}")
    print(f"Average Graph Size: {test_stats['ave_graph_size']}")

    print("\n>> Cluster A Per Node Stats")
    plot_per_group_associations(
        figs_folder=figs_folder,
        node_names=dataset.get_cluster_a_node_names(),
        cluster_name=dataset.cluster_a_name,
        num_list=test_stats["a"]["num"],
        num_pos_list=test_stats["a"]["num_pos"],
        c_pos=c_pos,
        c_neg=c_neg,
    )

    print("\n>> Cluster B Per Node Stats")
    plot_per_group_associations(
        figs_folder=figs_folder,
        node_names=dataset.get_cluster_b_node_names(),
        cluster_name=dataset.cluster_b_name,
        num_list=test_stats["b"]["num"],
        num_pos_list=test_stats["b"]["num_pos"],
        c_pos=c_pos,
        c_neg=c_neg,
    )

    print("\n>> Cluster A Per Node Stats (Summary)")
    plot_per_group_associations(
        figs_folder=figs_folder,
        node_names=dataset.get_cluster_a_node_names(),
        cluster_name=dataset.cluster_a_name,
        num_list=test_stats["a"]["num"],
        num_pos_list=test_stats["a"]["num_pos"],
        c_pos=c_pos,
        c_neg=c_neg,
        max_k=summary_size,
    )

    print("\n>> Cluster B Per Node Stats")
    plot_per_group_associations(
        figs_folder=figs_folder,
        node_names=dataset.get_cluster_b_node_names(),
        cluster_name=dataset.cluster_b_name,
        num_list=test_stats["b"]["num"],
        num_pos_list=test_stats["b"]["num_pos"],
        c_pos=c_pos,
        c_neg=c_neg,
        max_k=summary_size,
    )


def get_balanced_test_data_list(
    get_data: Callable,
    dataset_name: str,
    num_cross_validation: int = 5,
    k: int = 5,
    num_negative_sampling: int = 5,
    test_balance_method: str = "beta",
    test_balance_kwargs: dict = {},
    test_balance_negative_ratio: float = 1.0,
) -> List[BGData]:
    """Return a list of balanced test data by repeated cross validation.

    Args:
        get_data (Callable): A function that returns a BGData object.
        num_cross_validation (int, optional): number of cross validation. Defaults to 5.
        k (int, optional): number of folds. Defaults to 5.
        num_negative_sampling (int, optional): number of negative sampling. Defaults to 5.
        test_balance_method (str, optional): negative sampling method. Defaults to "beta".
        test_balance_kwargs (dict, optional): negative sampling method arguments. Defaults to {}.
        test_balance_negative_ratio (float, optional): negative ratio. Defaults to 1.0.

    Returns:
        List[BGData]: A list of balanced test data.
    """
    data_list = []
    with tqdm(
        total=num_cross_validation * k * num_negative_sampling,
        desc="Repeated Cross Validation",
    ) as pbar:
        for i in range(num_cross_validation):
            data = get_data()
            spliter = BGTrainTestSpliter(k=k, data=data, seed=i)
            for j in range(k):
                _, test_data = spliter.split(j)
                for l in range(num_negative_sampling):
                    save_name = f"dataset_{dataset_name}_{"test"}_cv_{i + 1}_fold_{j + 1}_neg_{l + 1}"
                    save_name += (
                        f"_met_{test_balance_method}_rat_{test_balance_negative_ratio}"
                    )
                    for key, value in test_balance_kwargs.items():
                        save_name += f"_{key}_{value}"
                    temp_test_data = copy.deepcopy(test_data)
                    temp_test_data.balance_data(
                        balance_method=test_balance_method,
                        negative_ratio=test_balance_negative_ratio,
                        seed=l,
                        save_name=save_name,
                        **test_balance_kwargs,
                    )
                    data_list.append(temp_test_data)
                    pbar.update(1)
    return data_list


def get_ave_stats(
    data_list: List[BGData],
    cluster_a_node_names: list,
    cluster_b_node_names: list,
) -> dict:
    """
    Calculate average statistics of n bipartite graph datasets.

    Args:
        data_list (List[BGData]): A list of BGData objects.
        cluster_a_node_names (list): A list of cluster A node names.
        cluster_b_node_names (list): A list of cluster B node names.

    Returns:
        dict: A nested dictionary of statistics for the dataset.
    """

    stats = _initialize_stats(cluster_a_node_names, cluster_b_node_names)

    graph_list = [data.associations for data in data_list]
    stats_list = [data.get_stats() for data in data_list]

    _aggregate_stats(stats, stats_list)

    # Add Pairwise Average Similarity between bipartite graphs
    pos_graph_list = [[e for e in g if e[2] == 1] for g in graph_list]
    neg_graph_list = [[e for e in g if e[2] == 0] for g in graph_list]
    stats["pas"] = _pairwise_average_similarity(graph_list)
    stats["pas_pos"] = _pairwise_average_similarity(pos_graph_list)
    stats["pas_neg"] = _pairwise_average_similarity(neg_graph_list)

    graph_sizes = [len(g) for g in graph_list]
    stats["ave_graph_size"] = np.mean(graph_sizes)

    return stats


def _initialize_stats(cluster_a_node_names, cluster_b_node_names) -> dict:
    """Initialize the statistics structure for the dataset."""

    def init_node_stats(size: int):
        return {
            "num_neg": np.zeros(size),
            "num_pos": np.zeros(size),
            "num": np.zeros(size),
            "r": np.zeros(size),
            "ent": np.zeros(1),
        }

    cluster_a_size = len(cluster_a_node_names)
    cluster_b_size = len(cluster_b_node_names)

    stats = {
        "ent": np.zeros(1),
        "a": init_node_stats(cluster_a_size),
        "b": init_node_stats(cluster_b_size),
    }
    return stats


def _aggregate_stats(stats, stats_list):
    """Aggregate the statistics from multiple datasets."""
    for s in stats_list:
        stats["ent"] += s["ent"]
        for node_type in ["a", "b"]:
            stats[node_type]["num_neg"] += s[node_type]["num_neg"]
            stats[node_type]["num_pos"] += s[node_type]["num_pos"]
            stats[node_type]["num"] += s[node_type]["num"]
            stats[node_type]["r"] += s[node_type]["r"]
            stats[node_type]["ent"] += s[node_type]["ent"]

    stats["ent"] /= len(stats_list)
    for node_type in ["a", "b"]:
        stats[node_type]["num_neg"] /= len(stats_list)
        stats[node_type]["num_pos"] /= len(stats_list)
        stats[node_type]["num"] /= len(stats_list)
        stats[node_type]["r"] /= len(stats_list)
        stats[node_type]["ent"] /= len(stats_list)


def _pairwise_average_similarity(lists):
    """
    Calculate the pairwise average Jaccard similarity for a list of lists
    containing nested lists (e.g., 3-element lists).

    Parameters:
        lists (list of lists of lists): The input lists, where each child is a 3-element list.

    Returns:
        float: The average Jaccard similarity.
    """
    # Convert child lists (e.g., [1, 2, 3]) into tuples to use as set elements
    sets = [set([tuple(sub_list) for sub_list in lst]) for lst in lists]

    # Generate all possible pairs of sets
    pairs = combinations(sets, 2)

    # Compute Jaccard similarity for each pair
    similarities = [len(a & b) / len(a | b) if len(a | b) > 0 else 0 for a, b in pairs]

    # Compute the average similarity
    return sum(similarities) / len(similarities) if similarities else 0
