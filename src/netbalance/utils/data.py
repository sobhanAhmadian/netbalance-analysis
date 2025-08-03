import copy
import os
from itertools import combinations
from typing import Callable, List

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt
from netbalance.configs.common import RESULTS_DIR

from netbalance.data.association_data import AData, ATrainTestSpliter
from netbalance.features.bipartite_graph_dataset import ADataset
from netbalance.visualization import plot_per_group_associations

from .logger import logging as prj_logger

logger = prj_logger.getLogger(__name__)

load_dotenv()


def analyse_datasest(
    dataset: ADataset,
    dataset_name: str,
    figs_folder: str,
    num_cross_validation: int,
    num_negative_sampling: int,
    k: int,
    test_balance_method: str = "beta",
    test_balance_kwargs: dict = {},
    test_balance_negative_ratio: float = 1.0,
    c_pos: str = "#66c2a5c5",
    c_neg: str = "#d53e50b5",
    summary_size: int = 30,
    with_negatives: bool = True,
) -> None:
    """Analyse the dataset.

    This function will print and plot the statistics of the dataset including entropy,
        pairwise average similarity, and per node stattistics.

    Args:7
        dataset (ADataset): The dataset to be analysed.
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
        with_negatives (bool, optional): Whether to generate negative edges. Defaults to True.
    """
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder, exist_ok=True)

    def get_data():
        return AData(
            associations=dataset.get_associations(with_negatives=with_negatives),
            node_names=dataset.get_node_names(),
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
        node_names=dataset.get_node_names(),
    )

    print("\n>> Entropy")
    for i in range(len(dataset.cluster_names)):
        symb = str(chr(97 + i))
        print(
            f"Cluster {symb.upper()} ({dataset.cluster_names[i]}) Entropy: {test_stats[symb]["ent"]}"
        )
    print(f"Mean of Two Entropies: {test_stats["ent"]}")

    print("\n>> Pairwise Average Similarity")
    print(f"Pairwise Average Similarity: {test_stats["pas"]}")
    print(f"Pairwise Average Similarity (Pos Edges): {test_stats["pas_pos"]}")
    print(f"Pairwise Average Similarity (Neg Edges): {test_stats["pas_neg"]}")
    print(f"Average Graph Size: {test_stats['ave_graph_size']}")

    for i in range(len(dataset.cluster_names)):
        symb = str(chr(97 + i))
        print(f"\n>> Cluster {symb.upper()} Per Node Stats")
        plot_per_group_associations(
            figs_folder=figs_folder,
            node_names=dataset.get_node_names()[i],
            cluster_name=dataset.cluster_names[i],
            num_list=test_stats[symb]["num"],
            num_pos_list=test_stats[symb]["num_pos"],
            c_pos=c_pos,
            c_neg=c_neg,
        )

        print(f"\n>> Cluster {symb.upper()} Per Node Stats (Summary)")
        plot_per_group_associations(
            figs_folder=figs_folder,
            node_names=dataset.get_node_names()[i],
            cluster_name=dataset.cluster_names[i],
            num_list=test_stats[symb]["num"],
            num_pos_list=test_stats[symb]["num_pos"],
            c_pos=c_pos,
            c_neg=c_neg,
            max_k=summary_size,
        )

        ratios = (test_stats[symb]["num_pos"]) / (test_stats[symb]["num"])
        dir_path = (
            f"{RESULTS_DIR}/numeric/data_analysis/{dataset_name}/{test_balance_method}"
        )
        os.makedirs(dir_path, exist_ok=True)
        filename = "ratios_" + dataset.cluster_names[i] + ".txt"
        np.savetxt(f"{dir_path}/{filename}", ratios, delimiter=",")
        print(f"\nSaved ratios list to {dir_path}/{filename}")

        file_name = "num_pos_" + dataset.cluster_names[i] + ".txt"
        np.savetxt(
            f"{dir_path}/{file_name}", test_stats[symb]["num_pos"], delimiter=","
        )
        print(f"Saved num_pos list to {dir_path}/{file_name}")

        file_name = "num_" + dataset.cluster_names[i] + ".txt"
        np.savetxt(f"{dir_path}/{file_name}", test_stats[symb]["num"], delimiter=",")
        print(f"Saved num list to {dir_path}/{file_name}")


def get_balanced_test_data_list(
    get_data: Callable,
    dataset_name: str,
    num_cross_validation: int = 5,
    k: int = 5,
    num_negative_sampling: int = 5,
    test_balance_method: str = "beta",
    test_balance_kwargs: dict = {},
    test_balance_negative_ratio: float = 1.0,
) -> List[AData]:
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
    with (
        tqdm(
            total=num_cross_validation * k * num_negative_sampling,
            desc="Repeated Cross Validation",
        ) as pbar,
    ):
        for i in range(num_cross_validation):
            data = get_data()
            spliter = ATrainTestSpliter(k=k, data=data, seed=i)
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

                    def process_data(data, *args, **kwargs):
                        data.balance_data(*args, **kwargs)
                        return data

                    data_list.append(
                        process_data(
                            temp_test_data,
                            balance_method=test_balance_method,
                            negative_ratio=test_balance_negative_ratio,
                            seed=l,
                            save_name=save_name,
                            **test_balance_kwargs,
                        )
                    )
                    pbar.update(1)

    return data_list


def get_ave_stats(data_list: List[AData], node_names: List[List[str]]) -> dict:
    """
    Calculate average statistics of n bipartite graph datasets.

    Args:
        data_list (List[AData]): A list of AData objects.
        cluster_a_node_names (list): A list of cluster A node names.
        cluster_b_node_names (list): A list of cluster B node names.

    Returns:
        dict: A nested dictionary of statistics for the dataset.
    """

    stats = _initialize_stats(node_names)

    graph_list = [data.associations for data in data_list]
    stats_list = [data.get_stats() for data in data_list]

    _aggregate_stats(stats, stats_list)

    # Add Pairwise Average Similarity between bipartite graphs
    pos_graph_list = [[e for e in g if e[-1] == 1] for g in graph_list]
    neg_graph_list = [[e for e in g if e[-1] == 0] for g in graph_list]
    stats["pas"] = _pairwise_average_similarity(graph_list)
    stats["pas_pos"] = _pairwise_average_similarity(pos_graph_list)
    stats["pas_neg"] = _pairwise_average_similarity(neg_graph_list)

    graph_sizes = [len(g) for g in graph_list]
    stats["ave_graph_size"] = np.mean(graph_sizes)

    return stats


def _initialize_stats(node_names: List[List[str]]) -> dict:
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

    for i, cluster_node_names in enumerate(node_names):
        # convert i to ith alphabetic letter for example 1 -> a, 2 -> b
        cluster_name = chr(i + 97)
        stats[cluster_name] = init_node_stats(len(cluster_node_names))

    return stats


def _aggregate_stats(stats: dict, stats_list: List[dict]):
    """Aggregate the statistics from multiple datasets."""
    node_types = list(stats.keys())
    node_types.remove("ent")

    for s in stats_list:
        stats["ent"] += s["ent"]
        for node_type in node_types:
            stats[node_type]["num_neg"] += s[node_type]["num_neg"]
            stats[node_type]["num_pos"] += s[node_type]["num_pos"]
            stats[node_type]["num"] += s[node_type]["num"]
            stats[node_type]["r"] += s[node_type]["r"]
            stats[node_type]["ent"] += s[node_type]["ent"]

    stats["ent"] /= len(stats_list)
    for node_type in node_types:
        stats[node_type]["num_neg"] /= len(stats_list)
        stats[node_type]["num_pos"] /= len(stats_list)
        stats[node_type]["num"] /= len(stats_list)
        stats[node_type]["r"] /= len(stats_list)
        stats[node_type]["ent"] /= len(stats_list)


def _pairwise_average_similarity(lists: List[List[int]]):
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
