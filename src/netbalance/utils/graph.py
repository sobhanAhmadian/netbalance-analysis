import itertools

import networkx as nx
import numpy as np

from .logger import get_header_format
from .logger import logging as prj_logger

logger = prj_logger.getLogger(__name__)


def _cohesiveness(graph, nodes, p=0):
    """Calculate the cohesiveness of a subgraph.

    Args:
        graph (networkx.Graph): The original graph.
        nodes (list): List of nodes in the subgraph.
        p (float, optional): A parameter to adjust the cohesiveness calculation. Defaults to 0.

    Returns:
        float: The cohesiveness of the subgraph.
    """
    if len(nodes) <= 1:
        return 0
    subgraph = graph.subgraph(nodes)
    w_in = sum([_sum_of_edge_weights(subgraph, node) for node in nodes]) / 2
    w_bound = 0
    for node in nodes:
        neighbors = set(graph.neighbors(node))
        w_bound += sum([1 for neighbor in neighbors if neighbor not in nodes])
    f = w_in / (w_in + w_bound + p)
    return f


def _merge_clusters(clusters, threshold=0.8):
    """Merges clusters based on a similarity threshold.

    Args:
        clusters (list): A list of clusters, where each cluster is represented as a list of elements.
        threshold (float, optional): The similarity threshold for merging clusters. Defaults to 0.8.

    Returns:
        list: A list of merged clusters.
    """
    merged_clusters = clusters.copy()
    while True:
        scores = np.zeros((len(merged_clusters), len(merged_clusters)))
        for i in range(len(merged_clusters)):
            for j in range(i + 1, len(merged_clusters)):
                scores[i, j] = len(
                    set(merged_clusters[i]) & set(merged_clusters[j])
                ) ** 2 / (len(merged_clusters[i]) * len(merged_clusters[j]))

        max_i, max_j = np.unravel_index(np.argmax(scores), scores.shape)
        if scores[max_i, max_j] >= threshold:
            new_cluster = list(
                set(merged_clusters[max_i]) | set(merged_clusters[max_j])
            )
            temp = merged_clusters.copy()
            temp.remove(merged_clusters[max_i])
            temp.remove(merged_clusters[max_j])
            temp.append(new_cluster)
            merged_clusters = temp
        else:
            break

    return merged_clusters


def _sum_of_edge_weights(graph, node):
    # Get the edges connected to the node
    edges = graph.edges(node, data=True)
    # Sum the weights of these edges
    weight_sum = sum(data["weight"] for _, _, data in edges)
    return weight_sum


def clusterone(graph, min_density=0.3, min_size=3, overlap_threshold=0.8):
    """Clusters nodes in a graph based on the ClusterONE algorithm.

    Args:
        graph (NetworkX graph): The input graph.
        min_density (float, optional): The minimum density threshold for a cluster to be considered valid. Defaults to 0.3.
        min_size (int, optional): The minimum size threshold for a cluster to be considered valid. Defaults to 3.
        overlap_threshold (float, optional): The threshold for merging overlapping clusters. Defaults to 0.8.

    Returns:
        list: A list of clusters, where each cluster is represented as a list of nodes.

    """

    # Step 1: Seed Expansion
    logger.info(get_header_format("ClusterONE"))
    seeds = set(list(graph.nodes))
    clusters = []
    while True:
        degree_dict = {node: _sum_of_edge_weights(graph, node) for node in seeds}
        logger.info(f"degree_dict {degree_dict}")
        seed_node = max(degree_dict, key=degree_dict.get)
        logger.info(
            f"new seed node: {seed_node} with cohesiveness: {_cohesiveness(graph, [seed_node])}",
        )

        cluster = [seed_node]
        expandable = True
        while expandable:
            neighbors = set(
                itertools.chain(*[list(graph.neighbors(n)) for n in cluster])
            ) - set(cluster)
            best_node = None
            best_quality = _cohesiveness(graph, cluster)
            for neighbor in neighbors:
                new_cluster = cluster + [neighbor]
                quality = _cohesiveness(graph, new_cluster)
                if quality > best_quality:
                    best_quality = quality
                    best_node = neighbor
                    logger.info(
                        f"add {best_node} to cluster. new cohesiveness: {best_quality}"
                    )
                    break
            if best_node is not None:
                cluster.append(best_node)
            else:
                expandable = False

        shrinkable = True
        while shrinkable:
            shrinkable = False
            for node in cluster:
                new_cluster = cluster.copy()
                new_cluster.remove(node)
                quality = _cohesiveness(graph, new_cluster)
                if quality > _cohesiveness(graph, cluster):
                    logger.info(
                        f"remove {node} from cluster. new cohesiveness: {quality}"
                    )
                    cluster = new_cluster
                    shrinkable = True
                    break

        logger.info(f"new cluster: {cluster}")
        clusters.append(cluster)

        seeds = seeds - set(cluster)
        if len(seeds) <= 0:
            break

    # Step 2: Merge Overlapping Clusters
    clusters = _merge_clusters(clusters, overlap_threshold)

    # Step 3: Filter Clusters
    final_clusters = []
    for cluster in clusters:
        subgraph = graph.subgraph(cluster)
        if len(cluster) >= min_size and nx.density(subgraph) >= min_density:
            final_clusters.append(cluster)

    return final_clusters
