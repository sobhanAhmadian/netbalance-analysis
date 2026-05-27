import numpy as np

from netbalance.data.association_data import AData


def power_law_sample(
    size: int, seed: int, alpha: float = 2.0, xmin: float = 1.0
) -> np.ndarray:
    """
    Generate samples from a continuous power-law (Pareto) distribution
    using inverse transform sampling.

    Args:
        size (int): Number of samples to generate.
        alpha (float): Power-law exponent (alpha > 1).
        xmin (float): Minimum value of the distribution.

    Returns:
        np.ndarray: Samples drawn from the power-law distribution.
    """
    if alpha <= 1:
        raise ValueError("alpha must be greater than 1.")
    if xmin <= 0:
        raise ValueError("xmin must be positive.")

    rng = np.random.default_rng(seed)

    u = rng.uniform(0, 1, size)
    samples = xmin * (1 - u) ** (-1 / (alpha - 1))

    return samples


def generate_samples(
    num_nodes_a: int,
    num_nodes_b: int,
    num_associations: int,
    positive_ratio: float,
    seed: int,
    alpha: float = 9.0,
) -> AData:
    """
    Generate an association dataset with the given parameters.


    Args:
        num_nodes_a (int): Number of nodes in set A.
        num_nodes_b (int): Number of nodes in set B.
        num_associations (int): Total number of associations to generate.
        positive_ratio (float): Ratio of positive associations (between 0 and 1).
        seed (int): Random seed for reproducibility.
        alpha (float): Power-law exponent for sampling weights.

    Returns:
        AData: Generated association dataset.
    """
    rng = np.random.default_rng(seed)

    nodes_a = [f"A{i}" for i in range(num_nodes_a)]
    nodes_b = [f"B{j}" for j in range(num_nodes_b)]

    a_weights = power_law_sample(num_nodes_a, seed=seed, alpha=alpha, xmin=1.0)
    b_weights = power_law_sample(num_nodes_b, seed=seed, alpha=alpha, xmin=1.0)
    a_weights = a_weights / np.sum(a_weights)
    b_weights = b_weights / np.sum(b_weights)

    weight_matrix = np.outer(a_weights, b_weights)
    weight_matrix = weight_matrix / np.sum(weight_matrix)

    num_positive = int(num_associations * positive_ratio)

    positive_indices = rng.choice(
        weight_matrix.size, size=num_positive, replace=False, p=weight_matrix.flatten()
    )
    associations = []
    for idx in positive_indices:
        i, j = divmod(idx, num_nodes_b)
        associations.append([int(i), int(j), 1])

    neg_weight_matrix = np.ones_like(weight_matrix)
    neg_weight_matrix[np.unravel_index(positive_indices, weight_matrix.shape)] = 0
    neg_weight_matrix = neg_weight_matrix / np.sum(neg_weight_matrix)

    num_negative = num_associations - num_positive
    negative_indices = rng.choice(
        weight_matrix.size,
        size=num_negative,
        replace=False,
        p=neg_weight_matrix.flatten(),
    )
    for idx in negative_indices:
        i, j = divmod(idx, num_nodes_b)
        associations.append([int(i), int(j), 0])

    associations = np.array(associations)

    data = AData(associations=associations, node_names=[nodes_a, nodes_b])
    return data


generate_samples(
    num_nodes_a=1000,
    num_nodes_b=1000,
    num_associations=10000,
    positive_ratio=0.5,
    alpha=1.1,
    seed=42,
)
