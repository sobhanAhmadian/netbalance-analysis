from .array import symmetrize_matrix, row_normalize, add_identity


def similarity_network_fusion(sim1, sim2, k, t):
    """Combines two similarity matrices using network fusion.

    Based on the paper "BRWMDA:Predicting microbe-disease associations
        based on similarities and bi-random walk on disease and microbe networks"

    Args:
        sim1 (numpy.ndarray): The first similarity matrix.
        sim2 (numpy.ndarray): The second similarity matrix.
        k (int): The number of nearest neighbors to consider.
        t (int): The number of iterations for network fusion.

    Returns:
        numpy.ndarray: The fused similarity matrix.
    """
    sim1 = symmetrize_matrix(sim1)
    sim2 = symmetrize_matrix(sim2)

    sim1 = add_identity(sim1)
    sim2 = add_identity(sim2)

    p1 = row_normalize(sim1)
    p2 = row_normalize(sim2)

    s1 = get_knn_kernel(p1, k)
    s2 = get_knn_kernel(p2, k)

    for i in range(t):
        p1_new = s1 @ p2 @ s1.T
        p2_new = s2 @ p1 @ s2.T

        p1 = row_normalize(p1_new)
        p2 = row_normalize(p2_new)

    return (p1 + p2) / 2


def get_knn_kernel(sim, k, normalize=True):
    """Calculates the k-nearest neighbors kernel matrix based on a similarity matrix.

    For each row, the top k elements are selected and the rest are set to zero. Then the
        matrix is row-normalized.

    Args:
        sim (numpy.ndarray): The similarity matrix.
        k (int): The number of nearest neighbors to consider.
        normalize (bool): whether to normalize output

    Returns:
        numpy.ndarray: The k-nearest neighbors kernel matrix.
    """
    w = sim.argsort(axis=1).argsort(axis=1) + 1  # ranks of each element in each row
    w = (w > sim.shape[1] - k).astype(int)  # select top k elements
    
    if normalize:
        return row_normalize(w * sim)
    else:
        return w * sim
