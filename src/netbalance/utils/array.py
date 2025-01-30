import numpy as np


def symmetrize_matrix(mat):
    return (mat + mat.T) / 2


def row_normalize(mat):
    row_sums = mat.sum(axis=1).reshape(-1, 1)
    row_sums = row_sums + (row_sums == 0)  # avoid division by zero
    return mat / row_sums


def add_identity(mat, eps=1e-5):
    return mat + eps * np.eye(mat.shape[0])


def sigmoid(x, c, d):
    """Calculate the sigmoid function for the given input.

    Parameters:
    x (float): The input value.
    c (float): The scaling factor.
    d (float): The shifting factor.

    Returns:
    float: The sigmoid value.
    """
    return 1 / (1 + np.exp(x * c + d))


def column_normalize(mat):
    col_sums = mat.sum(axis=0)
    return mat / col_sums
