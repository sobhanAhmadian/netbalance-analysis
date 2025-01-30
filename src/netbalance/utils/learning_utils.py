import random


def train_test_sampler(total_samples, train_ratio=0.7, seed=0):
    rng = random.Random(seed)
    train_samples = int(train_ratio * total_samples)
    test_samples = total_samples - train_samples
    train_indices = rng.sample([i for i in range(total_samples)], train_samples)
    test_indices = rng.sample([i for i in range(total_samples)], test_samples)
    return train_indices, test_indices


def statified_train_test_sampler(
    positive_samples: int,
    negative_samples: int,
    train_ratio: float = 0.7,
    seed: int = 0,
):
    """Stratified train-test sampler

    Args:
        positive_samples (int): number of positive samples
        negative_samples (int): number of negative samples
        train_ratio (float, optional): train ratio. Defaults to 0.7.
        seed (int, optional): random seed. Defaults to 0.

    Returns:
        tuple: train_positive_indices, train_negative_indices, test_positive_indices, test_negative_indices where each one are list of indices
    """
    rng = random.Random(seed)
    train_positive_samples = int(train_ratio * positive_samples)
    train_negative_samples = int(train_ratio * negative_samples)
    test_positive_samples = positive_samples - train_positive_samples
    test_negative_samples = negative_samples - train_negative_samples
    train_positive_indices = rng.sample(
        [i for i in range(positive_samples)], train_positive_samples
    )
    train_negative_indices = rng.sample(
        [i for i in range(negative_samples)], train_negative_samples
    )
    test_positive_indices = rng.sample(
        [i for i in range(positive_samples)], test_positive_samples
    )
    test_negative_indices = rng.sample(
        [i for i in range(negative_samples)], test_negative_samples
    )
    return (
        train_positive_indices,
        train_negative_indices,
        test_positive_indices,
        test_negative_indices,
    )
