import abc
import random

import torch

from netbalance.utils.logger import logging as base_logger

logger = base_logger.getLogger(__name__)


class Data(abc.ABC):
    """
    Abstract base class for data objects.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __len__(self):
        raise NotImplementedError


class TrainTestSplitter(abc.ABC):
    """
    Abstract base class for train-test splitters in cross-validation.

    Args:
        k (int): number of folds.
        data (Data): data object.
    """

    def __init__(self, k: int, data: Data, seed=42):
        self.rng = random.Random(seed)  # rng should be create before get_subsets
        self.k = k
        self.data = data
        self.data_size = self.get_data_size()
        self.subsets = self.get_subsets()

    @abc.abstractmethod
    def split(self, i):
        """
        Split the data into train and test sets for the i-th fold.

        Args:
            i (int): The index of the fold.

        Returns:
            tuple: A tuple containing the train and test data. <train_data, test_data>
        """
        logger.info("{:#^50}".format(f"  Splitting Fold {i + 1}   "))
        raise NotImplementedError

    @abc.abstractmethod
    def get_data_size(self):
        """
        Get the size of the data. This will be used to calculate the size of the subsets.

        Returns:
            int: The size of the data.
        """
        raise NotImplementedError

    def get_subsets(self):
        """
        Generate subsets of data.

        Returns:
            dict: A dictionary containing subsets of data, where the keys represent the subset index and the values
                    represent the indices of the elements in the subset.
        """
        subsets = dict()
        subset_size = int(self.data_size / self.k)
        remain = list(range(0, self.data_size))
        for i in range(self.k - 1):
            subsets[i] = self.rng.sample(remain, subset_size)
            remain = list(set(remain).difference(subsets[i]))
        subsets[self.k - 1] = remain
        return subsets


class PytorchData(Data):
    """
    A class representing PyTorch data.

    Attributes:
        X (torch.Tensor): The input data.
        y (torch.Tensor): The target data.

    Args:
        X (torch.Tensor): The input data.
        y (torch.Tensor): The target data.
        **kwargs: Additional keyword arguments.

    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]


class PytorchTrainTestSplitter(TrainTestSplitter):
    """
    A class that splits PyTorch data into train and test sets.

    Args:
        k (int): The number of subsets to create during the split.
        data (PytorchData): The PyTorch data object to be split.

    Attributes:
        k (int): The number of subsets to create during the split.
        data (PytorchData): The PyTorch data object to be split.

    Methods:
        split(i): Splits the data into train and test sets based on the i-th subset.
    """

    def __init__(self, k: int, data: PytorchData):
        super().__init__(k, data)

    def split(self, i):
        """
        Splits the data into train and test sets based on the i-th subset.

        Args:
            i (int): The index of the subset to use for the test set.

        Returns:
            tuple: A tuple containing the train and test data objects.

        """
        indices = set(range(0, self.data_size))
        test_indices = list(self.subsets[i])
        train_indices = list(indices.difference(self.subsets[i]))

        train_data = PytorchData(self.data.X[train_indices], self.data.y[train_indices])
        test_data = PytorchData(self.data.X[test_indices], self.data.y[test_indices])
        return train_data, test_data

    def get_data_size(self):
        return self.data.X.shape[0]
