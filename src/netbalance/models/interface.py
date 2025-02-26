import abc

import numpy as np

from netbalance.configs import ModelConfig
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class ModelHandler(abc.ABC):
    """
    Abstract base class for model handlers.

    Args:
        model_config (ModelConfig): Configuration of the model.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.model = self._build_model()
        self.fe = self._build_feature_extractor()

    @abc.abstractmethod
    def destroy(self):
        """
        Destroys the model.
        """
        logger.info(f"Model {self.model} has been deleted.")
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, **kwargs):
        """
        Makes predictions using the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def summary(self):
        """
        Prints a summary of the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_model(self):
        """
        Builds the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_feature_extractor(self):
        """
        Builds the feature extractor.
        """
        raise NotImplementedError


class HandlerFactory(abc.ABC):
    """Abstract base class for handler factories."""

    @abc.abstractmethod
    def create_handler(self) -> ModelHandler:
        """
        Abstract method to create a ModelHandler object.

        Returns:
            ModelHandler: The created ModelHandler object.
        """
        raise NotImplementedError


class AModelHandler(ModelHandler, abc.ABC):

    def predict(self, node_lists: list[np.ndarray]):
        num_samples = len(node_lists[0])
        for i in range(1, len(node_lists)):
            if len(node_lists[i]) != num_samples:
                raise ValueError(
                    "The number of samples in all node lists should be equal."
                )

        scores = self.predict_impl(node_lists)

        if not isinstance(scores, np.ndarray):
            raise ValueError(
                "The return value of predict_impl should be a numpy array."
            )

        if scores.shape != (num_samples,):
            raise ValueError(
                "The shape of the return value of predict_impl should be equal to the number of samples."
            )

        return scores

    @abc.abstractmethod
    def predict_impl(self, node_lists: list[np.ndarray]):
        raise NotImplementedError
