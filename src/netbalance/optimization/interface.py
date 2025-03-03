import abc

from netbalance.configs import OptimizerConfig
from netbalance.data.association_data import BGData
from netbalance.evaluation import Result
from netbalance.models import AModelHandler
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class Trainer(abc.ABC):
    """Abstract base class for trainers.
    Trainers are responsible for training a model using a given dataset and configuration.

    Methods:
        train: Abstract method for training a model.
    """

    @abc.abstractmethod
    def train(
        self, model_handler: AModelHandler, data: BGData, config: OptimizerConfig
    ) -> Result:
        """
        Train the model using the given dataset and configuration.

        Args:
            model_handler (ModelHandler): The model handler object.
            data (Data): The dataset object.
            config (OptimizerConfig): The configuration object.

        Returns:
            Result: The result of the training process.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
