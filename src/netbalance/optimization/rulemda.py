from netbalance.data.association_data import BGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.rulemda import RULEMDAHandlerFactory
from netbalance.utils import get_header_format, prj_logger

from .interface import Trainer

logger = prj_logger.getLogger(__name__)


class RULEMDATrainer(Trainer):

    def train(
        self,
        model_handler: RULEMDAHandlerFactory,
        data: BGData,
        config,
    ) -> Result:
        logger.info(get_header_format("Training the model"))
        model_handler.build(associations=data.associations)
        preds = model_handler.predict(dis_array=data.associations[:, 1])
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result
