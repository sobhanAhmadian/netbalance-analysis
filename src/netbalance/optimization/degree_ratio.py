import numpy as np

from netbalance.data.association_data import BGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.degree_ratio import ADegreeRatioModelHandler
from netbalance.utils import get_header_format, prj_logger

from .interface import Trainer

logger = prj_logger.getLogger(__name__)


class DegreeRatioTrainer(Trainer):

    def train(
        self,
        model_handler: ADegreeRatioModelHandler,
        data: BGData,
        config,
    ) -> Result:
        logger.info(get_header_format("Training the model"))
        model_handler.build(associations=data.associations)
        preds = model_handler.predict(
            [data.associations[:, i] for i in range(data.associations.shape[1] - 1)]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, -1], preds.reshape(-1), config.threshold
        )
        return result
