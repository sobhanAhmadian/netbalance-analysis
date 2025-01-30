import numpy as np

from netbalance.data.bipartite_graph_data import BGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.degree_ratio import DegreeRatioModelHandler
from netbalance.utils import get_header_format, prj_logger

from .interface import Trainer

logger = prj_logger.getLogger(__name__)


class DegreeRatioTrainer(Trainer):

    def train(
        self,
        model_handler: DegreeRatioModelHandler,
        data: BGData,
        config,
    ) -> Result:
        logger.info(get_header_format("Training the model"))
        model_handler.build(associations=data.associations)
        preds = model_handler.predict(
            a_nodes=data.associations[:, 0], b_nodes=data.associations[:, 1]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result
