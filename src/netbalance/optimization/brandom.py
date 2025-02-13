from netbalance.data.bipartite_graph_data import BGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.brandom import BRandomModelHandler
from netbalance.utils import prj_logger

from .interface import Trainer

logger = prj_logger.getLogger(__name__)


class BRandomTrainer(Trainer):

    def train(
        self,
        model_handler: BRandomModelHandler,
        data: BGData,
        config,
    ) -> Result:
        preds = model_handler.predict(
            a_nodes=data.associations[:, 0], b_nodes=data.associations[:, 1]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result
