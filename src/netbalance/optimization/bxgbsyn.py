import numpy as np

from netbalance.configs.bxgbsyn import BXGBSYNOptimizerConfig
from netbalance.data.association_data import TGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.bxgbsyn import BXGBSYNModelHandler
from netbalance.utils import get_header_format, prj_logger

from .interface import Trainer

logger = prj_logger.getLogger(__name__)


class BXGBSYNTrainer(Trainer):

    def train(
        self,
        model_handler: BXGBSYNModelHandler,
        data: TGData,
        config: BXGBSYNOptimizerConfig,
    ) -> Result:
        logger.info(get_header_format("Training the model"))
        model_handler.fe.build()

        associations = data.associations
        dp_embed = model_handler.fe.extract_features(
            associations[:, 0], associations[:, 1], associations[:, 2]
        )
        y = np.array(associations[:, -1].tolist(), dtype=np.float32).reshape(-1, 1)
        model_handler.model.fit(dp_embed, y)

        preds = model_handler.predict(
            [data.associations[:, i] for i in range(data.associations.shape[1] - 1)]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, -1], preds.reshape(-1), config.threshold
        )
        return result
