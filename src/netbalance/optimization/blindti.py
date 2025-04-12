import numpy as np

from netbalance.configs.blindti import BLINDTIOptimizerConfig
from netbalance.data.association_data import BGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.blindti import BLINDTIModelHandler
from netbalance.utils import get_header_format, prj_logger

from .interface import Trainer

logger = prj_logger.getLogger(__name__)


class BLINDTITrainer(Trainer):

    def train(
        self,
        model_handler: BLINDTIModelHandler,
        data: BGData,
        config: BLINDTIOptimizerConfig,
    ) -> Result:
        logger.info(get_header_format("Training the model"))

        if config.fair:
            model_handler.fe.build(data.associations)
        else:
            model_handler.fe.build()

        associations = data.associations
        dp_embed = model_handler.fe.extract_features(
            associations[:, 0], associations[:, 1]
        ).numpy()
        y = np.array(associations[:, 2].tolist(), dtype=np.float32).reshape(-1)
        model_handler.model.fit(dp_embed, y)

        preds = model_handler.predict(
            [data.associations[:, 0], data.associations[:, 1]]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result
