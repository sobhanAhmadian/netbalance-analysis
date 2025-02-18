import numpy as np

from netbalance.configs.xgbmda import XGBMDAOptimizerConfig
from netbalance.data.association_data import BGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.xgbmda import XGBMDAModelHandler
from netbalance.utils import get_header_format, prj_logger

from .interface import Trainer

logger = prj_logger.getLogger(__name__)


class XGBMDATrainer(Trainer):

    def train(
        self,
        model_handler: XGBMDAModelHandler,
        data: BGData,
        config: XGBMDAOptimizerConfig,
    ) -> Result:
        logger.info(get_header_format("Training the model"))

        model_handler.fe.build(config)

        associations = data.associations
        md_embed = (
            model_handler.fe.extract_features(associations[:, 0], associations[:, 1])
            .detach()
            .numpy()
        )
        y = np.array(associations[:, 2].tolist(), dtype=np.float32).reshape(-1, 1)

        model_handler.model.fit(md_embed, y)

        preds = model_handler.predict(
            a_nodes=data.associations[:, 0], b_nodes=data.associations[:, 1]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result
