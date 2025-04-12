import torch

from netbalance.configs.nvmda import NVMDAOptimizerConfig
from netbalance.data import PytorchData
from netbalance.data.association_data import BGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.nvmda import NVMDAModelHandler
from netbalance.utils import get_header_format, prj_logger

from .interface import Trainer
from .simple_pytorch import PytorchTrainer

logger = prj_logger.getLogger(__name__)


class NVMDATrainer(Trainer):

    def train(
        self,
        model_handler: NVMDAModelHandler,
        data: BGData,
        config: NVMDAOptimizerConfig,
    ) -> Result:
        logger.info(get_header_format("Training the model"))

        model_handler.fe.build(config)

        associations = data.associations
        md_embed = model_handler.fe.extract_features(
            associations[:, 0], associations[:, 1]
        ).to(config.device)
        y = (
            torch.tensor(associations[:, 2].tolist(), dtype=torch.float32)
            .reshape(-1, 1)
            .to(config.device)
        )
        simple_data = PytorchData(X=md_embed, y=y)

        pytorch_trainer = PytorchTrainer()
        pytorch_trainer.train(model_handler, simple_data, config)

        preds = model_handler.predict(
            [data.associations[:, 0], data.associations[:, 1]]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result
