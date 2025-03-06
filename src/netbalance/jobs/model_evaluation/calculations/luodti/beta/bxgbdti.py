import os

import torch

from netbalance.configs.bxgbdti import BXGBDTI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.configs.bxgbdti import BXGBDTIModelConfig as ModelConfig  # Parameter
from netbalance.configs.bxgbdti import (
    BXGBDTIOptimizerConfig as OptimizerConfig,
)  # Parameter
from netbalance.data.association_data import BGData, BGTrainTestSpliter
from netbalance.evaluation import repeated_cross_validation
from netbalance.features.luodti import LuoDTIDataset as Dataset  # Parameter
from netbalance.models.bxgbdti import (
    BXGBDTIHandlerFactory as HandlerFactory,
)  # Parameter
from netbalance.optimization.bxgbdti import BXGBDTITrainer as Trainer  # Parameter
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

model_name = "bxgbdti"  # Parameter
dataset = "luodti"  # Parameter
train_neg_samp_method = "beta"  # Parameter

num_cross_validation = 5  # Parameter

splitter_kwargs = {
    "k": 5,
    "train_balance": True,  # Parameter
    "train_balance_kwargs": {
        "balance_method": train_neg_samp_method,
        "negative_ratio": 1.0,  # Parameter
    },
}

model_result_dir = os.path.join(
    RESULTS_DIR,
    f"preds",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)

logger.info(
    f">>>>>>>>>>>>>>>>> Job: Model Evaluation - {dataset} - {model_name} - {train_neg_samp_method}"
)

ds = Dataset()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_config = ModelConfig()
model_config.drug_num = len(ds.get_cluster_a_node_names())
model_config.protein_num = len(ds.get_cluster_b_node_names())

optimizer_config = OptimizerConfig()  # Parameter
optimizer_config.n_epoch = 2000
optimizer_config.device = device
optimizer_config.es_patience = 50
optimizer_config.fair = True


def get_data():
    return BGData(
        associations=ds.get_associations(with_negatives=True),
        cluster_a_node_names=ds.get_cluster_a_node_names(),
        cluster_b_node_names=ds.get_cluster_b_node_names(),
    )


associations = ds.get_associations(with_negatives=True)

trainer = Trainer()
factory = HandlerFactory(model_config=model_config)  # Parameter

if __name__ == "__main__":
    repeated_cross_validation(
        get_data=get_data,
        SplitterClass=BGTrainTestSpliter,
        handler_factory=factory,
        trainer=trainer,
        optimizer_config=optimizer_config,
        num_cross_validation=num_cross_validation,
        save_preds_dir=model_result_dir,
        splitter_kwargs=splitter_kwargs,
    )
