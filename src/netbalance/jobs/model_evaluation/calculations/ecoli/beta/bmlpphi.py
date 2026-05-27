import os

from netbalance.configs.bmlpphi import BMLPPHI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.configs.bmlpphi import BMLPPHIModelConfig as ModelConfig  # Parameter
from netbalance.configs.bmlpphi import (
    BMLPPHIOptimizerConfig as OptimizerConfig,
)  # Parameter
from netbalance.data.association_data import BGData, BGTrainTestSpliter
from netbalance.evaluation import repeated_cross_validation
from netbalance.features.ecoli import EcoliDataset as Dataset  # Parameter
from netbalance.models.bmlpphi import (
    BMLPPHIHandlerFactory as HandlerFactory,
)  # Parameter
from netbalance.optimization.bmlpphi import BMLPPHITrainer as Trainer  # Parameter
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

model_name = "bmlpphi"  # Parameter
dataset = "ecoli"  # Parameter
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

model_config = ModelConfig()

optimizer_config = OptimizerConfig()  # Parameter
optimizer_config.n_epoch = 200
optimizer_config.lr = 0.001
optimizer_config.save = True
optimizer_config.save_path = os.path.join(
    RESULTS_DIR,
    f"models",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)
os.makedirs(optimizer_config.save_path, exist_ok=True)
optimizer_config.save_path = os.path.join(
    optimizer_config.save_path,
    f"n_epoch_{optimizer_config.n_epoch}_lr_{optimizer_config.lr}",
)


def get_data():
    return BGData(
        associations=ds.get_associations(with_negatives=False),
        cluster_a_node_names=ds.get_cluster_a_node_names(),
        cluster_b_node_names=ds.get_cluster_b_node_names(),
    )


associations = ds.get_associations(with_negatives=False)

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
