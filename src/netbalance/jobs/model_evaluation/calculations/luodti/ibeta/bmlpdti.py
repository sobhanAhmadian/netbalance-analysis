import os

from netbalance.configs.bmlpdti import BMLPDTI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.configs.bmlpdti import BMLPDTIModelConfig as ModelConfig  # Parameter
from netbalance.configs.bmlpdti import (
    BMLPDTIOptimizerConfig as OptimizerConfig,
)  # Parameter
from netbalance.data.association_data import BGData, BGTrainTestSpliter
from netbalance.evaluation import repeated_cross_validation
from netbalance.features.luodti import LuoDTIDataset as Dataset  # Parameter
from netbalance.models.bmlpdti import (
    BMLPDTIHandlerFactory as HandlerFactory,
)  # Parameter
from netbalance.optimization.bmlpdti import (
    BalanceBMLPDTITrainer as Trainer,
)  # Parameter
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

model_name = "bmlpdti"  # Parameter
dataset = "luodti"  # Parameter
train_neg_samp_method = "ibeta"  # Parameter

num_cross_validation = 5  # Parameter

splitter_kwargs = {
    "k": 5,
    "train_balance": False,  # Parameter
    "train_balance_kwargs": {},  # Parameter
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
model_config.drug_num = len(ds.get_cluster_a_node_names())
model_config.protein_num = len(ds.get_cluster_b_node_names())
model_config.input_dim = len(ds.get_cluster_a_node_names()) + len(
    ds.get_cluster_b_node_names()
)
model_config.hidden_dim = 64

optimizer_config = OptimizerConfig()  # Parameter
optimizer_config.i_balance_method = "beta"
optimizer_config.i_balance_kwargs = {}
optimizer_config.i_negative_ratio = 1.0
optimizer_config.i_max_num_bal = 30
optimizer_config.fair = True
optimizer_config.n_epoch = 600
optimizer_config.lr = 0.001


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
