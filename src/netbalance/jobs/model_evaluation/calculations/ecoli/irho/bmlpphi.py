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
from netbalance.optimization.bmlpphi import (
    BalanceBMLPPHITrainer as Trainer,
)  # Parameter
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

model_name = "bmlpphi"  # Parameter
dataset = "ecoli"  # Parameter
train_neg_samp_method = "irho"  # Parameter

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
model_config.hidden_dim = 256
model_config.output_dim = 1
model_config.num_layers = 3
model_config.dropout = 0.1

optimizer_config = OptimizerConfig()  # Parameter
optimizer_config.i_balance_method = "rho"
optimizer_config.i_balance_kwargs = {
    "max_iter": 20000,
    "delta": 0.1,
    "cooling_rate": 0.99,
    "initial_temp": 20.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}
optimizer_config.i_negative_ratio = 1.0
optimizer_config.i_max_num_bal = 30
optimizer_config.i_parallel_balance = True
optimizer_config.n_epoch = 200
optimizer_config.lr = 0.0001
optimizer_config.save = True
optimizer_config.save_path = os.path.join(
    RESULTS_DIR,
    f"models",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)
os.makedirs(optimizer_config.save_path, exist_ok=True)
name_suffix = (
    f"n_epoch_{optimizer_config.n_epoch}_lr_{optimizer_config.lr}_"
    + f"i_max_num_bal_{optimizer_config.i_max_num_bal}"
)
for key, value in optimizer_config.i_balance_kwargs.items():
    name_suffix += f"_i_balance_kwarg_{key}_{value}"
optimizer_config.save_path = os.path.join(optimizer_config.save_path, name_suffix)


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
