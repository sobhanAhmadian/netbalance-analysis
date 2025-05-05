import os

from netbalance.configs.bmlpsyn import (
    BMLPSYN_RESULTS_DIR as RESULTS_DIR,  # Parameter
)
from netbalance.configs.bmlpsyn import (
    BMLPSYNModelConfig as ModelConfig,  # Parameter
)
from netbalance.configs.bmlpsyn import (
    BMLPSYNOptimizerConfig as OptimizerConfig,  # Parameter
)
from netbalance.data.association_data import TGData, TGTrainTestSpliter
from netbalance.evaluation import repeated_cross_validation
from netbalance.features.sanger import SangerDataset as Dataset  # Parameter
from netbalance.models.bmlpsyn import (
    BMLPSYNHandlerFactory as HandlerFactory,  # Parameter
)
from netbalance.optimization.bmlpsyn import BalanceBMLPSYNTrainer as Trainer  # Parameter
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

model_name = "bmlpsyn"  # Parameter
dataset = "sanger"  # Parameter
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

optimizer_config = OptimizerConfig()  # Parameter
optimizer_config.i_balance_method = "rho"
optimizer_config.i_balance_kwargs = {
    "max_iter": 10000,
    "delta": 0.09,
    "cooling_rate": 0.99,
    "initial_temp": 10.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}
optimizer_config.i_negative_ratio = 1.0
optimizer_config.i_max_num_bal = 30
optimizer_config.n_epoch = 200

model_config = ModelConfig()  # Parameter

def get_data():
    return TGData(
        associations=ds.get_associations(with_negatives=False),  # Parameters
        cluster_a_node_names=ds.get_cluster_a_node_names(),
        cluster_b_node_names=ds.get_cluster_b_node_names(),
        cluster_c_node_names=ds.get_cluster_c_node_names(),
    )

associations = ds.get_associations(with_negatives=False)

trainer = Trainer()
factory = HandlerFactory(model_config=model_config)  # Parameter

if __name__ == "__main__":
    repeated_cross_validation(
        get_data=get_data,
        SplitterClass=TGTrainTestSpliter,
        handler_factory=factory,
        trainer=trainer,
        optimizer_config=optimizer_config,
        num_cross_validation=num_cross_validation,
        save_preds_dir=model_result_dir,
        splitter_kwargs=splitter_kwargs,
        parallel=False,
    )
