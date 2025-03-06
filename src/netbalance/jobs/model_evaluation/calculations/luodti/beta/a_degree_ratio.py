import os

from netbalance.configs import OptimizerConfig as OptimizerConfig  # Parameter
from netbalance.configs.a_degree_ratio import (
    A_DEGREE_RATIO_RESULTS_DIR as RESULTS_DIR,
)  # Parameter
from netbalance.data.association_data import BGData, BGTrainTestSpliter
from netbalance.evaluation import repeated_cross_validation
from netbalance.features.luodti import LuoDTIDataset as Dataset  # Parameter
from netbalance.models.degree_ratio import (
    BGDegreeRatioHandlerFactory as HandlerFactory,
)  # Parameter
from netbalance.optimization.degree_ratio import (
    DegreeRatioTrainer as Trainer,
)  # Parameter
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

model_name = "a_degree_ratio"  # Parameter
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

optimizer_config = OptimizerConfig()  # Parameter


def get_data():
    return BGData(
        associations=ds.get_associations(with_negatives=True),
        cluster_a_node_names=ds.get_cluster_a_node_names(),
        cluster_b_node_names=ds.get_cluster_b_node_names(),
    )


associations = ds.get_associations(with_negatives=True)

trainer = Trainer()
factory = HandlerFactory(  # Parameter
    a_node_ids=list(set(associations[:, 0])),
    b_node_ids=list(set(associations[:, 1])),
    use_a=True,
    use_b=False,
    reduction=None,
)


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
