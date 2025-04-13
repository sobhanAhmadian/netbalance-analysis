import os

from netbalance.configs.blindti import BLINDTI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.evaluation.general import get_result_of_rcv
from netbalance.features.luodti import LuoDTIDataset as Dataset  # Parameter
from netbalance.utils import prj_logger
from netbalance.utils.result import process_results

logger = prj_logger.getLogger(__name__)

model_name = "blindti"  # Parameter
dataset = "luodti"  # Parameter
train_neg_samp_method = "beta"  # Parameter

test_balance_method = "beta"  # Parameter
test_balance_kwargs = {}  # Parameter
test_balance_negative_ratio = 1.0  # Parameter

num_cross_validation = 5  # Parameter
num_negative_sampling = 5  # Parameter

model_result_dir = os.path.join(
    RESULTS_DIR,
    f"preds",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)

logger.info(
    f">>>>>>>>>>>>>>>>> Job: Model Evaluation - Results - {dataset} - {model_name} - {train_neg_samp_method}"
)

ds = Dataset()

if __name__ == "__main__":
    results = get_result_of_rcv(
        save_preds_dir=model_result_dir,
        node_names=ds.get_node_names(),
        num_cross_validation=num_cross_validation,
        num_negative_sampling=num_negative_sampling,
        test_balance_method=test_balance_method,
        test_balance_kwargs=test_balance_kwargs,
        test_balance_negative_ratio=test_balance_negative_ratio,
        dataset_name=dataset,
    )

    process_results(
        results=results,
        result_dir=RESULTS_DIR,
        dataset=dataset,
        train_balance_method=train_neg_samp_method,
        test_balance_method=test_balance_method,
        test_balance_kwargs=test_balance_kwargs,
    )
