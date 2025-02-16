import os

from netbalance.configs import cold_color2, warm_color1
from netbalance.configs.fmidti import FMIDTI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.configs.common import RESULTS_DIR as COMMON_RESULTS_DIR
from netbalance.evaluation.general import get_result_of_rcv
from netbalance.evaluation.utils import rho_hit_k_analyse
from netbalance.features.luodti import LuoDTIDataset as Dataset  # Parameter
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

model_name = "fmidti"  # Parameter
dataset = "luodti"  # Parameter
train_neg_samp_method = "beta"  # Parameter

test_balance_method = "rho"  # Parameter
test_balance_kwargs = {
    "max_iter": 100000,
    "delta": 0.1,
    "cooling_rate": 0.99,
    "initial_temp": 40.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}  # Parameter
test_balance_negative_ratio = 1.0  # Parameter

analyse = "rho_hit_k"  # Parameter
num_cross_validation = 5  # Parameter
num_negative_sampling = 5  # Parameter

logger.info(
    f">>>>>>>>>>>>>>>>> Job: Model Evaluation - Results - {dataset} - {model_name} - {train_neg_samp_method} - {analyse}"
)

model_result_dir = os.path.join(
    RESULTS_DIR,
    f"preds",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)

save_rho_hit_k_dir = os.path.join(
    RESULTS_DIR,
    f"rho_hit_k",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)
os.makedirs(save_rho_hit_k_dir, exist_ok=True)

figs_folder = os.path.join(
    COMMON_RESULTS_DIR,
    f"figs",
    "model_evaluation/results",
    dataset,
    train_neg_samp_method,
    analyse,
    model_name,
)

if not os.path.exists(figs_folder):
    os.makedirs(figs_folder, exist_ok=True)

ds = Dataset()

results = get_result_of_rcv(
    save_preds_dir=model_result_dir,
    cluster_a_node_names=ds.get_cluster_a_node_names(),
    cluster_b_node_names=ds.get_cluster_b_node_names(),
    num_cross_validation=num_cross_validation,
    num_negative_sampling=num_negative_sampling,
    test_balance_method=test_balance_method,
    test_balance_kwargs=test_balance_kwargs,
    test_balance_negative_ratio=test_balance_negative_ratio,
    dataset_name=dataset,
)

rho_hit_k_analyse(
    model_name, figs_folder, save_rho_hit_k_dir, results, cold_color2, warm_color1
)
