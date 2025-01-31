import os

from netbalance.configs import cold_color2, warm_color1
from netbalance.configs.common import RESULTS_DIR as COMMON_RESULTS_DIR
from netbalance.configs.weighted_mean_degree_ratio import (
    WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR as RESULTS_DIR,
)  # Parameter
from netbalance.evaluation.general import get_ent_vs_auc
from netbalance.features.hmdad import HMDADDataset as Dataset  # Parameter
from netbalance.utils import prj_logger
from netbalance.visualization import plot_x_vs_y_dist

logger = prj_logger.getLogger(__name__)

model_name = "weighted_mean_degree_ratio"  # Parameter
dataset = "hmdad"  # Parameter
train_neg_samp_method = "beta"  # Parameter
analyse = "balance_vs_auc"  # Parameter
test_balance_kwargs = {
    "max_iter": 40000,
    "delta": 0.13,
    "cooling_rate": 0.99,
    "initial_temp": 20.0,
    "shrinkage": 1.0,
}

logger.info(
    f">>>>>>>>>>>>>>>>> Job: Model Evaluation - Results - {dataset} - {model_name} - {train_neg_samp_method} - {analyse}"
)

model_result_dir = os.path.join(
    RESULTS_DIR,
    f"preds",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)

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

auc_list, auc_list_list, ent_list = get_ent_vs_auc(
    model_result_dir=model_result_dir,
    dataset_name=dataset,
    test_balance_kwargs=test_balance_kwargs,
    cluster_a_node_names=ds.get_cluster_a_node_names(),
    cluster_b_node_names=ds.get_cluster_b_node_names(),
)

plot_x_vs_y_dist(
    y_list=auc_list,
    y_list_list=auc_list_list,
    x_list=ent_list,
    figs_folder=figs_folder,
    warm_color=warm_color1,
    cold_color=cold_color2,
    x_name="Entropy",
    y_name="AUC",
    title=f"{dataset.upper()} Entropy vs {model_name.upper()} AUC Distribution",
)
