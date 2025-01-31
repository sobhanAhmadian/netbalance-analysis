import os

from netbalance.configs import warm_color1
from netbalance.configs.weighted_mean_degree_ratio import (
    WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR as RESULTS_DIR,
)  # Parameter
from netbalance.configs.common import RESULTS_DIR as COMMON_RESULTS_DIR
from netbalance.evaluation.general import get_result_of_rcv
from netbalance.features.hmdad import HMDADDataset as Dataset  # Parameter
from netbalance.utils import prj_logger
from netbalance.visualization import cluster_barplot

logger = prj_logger.getLogger(__name__)

model_name = "weighted_mean_degree_ratio"  # Parameter
dataset = "hmdad"  # Parameter
train_neg_samp_method = "beta"  # Parameter
analyse = "strat_ranking"  # Parameter
num_cross_validation = 5  # Parameter
num_negative_sampling = 1  # Parameter

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

results = get_result_of_rcv(
    save_preds_dir=model_result_dir,
    cluster_a_node_names=ds.get_cluster_a_node_names(),
    cluster_b_node_names=ds.get_cluster_b_node_names(),
    num_cross_validation=num_cross_validation,
    num_negative_sampling=num_negative_sampling,
    test_balance_method=None,
    dataset_name=dataset,
)

print(">> Cluster A Normalized Average Rank")
cluster_barplot(
    arr=results.result.norm_avg_rank_a,
    node_names=ds.get_cluster_a_node_names(),
    figs_folder=figs_folder,
    figure_name=f"Normalized Average Rank",
    cluster_name=ds.cluster_a_name,
    sorted=True,
    color=warm_color1,
)

print(">> Cluster B Normalized Average Rank")
cluster_barplot(
    arr=results.result.norm_avg_rank_b,
    node_names=ds.get_cluster_b_node_names(),
    figs_folder=figs_folder,
    figure_name=f"Normalized Average Rank",
    cluster_name=ds.cluster_b_name,
    sorted=True,
    color=warm_color1,
)
