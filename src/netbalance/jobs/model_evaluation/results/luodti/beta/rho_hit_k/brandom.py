import os

from netbalance.configs import cold_color2, warm_color1
from netbalance.configs.brandom import BRANDOM_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.configs.common import RESULTS_DIR as COMMON_RESULTS_DIR
from netbalance.evaluation.general import get_result_of_rcv
from netbalance.features.luodti import LuoDTIDataset as Dataset  # Parameter
from netbalance.utils import prj_logger
from netbalance.visualization import plot_x_vs_y_dist, plot_xs_vs_y_dist

logger = prj_logger.getLogger(__name__)

model_name = "brandom"  # Parameter
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

hit_k_list = results.result.hit_k_list
hit_k_accuracy_list = results.result.hit_k_accuracy_list
max_k = min(len(hit_k_list), 30)
hit_k_accuracy_list_list = []
for i in range(len(hit_k_list)):
    temp = []
    for r in results.fold_results:
        temp.append(r.hit_k_accuracy_list[i])
    hit_k_accuracy_list_list.append(temp)
per_fold_hit_k_accuracy_list = [r.hit_k_accuracy_list for r in results.fold_results]

plot_x_vs_y_dist(
    x_list=hit_k_list,
    y_list=hit_k_accuracy_list,
    y_list_list=hit_k_accuracy_list_list,
    figs_folder=figs_folder,
    cold_color=cold_color2,
    warm_color=warm_color1,
    ylim_down=-0.1,
    ylim_up=1.1,
    fig_width=20,
    fig_height=6,
    x_name="K",
    y_name="Hit@K Accuracy",
    title=f"{model_name.upper()} Hit@K Accuracy Distribution",
    max_k=max_k,
)

plot_xs_vs_y_dist(
    x_list=hit_k_list,
    y_list_list=per_fold_hit_k_accuracy_list,
    max_y_plot=4,
    figs_folder=figs_folder,
    cold_color=cold_color2,
    warm_color=warm_color1,
    ylim_down=-0.1,
    ylim_up=1.1,
    fig_width=20,
    fig_height=6,
    x_name="K",
    y_name="Hit@K Accuracy",
    title=f"{model_name.upper()} Hit@K Accuracy Distribution",
    max_k=max_k,
    seed=0,
)
