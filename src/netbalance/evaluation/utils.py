import numpy as np
import scipy.interpolate as interp
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from netbalance.data.association_data import AData
from netbalance.utils import prj_logger
from netbalance.visualization import plot_x_vs_y_dist

from .result import Result

logger = prj_logger.getLogger(__name__)


def evaluate_binary_classification(
    data: AData, y_predict: np.ndarray, threshold: float = 0.5
):
    """Evaluate binary classification predictions.

    Args:
        data (AData): The data object.
        y_predict (np.ndarray): The predicted scores.
        threshold (float): The threshold value for converting predicted probabilities to binary predictions.

    Returns:
        _type_: _description_
    """
    y_test = data.associations[:, -1]
    result = Result()

    # Entropy
    result.ent = data.get_stats()["ent"]

    # Add predictions to the result
    result.predictions = y_predict

    # Binary predictions
    binary_predict = (np.array(y_predict) >= threshold).astype(int)
    result.binary = binary_predict

    # Basic metrics
    result.acc = accuracy_score(y_test, binary_predict)
    result.f1 = f1_score(y_test, binary_predict, zero_division=1)
    result.recall = recall_score(y_test, binary_predict, zero_division=1)
    result.precision = precision_score(y_test, binary_predict, zero_division=1)
    result.mcc = matthews_corrcoef(y_test, binary_predict)

    # Advanced metrics
    _calc_max_f1_score(y_predict, y_test, result)
    _calc_aupr_metrics(y_predict, y_test, result)
    _calc_roc_metrics(y_predict, y_test, result)
    _calc_hit_k_scores(data, y_predict, result)
    sorted_indices = _calc_avg_rank(data, y_predict, result)
    result.sorted_edges = data.associations[sorted_indices]
    # _calc_edge_related(data, result)
    _calc_strat_avg_rank(data, y_predict, result)
    _calc_strat_hit_k(data, y_predict, result)

    return result


def _calc_aupr_metrics(y_predict, y_test, result):
    """Calculate the maximum F1 score."""
    precision, recall, _ = precision_recall_curve(y_test, y_predict)
    result.precision_curve = precision
    result.recall_curve = recall
    result.aupr = auc(recall, precision).item()


def _calc_max_f1_score(y_predict, y_test, result):
    """Calculate the maximum F1 score."""
    precision, recall, _ = precision_recall_curve(y_test, y_predict)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    result.max_f1 = np.max(f1_scores)


def _calc_roc_metrics(y_predict, y_test, result):
    """Calculate ROC AUC and related metrics."""
    fpr, tpr, _ = roc_curve(y_test, y_predict, pos_label=1)
    result.auc = auc(fpr, tpr).item()
    result.fpr = fpr
    result.tpr = tpr


def _calc_hit_k_scores(data, y_predict, result):
    """Compute Hit@K scores."""
    max_k = int(data.associations[:, -1].sum())
    hit_k_accuracy_list = [
        np.mean([data.associations[i, -1] for i in np.argsort(y_predict)[::-1][:k]])
        for k in range(1, max_k + 1)
    ]
    result.hit_k_list = np.arange(1, max_k + 1)
    result.hit_k_accuracy_list = np.array(hit_k_accuracy_list)
    result.num_test_positive = max_k
    result.num_test = len(data.associations)


def _calc_avg_rank(data, y_predict, result):
    """Calculate average rank and normalized average rank."""
    sorted_indices = np.argsort(y_predict)[::-1]
    positive_ranks = [
        i + 1
        for i in range(len(sorted_indices))
        if data.associations[sorted_indices[i], -1] == 1
    ]
    result.avg_rank = np.mean(positive_ranks)
    result.norm_avg_rank = result.avg_rank / (
        (data.associations[:, -1].sum() + 1) / 2 + 1e-6
    )
    return sorted_indices


def _calc_edge_related(data, result):
    """Compute edge endpoint degrees and ratios."""

    def compute_degrees(indices, col):
        pos_degrees = np.array(
            [
                data.associations[data.associations[:, col] == i, 2].sum()
                for i in indices
            ]
        )
        total_degrees = np.array(
            [data.associations[:, col].tolist().count(i) for i in indices]
        )
        neg_degrees = total_degrees - pos_degrees
        ratios = pos_degrees / (pos_degrees + neg_degrees + 1e-6)
        return pos_degrees, neg_degrees, ratios

    # Compute for both node types
    a_indices = data.associations[:, 0]
    b_indices = data.associations[:, 1]

    result.pos_degrees_a, result.neg_degrees_a, result.ratios_a = compute_degrees(
        a_indices, 0
    )
    result.pos_degrees_b, result.neg_degrees_b, result.ratios_b = compute_degrees(
        b_indices, 1
    )

    # Compute edge scores based on the ratios
    result.edge_scores = []
    for i in range(len(data.associations)):
        w_a = result.pos_degrees_a[i] / (
            result.pos_degrees_a[i] + result.neg_degrees_a[i] + 1e-6
        )
        w_b = result.pos_degrees_b[i] / (
            result.pos_degrees_b[i] + result.neg_degrees_b[i] + 1e-6
        )
        result.edge_scores.append(w_a * result.ratios_a[i] + w_b * result.ratios_b[i])


def _calc_strat_hit_k(data: AData, y_predict, result):
    """Calculate stratified Hit@K for both node types."""

    def compute_strat_hit_k(cluster_names, associations, predictions, col):
        results = []
        for idx in range(len(cluster_names)):
            # Filter associations and predictions for the current cluster
            cluster_associations = associations[associations[:, col] == idx]
            cluster_predictions = predictions[associations[:, col] == idx]

            # Sort predictions in descending order
            sorted_indices = np.argsort(cluster_predictions)[::-1]

            # Determine the number of positive associations (k)
            k = int(cluster_associations[:, -1].sum())

            # Calculate the hit@k accuracy for the cluster
            temp = [cluster_associations[i, -1] for i in sorted_indices[:k]]
            hit_k_accuracy = np.mean(temp if len(temp) > 0 else np.nan)

            results.append(hit_k_accuracy)

        return np.array(results)

    result.hit_k_accuracy_c = {}
    for i in range(len(data.node_names)):
        result.hit_k_accuracy_c[i] = compute_strat_hit_k(
            data.node_names[i], data.associations, y_predict, i
        )


def _calc_strat_avg_rank(data: AData, y_predict, result):
    """Calculate stratified average rank for both node types."""

    def compute_avg_rank(cluster_names, associations, predictions, col):
        avg_ranks = []
        norm_avg_ranks = []
        for idx in range(len(cluster_names)):
            node_associations = associations[associations[:, col] == idx]
            node_predictions = predictions[associations[:, col] == idx]
            positive_ranks = [
                rank + 1
                for rank, assoc in enumerate(np.argsort(node_predictions)[::-1])
                if node_associations[assoc, -1] == 1
            ]
            avg_rank = np.mean(positive_ranks) if len(positive_ranks) > 0 else np.nan
            avg_ranks.append(avg_rank)
            norm_avg_ranks.append(
                avg_rank / (len(node_associations) + 1e-6) if avg_rank else np.nan
            )
        return np.array(avg_ranks), np.array(norm_avg_ranks)

    result.avg_rank_c = {}
    result.norm_avg_rank_c = {}
    for i in range(len(data.node_names)):
        result.avg_rank_c[i], result.norm_avg_rank_c[i] = compute_avg_rank(
            data.node_names[i], data.associations, y_predict, i
        )


def evaluate_binary_classification_simple(y_test, y_predict, threshold):
    """
    Calculate various evaluation metrics for binary classification predictions.

    Args:
        y_test (array-like): True labels of the test set.
        y_predict (array-like): Predicted labels of the test set.
        threshold (float): Threshold value for converting predicted probabilities to binary predictions.

    Returns:
        Result: An object containing the calculated evaluation metrics.

    """

    result = Result()

    binary_predict = np.where(np.array(y_predict) >= threshold, 1, 0).tolist()
    result.binary = binary_predict

    result.acc = accuracy_score(y_test, binary_predict)
    result.f1 = f1_score(y_test, binary_predict, zero_division=1)
    result.recall = recall_score(y_test, binary_predict, zero_division=1)
    result.precision = precision_score(y_test, binary_predict, zero_division=1)
    result.mcc = matthews_corrcoef(y_test, binary_predict)

    # Calculate max F1 score
    precision, recall, _ = precision_recall_curve(y_test, y_predict)
    numerator = 2 * recall * precision
    denominator = recall + precision
    f1_scores = np.divide(
        numerator, denominator, out=np.zeros_like(denominator), where=(denominator != 0)
    )
    result.max_f1 = np.max(f1_scores)

    # Calculate AUC, TPR, FPR
    fpr, tpr, _ = roc_curve(y_test, y_predict, pos_label=1)
    result.auc = auc(fpr, tpr)
    result.fpr = fpr
    result.tpr = tpr

    return result


def rho_hit_k_analyse(
    model_name, figs_folder, save_rho_hit_k_dir, results, cold_color, warm_color
):
    def average_functions(functions, x_common):
        interpolated_ys = []
        for x, y in functions:
            f = interp.interp1d(
                x, y, kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            interpolated_ys.append(f(x_common))

        avg_y = np.mean(interpolated_ys, axis=0)
        return interpolated_ys, avg_y

    functions = []
    for r in results.fold_results:
        temp_hit_k_list = np.array(r.hit_k_list)
        temp_hit_k_accuracy_list = np.array(r.hit_k_accuracy_list)
        temp_hit_k_list = temp_hit_k_list / temp_hit_k_list.max()
        functions.append((temp_hit_k_list, temp_hit_k_accuracy_list))

    x_common = np.linspace(0, 1, 30)
    interpolated_ys, y_avg = average_functions(functions, x_common)

    per_x_ys = []
    for i in range(len(x_common)):
        temp = []
        for ys in interpolated_ys:
            temp.append(ys[i])
        per_x_ys.append(temp)

    max_k = len(x_common)

    plot_x_vs_y_dist(
        x_list=x_common,
        y_list=y_avg,
        y_list_list=per_x_ys,
        figs_folder=figs_folder,
        cold_color=cold_color,
        warm_color=warm_color,
        ylim_down=-1,
        ylim_up=2,
        xlim_left=-0.05,
        xlim_right=1.05,
        fig_width=20,
        fig_height=6,
        violon_width=1 / max_k / 2,
        x_name="K",
        y_name="Hit@K Accuracy",
        title=f"{model_name.upper()} Hit@K Accuracy Distribution",
        max_k=max_k,
    )

    file_name = f"{save_rho_hit_k_dir}/rho_hit_k"
    np.savetxt(file_name, y_avg, delimiter=",")
    print(f"\nSaved auc list to {file_name}")
