from typing import List

import numpy as np
from sklearn.metrics import RocCurveDisplay, auc

from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class Result:
    """
    Represents the evaluation result of a model.

    Attributes:
        auc (float): The Area Under the Curve (AUC) value.
        acc (float): The accuracy value.
        f1 (float): The F1 score value.
        aupr (float): The Area Under the Precision-Recall Curve (AUPR) value.
        loss (float): The loss value.
        recall (float): The recall value.
        precision (float): The precision value.
        mcc (float): The Matthews Correlation Coefficient (MCC) value.
        max_f1 (float): The maximum F1 score value for different thresholds.
        fpr (np.ndarray): The false positive rate values.
        tpr (np.ndarray): The true positive rate values.

    Methods:
        get_result(): Returns the evaluation result as a dictionary.
        add(result): Adds the values of another Result object to this Result object.
        divide(k): Divides the values of this Result object by a given number.
    """

    def __init__(self) -> None:
        self.auc = 0
        self.acc = 0
        self.f1 = 0
        self.aupr = 0
        self.loss = 0
        self.recall = 0
        self.precision = 0
        self.mcc = 0
        self.max_f1 = 0
        self.fpr = None
        self.tpr = None
        self.ent: float = 0.0

    def get_result(self):
        """
        Returns the evaluation result as a dictionary.

        Returns:
            dict: A dictionary containing the evaluation metrics and their corresponding values.
        """
        return {
            "AUC": self.auc,
            "ACC": self.acc,
            "F1 Score": self.f1,
            "AUPR": self.aupr,
            "Loss": self.loss,
            "Recall": self.recall,
            "Precision": self.precision,
            "MCC": self.mcc,
            "Max F1": self.max_f1,
        }


class CrossValidationResult:
    """
    Represents the result of cross-validation for evaluating a model's performance.

    Attributes:
        result (Result): The aggregated result of all folds. The tpr and fpr values are not aggregated.
        fold_results (list): List of individual fold results.
        is_result_calculated (bool): Indicates whether the result has been calculated.

    Functions:
        add_fold_result(test_result): Adds the result of a fold to the list of fold results.
        calculate_cv_result(): Calculates the aggregated result of all folds.
        get_roc_curve(ax): Plots the receiver operating characteristic (ROC) curve.
    """

    def __init__(self):
        self.result = Result()
        self.fold_results: List[Result] = []
        self.is_result_calculated: bool = False
        self.k: int = 0

    def add_fold_result(self, test_result):
        """
        Adds the result of a fold to the list of fold results.

        Args:
            test_result (Result): The result of a fold.
        """
        self.fold_results.append(test_result)
        self.k += 1

    def calculate_cv_result(self):
        """
        Calculates the aggregated result of all folds.
        """
        for test_result in self.fold_results:
            self._accumulate(test_result)
        self._divide(self.k)
        self.is_result_calculated = True

    def _divide(self, k):
        """
        Divides the aggregated result by the number of folds.

        Args:
            k (int): The number of folds.
        """
        self.result.acc = self.result.acc / k
        self.result.f1 = self.result.f1 / k
        self.result.auc = self.result.auc / k
        self.result.aupr = self.result.aupr / k
        self.result.loss = self.result.loss / k
        self.result.recall = self.result.recall / k
        self.result.precision = self.result.precision / k
        self.result.mcc = self.result.mcc / k
        self.result.max_f1 = self.result.max_f1 / k
        self.result.ent = self.result.ent / k

    def _accumulate(self, test_result):
        """
        Accumulates the result of a fold to the aggregated result.

        Args:
            test_result (Result): The result of a fold.
        """
        self.result.acc += test_result.acc
        self.result.f1 += test_result.f1
        self.result.auc += test_result.auc
        self.result.aupr += test_result.aupr
        self.result.loss += test_result.loss
        self.result.recall += test_result.recall
        self.result.precision += test_result.precision
        self.result.mcc += test_result.mcc
        self.result.max_f1 += test_result.max_f1
        self.result.ent += test_result.ent

    def get_roc_curve(self, ax, mean_fpr=np.linspace(0, 1, 100)):
        """
        Plots the receiver operating characteristic (ROC) curve.

        Args:
            ax: The matplotlib axes object to plot on.
            mean_fpr (array-like, optional): List of mean false positive rates for ROC curve interpolation.
                Defaults to np.linspace(0, 1, 100).
        """
        if not self.is_result_calculated:
            self.calculate_cv_result()

        tpr_list = []
        auc_list = []
        for r in self.fold_results:
            viz = RocCurveDisplay(fpr=r.fpr, tpr=r.tpr, roc_auc=r.auc)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0

            tpr_list.append(interp_tpr)
            auc_list.append(viz.roc_auc)

        tprs = np.array(tpr_list)
        aucs = np.array(auc_list)

        ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random", alpha=0.8
        )  # Plotting the random line

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.4f $\pm$ %0.4f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )  # Plotting the mean ROC curve

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.3,
            label=r"$\pm$ 1 std. dev.",
        )  # Plotting the standard deviation

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="Receiver operating characteristic",
        )
        ax.legend(loc="lower right")


class BGCCrossValidationResult(CrossValidationResult):

    def __init__(self):
        super().__init__()
        # self.result.hit_k_list = None
        # self.result.hit_k_accuracy_list = None
        self.result.avg_rank = 0
        self.result.norm_avg_rank = 0
        self.result.avg_rank_a = None
        self.result.norm_avg_rank_a = None
        self.result.avg_rank_b = None
        self.result.norm_avg_rank_b = None
        self.result.hit_k_accuracy_a = None
        self.result.hit_k_accuracy_b = None

        self.result.avg_rank_a_counter = None
        self.result.avg_rank_b_counter = None
        self.result.norm_avg_rank_a_counter = None
        self.result.norm_avg_rank_b_counter = None
        self.result.hit_k_accuracy_a_counter = None
        self.result.hit_k_accuracy_b_counter = None

    def _accumulate(self, test_result):
        super()._accumulate(test_result)

        # if self.result.hit_k_list is None:
        #     self.result.hit_k_list = test_result.hit_k_list
        #     self.result.hit_k_accuracy_list = test_result.hit_k_accuracy_list
        # else:
        #     self.result.hit_k_accuracy_list += test_result.hit_k_accuracy_list

        self.result.avg_rank += test_result.avg_rank
        self.result.norm_avg_rank += test_result.norm_avg_rank

        if self.result.avg_rank_a is None:
            self.avg_rank_a_counter = 1 - np.isnan(test_result.avg_rank_a)
            self.avg_rank_b_counter = 1 - np.isnan(test_result.avg_rank_b)

            self.result.avg_rank_a = np.nan_to_num(test_result.avg_rank_a)
            self.result.norm_avg_rank_a = np.nan_to_num(test_result.norm_avg_rank_a)
        else:
            self.avg_rank_a_counter += 1 - np.isnan(test_result.avg_rank_a)
            self.avg_rank_b_counter += 1 - np.isnan(test_result.avg_rank_b)

            self.result.avg_rank_a += np.nan_to_num(test_result.avg_rank_a)
            self.result.norm_avg_rank_a += np.nan_to_num(test_result.norm_avg_rank_a)

        if self.result.avg_rank_b is None:
            self.result.avg_rank_b = np.nan_to_num(test_result.avg_rank_b)
            self.result.norm_avg_rank_b = np.nan_to_num(test_result.norm_avg_rank_b)
        else:
            self.result.avg_rank_b += np.nan_to_num(test_result.avg_rank_b)
            self.result.norm_avg_rank_b += np.nan_to_num(test_result.norm_avg_rank_b)

        if self.result.hit_k_accuracy_a is None:
            self.result.hit_k_accuracy_a = np.nan_to_num(test_result.hit_k_accuracy_a)
            self.result.hit_k_accuracy_b = np.nan_to_num(test_result.hit_k_accuracy_b)
            self.result.hit_k_accuracy_a_counter = 1 - np.isnan(
                test_result.hit_k_accuracy_a
            )
            self.result.hit_k_accuracy_b_counter = 1 - np.isnan(
                test_result.hit_k_accuracy_b
            )
        else:
            self.result.hit_k_accuracy_a += np.nan_to_num(test_result.hit_k_accuracy_a)
            self.result.hit_k_accuracy_b += np.nan_to_num(test_result.hit_k_accuracy_b)
            self.result.hit_k_accuracy_a_counter += 1 - np.isnan(
                test_result.hit_k_accuracy_a
            )
            self.result.hit_k_accuracy_b_counter += 1 - np.isnan(
                test_result.hit_k_accuracy_b
            )

    def _divide(self, k):
        super()._divide(k)
        # self.result.hit_k_accuracy_list /= k
        self.result.avg_rank /= k
        self.result.norm_avg_rank /= k
        self.result.avg_rank_a = self.result.avg_rank_a / (
            self.avg_rank_a_counter + 1e-6
        )
        self.result.norm_avg_rank_a = self.result.norm_avg_rank_a / (
            self.avg_rank_a_counter + 1e-6
        )
        self.result.avg_rank_b = self.result.avg_rank_b / (
            self.avg_rank_b_counter + 1e-6
        )
        self.result.norm_avg_rank_b = self.result.norm_avg_rank_b / (
            self.avg_rank_b_counter + 1e-6
        )
        self.result.hit_k_accuracy_a = self.result.hit_k_accuracy_a / (
            self.result.hit_k_accuracy_a_counter + 1e-6
        )
        self.result.hit_k_accuracy_b = self.result.hit_k_accuracy_b / (
            self.result.hit_k_accuracy_b_counter + 1e-6
        )
