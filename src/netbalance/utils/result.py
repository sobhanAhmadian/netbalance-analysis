import os

import numpy as np
import pandas as pd
from scipy import stats

from netbalance.configs import RESULTS_DIR_DICT
from netbalance.configs.common import RESULTS_DIR

from .logger import logging as prj_logger

logger = prj_logger.getLogger(__name__)


def _prepare_result(result, model_name):
    new_result = {}
    new_result["model"] = [model_name]
    for k in result.keys():
        new_result[k] = [result[k]]
    return pd.DataFrame(new_result)


def save_results(
    cv_result, model_name, dataset, expr_serie, analyse_name, negative_ratio=None
):

    dir_path = f"{RESULTS_DIR}/{dataset}/{analyse_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    file_path = (
        f"{dir_path}/{expr_serie}.csv"
        if negative_ratio is None
        else f"{dir_path}/{expr_serie}_nr_{negative_ratio}.csv"
    )

    result = cv_result.result.get_result()
    result_df = _prepare_result(result, model_name)

    if os.path.exists(file_path):
        logger.info(f"Appending result to {file_path}")
        df = pd.read_csv(file_path)
        df = df.loc[df["model"] != model_name]
        df = pd.concat([df, result_df], ignore_index=True)
        df.to_csv(file_path, index=False)
    else:
        logger.info(f"Creating new file {file_path}")
        result_df.to_csv(file_path, index=False)


def save_auc_of_cv_folds(cv_result, dir_path, filename):
    auc_list = [result.auc for result in cv_result.fold_results]
    auc_arr = np.array(auc_list)
    np.savetxt(f"{dir_path}/{filename}", auc_arr, delimiter=",")


def get_auc_of_cv_folds(
    model_name,
    dataset,
    expr_serie,
    analysis_name="5_fold",
    num_neg=100,
    num_cv=10,
    negative_ratio=None,
):
    dir_path = RESULTS_DIR_DICT[model_name]
    file_name = (
        f"list_auc_{analysis_name}_num_neg_{num_neg}_num_cv_{num_cv}_nr_{negative_ratio}.csv"
        if negative_ratio
        else f"list_auc_{analysis_name}_num_neg_{num_neg}_num_cv_{num_cv}.csv"
    )
    auc_arr = np.loadtxt(
        f"{dir_path}/{dataset}/{expr_serie}/{file_name}", delimiter=","
    )
    return auc_arr


def pared_ttest_of_auc_of_cv_folds(
    first_model,
    second_model,
    dataset="hmdad",
    experiment_name="beta",
    analysis_name="5_fold",
    num_neg=100,
    num_cv=10,
):
    """One-sided pared ttest for auc of two models

    Args:
        first_model (str): name of the first model
        second_model (str): name of the second model
        dataset (str): name of dataset
        experiment_name (str, optional): experiment name which is beta or gamma. Defaults to "beta".
        analysis_name (str, optional): the method that used for evaluation. Defaults to "5_fold".
        num_neg (int, optional): number of negative sampling used for evaluation. Defaults to 100.
        num_cv (int, optional): number of cross_validation used for evaluation. Defaults to 10.

    Raises:
        ValueError: first_model should be in MODEL_TO_GET_AUC_FUNC.keys()
        ValueError: second_model should be in MODEL_TO_GET_AUC_FUNC.keys()

    Returns:
        float: p_value of ttest
    """
    if first_model not in RESULTS_DIR_DICT.keys:
        raise ValueError(f"Unknown model name {first_model}!")
    elif second_model not in RESULTS_DIR_DICT.keys:
        raise ValueError(f"Unknown model name {second_model}!")

    first_aucs = get_auc_of_cv_folds(
        model_name=first_model,
        dataset=dataset,
        expr_serie=experiment_name,
        analysis_name=analysis_name,
        num_neg=num_neg,
        num_cv=num_cv,
    )
    second_aucs = get_auc_of_cv_folds(
        model_name=second_model,
        dataset=dataset,
        expr_serie=experiment_name,
        analysis_name=analysis_name,
        num_neg=num_neg,
        num_cv=num_cv,
    )
    _, p_value = stats.ttest_rel(
        first_aucs, second_aucs, alternative="greater"
    )  # first greater than second
    return p_value


def save_all_results_for_expr(
    general_cv_result,
    fig,
    save_figs,
    analyse_name,
    model_name,
    model_result_dir,
    dataset,
    expr_serie,
    num_negative_sampling,
    num_cross_validation,
    negative_ratio=None,
):

    model_figs_folder = f"result_figs/{model_name}"
    auc_result_folder = f"{model_result_dir}/{dataset}/{expr_serie}"

    if not os.path.exists(model_figs_folder):
        os.makedirs(model_figs_folder, exist_ok=True)

    if not os.path.exists(auc_result_folder):
        os.makedirs(auc_result_folder, exist_ok=True)

    if save_figs:
        fig.savefig(f"{model_figs_folder}/roc_{analyse_name}.png", dpi=300)

    file_name = (
        f"list_auc_{analyse_name}_num_neg_{num_negative_sampling}_num_cv_{num_cross_validation}_nr_{negative_ratio}.csv"
        if negative_ratio
        else f"list_auc_{analyse_name}_num_neg_{num_negative_sampling}_num_cv_{num_cross_validation}.csv"
    )
    save_auc_of_cv_folds(
        general_cv_result,
        auc_result_folder,
        file_name,
    )

    save_results(
        cv_result=general_cv_result,
        model_name=model_name,
        dataset=dataset,
        expr_serie=expr_serie,
        analyse_name=analyse_name,
        negative_ratio=negative_ratio,
    )
