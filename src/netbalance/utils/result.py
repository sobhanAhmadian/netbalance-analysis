import os
from typing import List, Union

import numpy as np
import pandas as pd
from scipy import stats

from netbalance.configs import RESULTS_DIR_DICT
from netbalance.configs.common import RESULTS_DIR
from netbalance.evaluation.result import ACrossValidationResult

from .logger import logging as prj_logger

logger = prj_logger.getLogger(__name__)


def _prepare_result(result, model_name):
    new_result = {}
    new_result["model"] = [model_name]
    for k in result.keys():
        new_result[k] = [result[k]]
    return pd.DataFrame(new_result)


def _get_aucs_dir_name(
    base_dir: str,
    dataset: str,
    train_balance_method: str,
    test_balance_method: str,
    test_balance_kwargs: dict,
):
    aucs_results_dir = f"{base_dir}/aucs/dataset_{dataset}/train_neg_samp_{train_balance_method}/test_neg_samp_{test_balance_method}"
    for key, value in test_balance_kwargs.items():
        aucs_results_dir += f"_{key}_{value}"
    return aucs_results_dir


def _get_max_f1_dir_name(
    base_dir: str,
    dataset: str,
    train_balance_method: str,
    test_balance_method: str,
    test_balance_kwargs: dict,
):
    aucs_results_dir = f"{base_dir}/max_f1/dataset_{dataset}/train_neg_samp_{train_balance_method}/test_neg_samp_{test_balance_method}"
    for key, value in test_balance_kwargs.items():
        aucs_results_dir += f"_{key}_{value}"
    return aucs_results_dir


def _process_results_temp(
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


def _save_auc_of_cv_folds(
    results: ACrossValidationResult, dir_path: str, filename: str
):
    auc_list = [result.auc for result in results.fold_results]
    auc_arr = np.array(auc_list)
    np.savetxt(f"{dir_path}/{filename}", auc_arr, delimiter=",")
    print(f"\nSaved auc list to {dir_path}/{filename}")


def _save_max_f1_of_cv_folds(
    results: ACrossValidationResult, dir_path: str, filename: str
):
    max_f1_list = [result.max_f1 for result in results.fold_results]
    max_f1_arr = np.array(max_f1_list)
    np.savetxt(f"{dir_path}/{filename}", max_f1_arr, delimiter=",")
    print(f"\nSaved max f1 list to {dir_path}/{filename}")


def get_auc_of_cv_folds(
    result_dir,
    dataset,
    train_balance_method,
    test_balance_method,
    test_balance_kwargs,
):
    dir_path = _get_aucs_dir_name(
        base_dir=result_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method,
        test_balance_kwargs=test_balance_kwargs,
    )
    file_name = f"{dir_path}/aucs.txt"
    auc_arr = np.loadtxt(file_name, delimiter=",")
    return auc_arr


def get_max_f1_of_cv_folds(
    result_dir,
    dataset,
    train_balance_method,
    test_balance_method,
    test_balance_kwargs,
):
    dir_path = _get_max_f1_dir_name(
        base_dir=result_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method,
        test_balance_kwargs=test_balance_kwargs,
    )
    file_name = f"{dir_path}/max_f1s.txt"
    auc_arr = np.loadtxt(file_name, delimiter=",")
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
    _save_auc_of_cv_folds(
        general_cv_result,
        auc_result_folder,
        file_name,
    )

    _process_results_temp(
        cv_result=general_cv_result,
        model_name=model_name,
        dataset=dataset,
        expr_serie=expr_serie,
        analyse_name=analyse_name,
        negative_ratio=negative_ratio,
    )


def process_results(
    results: ACrossValidationResult,
    result_dir: str,
    dataset: str,
    train_balance_method: str,
    test_balance_method: str,
    test_balance_kwargs: str,
):
    # Print Average Results
    print("\nAverage Results:")
    print(results.result.get_result())

    # Save Fold AUCs
    aucs_results_dir = _get_aucs_dir_name(
        base_dir=result_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method,
        test_balance_kwargs=test_balance_kwargs,
    )
    max_f1s_results_dir = _get_max_f1_dir_name(
        base_dir=result_dir,
        dataset=dataset,
        train_balance_method=train_balance_method,
        test_balance_method=test_balance_method,
        test_balance_kwargs=test_balance_kwargs,
    )
    os.makedirs(aucs_results_dir, exist_ok=True)
    _save_auc_of_cv_folds(results, aucs_results_dir, filename="aucs.txt")

    os.makedirs(max_f1s_results_dir, exist_ok=True)
    _save_max_f1_of_cv_folds(results, max_f1s_results_dir, filename="max_f1s.txt")
