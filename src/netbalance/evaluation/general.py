import os
from typing import Callable, List, Tuple, Union

import dask
import numpy as np
import pandas as pd
import torch
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from netbalance.configs import OptimizerConfig
from netbalance.data import TrainTestSplitter
from netbalance.data.association_data import AData
from netbalance.models import HandlerFactory
from netbalance.optimization.interface import Trainer
from netbalance.utils import get_header_format, prj_logger

from .result import ACrossValidationResult
from .utils import evaluate_binary_classification

logger = prj_logger.getLogger(__name__)


def get_ent_vs_auc(
    model_result_dir: str,
    dataset_name: str,
    test_balance_kwargs: dict,
    node_names: list[list[int]],
    num_cross_validation: int = 2,
    num_negative_sampling: int = 2,
) -> Tuple[List[float], List[List[float]], List[float]]:
    """Get the AUC results of a model for different desired dataset entropy values.

    Args:
        model_result_dir (str): The directory containing the prediction files.
        dataset_name (str): The name of the dataset.
        test_balance_kwargs (dict): The keyword arguments for balancing the test data using rho method.
        node_names (list[list[int]]): The node names of the dataset.
        num_cross_validation (int, optional): The number of cross-validation (for each entropy). Defaults to 1.
        num_negative_sampling (int, optional): The number of negative samples (for each entropy). Defaults to 1.

    Returns:
        Tuple[List[float], List[List[float]], List[float]]: A tuple containing the AUC results, the AUC results for each fold, and the entropy values.
    """

    desired_ent_list = np.arange(0.0, 1.05, 0.05).tolist()
    auc_list = []
    auc_list_list = []
    ent_list = []

    for ent in desired_ent_list:
        print(
            f"Desired Entropy: {round(ent, 2)} [{desired_ent_list.index(ent) + 1}/{len(desired_ent_list)}]"
        )
        results = get_result_of_rcv(
            save_preds_dir=model_result_dir,
            node_names=node_names,
            num_cross_validation=num_cross_validation,
            num_negative_sampling=num_negative_sampling,
            dataset_name=dataset_name,
            test_balance_method="rho",
            test_balance_negative_ratio=1.0,
            test_balance_kwargs={
                **test_balance_kwargs,
                "ent_desired": ent,
            },
        )
        ent_list.append(results.result.ent)
        auc_list.append(results.result.auc)
        auc_list_list.append([r.auc for r in results.fold_results])
    return auc_list, auc_list_list, ent_list


def repeated_cross_validation(
    get_data: Callable,
    SplitterClass: Callable,
    handler_factory: HandlerFactory,
    trainer: Trainer,
    optimizer_config: OptimizerConfig,
    num_cross_validation: int,
    save_preds_dir: str,
    splitter_kwargs: dict = {},
    test_batch_size: int = 1000,
    parallel: bool = False,
):
    """Perform repeated cross validation using the given components and configuration,
    and save the predictions for test data of each fold.

    Args:
        get_data (Callable):
        SplitterClass (Callable): _description_
        handler_factory (HandlerFactory): _description_
        trainer (Trainer): _description_
        optimizer_config (OptimizerConfig): _description_
        num_cross_validation (int): _description_
        save_preds_dir (str): _description_
        splitter_kwargs (dict, optional): _description_. Defaults to {}.
        test_batch_size (int, optional): _description_. Defaults to 1000.
        parallel (bool, optional): Whether to run the cross validation in parallel. Defaults to True.
    """
    logger.info(get_header_format("Repeated Cross Validation"))

    with tqdm(total=num_cross_validation * 5, desc="Repeated Cross Validation") as pbar:
        for j in range(num_cross_validation):
            save_preds_dir_re = os.path.join(save_preds_dir, f"cv_{j + 1}")
            data = get_data()

            spliter = SplitterClass(data=data, seed=j, **splitter_kwargs)
            cross_validation(
                train_test_spliter=spliter,
                handler_factory=handler_factory,
                trainer=trainer,
                config=optimizer_config,
                save_preds_dir=save_preds_dir_re,
                test_batch_size=test_batch_size,
                pbar=pbar,
                parallel=parallel,
            )


def cross_validation(
    train_test_spliter: TrainTestSplitter,
    handler_factory: HandlerFactory,
    trainer: Trainer,
    config: OptimizerConfig,
    save_preds_dir: str,
    test_batch_size: int = 1000,
    pbar: tqdm = None,
    parallel: bool = True,
):
    """
    Perform k-fold cross validation using the given components and configuration.
    The predictions for test data of each fold are saved in save_preds_dir.

    Args:
        train_test_spliter (TrainTestSplitter): The train-test splitter object.
        handler_factory (HandlerFactory): The handler factory object.
        trainer (Trainer): The trainer object.
        config (OptimizerConfig): The optimizer configuration object.
        save_preds_dir (str): The directory to save the predictions.
        test_batch_size (int, optional): The batch size for test data. Defaults to 1000.
        parallel (bool, optional): Whether to run the cross validation in parallel. Defaults to True.
    """

    k = train_test_spliter.k
    logger.info(f"Start {k}-fold Cross Validation with config: {config.exp_name}")

    def task(i: int):
        logger.info("{:#^50}".format(f"   Fold {i + 1}   "))
        os.makedirs(save_preds_dir, exist_ok=True)
        save_preds_file = f"{save_preds_dir}/fold_{i + 1}.csv"

        # Split the data
        train_data, test_data = train_test_spliter.split(i)

        # Create model handler
        model_handler = handler_factory.create_handler()

        # Train the model
        trainer.train(model_handler=model_handler, data=train_data, config=config)

        # Save Test Predictions
        if isinstance(model_handler.model, torch.nn.Module):
            model_handler.model.eval()

        preds = np.zeros(test_data.associations.shape[0])
        for j in range(0, test_data.associations.shape[0], test_batch_size):
            preds[j : j + test_batch_size] = model_handler.predict(
                [
                    test_data.associations[j : j + test_batch_size, r]
                    for r in range(test_data.associations.shape[1] - 1)
                ],
            )
        logger.info("Predictions generated.")

        _save_predictions(preds, test_data.associations, save_preds_file)

        # Destroy the model handler
        model_handler.destroy()

        if pbar is not None:
            pbar.update(1)

    if parallel:
        tasks = [dask.delayed(task)(i) for i in range(k)]
        dask.compute(tasks)
    else:
        for i in range(k):
            task(i)


def _save_predictions(predictions: np.ndarray, associations: np.ndarray, file: str):
    """_summary_

    Args:
        predictions (np.ndarray): _description_
        associations (np.ndarray): _description_
        file (str): _description_
    """
    with open(file, "w") as f:
        n = associations.shape[1] - 1
        for i in range(n):
            f.write(f"Node {str(chr(i + 97)).upper()},")
        f.write("Association, Score\n")
        for i in range(len(associations)):
            for j in range(n):
                f.write(f"{associations[i, j]},")
            f.write(f"{associations[i, -1]},{predictions[i]}\n")
        logger.info(f"Predictions saved to {file}")


def get_result_of_rcv(
    save_preds_dir: str,
    node_names: list[list[str]],
    num_negative_sampling: int,
    num_cross_validation: int,
    test_balance_method: Union[str, None] = "beta",
    test_balance_kwargs: dict = {},
    test_balance_negative_ratio: float = 1.0,
    dataset_name: str = None,
    parallel: bool = True,
):
    logger.info(get_header_format("Repeated Cross Validation From Prediction Files"))
    general_cv_result = ACrossValidationResult()

    def task(i: int, k: int, j: int, associations: np.ndarray, df: pd.DataFrame):
        save_name = None
        if dataset_name is not None and test_balance_method is not None:
            save_name = (
                f"dataset_{dataset_name}_{"test"}_cv_{i + 1}_fold_{k + 1}_neg_{j + 1}"
            )
            save_name += f"_met_{test_balance_method}_rat_{test_balance_negative_ratio}"
            for key, value in test_balance_kwargs.items():
                save_name += f"_{key}_{value}"
        data = AData(associations=associations, node_names=node_names)
        if test_balance_method is not None:
            data.balance_data(
                balance_method=test_balance_method,
                negative_ratio=test_balance_negative_ratio,
                seed=j,
                save_name=save_name,
                **test_balance_kwargs,
            )
            temp_df = pd.DataFrame(
                data.associations[:, :-1], columns=df.columns[:-2].tolist()
            )
            reduced_df = df.merge(temp_df, on=df.columns[:-2].tolist(), how="right")
            reduced_preds = reduced_df.iloc[:, -1].to_numpy().flatten()
        else:
            reduced_preds = df.iloc[:, -1].to_numpy()
        result = evaluate_binary_classification(data, reduced_preds, threshold=0.5)
        logger.info(
            f"AUC Result of fold {k + 1} of cv {i + 1} of neg {j + 1} is {result.auc}"
        )
        # general_cv_result.add_fold_result(result)
        return result

    tasks = []
    for i in range(num_cross_validation):
        for k in range(5):
            preds_file = os.path.join(
                save_preds_dir, f"cv_{i + 1}", f"fold_{k + 1}.csv"
            )
            logger.info(f"Reading predictions from {preds_file}")
            df = pd.read_csv(preds_file)
            associations = df.iloc[:, :-1].to_numpy()

            for j in range(num_negative_sampling):
                tasks.append(dask.delayed(task)(i, k, j, associations, df))
    if parallel:
        local_cluster = LocalCluster(
            n_workers=int(os.getenv("NUM_WORKERS")),
            threads_per_worker=int(os.getenv("THREADS_PER_WORKER")),
        )
        with (
            Client(local_cluster) as client,
            tqdm(total=len(tasks), desc="Calc Result of RCV") as pbar,
        ):
            futures = client.compute(tasks)

            for future in dask.distributed.as_completed(futures):
                pbar.update(1)
                general_cv_result.add_fold_result(future.result())

            local_cluster.close()
    else:
        for task in tqdm(tasks, desc="Calc Result of RCV"):
            general_cv_result.add_fold_result(task.compute())

    general_cv_result.calculate_cv_result()
    return general_cv_result
