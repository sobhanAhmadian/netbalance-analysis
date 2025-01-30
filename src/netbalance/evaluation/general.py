import os
from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from netbalance.configs import OptimizerConfig
from netbalance.data import TrainTestSplitter
from netbalance.data.bipartite_graph_data import BGData
from netbalance.models import HandlerFactory
from netbalance.optimization.interface import Trainer
from netbalance.utils import get_header_format, prj_logger

from .result import BGCCrossValidationResult
from .utils import evaluate_binary_classification

logger = prj_logger.getLogger(__name__)


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
            )


def cross_validation(
    train_test_spliter: TrainTestSplitter,
    handler_factory: HandlerFactory,
    trainer: Trainer,
    config: OptimizerConfig,
    save_preds_dir: str,
    test_batch_size: int = 1000,
    pbar: tqdm = None,
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
    """

    k = train_test_spliter.k
    logger.info(f"Start {k}-fold Cross Validation with config: {config.exp_name}")

    for i in range(k):
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
                a_nodes=test_data.associations[j : j + test_batch_size, 0],
                b_nodes=test_data.associations[j : j + test_batch_size, 1],
            )
        logger.info("Predictions generated.")

        _save_predictions(preds, test_data.associations, save_preds_file)

        # Destroy the model handler
        model_handler.destroy()

        if pbar is not None:
            pbar.update(1)


def _save_predictions(predictions: np.ndarray, associations: np.ndarray, file: str):
    """_summary_

    Args:
        predictions (np.ndarray): _description_
        associations (np.ndarray): _description_
        file (str): _description_
    """
    with open(file, "w") as f:
        f.write("Node A,Node B,Association,Score\n")
        for i in range(len(associations)):
            f.write(
                f"{associations[i, 0]},{associations[i, 1]},{associations[i, 2]},{predictions[i]}\n"
            )
        logger.info(f"Predictions saved to {file}")


def get_result_of_rcv(
    save_preds_dir: str,
    cluster_a_node_names: list,
    cluster_b_node_names: list,
    num_negative_sampling: int,
    num_cross_validation: int,
    test_balance_method: Union[str, None] = "beta",
    test_balance_kwargs: dict = {},
    test_balance_negative_ratio: float = 1.0,
    dataset_name: str = None,
):
    logger.info(get_header_format("Repeated Cross Validation From Prediction Files"))
    general_cv_result = BGCCrossValidationResult()

    with tqdm(
        total=num_cross_validation * 5 * num_negative_sampling,
        desc="Calc Result of RCV",
    ) as pbar:
        for i in range(num_cross_validation):
            for k in range(5):
                preds_file = os.path.join(
                    save_preds_dir, f"cv_{i + 1}", f"fold_{k + 1}.csv"
                )
                logger.info(f"Reading predictions from {preds_file}")
                df = pd.read_csv(preds_file)
                associations = df.iloc[:, :3].to_numpy()

                for j in range(num_negative_sampling):
                    save_name = None
                    if dataset_name is not None and test_balance_method is not None:
                        save_name = f"dataset_{dataset_name}_{"test"}_cv_{i + 1}_fold_{k + 1}_neg_{j + 1}"
                        save_name += f"_met_{test_balance_method}_rat_{test_balance_negative_ratio}"
                        for key, value in test_balance_kwargs.items():
                            save_name += f"_{key}_{value}"
                    data = BGData(
                        associations=associations,
                        cluster_a_node_names=cluster_a_node_names,
                        cluster_b_node_names=cluster_b_node_names,
                    )
                    if test_balance_method is not None:
                        data.balance_data(
                            balance_method=test_balance_method,
                            negative_ratio=test_balance_negative_ratio,
                            seed=j,
                            save_name=save_name,
                            **test_balance_kwargs,
                        )
                        reduced_preds = np.array(
                            [
                                df.loc[df.iloc[:, 0] == indi]
                                .loc[df.iloc[:, 1] == indj]
                                .iloc[:, 3]
                                .item()
                                for indi, indj, _ in data.associations
                            ]
                        ).flatten()
                    else:
                        reduced_preds = df.iloc[:, 3].to_numpy()
                    result = evaluate_binary_classification(
                        data, reduced_preds, threshold=0.5
                    )
                    logger.info(
                        f"AUC Result of fold {k + 1} of cv {i + 1} of neg {j + 1} is {result.auc}"
                    )
                    general_cv_result.add_fold_result(result)
                    pbar.update(1)

    general_cv_result.calculate_cv_result()
    return general_cv_result
