import copy
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor

import dask
import numpy as np
import torch
from dask.distributed import Client, LocalCluster
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from netbalance.configs.bmlpdti import BMLPDTIOptimizerConfig
from netbalance.data import PytorchData
from netbalance.data.association_data import BGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.bmlpdti import BMLPDTIModelHandler
from netbalance.optimization.simple_pytorch import PytorchTrainer
from netbalance.utils import get_header_format, prj_logger

from .interface import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hash_numpy_array(arr):
    """Hashes a NumPy array using SHA256."""
    arr_bytes = arr.tobytes()  # Convert to bytes
    hash_obj = hashlib.sha256(arr_bytes)  # Compute hash
    return hash_obj.hexdigest()  # Return as hexadecimal string


logger = prj_logger.getLogger(__name__)


class BMLPDTITrainer(Trainer):

    def train(
        self,
        model_handler: BMLPDTIModelHandler,
        data: BGData,
        config: BMLPDTIOptimizerConfig,
    ) -> Result:
        logger.info(get_header_format("Training the model"))

        if config.fair:
            model_handler.fe.build(data.associations)
        else:
            model_handler.fe.build()

        associations = data.associations
        dp_embed = model_handler.fe.extract_features(
            associations[:, 0], associations[:, 1]
        ).numpy()
        y = np.array(associations[:, 2].tolist(), dtype=np.float32).reshape(-1, 1)

        model_handler.model.to(config.device)

        simple_data = PytorchData(
            X=torch.tensor(dp_embed).to(config.device),
            y=torch.tensor(y).to(config.device),
        )
        pytorch_trainer = PytorchTrainer()
        pytorch_trainer.train(model_handler, simple_data, config)

        preds = model_handler.predict(
            [data.associations[:, 0], data.associations[:, 1]]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result


class BalanceBMLPDTITrainer(Trainer):

    def train(
        self,
        model_handler: BMLPDTIModelHandler,
        data: BGData,
        config: BMLPDTIOptimizerConfig,
    ) -> Result:
        logger.info(get_header_format("Training the model"))

        if config.fair:
            model_handler.fe.build(data.associations)
        else:
            model_handler.fe.build()

        ###########################################################
        hash_associations = hash_numpy_array(data.associations)
        logger.info(f"hash of associations: {hash_associations}")

        e_config = copy.deepcopy(config)
        e_config.n_epoch = 1

        def task(num_bal, input_data):
            save_name = (
                f"i{config.i_balance_method}_bmlpdti_{hash_associations}_{num_bal}"
            )
            logger.info(
                f"Parallel balancing data with {config.i_balance_method} in epoch {num_bal}"
            )
            input_data.balance_data(
                balance_method=config.i_balance_method,
                negative_ratio=config.i_negative_ratio,
                seed=num_bal,
                save_name=save_name,
                **config.i_balance_kwargs,
            )
            logger.info(
                f"balanced data with {config.i_balance_method} in epoch {num_bal}"
            )

        tasks = [
            dask.delayed(task)(num_bal, copy.deepcopy(data))
            for num_bal in range(0, config.i_max_num_bal)
        ]

        dask.compute(*tasks)

        for e in range(config.n_epoch):
            num_bal = e % config.i_max_num_bal
            save_name = (
                f"i{config.i_balance_method}_bmlpdti_{hash_associations}_{num_bal}"
            )

            e_data = copy.deepcopy(data)
            e_data.balance_data(
                balance_method=config.i_balance_method,
                negative_ratio=config.i_negative_ratio,
                seed=num_bal,
                save_name=save_name,
                **config.i_balance_kwargs,
            )
            logger.info(f"balanced data with {config.i_balance_method} in epoch {e}")

            associations = e_data.associations
            dp_embed = model_handler.fe.extract_features(
                associations[:, 0], associations[:, 1]
            ).numpy()
            y = np.array(associations[:, 2].tolist(), dtype=np.float32).reshape(-1, 1)
            simple_data = PytorchData(
                X=torch.tensor(dp_embed).to(device),
                y=torch.tensor(y).to(device),
            )

            pytorch_trainer = PytorchTrainer()
            pytorch_trainer.train(model_handler, simple_data, e_config)

        ###########################################################

        test_batch_size = 1000
        preds = np.zeros(data.associations.shape[0])
        for j in range(0, data.associations.shape[0], test_batch_size):
            preds[j : j + test_batch_size] = model_handler.predict(
                [
                    data.associations[j : j + test_batch_size, 0],
                    data.associations[j : j + test_batch_size, 1],
                ]
            )
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result


class WeightedBMLPDTITrainer(Trainer):

    def train(
        self,
        model_handler: BMLPDTIModelHandler,
        data: BGData,
        config: BMLPDTIOptimizerConfig,
    ) -> Result:
        logger.info(get_header_format("Training the model"))

        if config.fair:
            model_handler.fe.build(data.associations)
        else:
            model_handler.fe.build()
        logger.info("feature extractor has been built")

        associations = data.associations
        dp_embed = model_handler.fe.extract_features(
            associations[:, 0], associations[:, 1]
        ).numpy()
        y = np.array(associations[:, 2].tolist(), dtype=np.float32).reshape(-1, 1)
        logger.info(f"Initial X shape: {dp_embed.shape}, y shape: {y.shape}")

        model_handler.model.to(config.device)

        ###########################################################

        w = self._get_weights(associations)
        logger.info(
            f"weights have been calculated with shape: {w.shape}, with sum: {w.sum()}"
        )

        model_handler.model = model_handler.model.to(config.device)
        dataset = TensorDataset(
            torch.tensor(dp_embed).to(config.device),
            torch.tensor(y).to(config.device),
            torch.tensor(w, dtype=torch.float).to(config.device),
        )
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        self._batch_optimize(loader, model_handler, config)

        ###########################################################

        preds = model_handler.predict(
            [data.associations[:, 0], data.associations[:, 1]]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result

    def _batch_optimize(self, loader, model_handler, config):
        model_handler.model.train()
        optimizer = torch.optim.Adam(model_handler.model.parameters(), lr=config.lr)
        loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")

        for epoch in range(config.n_epoch):
            running_loss = 0.0
            for j, data in enumerate(loader, 0):
                X, y, w = data

                pred = model_handler.model(X)
                per_sample_loss = loss_function(pred, y)
                loss = w.reshape(1, -1) @ per_sample_loss

                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.info(f"epoch: {epoch}, loss: {running_loss}")

    @staticmethod
    def _get_weights(associations):

        a_pos = []
        b_pos = []
        a_neg = []
        b_neg = []

        a_node_ids = list(set(associations[:, 0]))
        b_node_ids = list(set(associations[:, 1]))

        for node_id in a_node_ids:
            p_count = 0
            n_count = 0
            for a in associations:
                if a[0] == node_id:
                    if a[2] == 1:
                        p_count += 1
                    else:
                        n_count += 1
            a_pos.append(p_count)
            a_neg.append(n_count)

        for node_id in b_node_ids:
            p_count = 0
            n_count = 0
            for a in associations:
                if a[1] == node_id:
                    if a[2] == 1:
                        p_count += 1
                    else:
                        n_count += 1

            b_pos.append(p_count)
            b_neg.append(n_count)

        a_nodes = associations[:, 0].tolist()
        b_nodes = associations[:, 1].tolist()
        scores = []
        for i in range(len(a_nodes)):
            a_node_id = a_node_ids.index(a_nodes[i])
            b_node_id = b_node_ids.index(b_nodes[i])

            a_node_pos = a_pos[a_node_id]
            a_node_neg = a_neg[a_node_id]
            b_node_pos = b_pos[b_node_id]
            b_node_neg = b_neg[b_node_id]

            if associations[i][2] == 1:
                a_score = a_node_neg / (a_node_pos * a_node_neg + 1e-6)
                b_score = b_node_neg / (b_node_pos * b_node_neg + 1e-6)
            else:
                a_score = a_node_pos / (a_node_pos * a_node_neg + 1e-6)
                b_score = b_node_pos / (b_node_pos * b_node_neg + 1e-6)

            w_a = (a_node_pos + a_node_neg) / (
                a_node_pos + a_node_neg + b_node_pos + b_node_neg + 1e-6
            )
            w_b = (b_node_pos + b_node_neg) / (
                a_node_pos + a_node_neg + b_node_pos + b_node_neg + 1e-6
            )

            s = w_a * a_score + w_b * b_score
            scores.append(s)

        scores = np.array(scores)
        scores = scores / scores.max()
        logger.info(f"scores between 0 and 1: {scores.min()}, {scores.max()}")
        return scores
