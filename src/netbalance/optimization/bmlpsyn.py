import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from netbalance.configs.bmlpsyn import BMLPSYNOptimizerConfig
from netbalance.data import PytorchData
from netbalance.data.association_data import TGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.bmlpsyn import BMLPSYNModelHandler
from netbalance.optimization.simple_pytorch import PytorchTrainer
from netbalance.utils import get_header_format, prj_logger

from .interface import Trainer

logger = prj_logger.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hash_numpy_array(arr):
    """Hashes a NumPy array using SHA256."""
    arr_bytes = arr.tobytes()  # Convert to bytes
    hash_obj = hashlib.sha256(arr_bytes)  # Compute hash
    return hash_obj.hexdigest()  # Return as hexadecimal string


class BMLPSYNTrainer(Trainer):

    def train(
        self,
        model_handler: BMLPSYNModelHandler,
        data: TGData,
        config: BMLPSYNOptimizerConfig,
    ) -> Result:
        logger.info(get_header_format("Training the model"))
        model_handler.fe.build()

        associations = data.associations
        dp_embed = model_handler.fe.extract_features(
            associations[:, 0], associations[:, 1], associations[:, 2]
        )
        y = np.array(associations[:, -1].tolist(), dtype=np.float32).reshape(-1, 1)

        model_handler.model.to(device)

        simple_data = PytorchData(
            X=torch.tensor(dp_embed, dtype=torch.float32).to(device),
            y=torch.tensor(y, dtype=torch.float32).to(device),
        )
        pytorch_trainer = PytorchTrainer()
        pytorch_trainer.train(model_handler, simple_data, config)

        preds = model_handler.predict(
            [data.associations[:, i] for i in range(data.associations.shape[1] - 1)]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, -1], preds.reshape(-1), config.threshold
        )
        return result


class BalanceBMLPSYNTrainer(Trainer):

    def train(
        self,
        model_handler: BMLPSYNModelHandler,
        data: TGData,
        config: BMLPSYNOptimizerConfig,
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

        def task(num_bal):
            save_name = (
                f"i{config.i_balance_method}_bmlpsyn/{hash_associations}_{num_bal}"
            )
            logger.info(
                f"Parallel balancing data with {config.i_balance_method} in epoch {num_bal}"
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

        # Multi thread run tasks for num_bal in range 1 to 20
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(task, range(0, 20))

        for e in range(config.n_epoch):
            num_bal = e % 20
            save_name = (
                f"i{config.i_balance_method}_bmlpsyn/{hash_associations}_{num_bal}"
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
                X=torch.tensor(dp_embed).to(config.device),
                y=torch.tensor(y).to(config.device),
            )

            pytorch_trainer = PytorchTrainer()
            pytorch_trainer.train(model_handler, simple_data, e_config)

        ###########################################################

        test_batch_size = 1000
        preds = np.zeros(data.associations.shape[0])
        for j in range(0, data.associations.shape[0], test_batch_size):
            preds[j : j + test_batch_size] = model_handler.predict(
                a_nodes=data.associations[j : j + test_batch_size, 0],
                b_nodes=data.associations[j : j + test_batch_size, 1],
            )
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result
