from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import SGD

from netbalance.configs.midti import MIDTIOptimizerConfig
from netbalance.data.association_data import BGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.midti import MIDTIModelHandler
from netbalance.utils import get_header_format, prj_logger
from netbalance.utils.learning_utils import statified_train_test_sampler

from .interface import Trainer

logger = prj_logger.getLogger(__name__)


class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, k=0, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

        self.optimizer

        self.defaults = optimizer.defaults
        # Add required optimizer hooks
        self._optimizer_step_pre_hooks = {}
        self._optimizer_step_post_hooks = {}
        self._optimizer_state_dict_hooks = {}
        self._optimizer_load_state_dict_pre_hooks = {}
        self._optimizer_load_state_dict_post_hooks = {}

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class MIDTITrainer(Trainer):

    def train(
        self,
        model_handler: MIDTIModelHandler,
        data: BGData,
        config: MIDTIOptimizerConfig,
    ) -> Result:
        logger.info(get_header_format("Training the model"))

        positive_associations = data.associations[data.associations[:, 2] == 1]
        negative_associations = data.associations[data.associations[:, 2] == 0]
        (
            train_positive_indices,
            train_negative_indices,
            dev_positive_indices,
            dev_negative_indices,
        ) = statified_train_test_sampler(
            positive_associations.shape[0],
            negative_associations.shape[0],
            train_ratio=0.9,
            seed=0,
        )
        train_positive_associations = positive_associations[train_positive_indices]
        train_negative_associations = negative_associations[train_negative_indices]
        dev_positive_associations = positive_associations[dev_positive_indices]
        dev_negative_associations = negative_associations[dev_negative_indices]
        train_associations = np.concatenate(
            [train_positive_associations, train_negative_associations], axis=0
        )
        dev_associations = np.concatenate(
            [dev_positive_associations, dev_negative_associations], axis=0
        )

        train_data = BGData(
            associations=train_associations,
            cluster_a_node_names=data.cluster_a_node_names,
            cluster_b_node_names=data.cluster_b_node_names,
        )
        val_data = BGData(
            associations=dev_associations,
            cluster_a_node_names=data.cluster_a_node_names,
            cluster_b_node_names=data.cluster_b_node_names,
        )
        data = train_data

        if config.fair:
            model_handler.fe.build(train_data.associations)
        else:
            model_handler.fe.build()

        optimizer_inner = SGD(
            model_handler.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)

        es = 0
        best_auc = 0.0
        for epoch in range(config.n_epoch):
            logger.info("---------epoch{}---------".format(epoch))
            loss = self._train_batch(
                drug_nodes=torch.tensor(data.associations[:, 0], dtype=torch.long).to(
                    config.device
                ),
                protein_nodes=torch.tensor(
                    data.associations[:, 1], dtype=torch.long
                ).to(config.device),
                interactions=torch.tensor(data.associations[:, 2], dtype=torch.long).to(
                    config.device
                ),
                model_handler=model_handler,
                optimizer=optimizer,
                criterion=config.criterion,
                device=config.device,
            )
            logger.info("loss:{}".format(loss))

            # validation
            val_preds = model_handler.predict(
                [val_data.associations[:, 0], val_data.associations[:, 1]]
            )
            val_result = evaluate_binary_classification_simple(
                val_data.associations[:, 2], val_preds.reshape(-1), config.threshold
            )
            val_auc = val_result.auc
            logger.info("val_auc:{}".format(val_auc))
            if val_auc > best_auc:
                best_auc = val_auc
                es = 0  # early stop mechanism
                logger.info("get higher val_auc")
                model_handler.save_model()
            else:
                es += 1
                if es > config.es_patience:
                    logger.info(
                        f"Early stopping counter reaches to {es}, the training will stop"
                    )
                    break

        model_handler.load_model()

        preds = model_handler.predict(
            [data.associations[:, 0], data.associations[:, 1]]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, 2], preds.reshape(-1), config.threshold
        )
        return result

    @staticmethod
    def _train_batch(
        drug_nodes: torch.Tensor,
        protein_nodes: torch.Tensor,
        interactions: torch.Tensor,
        model_handler: MIDTIModelHandler,
        criterion,
        optimizer: Lookahead,
        device: str = "cpu",
    ):

        _, predicted = model_handler.model(
            drug_nodes, protein_nodes, model_handler.fe.feature_dataset
        )

        criterion = criterion.to(device)
        loss = criterion(predicted, interactions)

        loss.backward()
        clip_grad_norm_(parameters=model_handler.model.parameters(), max_norm=5)
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
