import os.path

import torch
from torch.utils.data import DataLoader, TensorDataset

from netbalance.configs import OptimizerConfig
from netbalance.data import PytorchData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models import ModelHandler
from netbalance.utils import prj_logger

from .interface import Trainer

logger = prj_logger.getLogger(__name__)


def _predict_error(X, loss_function, model_handler: ModelHandler, running_loss, y):
    pred = model_handler.model(X)
    loss = loss_function(pred, y)
    running_loss += loss.item()
    return loss, running_loss


def _backpropagation(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _batch_optimize(
    loader,
    model_handler: ModelHandler,
    config: OptimizerConfig,
):
    model_handler.model.train()
    optimizer = config.optimizer(model_handler.model.parameters(), lr=config.lr)

    for epoch in range(config.n_epoch):
        running_loss = 0.0
        for j, data in enumerate(loader, 0):
            X, y = data

            loss, running_loss = _predict_error(
                X, config.criterion, model_handler, running_loss, y
            )
            _backpropagation(loss, optimizer)

            if j % config.report_size == config.report_size - 1:
                loss = running_loss / (config.report_size * config.batch_size)
                logger.info(f"loss: {loss:.4f}    [{epoch + 1}, {j + 1:5d}]")
                running_loss = 0

        if config.save:
            if epoch % 5 == 0 or epoch == config.n_epoch - 1:
                m = os.path.join(
                    config.save_path, model_handler.model_config.model_name + ".pth"
                )
                torch.save(model_handler.classifier.state_dict(), m)


def _evaluate(model: ModelHandler, loader, config: OptimizerConfig):
    model.model.eval()
    total_labels = []
    total_predictions = []
    total_loss = 0.0
    for data in loader:
        inputs, labels = data
        outputs = model.model(inputs).detach()
        total_loss += config.criterion(outputs, labels).item()
        total_labels.extend(labels.cpu().numpy())
        total_predictions.extend(torch.sigmoid(outputs).cpu().numpy())
    result = evaluate_binary_classification_simple(
        total_labels, total_predictions, config.threshold
    )
    result.loss = total_loss / len(loader)
    return result


class PytorchTrainer(Trainer):
    def train(
        self,
        model_handler: ModelHandler,
        data: PytorchData,
        config: OptimizerConfig,
    ) -> Result:
        """
        Trains the Pytorch model using the provided data and configuration.

        Args:
            model_handler (ModelHandler): The model handler object with a PyTorch model.
            data (PytorchData): The PyTorch data object containing the training data.
            config (OptimizerConfig): The configuration object for the optimizer.

        Returns:
            Result: The result of the training process.
        """
        logger.info(
            "{:#^50}".format(f"   Running PyTorch Trainer : {config.exp_name}  ")
        )

        model_handler.model = model_handler.model.to(config.device)
        dataset = TensorDataset(data.X.to(config.device), data.y.to(config.device))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        logger.info(f"Model and data moved to {config.device}")

        _batch_optimize(loader, model_handler, config)
        result = _evaluate(model_handler, loader, config)

        logger.info(f"Result on Train Data: {result.get_result()}")
        return result
