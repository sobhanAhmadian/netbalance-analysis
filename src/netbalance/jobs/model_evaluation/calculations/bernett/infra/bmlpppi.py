import os

import numpy as np
import torch

from netbalance.configs.bmlpppi import (
    BMLPPPI_RESULTS_DIR as RESULTS_DIR,  # Parameter
)
from netbalance.configs.bmlpppi import (
    BMLPPPIModelConfig as ModelConfig,  # Parameter
)
from netbalance.configs.bmlpppi import (
    BMLPPPIOptimizerConfig as OptimizerConfig,  # Parameter
)
from netbalance.data.association_data import BGData, BGTrainTestSpliter
from netbalance.features.bernett import BernettDataset as Dataset  # Parameter
from netbalance.models.bmlpppi import (
    BMLPPPIHandlerFactory as HandlerFactory,  # Parameter
)
from netbalance.optimization.bmlpppi import BMLPPPITrainer as Trainer  # Parameter
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


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


model_name = "bmlpppi"  # Parameter
dataset = "sanger"  # Parameter
train_neg_samp_method = "beta"  # Parameter
test_batch_size = 1000  # Parameter

model_result_dir = os.path.join(
    RESULTS_DIR,
    f"preds",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)
os.makedirs(model_result_dir, exist_ok=True)


logger.info(
    f">>>>>>>>>>>>>>>>> Job: Model Evaluation - {dataset} - {model_name} - {train_neg_samp_method}"
)

ds_infra0 = Dataset(type="intra0")
ds_infra1 = Dataset(type="intra1")
ds_infra2 = Dataset(type="intra2")

optimizer_config = OptimizerConfig()  # Parameter
optimizer_config.n_epoch = 100
model_config = ModelConfig()  # Parameter

train_data = BGData(
    associations=ds_infra1.get_associations(with_negatives=False),
    cluster_a_node_names=ds_infra1.get_cluster_a_node_names(),
    cluster_b_node_names=ds_infra1.get_cluster_b_node_names(),
)
val_data = BGData(
    associations=ds_infra0.get_associations(with_negatives=False),
    cluster_a_node_names=ds_infra1.get_cluster_a_node_names(),
    cluster_b_node_names=ds_infra1.get_cluster_b_node_names(),
)
test_data = BGData(
    associations=ds_infra2.get_associations(with_negatives=False),
    cluster_a_node_names=ds_infra1.get_cluster_a_node_names(),
    cluster_b_node_names=ds_infra1.get_cluster_b_node_names(),
)

trainer = Trainer()
factory = HandlerFactory(model_config=model_config)
model_handler = factory.create_handler()
result = trainer.train(
    model_handler=model_handler, data=train_data, config=optimizer_config
)
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

save_preds_file = os.path.join(
    model_result_dir,
    "infra2.csv"
)
_save_predictions(preds, test_data.associations, save_preds_file)

# Destroy the model handler
model_handler.destroy()
