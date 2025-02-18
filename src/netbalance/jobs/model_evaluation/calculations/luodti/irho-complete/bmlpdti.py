import os

import numpy as np

from netbalance.configs.bmlpdti import BMLPDTI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.configs.bmlpdti import BMLPDTIModelConfig as ModelConfig  # Parameter
from netbalance.configs.bmlpdti import (
    BMLPDTIOptimizerConfig as OptimizerConfig,
)  # Parameter
from netbalance.data.association_graph_data import BGData
from netbalance.features.luodti import LuoDTIDataset as Dataset  # Parameter
from netbalance.models.bmlpdti import (
    BMLPDTIHandlerFactory as HandlerFactory,
)  # Parameter
from netbalance.optimization.bmlpdti import (
    BalanceBMLPDTITrainer as Trainer,
)  # Parameter
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

model_name = "bmlpdti"  # Parameter
dataset = "luodti"  # Parameter
train_neg_samp_method = "irho_complete"  # Parameter

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

ds = Dataset()

model_config = ModelConfig()
model_config.drug_num = len(ds.get_cluster_a_node_names())
model_config.protein_num = len(ds.get_cluster_b_node_names())
model_config.input_dim = len(ds.get_cluster_a_node_names()) + len(
    ds.get_cluster_b_node_names()
)
model_config.hidden_dim = 64

optimizer_config = OptimizerConfig()  # Parameter
optimizer_config.i_balance_method = "rho"
optimizer_config.i_balance_kwargs = {
    "max_iter": 40000,
    "delta": 0.1,
    "cooling_rate": 0.99,
    "initial_temp": 40.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}
optimizer_config.i_negative_ratio = 1.0
optimizer_config.fair = True
optimizer_config.n_epoch = 10
optimizer_config.lr = 0.001

associations = ds.get_associations(with_negatives=True)

trainer = Trainer()
factory = HandlerFactory(model_config=model_config)  # Parameter
model_handler = factory.create_handler()
data = BGData(
    associations=ds.get_associations(with_negatives=True),
    cluster_a_node_names=ds.get_cluster_a_node_names(),
    cluster_b_node_names=ds.get_cluster_b_node_names(),
)

# Train the model
trainer.train(model_handler=model_handler, data=data, config=optimizer_config)

# Generate Predictions
model_handler.model.eval()
batch_size = 1000
preds = np.zeros(data.associations.shape[0])
for j in range(0, data.associations.shape[0], batch_size):
    preds[j : j + batch_size] = model_handler.predict(
        a_nodes=data.associations[j : j + batch_size, 0],
        b_nodes=data.associations[j : j + batch_size, 1],
    )
logger.info("Predictions generated.")
file = f"{model_result_dir}/preds.csv"
with open(file, "w") as f:
    f.write("Node A,Node B,Association,Score\n")
    for i in range(len(data.associations)):
        f.write(
            f"{data.associations[i, 0]},{data.associations[i, 1]},{data.associations[i, 2]},{preds[i]}\n"
        )
    print(f"Predictions saved to {file}")
