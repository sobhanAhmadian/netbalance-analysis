import copy
import hashlib
import os

import numpy as np

from netbalance.configs.bmlpdti import BMLPDTIModelConfig as ModelConfig
from netbalance.configs.bmlpdti import BMLPDTIOptimizerConfig as OptimizerConfig
from netbalance.configs.common import RESULTS_DIR
from netbalance.data.association_data import BGData, BGTrainTestSpliter
from netbalance.features.luodti import LuoDTIDataset as Dataset
from netbalance.models.bmlpdti import BMLPDTIHandlerFactory as HandlerFactory
from netbalance.optimization.bmlpdti import BalanceBMLPDTITrainer2 as Trainer
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


def _save_predictions(predictions: np.ndarray, associations: np.ndarray, file: str):
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


dataset = "luodti"

save_dir = os.path.join(
    RESULTS_DIR, "numeric", "other", f"dataset-{dataset}-delta-comparison", "preds"
)
os.makedirs(save_dir, exist_ok=True)

ds = Dataset()


def hash_numpy_array(arr):
    """Hashes a NumPy array using SHA256."""
    arr_bytes = arr.tobytes()  # Convert to bytes
    hash_obj = hashlib.sha256(arr_bytes)  # Compute hash
    return hash_obj.hexdigest()  # Return as hexadecimal string


associations = ds.get_associations(with_negatives=True)
bg_data = BGData(
    associations=ds.get_associations(with_negatives=True),
    cluster_a_node_names=ds.get_cluster_a_node_names(),
    cluster_b_node_names=ds.get_cluster_b_node_names(),
)

spliter = BGTrainTestSpliter(data=bg_data, seed=0, k=5)

train_data, test_data = spliter.split(0)


def balance_data(
    input_data,
    seed,
    max_iter,
    delta,
    cooling_rate,
    initial_temp,
):
    save_name = f"sa_parameters_irho_bmlpdti_seed{seed}_iter{max_iter}_delta{delta}_cooling{cooling_rate}_temp{initial_temp}"
    input_data.balance_data(
        balance_method="rho",
        negative_ratio=1.0,
        seed=seed,
        save_name=save_name,
        ent_desired=1.0,
        shrinkage=1.0,
        max_iter=max_iter,
        delta=delta,
        cooling_rate=cooling_rate,
        initial_temp=initial_temp,
    )
    return input_data.associations


# delta
for delta in [
    0.0,
    0.1,
    2.0,
]:
    temp_train_data = copy.deepcopy(train_data)

    model_config = ModelConfig()
    model_config.drug_num = len(ds.get_cluster_a_node_names())
    model_config.protein_num = len(ds.get_cluster_b_node_names())
    model_config.input_dim = len(ds.get_cluster_a_node_names()) + len(
        ds.get_cluster_b_node_names()
    )
    model_config.hidden_dim = 64

    optimizer_config = OptimizerConfig()  # Parameter
    optimizer_config.fair = True
    optimizer_config.n_epoch = 600
    optimizer_config.lr = 0.001
    optimizer_config.associations_list = [
        balance_data(
            input_data=copy.deepcopy(train_data),
            seed=seed,
            max_iter=40000,
            delta=delta,
            cooling_rate=0.99,
            initial_temp=40.0,
        )
        for seed in range(30)
    ]
    print("Finished balancing data with delta =", delta)

    trainer = Trainer()
    factory = HandlerFactory(model_config=model_config)

    save_preds_file = os.path.join(
        save_dir,
        f"delta_{delta}_preds.csv",
    )

    model_handler = factory.create_handler()

    trainer.train(
        model_handler=model_handler, data=temp_train_data, config=optimizer_config
    )
    print("Finished training model with delta =", delta)

    test_batch_size = 64
    preds = np.zeros(test_data.associations.shape[0])
    for j in range(0, test_data.associations.shape[0], test_batch_size):
        preds[j : j + test_batch_size] = model_handler.predict(
            [
                test_data.associations[j : j + test_batch_size, r]
                for r in range(test_data.associations.shape[1] - 1)
            ],
        )

    _save_predictions(preds, test_data.associations, save_preds_file)
    print(f"Saved predictions to {save_preds_file}")
