import os

import numpy as np
import shap
import torch

from netbalance.configs.bmlpphi import BMLPPHI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.configs.bmlpphi import BMLPPHIModelConfig as ModelConfig  # Parameter
from netbalance.configs.bmlpphi import (
    BMLPPHIOptimizerConfig as OptimizerConfig,  # Parameter
)
from netbalance.data.association_data import BGData, BGTrainTestSpliter
from netbalance.features.ecoli import EcoliDataset as Dataset  # Parameter
from netbalance.models.bmlpphi import (
    BMLPPHIHandlerFactory as HandlerFactory,  # Parameter
)
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

dataset = "ecoli"  # Parameter

save_dir = os.path.join(
    RESULTS_DIR,
    "numeric",
    "other",
    f"feature_importance",
    f"dataset-{dataset}",
    "train_neg_samp-irho",
)
os.makedirs(save_dir, exist_ok=True)

model_name = "bmlpphi"  # Parameter
train_neg_samp_method = "irho"  # Parameter


splitter_kwargs = {
    "k": 5,
    "train_balance": False,  # Parameter
    "train_balance_kwargs": {},  # Parameter
}

ds = Dataset()

associations = ds.get_associations(with_negatives=False)
bg_data = BGData(
    associations=ds.get_associations(with_negatives=False),
    cluster_a_node_names=ds.get_cluster_a_node_names(),
    cluster_b_node_names=ds.get_cluster_b_node_names(),
)

spliter = BGTrainTestSpliter(data=bg_data, seed=0, **splitter_kwargs)


model_config = ModelConfig()
model_config.hidden_dim = 256
model_config.output_dim = 1
model_config.num_layers = 3
model_config.dropout = 0.1

optimizer_config = OptimizerConfig()  # Parameter
optimizer_config.i_balance_method = "rho"
optimizer_config.i_balance_kwargs = {
    "max_iter": 20000,
    "delta": 0.1,
    "cooling_rate": 0.99,
    "initial_temp": 20.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}
optimizer_config.i_negative_ratio = 1.0
optimizer_config.i_max_num_bal = 30
optimizer_config.i_parallel_balance = True
optimizer_config.n_epoch = 200
optimizer_config.lr = 0.0001
optimizer_config.save = True
optimizer_config.save_path = os.path.join(
    RESULTS_DIR,
    f"models",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)
os.makedirs(optimizer_config.save_path, exist_ok=True)
name_suffix = (
    f"n_epoch_{optimizer_config.n_epoch}_lr_{optimizer_config.lr}_"
    + f"i_max_num_bal_{optimizer_config.i_max_num_bal}"
)
for key, value in optimizer_config.i_balance_kwargs.items():
    name_suffix += f"_i_balance_kwarg_{key}_{value}"
optimizer_config.save_path = os.path.join(optimizer_config.save_path, name_suffix)


factory = HandlerFactory(model_config=model_config)
for fold in range(splitter_kwargs["k"]):

    train_data, test_data = spliter.split(fold)

    model_handler = factory.create_handler()
    model_handler.fe.build()
    model_handler.load_model(optimizer_config.save_path + f"_cv_1_fold_{fold + 1}")

    model = model_handler.model.eval()

    a_nodes, b_nodes = test_data.associations[:, :2].T
    features = model_handler.fe.extract_features(a_nodes, b_nodes)
    features_tensor = torch.tensor(features, dtype=torch.float32).to(
        model_handler.model_config.device
    )

    # Background samples for SHAP (using a subset of the data)
    background_size = min(100, len(features_tensor))
    background_idx = np.random.choice(
        len(features_tensor), background_size, replace=False
    )
    background = features_tensor[background_idx]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(
        features_tensor
    )  # numpy array with the shape (num_test_samples, num_features, 1)
    shap_values = (
        shap_values.cpu().numpy() if torch.is_tensor(shap_values) else shap_values
    )

    shap_path = os.path.join(save_dir, f"shap_values_fold_{fold + 1}.npy")
    np.save(shap_path, shap_values)
    print(f"SHAP values saved to: {shap_path} with shape {shap_values.shape}")

    features_path = os.path.join(save_dir, f"features_fold_{fold + 1}.npy")
    np.save(features_path, features)
    print(f"Features saved to: {features_path} with shape {features.shape}")
