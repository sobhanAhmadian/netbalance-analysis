import csv
import os
from typing import Union

from anyio import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import sigmoid

from netbalance.configs.bmlpphi import BMLPPHI_PROCESSED_DATA_DIR, BMLPPHIModelConfig
from netbalance.configs.ecoli import ECOLI_PROCESSED_DATA_DIR
from netbalance.methods import FeatureExtractor
from netbalance.models.interface import AModelHandler, HandlerFactory
from netbalance.models.modules.simple_mlp import SimpleMLP
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BMLPPHIFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
    ):
        super().__init__()
        self.strain_file = Path(f"{ECOLI_PROCESSED_DATA_DIR}/strains-features.csv")
        self.phage_file = Path(f"{ECOLI_PROCESSED_DATA_DIR}/phages-features.csv")

    def build(self):
        self.strain_features = np.loadtxt(self.strain_file, delimiter=",").astype(
            np.float32
        )
        self.phage_features = np.loadtxt(self.phage_file, delimiter=",").astype(
            np.float32
        )

    def extract_features(
        self,
        a_nodes: Union[list[int], np.ndarray],
        b_nodes: Union[list[int], np.ndarray],
    ):
        bac = self.strain_features[a_nodes]
        pha = self.phage_features[b_nodes]
        features = np.concatenate([bac, pha], axis=1)
        return features


class BMLPPHIModelHandler(AModelHandler):

    def __init__(self, model_config: BMLPPHIModelConfig) -> None:
        super().__init__(model_config)

    def destroy(self):
        del self.model
        del self.fe

    def predict_impl(self, node_lists: list[np.ndarray]):
        a_nodes, b_nodes = node_lists
        dp_embedd = self.fe.extract_features(a_nodes, b_nodes)
        dp_embedd = torch.tensor(dp_embedd).to(self.model_config.device)
        preds = sigmoid(self.model(dp_embedd).flatten()).cpu().detach().numpy()
        return preds

    def summary(self):
        pass

    def _build_model(self):
        return SimpleMLP(**self.model_config.get_model_kwargs())

    def _build_feature_extractor(self):
        return BMLPPHIFeatureExtractor()

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path + ".pt")

    def load_model(self, path: str):
        self.model.load_state_dict(
            torch.load(path + ".pt", map_location=self.model_config.device)
        )
        self.model.to(self.model_config.device)


class BMLPPHIHandlerFactory(HandlerFactory):

    def __init__(self, model_config: BMLPPHIModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

    def create_handler(self) -> BMLPPHIModelHandler:
        return BMLPPHIModelHandler(self.model_config)


if __name__ == "__main__":
    feature_extractor = BMLPPHIFeatureExtractor()
    feature_extractor.build()
    print(feature_extractor.extract_features([0, 1], [0, 3]).shape)

    model_config = BMLPPHIModelConfig()
    model_config.input_dim = feature_extractor.extract_features([0, 1], [0, 3]).shape[1]
    model_handler = BMLPPHIModelHandler(model_config)
    model_handler.fe.build()
    print(model_handler.predict_impl([[0, 1, 2], [0, 3, 4]]))
