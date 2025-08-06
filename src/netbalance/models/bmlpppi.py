from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch

from netbalance.configs.bernett import (
    BERNETT_NUCLEOTIDE_FEATURES_FILE,
)
from netbalance.configs.bmlpppi import (
    BMLPPPIModelConfig,
)
from netbalance.methods.general import FeatureExtractor
from netbalance.models.modules.simple_mlp import SimpleMLP
from netbalance.utils import prj_logger

from .interface import AModelHandler, HandlerFactory

logger = prj_logger.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BMLPPPIFeatureExtractor(FeatureExtractor):

    def build(self):
        self.features = np.loadtxt(BERNETT_NUCLEOTIDE_FEATURES_FILE, delimiter=",")

    def extract_features(
        self,
        a_nodes: Union[list[int], np.ndarray],
        b_nodes: Union[list[int], np.ndarray],
    ):
        d1 = self.features[a_nodes]
        d2 = self.features[b_nodes]
        features = np.concatenate([d1, d2], axis=1)
        return features


class BMLPPPIModelHandler(AModelHandler):
    """Baseline model which generates a random number between 0 and 1 as prediction."""

    def __init__(
        self,
        model_config: BMLPPPIModelConfig,
    ):
        super().__init__(model_config)

    def predict_impl(self, node_lists: Union[list[int], np.ndarray]):
        a_nodes, b_nodes = node_lists
        features = self.fe.extract_features(a_nodes, b_nodes)
        features = torch.tensor(features, dtype=torch.float32).to(device)
        return torch.sigmoid(self.model(features)).reshape(-1).cpu().detach().numpy()

    def destroy(self):
        del self.model
        del self.fe

    def summary(self):
        raise NotImplementedError

    def _build_model(self):
        return SimpleMLP(**self.model_config.get_model_kwargs()).to(device)

    def _build_feature_extractor(self):
        return BMLPPPIFeatureExtractor()


class BMLPPPIHandlerFactory(HandlerFactory):
    def __init__(self, model_config: BMLPPPIModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

    def create_handler(self) -> BMLPPPIModelHandler:
        return BMLPPPIModelHandler(self.model_config)
