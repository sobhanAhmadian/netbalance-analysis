from typing import Union

import numpy as np
import torch
from anyio import Path
from sklearn.linear_model import LogisticRegression
from torch import sigmoid

from netbalance.configs.blinphi import BLINPHI_PROCESSED_DATA_DIR, BLINPHIModelConfig
from netbalance.configs.ecoli import ECOLI_PROCESSED_DATA_DIR
from netbalance.methods import FeatureExtractor
from netbalance.models.interface import AModelHandler, HandlerFactory
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BLINPHIFeatureExtractor(FeatureExtractor):

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


class BLINPHIModelHandler(AModelHandler):

    def __init__(self, model_config: BLINPHIModelConfig) -> None:
        super().__init__(model_config)

    def destroy(self):
        del self.model
        del self.fe

    def predict_impl(self, node_lists: list[np.ndarray]):
        a_nodes, b_nodes = node_lists
        features = self.fe.extract_features(a_nodes, b_nodes)
        preds = self.model.predict_proba(features)[:, 1]
        return preds

    def summary(self):
        pass

    def _build_model(self):
        return LogisticRegression(random_state=0, max_iter=10000)

    def _build_feature_extractor(self):
        return BLINPHIFeatureExtractor()


class BLINPHIHandlerFactory(HandlerFactory):

    def __init__(self, model_config: BLINPHIModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

    def create_handler(self) -> BLINPHIModelHandler:
        return BLINPHIModelHandler(self.model_config)
