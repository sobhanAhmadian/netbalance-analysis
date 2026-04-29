from typing import Union

import numpy as np
from anyio import Path
from xgboost import XGBClassifier

from netbalance.configs.bxgbphi import BXGBPHIModelConfig
from netbalance.configs.ecoli import ECOLI_PROCESSED_DATA_DIR
from netbalance.methods import FeatureExtractor
from netbalance.models.interface import AModelHandler, HandlerFactory
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class BXGBPHIFeatureExtractor(FeatureExtractor):

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


class BXGBPHIModelHandler(AModelHandler):

    def __init__(self, model_config: BXGBPHIModelConfig) -> None:
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
        return XGBClassifier(
            n_estimators=1000,
            max_depth=2,
            learning_rate=0.01,
            objective="binary:logistic",
        )

    def _build_feature_extractor(self):
        return BXGBPHIFeatureExtractor()


class BXGBPHIHandlerFactory(HandlerFactory):

    def __init__(self, model_config: BXGBPHIModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

    def create_handler(self) -> BXGBPHIModelHandler:
        return BXGBPHIModelHandler(self.model_config)
