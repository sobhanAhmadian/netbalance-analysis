from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from netbalance.configs.blinsyn import (
    BLINSYN_CELL_FEATURES_DIR,
    BLINSYN_DRUG_FEATURES_DIR,
    BLINSYNModelConfig,
)
from netbalance.methods.general import FeatureExtractor
from netbalance.utils import prj_logger

from .interface import AModelHandler, HandlerFactory

logger = prj_logger.getLogger(__name__)


DRUG_FEATURE_NAMES = Literal[
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
]
CELL_FEATURE_NAMES = Literal[
    "Cell1",
    "Cell2",
    "Cell3",
    "Cell4",
    "Cell5",
]

class BLINSYNFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
        drug_feature_name: DRUG_FEATURE_NAMES,
        cell_feature_name: CELL_FEATURE_NAMES,
    ):
        super().__init__()

        self.drug_feature_name = drug_feature_name
        self.cell_feature_name = cell_feature_name
        self.drug_file = Path(f"{BLINSYN_DRUG_FEATURES_DIR}/{drug_feature_name}.txt")
        self.cell_file = Path(f"{BLINSYN_CELL_FEATURES_DIR}/{cell_feature_name}.txt")

    def build(self):
        self.drug_features = np.loadtxt(self.drug_file, delimiter=",")
        self.cell_features = np.loadtxt(self.cell_file, delimiter=",")

    def extract_features(
        self,
        a_nodes: Union[list[int], np.ndarray],
        b_nodes: Union[list[int], np.ndarray],
        c_nodes: Union[list[int], np.ndarray],
    ):
        d1 = self.drug_features[a_nodes]
        d2 = self.drug_features[b_nodes]
        c = self.cell_features[c_nodes]
        features = np.concatenate([d1, d2, c], axis=1)
        return features


class BLINSYNModelHandler(AModelHandler):
    """Baseline model which generates a random number between 0 and 1 as prediction."""

    def __init__(
        self,
        model_config: BLINSYNModelConfig,
    ):
        super().__init__(model_config)

    def predict_impl(self, node_lists: Union[list[int], np.ndarray]):
        a_nodes, b_nodes, c_nodes = node_lists
        features = self.fe.extract_features(a_nodes, b_nodes, c_nodes)
        preds = self.model.predict_proba(features)[:, 1]
        return preds

    def destroy(self):
        del self.model
        del self.fe

    def summary(self):
        raise NotImplementedError

    def _build_model(self):
        return LogisticRegression(random_state=0, max_iter=10000)

    def _build_feature_extractor(self):
        return BLINSYNFeatureExtractor(
            self.model_config.drug_feature_name, self.model_config.cell_feature_name
        )


class BLINSYNHandlerFactory(HandlerFactory):
    def __init__(self, model_config: BLINSYNModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

    def create_handler(self) -> BLINSYNModelHandler:
        return BLINSYNModelHandler(self.model_config)
