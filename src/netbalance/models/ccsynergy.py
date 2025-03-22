from pathlib import Path
from typing import Literal, Union

import keras
import numpy as np
from keras import callbacks, layers, models

from netbalance.configs.ccsynergy import (
    CCSYNERGY_CELL_FEATURES_DIR,
    CCSYNERGY_DRUG_FEATURES_DIR,
    CCSynergyModelConfig,
)
from netbalance.methods.general import FeatureExtractor
from netbalance.utils import prj_logger

from .interface import AModelHandler, HandlerFactory

logger = prj_logger.getLogger(__name__)

import os

# keras run on cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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


def DNN(inputLength, n1=2000, n2=1000, n3=500, lr=0.0001):
    model = models.Sequential()
    model.add(
        layers.Dense(n1, kernel_initializer="he_normal", input_shape=[inputLength])
    )
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n2, activation="relu", kernel_initializer="he_normal"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(n3, activation="tanh", kernel_initializer="he_normal"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(lr), beta_1=0.9, beta_2=0.999, amsgrad=False
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CCSynergyFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
        drug_feature_name: DRUG_FEATURE_NAMES,
        cell_feature_name: CELL_FEATURE_NAMES,
    ):
        super().__init__()

        self.drug_feature_name = drug_feature_name
        self.cell_feature_name = cell_feature_name
        self.drug_file = Path(f"{CCSYNERGY_DRUG_FEATURES_DIR}/{drug_feature_name}.txt")
        self.cell_file = Path(f"{CCSYNERGY_CELL_FEATURES_DIR}/{cell_feature_name}.txt")

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


class CCSynergyModelHandler(AModelHandler):
    """Baseline model which generates a random number between 0 and 1 as prediction."""

    def __init__(
        self,
        model_config: CCSynergyModelConfig,
    ):
        super().__init__(model_config)

    def predict_impl(self, node_lists: Union[list[int], np.ndarray]):
        a_nodes, b_nodes, c_nodes = node_lists
        features = self.fe.extract_features(a_nodes, b_nodes, c_nodes)
        return self.model.predict(features).reshape(-1)

    def destroy(self):
        del self.model
        del self.fe

    def summary(self):
        raise NotImplementedError

    def _build_model(self):
        return DNN(
            inputLength=self.model_config.inputLength,
            n1=self.model_config.n1,
            n2=self.model_config.n2,
            n3=self.model_config.n3,
            lr=self.model_config.lr,
        )

    def _build_feature_extractor(self):
        return CCSynergyFeatureExtractor(
            self.model_config.drug_feature_name, self.model_config.cell_feature_name
        )


class CCSynergyHandlerFactory(HandlerFactory):
    def __init__(self, model_config: CCSynergyModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

    def create_handler(self) -> CCSynergyModelHandler:
        return CCSynergyModelHandler(self.model_config)
