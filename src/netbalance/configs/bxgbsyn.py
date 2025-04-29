import os

import torch
from torch.nn.modules.loss import CrossEntropyLoss

from .common import MODEL_SAVED_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .general import ModelConfig, OptimizerConfig

BXGBSYN_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "bxgbsyn")
BXGBSYN_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "bxgbsyn")
BXGBSYN_MODEL_SAVED_DIR = os.path.join(MODEL_SAVED_DIR, "bxgbsyn")
BXGBSYN_RESULTS_DIR = os.path.join(BXGBSYN_PROCESSED_DATA_DIR, "results")

BXGBSYN_CELL_FEATURES_DIR = os.path.join(BXGBSYN_PROCESSED_DATA_DIR, "features/cells")
BXGBSYN_DRUG_FEATURES_DIR = os.path.join(BXGBSYN_PROCESSED_DATA_DIR, "features/drugs")


class BXGBSYNModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()

        self.drug_feature_name = "C4"
        self.cell_feature_name = "Cell2"

    def get_configuration(self):
        return super().get_configuration()

    def get_feature_extractor_kwargs(self):
        return {}

    def get_model_kwargs(self):
        return {}


class BXGBSYNOptimizerConfig(OptimizerConfig):

    def __init__(self) -> None:
        super().__init__()
        self.exp_name = "BXGBSYN optimizer"
        self.threshold = 0.5

    def get_configuration(self):
        return super().get_configuration()

    def get_fe_loader_configuration(self):
        return {}
