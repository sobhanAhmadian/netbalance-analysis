import os

from .common import MODEL_SAVED_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .general import ModelConfig, OptimizerConfig

BLINPHI_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "blinphi")
BLINPHI_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "blinphi")
BLINPHI_MODEL_SAVED_DIR = os.path.join(MODEL_SAVED_DIR, "blinphi")
BLINPHI_RESULTS_DIR = os.path.join(BLINPHI_PROCESSED_DATA_DIR, "results")


class BLINPHIModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()

    def get_configuration(self):
        return super().get_configuration()

    def get_feature_extractor_kwargs(self):
        return {}


class BLINPHIOptimizerConfig(OptimizerConfig):

    def __init__(self) -> None:
        super().__init__()

        self.exp_name = "BLINPHI optimizer"
        self.threshold = 0.5

    def get_configuration(self):
        return super().get_configuration()

    def get_fe_loader_configuration(self):
        return {}
