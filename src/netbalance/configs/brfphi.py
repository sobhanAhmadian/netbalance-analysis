import os

from .common import MODEL_SAVED_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .general import ModelConfig, OptimizerConfig

BRFPHI_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "brfphi")
BRFPHI_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "brfphi")
BRFPHI_MODEL_SAVED_DIR = os.path.join(MODEL_SAVED_DIR, "brfphi")
BRFPHI_RESULTS_DIR = os.path.join(BRFPHI_PROCESSED_DATA_DIR, "results")


class BRFPHIModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()

    def get_configuration(self):
        return super().get_configuration()

    def get_feature_extractor_kwargs(self):
        return {}


class BRFPHIOptimizerConfig(OptimizerConfig):

    def __init__(self) -> None:
        super().__init__()

        self.exp_name = "BRFPHI optimizer"
        self.threshold = 0.5

    def get_configuration(self):
        return super().get_configuration()

    def get_fe_loader_configuration(self):
        return {}
