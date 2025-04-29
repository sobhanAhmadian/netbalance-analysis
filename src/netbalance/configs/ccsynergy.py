import os

from torch.nn.modules.loss import CrossEntropyLoss

from .common import MODEL_SAVED_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .general import ModelConfig, OptimizerConfig

CCSYNERGY_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "ccsynergy")
CCSYNERGY_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "ccsynergy")
CCSYNERGY_MODEL_SAVED_DIR = os.path.join(MODEL_SAVED_DIR, "ccsynergy")
CCSYNERGY_RESULTS_DIR = os.path.join(CCSYNERGY_PROCESSED_DATA_DIR, "results")

CCSYNERGY_CELL_FEATURES_DIR = os.path.join(
    CCSYNERGY_PROCESSED_DATA_DIR, "features/cells"
)
CCSYNERGY_DRUG_FEATURES_DIR = os.path.join(
    CCSYNERGY_PROCESSED_DATA_DIR, "features/drugs"
)


class CCSynergyModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.inputLength = 356
        self.n1 = 2000
        self.n2 = 1000
        self.n3 = 500
        self.lr = 0.0001
        self.drug_feature_name = "C4"
        self.cell_feature_name = "Cell2"

    def get_configuration(self):
        return super().get_configuration()

    def get_feature_extractor_kwargs(self):
        return {}

    def get_model_kwargs(self):
        return {}


class CCSynergyOptimizerConfig(OptimizerConfig):

    def __init__(self) -> None:
        super().__init__()
        self.batch_size = 128
        self.n_epoch = 1000

    def get_configuration(self):
        return super().get_configuration()

    def get_fe_loader_configuration(self):
        return {}
