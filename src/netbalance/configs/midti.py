import os

from torch.nn.modules.loss import CrossEntropyLoss

from .common import MODEL_SAVED_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .general import ModelConfig, OptimizerConfig

MIDTI_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "midti")
MIDTI_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "midti")
MIDTI_MODEL_SAVED_DIR = os.path.join(MODEL_SAVED_DIR, "midti")
MIDTI_RESULTS_DIR = os.path.join(MIDTI_PROCESSED_DATA_DIR, "results")


class MIDTIModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.drug_num = None
        self.protein_num = None
        self.dim = 512
        self.layer_output = 3
        self.layer_IA = 3
        self.nhead = 8
        self.dropout = 0.1

    def get_configuration(self):
        return super().get_configuration()

    def get_feature_extractor_kwargs(self):
        return {}

    def get_model_kwargs(self):
        return {}


class MIDTIOptimizerConfig(OptimizerConfig):

    def __init__(self) -> None:
        super().__init__()
        self.device = None
        self.criterion = CrossEntropyLoss()
        self.n_epoch = 2000
        self.lr = 0.1
        self.weight_decay = 5e-4
        self.es_patience = 50
        self.fair = False

    def get_configuration(self):
        return super().get_configuration()

    def get_fe_loader_configuration(self):
        return {}
