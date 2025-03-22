import os

import torch
from torch.nn.modules.loss import CrossEntropyLoss

from .common import MODEL_SAVED_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .general import ModelConfig, OptimizerConfig

BMLPSYN_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "bmlpsyn")
BMLPSYN_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "bmlpsyn")
BMLPSYN_MODEL_SAVED_DIR = os.path.join(MODEL_SAVED_DIR, "bmlpsyn")
BMLPSYN_RESULTS_DIR = os.path.join(BMLPSYN_PROCESSED_DATA_DIR, "results")

BMLPSYN_CELL_FEATURES_DIR = os.path.join(BMLPSYN_PROCESSED_DATA_DIR, "features/cells")
BMLPSYN_DRUG_FEATURES_DIR = os.path.join(BMLPSYN_PROCESSED_DATA_DIR, "features/drugs")


class BMLPSYNModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()

        self.input_dim = 356
        self.hidden_dim = 32
        self.output_dim = 1
        self.num_layers = 3
        self.dropout = 0.1

        self.drug_feature_name = "C4"
        self.cell_feature_name = "Cell2"

    def get_configuration(self):
        return super().get_configuration()

    def get_feature_extractor_kwargs(self):
        return {}

    def get_model_kwargs(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class BMLPSYNOptimizerConfig(OptimizerConfig):

    def __init__(self) -> None:
        super().__init__()
        self.optimizer = torch.optim.Adam
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = 0.01
        self.batch_size = 32
        self.n_epoch = 50
        self.exp_name = "BMLPSYN optimizer"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.report_size = 10
        self.threshold = 0.5

    def get_configuration(self):
        return super().get_configuration()

    def get_fe_loader_configuration(self):
        return {}
