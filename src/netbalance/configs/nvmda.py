import os

from .common import MODEL_SAVED_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .general import ModelConfig, OptimizerConfig

NVMDA_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "nvmda")
NVMDA_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "nvmda")
NVMDA_MODEL_SAVED_DIR = os.path.join(MODEL_SAVED_DIR, "nvmda")
NVMDA_RESULTS_DIR = os.path.join(NVMDA_PROCESSED_DATA_DIR, "results")


class NVMDAModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.fe_embedding_dim = 32
        self.fe_walk_length = 50
        self.fe_context_size = 10
        self.fe_walks_per_node = 10
        self.fe_num_negative_samples = 1
        self.fe_p = 1.0
        self.fe_q = 1.0
        self.fe_num_nodes = None
        self.fe_sparse = True

        self.input_dim = None
        self.hidden_dim = 32
        self.output_dim = 1
        self.num_layers = 3
        self.dropout = 0.1

        self.device = "cpu"

    def get_configuration(self):
        return {
            **super().get_configuration(),
            **{
                "fe_embedding_dim": self.fe_embedding_dim,
                "fe_walk_length": self.fe_walk_length,
                "fe_context_size": self.fe_context_size,
                "fe_walks_per_node": self.fe_walks_per_node,
                "fe_num_negative_samples": self.fe_num_negative_samples,
                "fe_p": self.fe_p,
                "fe_q": self.fe_q,
                "fe_num_nodes": self.fe_num_nodes,
                "fe_sparse": self.fe_sparse,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "device": self.device,
            },
        }

    def get_feature_extractor_kwargs(self):
        return {
            "embedding_dim": self.fe_embedding_dim,
            "walk_length": self.fe_walk_length,
            "context_size": self.fe_context_size,
            "walks_per_node": self.fe_walks_per_node,
            "num_negative_samples": self.fe_num_negative_samples,
            "p": self.fe_p,
            "q": self.fe_q,
            "num_nodes": self.fe_num_nodes,
            "sparse": self.fe_sparse,
        }

    def get_model_kwargs(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class NVMDAOptimizerConfig(OptimizerConfig):

    def __init__(self) -> None:
        super().__init__()
        self.fe_shuffle = True
        self.fe_num_workers = 0
        self.fe_batch_size = 64
        self.fe_lr = 0.01
        self.fe_optimizer = None
        self.fe_report_size = 1000

        self.optimizer = None
        self.criterion = None
        self.lr = 0.01
        self.batch_size = 32
        self.n_epoch = 50
        self.exp_name = "NVMDA optimizer"
        self.device = "cpu"
        self.report_size = 10
        self.threshold = 0.5

    def get_configuration(self):
        return {
            **super().get_configuration(),
            **{
                "fe_shuffle": self.fe_shuffle,
                "fe_num_workers": self.fe_num_workers,
                "fe_batch_size": self.fe_batch_size,
                "fe_lr": self.fe_lr,
                "fe_optimizer": self.fe_optimizer,
                "device": self.device,
                "report_size": self.report_size,
                "batch_size": self.batch_size,
                "criterion": self.criterion,
                "threshold": self.threshold,
            },
        }

    def get_fe_loader_configuration(self):
        return {
            "shuffle": self.fe_shuffle,
            "num_workers": self.fe_num_workers,
            "batch_size": self.fe_batch_size,
        }
