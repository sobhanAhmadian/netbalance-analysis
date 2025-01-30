import abc


class Config(abc.ABC):
    @abc.abstractmethod
    def get_configuration(self):
        pass


class ModelConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.embedding_dim = None  # dimension of embedding
        self.device = None

    def get_configuration(self):
        return {
            "model_name": self.model_name,
            "embed_dim": self.embedding_dim,
            "device": self.device,
        }


class MethodConfig(Config):
    def __init__(self):
        super().__init__()
        self.method_name = None

    def get_configuration(self):
        return {
            "method_name": self.method_name,
        }


class OptimizerConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = None
        self.criterion = None
        self.lr = None  # learning rate
        self.batch_size = None
        self.n_epoch = None
        self.exp_name = None
        self.save = False
        self.save_path = None
        self.device = "cpu"
        self.report_size = 100  # batch to report ratio
        self.threshold = 0.5

    def get_configuration(self):
        return {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_epoch": self.n_epoch,
            "exp_name": self.exp_name,
            "save": self.save,
            "save_path": self.save_path,
            "device": self.device,
            "report_size": self.report_size,
            "threshold": self.threshold,
        }
