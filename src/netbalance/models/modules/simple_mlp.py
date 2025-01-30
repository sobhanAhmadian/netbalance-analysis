from torch import nn

from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


class SimpleMLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=32, output_dim=1, num_layers=3, dropout=0.1
    ):
        super().__init__()

        logger.info(
            f"""Initial SimpleMLP with {input_dim} input dimension, {hidden_dim} hidden dimension, {output_dim} 
            output dimension, {num_layers} layers and with {dropout} dropout"""
        )

        modules = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(hidden_dim, output_dim))

        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)
