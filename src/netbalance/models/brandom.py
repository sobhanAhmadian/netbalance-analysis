import numpy as np

from netbalance.utils import prj_logger

from .interface import AModelHandler, HandlerFactory

logger = prj_logger.getLogger(__name__)


class BRandomModelHandler(AModelHandler):
    """Baseline model which generates a random number between 0 and 1 as prediction."""

    def __init__(
        self,
    ):
        super().__init__(None)

        self.rng = np.random.default_rng(0)
        logger.info("defult_rng created with seed 0")

    def predict_impl(self, node_lists: list[np.ndarray]):
        a_nodes = node_lists[0]
        return self.rng.uniform(0, 1, size=(len(a_nodes),))

    def destroy(self):
        pass

    def summary(self):
        return None

    def _build_model(self):
        return None

    def _build_feature_extractor(self):
        return None


class BRandomHandlerFactory(HandlerFactory):
    """Factory for baseline model which generates a random number between 0 and 1 as prediction."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def create_handler(self) -> BRandomModelHandler:
        return BRandomModelHandler()
