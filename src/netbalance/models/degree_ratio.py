from typing import Literal, Union

import numpy as np

from netbalance.utils import prj_logger

from .interface import BGCModelHandler, HandlerFactory

logger = prj_logger.getLogger(__name__)


class DegreeRatioModelHandler(BGCModelHandler):
    """Baseline model which uses degree ratio of nodes to predict probability of association.

    Args:
        a_node_ids (list): List of node ids in cluster A.
        b_node_ids (list): List of node ids in cluster B.
        use_a (bool): If True, use degree ratio of nodes in cluster A.
        use_b (bool): If True, use degree ratio of nodes in cluster B.
        reduction (str): Reduction method to combine degree ratios of nodes in clusters A and B. Used when both use_a and use_b are True.
            - "mean": Mean of degree ratios of nodes in clusters A and B.
            - "max": Maximum of degree ratios of nodes in clusters A and B.
            - "multiply": Product of degree ratios of nodes in clusters A and B.
            - "weighted_mean": Weighted mean of degree ratios of nodes in clusters A and B.
    """

    def __init__(
        self,
        a_node_ids: list,
        b_node_ids: list,
        use_a: bool,
        use_b: bool,
        reduction: Union[
            Literal["mean", "max", "multiply", "weighted_mean"], None
        ] = "mean",
    ):
        super().__init__(None)

        if not use_a and not use_b:
            raise ValueError("At least one of use_a and use_b should be True.")

        self.a_node_ids = a_node_ids
        self.b_node_ids = b_node_ids
        self.use_a = use_a
        self.use_b = use_b
        self.reduction = reduction

        self.a_ratios = []
        self.b_ratios = []

        self.a_num = []
        self.b_num = []

        self.a_pos_num = []
        self.b_pos_num = []

    def build(self, associations):
        self.b_num, self.b_pos_num, self.b_ratios = self._calculate_node_statistics(
            associations,
            self.b_node_ids,
            1,
        )
        self.a_num, self.a_pos_num, self.a_ratios = self._calculate_node_statistics(
            associations,
            self.a_node_ids,
            0,
        )

    @staticmethod
    def _calculate_node_statistics(associations, node_ids, index=1):
        num = []
        pos_num = []
        ratios = []
        for node_id in node_ids:
            p_count = 0
            n_count = 0
            for a in associations:
                if a[index] == node_id:
                    if a[2] == 1:
                        p_count += 1
                    else:
                        n_count += 1

            num.append(p_count + n_count)
            pos_num.append(p_count)
            if n_count == 0 and p_count == 0:
                ratios.append(0.5)
            else:
                ratios.append(p_count / (p_count + n_count))
        return num, pos_num, ratios

    def predict_impl(self, a_nodes, b_nodes):
        if a_nodes.shape[0] != b_nodes.shape[0]:
            raise ValueError(
                "The number of samples in a_nodes and b_nodes should be equal."
            )

        scores = []
        for i in range(len(a_nodes)):
            a_node_id = self.a_node_ids.index(a_nodes[i])
            b_node_id = self.b_node_ids.index(b_nodes[i])

            b_node_score = self.b_ratios[b_node_id]
            a_node_score = self.a_ratios[a_node_id]

            if not self.use_a:
                scores.append(b_node_score)
            elif not self.use_b:
                scores.append(a_node_score)
            else:
                if self.reduction == "weighted_mean":
                    t = self.b_pos_num[b_node_id] + self.a_pos_num[a_node_id] + 1
                    a_node_w = self.a_pos_num[a_node_id] / t
                    b_node_w = self.b_pos_num[b_node_id] / t
                    s = b_node_score * b_node_w + a_node_score * a_node_w
                elif self.reduction == "mean":
                    s = (b_node_score + a_node_score) / 2
                elif self.reduction == "max":
                    s = max(b_node_score, a_node_score)
                elif self.reduction == "multiply":
                    s = b_node_score * a_node_score
                scores.append(s)
        return np.array(scores)

    def destroy(self):
        del self.b_ratios
        del self.a_ratios

    def summary(self):
        return None

    def _build_model(self):
        return None

    def _build_feature_extractor(self):
        return None


class DegreeRatioHandlerFactory(HandlerFactory):
    """Factory for baseline model which uses degree ratio of nodes to predict probability of association.

    Args:
        a_node_ids (list): List of node ids in cluster A.
        b_node_ids (list): List of node ids in cluster B.
        use_a (bool): If True, use degree ratio of nodes in cluster A.
        use_b (bool): If True, use degree ratio of nodes in cluster B.
        reduction (str): Reduction method to combine degree ratios of nodes in clusters A and B. Used when both use_a and use_b are True.
            - "mean": Mean of degree ratios of nodes in clusters A and B.
            - "max": Maximum of degree ratios of nodes in clusters A and B.
            - "multiply": Product of degree ratios of nodes in clusters A and B.
            - "weighted_mean": Weighted mean of degree ratios of nodes in clusters A and B.
    """

    def __init__(
        self,
        a_node_ids: list,
        b_node_ids: list,
        use_a: bool = True,
        use_b: bool = True,
        reduction: Union[
            Literal["mean", "max", "multiply", "weighted_mean"], None
        ] = "weighted_mean",
    ) -> None:
        super().__init__()
        self.b_node_ids = b_node_ids
        self.a_node_ids = a_node_ids
        self.use_a = use_a
        self.use_b = use_b
        self.reduction = reduction

    def create_handler(self) -> DegreeRatioModelHandler:
        return DegreeRatioModelHandler(
            self.a_node_ids, self.b_node_ids, self.use_a, self.use_b, self.reduction
        )
