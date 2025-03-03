from typing import Literal, Union

import numpy as np

from netbalance.utils import prj_logger

from .interface import AModelHandler, HandlerFactory

logger = prj_logger.getLogger(__name__)


class ADegreeRatioModelHandler(AModelHandler):

    def __init__(
        self,
        node_ids: list[list[int]],
        use_nodes: list[bool],
        reduction: Union[
            Literal["mean", "max", "multiply", "weighted_mean"], None
        ] = "mean",
    ):
        super().__init__(None)

        if not any(use_nodes):
            raise ValueError("At least one of use_nodes should be True.")

        self.node_ids = node_ids
        self.use_nodes = use_nodes
        self.reduction = reduction

        self.ratios: list[list[float]] = []
        self.nums: list[list[int]] = []
        self.pos_nums: list[list[int]] = []

    def build(self, associations):
        for i in range(len(self.node_ids)):
            nums, pos_nums, ratios = self._calculate_node_statistics(
                associations,
                self.node_ids[i],
                i,
            )
            self.ratios.append(ratios)
            self.nums.append(nums)
            self.pos_nums.append(pos_nums)

    @staticmethod
    def _calculate_node_statistics(associations, node_ids, index):
        num = []
        pos_num = []
        ratios = []
        for node_id in node_ids:
            p_count = 0
            n_count = 0
            for a in associations:
                if a[index] == node_id:
                    if a[-1] == 1:
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

    def predict_impl(self, node_lists: list[np.ndarray]):
        num_samples = len(node_lists[0])
        for i in range(1, len(node_lists)):
            if len(node_lists[i]) != num_samples:
                raise ValueError(
                    "The number of samples in all node lists should be equal."
                )

        scores = []
        for i in range(num_samples):
            node_ids = []
            node_scores = []
            node_pos_nums = []
            for j in range(len(node_lists)):
                if not self.use_nodes[j]:
                    continue
                node_id = self.node_ids[j].index(node_lists[j][i])
                node_ids.append(node_id)
                node_scores.append(self.ratios[j][node_id])
                node_pos_nums.append(self.pos_nums[j][node_id])

            if len(node_scores) == 1:
                scores.append(node_scores[0])
            else:
                if self.reduction == "weighted_mean":
                    t = sum(node_pos_nums) + 1
                    w = [pos_num / t for pos_num in node_pos_nums]
                    s = sum([score * w[i] for i, score in enumerate(node_scores)])
                elif self.reduction == "mean":
                    s = sum(node_scores) / len(node_scores)
                elif self.reduction == "max":
                    s = max(node_scores)
                elif self.reduction == "multiply":
                    s = np.prod(node_scores)
                scores.append(s)
        return np.array(scores)

    def destroy(self):
        del self.ratios
        del self.nums
        del self.pos_nums

    def summary(self):
        return None

    def _build_model(self):
        return None

    def _build_feature_extractor(self):
        return None


class BGDegreeRatioModelHandler(ADegreeRatioModelHandler):
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
        super().__init__([a_node_ids, b_node_ids], [use_a, use_b], reduction)


class BGDegreeRatioHandlerFactory(HandlerFactory):
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

    def create_handler(self) -> BGDegreeRatioModelHandler:
        return BGDegreeRatioModelHandler(
            self.a_node_ids, self.b_node_ids, self.use_a, self.use_b, self.reduction
        )


class TGDegreeRatioModelHandler(ADegreeRatioModelHandler):
    """Baseline model which uses degree ratio of nodes to predict probability of association.

    Args:
        a_node_ids (list): List of node ids in cluster A.
        b_node_ids (list): List of node ids in cluster B.
        c_node_ids (list): List of node ids in cluster C.
        use_a (bool): If True, use degree ratio of nodes in cluster A.
        use_b (bool): If True, use degree ratio of nodes in cluster B.
        use_c (bool): If True, use degree ratio of nodes in cluster C.
        reduction (str): Reduction method to combine degree ratios of nodes in clusters.
            - "mean": Mean of degree ratios of nodes in clusters A and B.
            - "max": Maximum of degree ratios of nodes in clusters A and B.
            - "multiply": Product of degree ratios of nodes in clusters A and B.
            - "weighted_mean": Weighted mean of degree ratios of nodes in clusters A and B.
    """

    def __init__(
        self,
        a_node_ids: list,
        b_node_ids: list,
        c_node_ids: list,
        use_a: bool,
        use_b: bool,
        use_c: bool,
        reduction: Union[
            Literal["mean", "max", "multiply", "weighted_mean"], None
        ] = "mean",
    ):
        super().__init__(
            [a_node_ids, b_node_ids, c_node_ids], [use_a, use_b, use_c], reduction
        )


class TGDegreeRatioHandlerFactory(HandlerFactory):
    """Factory for baseline model which uses degree ratio of nodes to predict probability of association.

    Args:
        a_node_ids (list): List of node ids in cluster A.
        b_node_ids (list): List of node ids in cluster B.
        c_node_ids (list): List of node ids in cluster C.
        use_a (bool): If True, use degree ratio of nodes in cluster A.
        use_b (bool): If True, use degree ratio of nodes in cluster B.
        use_c (bool): If True, use degree ratio of nodes in cluster C.
        reduction (str): Reduction method to combine degree ratios of nodes in clusters.
            - "mean": Mean of degree ratios of nodes in clusters A and B.
            - "max": Maximum of degree ratios of nodes in clusters A and B.
            - "multiply": Product of degree ratios of nodes in clusters A and B.
            - "weighted_mean": Weighted mean of degree ratios of nodes in clusters A and B.
    """

    def __init__(
        self,
        a_node_ids: list,
        b_node_ids: list,
        c_node_ids: list,
        use_a: bool = True,
        use_b: bool = True,
        use_c: bool = True,
        reduction: Union[
            Literal["mean", "max", "multiply", "weighted_mean"], None
        ] = "weighted_mean",
    ) -> None:
        super().__init__()
        self.b_node_ids = b_node_ids
        self.a_node_ids = a_node_ids
        self.c_node_ids = c_node_ids
        self.use_a = use_a
        self.use_b = use_b
        self.use_c = use_c
        self.reduction = reduction

    def create_handler(self) -> TGDegreeRatioModelHandler:
        return TGDegreeRatioModelHandler(
            self.a_node_ids,
            self.b_node_ids,
            self.c_node_ids,
            self.use_a,
            self.use_b,
            self.use_c,
            self.reduction,
        )
