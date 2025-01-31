from .a_degree_ratio import A_DEGREE_RATIO_RESULTS_DIR
from .b_degree_ratio import B_DEGREE_RATIO_RESULTS_DIR
from .common import LOG_DIR
from .general import MethodConfig, ModelConfig, OptimizerConfig
from .nvmda import NVMDA_RESULTS_DIR
from .weighted_mean_degree_ratio import WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR

RESULTS_DIR_DICT = {
    "nvmda": NVMDA_RESULTS_DIR,
    "a_degree_ratio": A_DEGREE_RATIO_RESULTS_DIR,
    "b_degree_ratio": B_DEGREE_RATIO_RESULTS_DIR,
    "weighted_mean_degree_ratio": WEIGHTED_MEAN_DEGREE_RATIO_RESULTS_DIR,
}

warm_color1 = "#ffafcc"
cold_color1 = "#a2d2ff"
cold_color2 = "#82c0fa"
