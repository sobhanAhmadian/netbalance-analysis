from netbalance.configs import cold_color1, warm_color1
from netbalance.configs.common import RESULTS_DIR
from netbalance.features import LuoDTIDataset as Dataset  # Parameter
from netbalance.utils.data import analyse_datasest

c_pos = "#66c2a5d3"
c_neg = "#d53e50d5"
save_figs = True  # Parameter
dataset = "luodti"  # Parameter
num_cross_validation = 5  # Parameter
num_negative_sampling = 5  # Parameter
k = 5  # Parameter
test_balance_method = "rho"  # Parameter
test_balance_kwargs = {
    "max_iter": 100000,
    "delta": 0.1,
    "cooling_rate": 0.99,
    "initial_temp": 40.0,
    "ent_desired": 1.0,
    "shrinkage": 1.0,
}  # Parameter
test_balance_negative_ratio = 1.0  # Parameter
figs_folder = f"{RESULTS_DIR}/figs/data_analysis/{dataset}/{test_balance_method}"

if __name__ == "__main__":
    analyse_datasest(
        dataset=Dataset(),
        dataset_name=dataset,
        figs_folder=figs_folder,
        num_cross_validation=num_cross_validation,
        num_negative_sampling=num_negative_sampling,
        k=k,
        test_balance_method=test_balance_method,
        test_balance_kwargs=test_balance_kwargs,
        test_balance_negative_ratio=test_balance_negative_ratio,
        c_pos=c_pos,
        c_neg=c_neg,
    )
