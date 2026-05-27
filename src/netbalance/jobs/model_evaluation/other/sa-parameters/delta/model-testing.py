import os
from typing import Union

import numpy as np
import pandas as pd

from netbalance.configs.common import RESULTS_DIR
from netbalance.data.association_data import AData
from netbalance.evaluation.utils import evaluate_binary_classification
from netbalance.features.luodti import LuoDTIDataset as Dataset
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)


def get_result_of_rcv(
    delta: float,
    node_names: list[list[str]],
    num_negative_sampling: int,
    test_balance_method: Union[str, None] = "beta",
    test_balance_kwargs: dict = {},
    test_balance_negative_ratio: float = 1.0,
    dataset_name: str = None,
):

    def task(i: int, k: int, j: int, associations: np.ndarray, df: pd.DataFrame):
        save_name = None
        if dataset_name is not None and test_balance_method is not None:
            save_name = (
                f"dataset_{dataset_name}_{"test"}_cv_{i + 1}_fold_{k + 1}_neg_{j + 1}"
            )
            save_name += f"_met_{test_balance_method}_rat_{test_balance_negative_ratio}"
            for key, value in test_balance_kwargs.items():
                save_name += f"_{key}_{value}"
        data = AData(associations=associations, node_names=node_names)
        if test_balance_method is not None:
            data.balance_data(
                balance_method=test_balance_method,
                negative_ratio=test_balance_negative_ratio,
                seed=j,
                save_name=save_name,
                **test_balance_kwargs,
            )
            temp_df = pd.DataFrame(
                data.associations[:, :-1], columns=df.columns[:-2].tolist()
            )
            reduced_df = df.merge(temp_df, on=df.columns[:-2].tolist(), how="right")
            reduced_preds = reduced_df.iloc[:, -1].to_numpy().flatten()
        else:
            reduced_preds = df.iloc[:, -1].to_numpy()
        result = evaluate_binary_classification(data, reduced_preds, threshold=0.5)
        logger.info(
            f"AUC Result of fold {k + 1} of cv {i + 1} of neg {j + 1} is {result.auc}"
        )
        # general_cv_result.add_fold_result(result)
        return result

    preds_file = os.path.join(
        save_dir,
        f"delta_{delta}_preds.csv",
    )

    df = pd.read_csv(preds_file)
    associations = df.iloc[:, :-1].to_numpy()

    auc_results = []
    for j in range(num_negative_sampling):
        result = task(i=0, k=0, j=j, associations=associations, df=df)
        auc_results.append(result.auc)
    return auc_results


dataset = "luodti"

save_dir = os.path.join(
    RESULTS_DIR, "numeric", "other", f"dataset-{dataset}-delta-comparison", "preds"
)
print(save_dir)
ds = Dataset()


#############

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

print("Rho Balance:")
for delta in [0.0, 0.1, 2.0]:
    auc_results = get_result_of_rcv(
        delta=delta,
        node_names=ds.get_node_names(),
        num_negative_sampling=5,
        test_balance_method=test_balance_method,
        test_balance_kwargs=test_balance_kwargs,
        test_balance_negative_ratio=test_balance_negative_ratio,
        dataset_name=dataset,
    )

    print(f"Delta: {delta}, AUC: {np.mean(auc_results)}")


test_balance_method = "beta"  # Parameter
test_balance_kwargs = {}  # Parameter
test_balance_negative_ratio = 1.0  # Parameter

print("Beta Balance:")
for delta in [0.0, 0.1, 2.0]:
    auc_results = get_result_of_rcv(
        delta=delta,
        node_names=ds.get_node_names(),
        num_negative_sampling=5,
        test_balance_method=test_balance_method,
        test_balance_kwargs=test_balance_kwargs,
        test_balance_negative_ratio=test_balance_negative_ratio,
        dataset_name=dataset,
    )

    print(f"Delta: {delta}, AUC: {np.mean(auc_results)}")
