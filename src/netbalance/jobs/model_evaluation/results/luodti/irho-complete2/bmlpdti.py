import os

import pandas as pd

from netbalance.configs.bmlpdti import BMLPDTI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.configs.common import RESULTS_DIR as COMMON_RESULTS_DIR
from netbalance.features.luodti import LuoDTIDataset as Dataset  # Parameter
from netbalance.utils import prj_logger
import glob

logger = prj_logger.getLogger(__name__)

model_name = "bmlpdti"  # Parameter
dataset = "luodti"  # Parameter
train_neg_samp_method = "irho"  # Parameter

analyse = "watch hit at k"  # Parameter

logger.info(
    f">>>>>>>>>>>>>>>>> Job: Model Evaluation - Results - {dataset} - {model_name} - {train_neg_samp_method} - {analyse}"
)

model_result_dir = os.path.join(
    RESULTS_DIR,
    f"preds",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)

ds = Dataset()

csv_files = glob.glob(f"{model_result_dir}/cv_1/*.csv")  # Replace with the actual path
df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

# preds_file = os.path.join(model_result_dir, f"preds.csv")
# df = pd.read_csv(preds_file)

cluster_a_node_names = ds.get_cluster_a_node_names()
cluster_b_node_names = ds.get_cluster_b_node_names()
a_node_names = [cluster_a_node_names[i] for i in df.iloc[:, 0]]
b_node_names = [cluster_b_node_names[i] for i in df.iloc[:, 1]]
df["Drug"] = a_node_names
df["Target"] = b_node_names

filtered_df = df[df.iloc[:, 2] == 1]
counts = filtered_df.iloc[:, 4].value_counts()
df["Drug_Pos"] = [counts.get(d, 0) for d in a_node_names]
counts = filtered_df.iloc[:, 5].value_counts()
df["Protein_Pos"] = [counts.get(p, 0) for p in b_node_names]


zero_df = df.loc[df.iloc[:, 2] == 0]
zero_df = zero_df.sort_values(by=["Score"], ascending=False)

zero_df.to_csv(
    "/Users/sobhan.ahmadian.moghadam/PycharmProjects/netbalance/src/netbalance/data_repository/results/zero_df_2.csv",
    index=False,
)

# zero_df = zero_df.loc[zero_df.iloc[:, 4] == "DB00674"]

print(zero_df.head(10))
