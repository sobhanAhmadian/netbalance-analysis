import os

import matplotlib.pyplot as plt
import pandas as pd

from netbalance.configs.bmlpdti import BMLPDTI_RESULTS_DIR as RESULTS_DIR  # Parameter
from netbalance.configs.common import RESULTS_DIR as COMMON_RESULTS_DIR
from netbalance.features.luodti import LuoDTIDataset as Dataset  # Parameter
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

model_name = "bmlpdti"  # Parameter
dataset = "luodti"  # Parameter
train_neg_samp_method = "irho_complete"  # Parameter

analyse = "watch_predicts_for_unkowns"  # Parameter

logger.info(
    f">>>>>>>>>>>>>>>>> Job: Model Evaluation - Results - {dataset} - {model_name} - {train_neg_samp_method} - {analyse}"
)

model_result_dir = os.path.join(
    RESULTS_DIR,
    f"preds",
    f"dataset_{dataset}",
    f"train_neg_samp_{train_neg_samp_method}",
)

figs_folder = (
    f"{COMMON_RESULTS_DIR}/figs/model_evaluation/results/{dataset}/{analyse}/method1"
)
os.makedirs(figs_folder, exist_ok=True)

ds = Dataset()

preds_file = os.path.join(model_result_dir, f"preds.csv")
df = pd.read_csv(preds_file)

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

# Save Dataframe
file_name = f"{COMMON_RESULTS_DIR}/{analyse}_method1.csv"
zero_df.to_csv(file_name, index=False)
print(f"\nZero df saved in {file_name}")

####################################
per_drug_average_scores = []
per_drug_var_scores = []
for i in range(len(cluster_a_node_names)):
    temp = zero_df.loc[zero_df.iloc[:, 0] == i]
    per_drug_average_scores.append(temp["Score"].mean())
    per_drug_var_scores.append(temp["Score"].var())

per_target_average_scores = []
per_target_var_scores = []
for i in range(len(cluster_b_node_names)):
    temp = zero_df.loc[zero_df.iloc[:, 1] == i]
    per_target_average_scores.append(temp["Score"].mean())
    per_target_var_scores.append(temp["Score"].var())

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
axes[0][0].hist(per_drug_average_scores, bins=20)
axes[0][0].set_title("Per-drug Average Scores")
axes[0][1].hist(per_drug_var_scores, bins=20)
axes[0][1].set_title("Per-drug Variance Scores")

axes[1][0].hist(per_target_average_scores, bins=20)
axes[1][0].set_title("Per-target Average Scores")
axes[1][1].hist(per_target_var_scores, bins=20)
axes[1][1].set_title("Per-target Variance Scores")

for ax in axes.flat:
    ax.set_xlim(0, 1)

fig.tight_layout()

file_name = f"{figs_folder}/average_and_variance_scores.svg"
fig.savefig(file_name)
print(f"\nFigs of average and variance scores saved in {file_name}")
