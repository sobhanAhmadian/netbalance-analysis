import os

import numpy as np
import pandas as pd

from netbalance.configs.common import PROCESSED_DATA_DIR
from netbalance.configs.noonan import NOONAN_RAW_DATA_DIR

datasets = {
    "ecoli_subset": f"{NOONAN_RAW_DATA_DIR}/ecoli_interaction_matrix_subset.csv",
    "ecoli": f"{NOONAN_RAW_DATA_DIR}/ecoli_interaction_matrix.csv",
    "klebsiella1": f"{NOONAN_RAW_DATA_DIR}/klebsiella1_interaction_matrix.csv",
    "klebsiella2": f"{NOONAN_RAW_DATA_DIR}/klebsiella2_interaction_matrix.csv",
    "pseudomonas": f"{NOONAN_RAW_DATA_DIR}/pseudomonas_interaction_matrix.csv",
    "vibrio": f"{NOONAN_RAW_DATA_DIR}/vibrio_interaction_matrix.csv",
}


def process_datasets():
    df_dict = {name: pd.read_csv(path, index_col=0) for name, path in datasets.items()}

    for name, df in df_dict.items():
        df = df.reset_index()

        strains = pd.DataFrame(pd.unique(df["strain"]), columns=["strain"])
        phages = pd.DataFrame(pd.unique(df["phage"]), columns=["phage"])
        strain_to_idx = {s: i for i, s in enumerate(strains["strain"])}
        phage_to_idx = {p: i for i, p in enumerate(phages["phage"])}

        df["strain"] = df["strain"].map(strain_to_idx)
        df["phage"] = df["phage"].map(phage_to_idx)

        mat = df.to_numpy()

        dirname = f"{PROCESSED_DATA_DIR}/{name}"
        os.makedirs(dirname, exist_ok=True)
        np.savetxt(
            os.path.join(dirname, name + ".txt"),
            mat.astype(int),
            fmt="%d",
            delimiter=",",
        )
        strains.to_csv(os.path.join(dirname, "strains.csv"), index=True)
        phages.to_csv(os.path.join(dirname, "phages.csv"), index=True)


if __name__ == "__main__":
    process_datasets()
