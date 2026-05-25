import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from netbalance.configs.bmlpphi import BMLPPHI_RESULTS_DIR as RESULTS_DIR
from netbalance.utils import prj_logger
import requests

plt.rcParams.update(
    {
        "font.weight": "normal",  # options: 'normal', 'light', 'regular'
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelweight": "regular",
        "axes.titleweight": "regular",
    }
)

logger = prj_logger.getLogger(__name__)


def fetch_go_descriptions(go_terms):
    """Fetch GO term descriptions from QuickGO API."""
    descriptions = {}
    # QuickGO supports batch queries
    batch_size = 200
    terms_list = list(go_terms)

    for i in range(0, len(terms_list), batch_size):
        batch = terms_list[i : i + batch_size]
        ids_param = ",".join(batch)
        url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{ids_param}"
        response = requests.get(url, headers={"Accept": "application/json"})

        if response.ok:
            data = response.json()
            for result in data.get("results", []):
                descriptions[result["id"]] = result.get("name", "N/A")
        else:
            logger.warning(
                f"QuickGO batch request failed with status {response.status_code}"
            )

    return descriptions


dataset = "ecoli"  # Parameter
figs_folder = f"{RESULTS_DIR}/figs/other/feature_importance/{dataset}/bmlpphi"
save_dir = os.path.join(
    RESULTS_DIR,
    "numeric",
    "other",
    f"feature_importance",
    f"dataset-{dataset}",
    "go-analyses",
)

n = 500
beta_file_name = f"{save_dir}/top-{n}-categories-beta.csv"
irho_file_name = f"{save_dir}/top-{n}-categories-irho.csv"
beta_results = pd.read_csv(beta_file_name)
irho_results = pd.read_csv(irho_file_name)

print("Beta top categories:")
print(beta_results.iloc[:10].to_string(index=False))
print("IRho top categories:")
print(irho_results.iloc[:10].to_string(index=False))


merged = pd.merge(
    beta_results[["category", "fg_count", "enriched"]],
    irho_results[["category", "fg_count", "enriched"]],
    on="category",
    suffixes=("_beta", "_irho"),
)
merged = merged.sort_values("fg_count_irho", ascending=True)
merged = merged[merged["category"] != "-"]
merged = merged[merged["fg_count_beta"] + merged["fg_count_irho"] > 0]

go_descriptions = fetch_go_descriptions(merged["category"])
merged["description"] = merged["category"].map(go_descriptions).fillna("N/A")
categories = merged["category"]
y = np.arange(len(categories))


# Compare absolute numbers
fig, ax = plt.subplots(
    figsize=(12, len(categories) * 0.3)
)  # Adjust height based on number of categories
bar_height = 0.4

beta_color = "#a6dba0"  # green
irho_color = "#9970ab"  # red


def draw_bars(ax, values, enriched_col, y_positions, height, color):
    for i, (val, e) in enumerate(zip(values, enriched_col)):
        hatch = "xx" if e == "enriched" else None
        ax.barh(
            y_positions[i],
            val,
            height=height,
            color=color,
            edgecolor="white",
            hatch=hatch,
        )


draw_bars(
    ax,
    merged["fg_count_beta"],
    merged["enriched_beta"],
    y + bar_height / 2,
    bar_height,
    beta_color,
)
draw_bars(
    ax,
    merged["fg_count_irho"],
    merged["enriched_irho"],
    y - bar_height / 2,
    bar_height,
    irho_color,
)

ax.set_yticks(y)
ax.set_yticklabels(
    [
        f"{cat} - {desc}"
        for cat, desc in zip(categories, merged.loc[categories.index, "description"])
    ]
)
ax.set_xlabel(f"Count in top {n} features")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()
file_name = f"{figs_folder}/go-analyses-top-{n}.svg"
plt.savefig(file_name, transparent=True)
print(f"\nFigure Saved: {file_name}")

merged.to_csv(f"test.csv", index=False)