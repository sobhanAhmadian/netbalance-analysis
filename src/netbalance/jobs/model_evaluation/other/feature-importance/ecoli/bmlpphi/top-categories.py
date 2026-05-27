import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from netbalance.configs.bmlpphi import BMLPPHI_RESULTS_DIR as RESULTS_DIR
from netbalance.utils import prj_logger

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

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

figs_folder = f"{RESULTS_DIR}/figs/other/feature_importance/ecoli/bmlpphi"
os.makedirs(figs_folder, exist_ok=True)

logger = prj_logger.getLogger(__name__)

dataset = "ecoli"  # Parameter

beta_save_dir = os.path.join(
    RESULTS_DIR,
    "numeric",
    "other",
    f"feature_importance",
    f"dataset-{dataset}",
    "train_neg_samp-beta",
)
irho_save_dir = os.path.join(
    RESULTS_DIR,
    "numeric",
    "other",
    f"feature_importance",
    f"dataset-{dataset}",
    "train_neg_samp-irho",
)

beta_fa_sorted_path = os.path.join(
    beta_save_dir, "feature_annotations_sorted_by_shap.csv"
)
irho_fa_sorted_path = os.path.join(
    irho_save_dir, "feature_annotations_sorted_by_shap.csv"
)

beta_df = pd.read_csv(beta_fa_sorted_path)
irho_df = pd.read_csv(irho_fa_sorted_path)
print(f"Reading sorted feature annotations from {beta_fa_sorted_path}")
print(f"Reading sorted feature annotations from {irho_fa_sorted_path}")


def category_enrichment(
    df, n, category_col="Category", fdr_method="fdr_bh", expand_sep=None
):
    """
    Test which categories are enriched in the top-n rows vs. the full background.

    Parameters
    ----------
    df : pd.DataFrame
        Rows sorted by importance (descending). Must contain `category_col`.
    n : int
        Number of top features to treat as the "foreground".
    category_col : str
        Column name holding annotation categories.
    fdr_method : str
        Multiple-testing correction method passed to statsmodels (default: Benjamini-Hochberg).
    expand_sep : str or None
        If categories are semicolon-separated, set this to ";" to treat each category separately.

    Returns
    -------
    pd.DataFrame
        One row per category, sorted by p-value, with columns:
        category, fg_count, fg_total, bg_count, bg_total,
        fg_pct, bg_pct, odds_ratio, pvalue, padj, enriched
    """
    foreground = df.iloc[:n]
    background = df.iloc[n:]  # everything NOT in foreground
    all_categories = df[category_col].unique()

    if expand_sep is not None:
        # Expand categories into separate rows
        foreground = foreground.assign(
            **{category_col: foreground[category_col].str.split(expand_sep)}
        ).explode(category_col)
        background = background.assign(
            **{category_col: background[category_col].str.split(expand_sep)}
        ).explode(category_col)

        all_categories = pd.concat(
            [foreground[category_col], background[category_col]]
        ).unique()

    fg_total = len(foreground)
    bg_total = len(background)

    results = []
    for cat in all_categories:
        fg_in = (foreground[category_col] == cat).sum()
        fg_out = fg_total - fg_in
        bg_in = (background[category_col] == cat).sum()
        bg_out = bg_total - bg_in

        # 2×2 contingency table
        table = [[fg_in, fg_out], [bg_in, bg_out]]
        odds_ratio, pvalue = fisher_exact(table, alternative="two-sided")

        results.append(
            {
                "category": cat,
                "fg_count": fg_in,
                "fg_total": fg_total,
                "bg_count": bg_in,
                "bg_total": bg_total,
                "fg_pct": round(100 * fg_in / fg_total, 2),
                "bg_pct": round(100 * bg_in / bg_total, 2),
                "odds_ratio": round(odds_ratio, 3),
                "pvalue": pvalue,
            }
        )

    result_df = pd.DataFrame(results).sort_values("pvalue")

    # Multiple-testing correction
    reject, padj, _, _ = multipletests(result_df["pvalue"], method=fdr_method)
    result_df["padj"] = padj
    result_df["enriched"] = result_df.apply(
        lambda r: "enriched" if r.fg_pct > r.bg_pct else "depleted", axis=1
    )
    result_df.loc[~reject, "enriched"] = "ns"

    return result_df.reset_index(drop=True)


# --- Usage ---
n = 100  # ← adjust as needed
beta_results = category_enrichment(beta_df, n=n, category_col="COG_category", expand_sep=",")
irho_results = category_enrichment(irho_df, n=n, category_col="COG_category", expand_sep=",")

print("Beta top categories:")
print(beta_results.to_string(index=False))
print("IRho top categories:")
print(irho_results.to_string(index=False))

# Compare absolute numbers
fig, ax = plt.subplots(figsize=(6, 13))

merged = pd.merge(
    beta_results[["category", "fg_count", "enriched"]],
    irho_results[["category", "fg_count", "enriched"]],
    on="category",
    suffixes=("_beta", "_irho"),
)
merged = merged.sort_values("fg_count_irho", ascending=True)
merged = merged[merged["category"] != "Other / Unclassified"]
merged = merged[merged["category"] != "Hypothetical & Unknown Function"]

categories = merged["category"]
y = np.arange(len(categories))
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
ax.set_yticklabels(categories)
ax.set_xlabel(f"Count in top {n} features")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()
file_name = f"{figs_folder}/top-{n}-categories.svg"
plt.savefig(file_name, transparent=True)
print(f"\nFigure Saved: {file_name}")
