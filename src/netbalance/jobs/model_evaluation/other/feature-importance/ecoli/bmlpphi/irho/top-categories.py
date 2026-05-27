import os

import matplotlib.pyplot as plt
import pandas as pd

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

logger = prj_logger.getLogger(__name__)

dataset = "ecoli"  # Parameter

save_dir = os.path.join(
    RESULTS_DIR,
    "numeric",
    "other",
    f"feature_importance",
    f"dataset-{dataset}",
    "train_neg_samp-irho",
)

fa_sorted_path = os.path.join(save_dir, "feature_annotations_sorted_by_shap.csv")
df = pd.read_csv(fa_sorted_path)
print(f"Reading sorted feature annotations from: {fa_sorted_path}")


def category_enrichment(df, n, category_col="Category", fdr_method="fdr_bh"):
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

    Returns
    -------
    pd.DataFrame
        One row per category, sorted by p-value, with columns:
        category, fg_count, fg_total, bg_count, bg_total,
        fg_pct, bg_pct, odds_ratio, pvalue, padj, enriched
    """
    foreground = df.iloc[:n]
    background = df.iloc[n:]  # everything NOT in foreground

    fg_total = len(foreground)
    bg_total = len(background)

    all_categories = df[category_col].unique()
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
results = category_enrichment(df, n=n)
print(results.to_string(index=False))
