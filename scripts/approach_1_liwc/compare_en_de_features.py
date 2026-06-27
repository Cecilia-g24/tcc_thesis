"""
Quick comparison of LIWC-22 feature output between the EN and DE result CSVs.

The two CSVs come from different LIWC dictionaries (EN: LIWC-22 dictionary,
DE: an older/German dictionary with different category names), so they don't
share a 1:1 feature set. This script reports:
    1. Which numeric columns exist only in EN, only in DE, or in both.
    2. For the shared (common) numeric columns, per-language descriptive
       stats and the row-wise correlation between EN and DE values (rows
       are aligned 1:1 by `id`, since each row is the same transcript
       segment in both languages).

Outputs (results/approach1/feature_comparison/):
    - column_diff.csv             EN-only / DE-only / common column lists
    - common_feature_stats.csv    mean/std per language + correlation, per common feature
    - column_overlap.png         bar chart of EN-only / common / DE-only column counts
    - common_feature_corr.png    horizontal bar chart of row-wise EN/DE correlation per common feature
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "approach1" / "feature_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS = {
    "id",
    "dimension",
    "text",
    "Text",
    "ColumnID",
    "Segment",
    "rater_one",
    "rater_two",
    "rater_three",
    "average_score",
}


def load_paths_config() -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "configs" / "paths.json").read_text(encoding="utf-8"))


def load_results(paths: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = PROJECT_ROOT / paths["data_clean"] / paths["liwc_results_subdir"]
    en = pd.read_csv(base / paths["liwc_en_results_filename"])
    de = pd.read_csv(base / paths["liwc_de_results_filename"])
    return en, de


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    return sorted(numeric - NON_FEATURE_COLS)


def plot_column_overlap(n_en_only: int, n_common: int, n_de_only: int, path: Path) -> None:
    labels = ["EN-only", "Common", "DE-only"]
    counts = [n_en_only, n_common, n_de_only]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts, color=["#1f77b4", "#2ca02c", "#d62728"])
    for i, count in enumerate(counts):
        ax.text(i, count + 1, str(count), ha="center")
    ax.set_ylabel("Number of numeric feature columns")
    ax.set_title("EN vs DE LIWC output: column overlap")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_common_feature_corr(stats_df: pd.DataFrame, path: Path) -> None:
    plot_df = stats_df.dropna(subset=["row_wise_corr"]).sort_values("row_wise_corr")
    fig, ax = plt.subplots(figsize=(7, max(6, 0.18 * len(plot_df))))
    colors = ["#d62728" if c < 0.5 else "#2ca02c" for c in plot_df["row_wise_corr"]]
    ax.barh(plot_df["feature"], plot_df["row_wise_corr"], color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Row-wise EN/DE correlation (same transcript segments)")
    ax.set_title("Common LIWC feature consistency between EN and DE output")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    paths = load_paths_config()
    en, de = load_results(paths)

    if en["id"].tolist() != de["id"].tolist():
        raise ValueError("EN and DE rows are not aligned by id; row-wise correlation would be invalid.")

    en_feats = numeric_feature_cols(en)
    de_feats = numeric_feature_cols(de)
    common = sorted(set(en_feats) & set(de_feats))
    en_only = sorted(set(en_feats) - set(de_feats))
    de_only = sorted(set(de_feats) - set(en_feats))

    diff_rows = (
        [{"set": "common", "column": c} for c in common]
        + [{"set": "en_only", "column": c} for c in en_only]
        + [{"set": "de_only", "column": c} for c in de_only]
    )
    pd.DataFrame(diff_rows).to_csv(OUT_DIR / "column_diff.csv", index=False)

    stats_rows = []
    for feat in common:
        en_vals = pd.to_numeric(en[feat], errors="coerce")
        de_vals = pd.to_numeric(de[feat], errors="coerce")
        valid = en_vals.notna() & de_vals.notna()
        corr = en_vals[valid].corr(de_vals[valid]) if valid.sum() > 1 else np.nan
        stats_rows.append(
            {
                "feature": feat,
                "en_mean": en_vals.mean(),
                "en_std": en_vals.std(),
                "de_mean": de_vals.mean(),
                "de_std": de_vals.std(),
                "mean_diff_en_minus_de": en_vals.mean() - de_vals.mean(),
                "row_wise_corr": corr,
                "n_valid_rows": int(valid.sum()),
            }
        )
    stats_df = pd.DataFrame(stats_rows).sort_values("row_wise_corr")
    stats_df.to_csv(OUT_DIR / "common_feature_stats.csv", index=False)

    overlap_path = OUT_DIR / "column_overlap.png"
    corr_path = OUT_DIR / "common_feature_corr.png"
    plot_column_overlap(len(en_only), len(common), len(de_only), overlap_path)
    plot_common_feature_corr(stats_df, corr_path)

    print("=" * 90)
    print(f"EN numeric feature columns: {len(en_feats)}")
    print(f"DE numeric feature columns: {len(de_feats)}")
    print(f"Common: {len(common)}  |  EN-only: {len(en_only)}  |  DE-only: {len(de_only)}")
    print()
    print("EN-only columns (LIWC-22 category names not present in DE dictionary output):")
    print(en_only)
    print()
    print("DE-only columns (older/German-specific category names not present in EN output):")
    print(de_only)
    print()
    print("Lowest row-wise EN/DE correlation among common features (most divergent):")
    print(stats_df.head(10).to_string(index=False))
    print()
    print("Highest row-wise EN/DE correlation among common features (most consistent):")
    print(stats_df.tail(10).to_string(index=False))
    print("=" * 90)
    print(f"Saved: {OUT_DIR / 'column_diff.csv'}")
    print(f"Saved: {OUT_DIR / 'common_feature_stats.csv'}")
    print(f"Saved: {overlap_path}")
    print(f"Saved: {corr_path}")


if __name__ == "__main__":
    main()
