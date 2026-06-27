"""
Approach 1 (LIWC): are results meaningfully different across the 4 result sets
(de+mae, de+rmse, en+mae, en+rmse)?

Each variant trains the same nested-CV pipeline search separately, optimizing
for a different (language, selection-metric) combination, but all four reuse
the exact same outer_cv KFold(n_splits=5, random_state=42) applied to rows
that are in the same order across languages (DE/EN texts are parallel
translations evaluated by the same raters, in the same row order). That means
`row_position_within_dimension` in outer_cv_predictions.csv identifies the
*same* underlying text in all four variants, so per-row absolute errors of
each variant's best pipeline can be paired and compared with a Wilcoxon
signed-rank test instead of only comparing single summary metrics across 5
dimensions.

Outputs (written to results/approach1/variant_comparison/):
    - combined_best_model_metrics.csv   (best model per dimension x variant)
    - pairwise_row_level_tests.csv       (Wilcoxon signed-rank, paired by row)
    - friedman_omnibus_test.csv          (omnibus check across all 4 variants)
    - best_model_family_matrix.csv       (which model family wins, per dimension)
    - metric_comparison_with_significance.png
    - best_model_family_heatmap.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "approach1"
OUT_DIR = RESULTS_ROOT / "variant_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "de_mae": RESULTS_ROOT / "de_results" / "mae_as_goal",
    "de_rmse": RESULTS_ROOT / "de_results" / "rmse_as_goal",
    "en_mae": RESULTS_ROOT / "en_results" / "mae_as_goal",
    "en_rmse": RESULTS_ROOT / "en_results" / "rmse_as_goal",
}

# The 4 primary paired contrasts of interest (language x optimization goal).
PRIMARY_CONTRASTS = [
    ("de_mae", "de_rmse"),   # goal effect, German
    ("en_mae", "en_rmse"),   # goal effect, English
    ("de_mae", "en_mae"),    # language effect, MAE-optimized
    ("de_rmse", "en_rmse"),  # language effect, RMSE-optimized
]

METRIC_COLS = ["MAE", "RMSE", "R2", "Pearson_r", "Spearman_rho"]


def load_best_model_metrics() -> pd.DataFrame:
    frames = []
    for variant, folder in VARIANTS.items():
        df = pd.read_csv(folder / "best_model_per_dimension.csv")
        df = df[["dimension", "model_family", "pipeline", *METRIC_COLS, "n_predictions"]].copy()
        df.insert(0, "variant", variant)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_best_pipeline_row_errors(combined: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    """For each (variant, dimension), return the best pipeline's per-row absolute
    error, indexed by row_position_within_dimension so different variants align."""
    errors: dict[tuple[str, str], pd.Series] = {}
    for variant, folder in VARIANTS.items():
        preds = pd.read_csv(folder / "outer_cv_predictions.csv")
        variant_best = combined[combined["variant"] == variant]
        for _, row in variant_best.iterrows():
            dimension = row["dimension"]
            mask = (preds["dimension"] == dimension) & (preds["pipeline"] == row["pipeline"])
            subset = preds.loc[mask].drop_duplicates("row_position_within_dimension")
            subset = subset.set_index("row_position_within_dimension").sort_index()
            errors[(variant, dimension)] = subset["absolute_error"]
    return errors


def pairwise_row_level_tests(
    combined: pd.DataFrame, errors: dict[tuple[str, str], pd.Series]
) -> pd.DataFrame:
    dimensions = sorted(combined["dimension"].unique())
    rows = []
    for variant_a, variant_b in PRIMARY_CONTRASTS:
        for dimension in dimensions:
            err_a = errors[(variant_a, dimension)]
            err_b = errors[(variant_b, dimension)]
            common_idx = err_a.index.intersection(err_b.index)
            a = err_a.loc[common_idx].to_numpy()
            b = err_b.loc[common_idx].to_numpy()
            diff = a - b
            if np.allclose(diff, 0):
                stat, p_value = np.nan, 1.0
            else:
                stat, p_value = wilcoxon(a, b, zero_method="wilcox")
            rows.append(
                {
                    "contrast": f"{variant_a}_vs_{variant_b}",
                    "dimension": dimension,
                    "n_paired_rows": len(common_idx),
                    "mean_abs_error_a": float(np.mean(a)),
                    "mean_abs_error_b": float(np.mean(b)),
                    "mean_diff_a_minus_b": float(np.mean(diff)),
                    "wilcoxon_stat": stat,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["p_value_fdr_bh"] = multipletests(result["p_value"], method="fdr_bh")[1]
    result["significant_fdr_0.05"] = result["p_value_fdr_bh"] < 0.05
    return result.sort_values(["contrast", "dimension"]).reset_index(drop=True)


def friedman_omnibus(combined: pd.DataFrame) -> pd.DataFrame:
    """Omnibus check: do MAE (and RMSE) ranks differ across the 4 variants,
    blocking on dimension? n=5 blocks is low-powered; treat as descriptive."""
    rows = []
    for metric in ["MAE", "RMSE"]:
        wide = combined.pivot(index="dimension", columns="variant", values=metric)
        wide = wide[list(VARIANTS.keys())]
        stat, p_value = friedmanchisquare(*[wide[col].to_numpy() for col in wide.columns])
        rows.append({"metric": metric, "n_blocks_dimensions": len(wide), "statistic": stat, "p_value": p_value})
    return pd.DataFrame(rows)


def best_model_family_matrix(combined: pd.DataFrame) -> pd.DataFrame:
    matrix = combined.pivot(index="dimension", columns="variant", values="model_family")
    matrix = matrix[list(VARIANTS.keys())]
    matrix["all_agree"] = matrix.nunique(axis=1) == 1
    return matrix


def plot_metric_comparison(combined: pd.DataFrame, tests: pd.DataFrame, path: Path) -> None:
    dimensions = sorted(combined["dimension"].unique())
    variants = list(VARIANTS.keys())
    x = np.arange(len(dimensions))
    width = 0.2

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), height_ratios=[3, 1.4])
    ax_bar, ax_sig = axes

    colors = {"de_mae": "#1f77b4", "de_rmse": "#aec7e8", "en_mae": "#d62728", "en_rmse": "#ff9896"}
    for i, variant in enumerate(variants):
        values = [
            combined.loc[(combined.variant == variant) & (combined.dimension == d), "MAE"].iloc[0]
            for d in dimensions
        ]
        ax_bar.bar(x + (i - 1.5) * width, values, width, label=variant, color=colors[variant])

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(dimensions, rotation=20, ha="right")
    ax_bar.set_ylabel("MAE (best pipeline, outer-CV)")
    ax_bar.set_title("Best-model MAE per dimension across the 4 result sets")
    ax_bar.legend(title="variant")
    ax_bar.grid(axis="y", alpha=0.3)

    # Significance strip: one row per primary contrast, stars where FDR-corrected p < .05.
    contrast_labels = [f"{a} vs {b}" for a, b in PRIMARY_CONTRASTS]
    sig_matrix = np.zeros((len(PRIMARY_CONTRASTS), len(dimensions)))
    for r, (a, b) in enumerate(PRIMARY_CONTRASTS):
        contrast_name = f"{a}_vs_{b}"
        for c, dimension in enumerate(dimensions):
            row = tests[(tests.contrast == contrast_name) & (tests.dimension == dimension)].iloc[0]
            sig_matrix[r, c] = 1.0 if row["significant_fdr_0.05"] else 0.0

    ax_sig.imshow(sig_matrix, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax_sig.set_xticks(x)
    ax_sig.set_xticklabels(dimensions, rotation=20, ha="right")
    ax_sig.set_yticks(range(len(contrast_labels)))
    ax_sig.set_yticklabels(contrast_labels)
    ax_sig.set_title("Row-level Wilcoxon signed-rank, significant after FDR correction (red = p<.05)")
    for r in range(sig_matrix.shape[0]):
        for c in range(sig_matrix.shape[1]):
            contrast_name = f"{PRIMARY_CONTRASTS[r][0]}_vs_{PRIMARY_CONTRASTS[r][1]}"
            p = tests[(tests.contrast == contrast_name) & (tests.dimension == dimensions[c])]["p_value_fdr_bh"].iloc[0]
            ax_sig.text(c, r, f"{p:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if sig_matrix[r, c] else "black")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_best_model_family_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    variants = list(VARIANTS.keys())
    dimensions = matrix.index.tolist()
    families = sorted(pd.unique(matrix[variants].to_numpy().ravel()))
    family_to_idx = {fam: i for i, fam in enumerate(families)}
    grid = matrix[variants].apply(lambda col: col.map(family_to_idx)).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(grid, cmap="tab20", aspect="auto", vmin=0, vmax=max(len(families) - 1, 1))
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_yticks(range(len(dimensions)))
    ax.set_yticklabels(dimensions)
    ax.set_title("Best model family per dimension and variant\n(non-matching rows = key finding differs across variants)")

    for r, dimension in enumerate(dimensions):
        for c, variant in enumerate(variants):
            label = matrix.loc[dimension, variant]
            ax.text(c, r, label, ha="center", va="center", fontsize=8, color="black")
        if not matrix.loc[dimension, "all_agree"]:
            ax.add_patch(plt.Rectangle((-0.5, r - 0.5), len(variants), 1, fill=False, edgecolor="red", linewidth=2))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    combined = load_best_model_metrics()
    combined.to_csv(OUT_DIR / "combined_best_model_metrics.csv", index=False)

    errors = load_best_pipeline_row_errors(combined)
    tests = pairwise_row_level_tests(combined, errors)
    tests.to_csv(OUT_DIR / "pairwise_row_level_tests.csv", index=False)

    omnibus = friedman_omnibus(combined)
    omnibus.to_csv(OUT_DIR / "friedman_omnibus_test.csv", index=False)

    matrix = best_model_family_matrix(combined)
    matrix.to_csv(OUT_DIR / "best_model_family_matrix.csv")

    plot_metric_comparison(combined, tests, OUT_DIR / "metric_comparison_with_significance.png")
    plot_best_model_family_heatmap(matrix, OUT_DIR / "best_model_family_heatmap.png")

    n_sig = int(tests["significant_fdr_0.05"].sum())
    n_total = len(tests)
    n_family_mismatches = int((~matrix["all_agree"]).sum())

    summary = {
        "n_row_level_pairwise_tests": n_total,
        "n_significant_after_fdr": n_sig,
        "primary_contrasts": [f"{a}_vs_{b}" for a, b in PRIMARY_CONTRASTS],
        "dimensions_with_diverging_best_model_family": matrix.index[~matrix["all_agree"]].tolist(),
        "n_dimensions_with_diverging_best_model_family": n_family_mismatches,
        "friedman_omnibus": omnibus.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print("4-VARIANT COMPARISON SUMMARY (de+mae, de+rmse, en+mae, en+rmse)")
    print("=" * 80)
    print(f"Row-level Wilcoxon tests: {n_sig}/{n_total} significant after FDR correction (q<0.05)")
    print(tests.to_string(index=False))
    print()
    print("Friedman omnibus test (MAE/RMSE across all 4 variants, blocked by dimension):")
    print(omnibus.to_string(index=False))
    print()
    print(f"Best model family diverges across variants in {n_family_mismatches}/5 dimensions:")
    print(matrix.to_string())
    print()
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
