"""
Approach 1 (English, MAE-optimized variant): which LIWC features matter most?

Feasibility check (done once, see bottom of this docstring): the en_results/mae_as_goal
permuter checkpoints already contain everything needed to answer this without retraining:
    - param_searches[pipeline_key]["best_estimator"]: a fitted sklearn Pipeline per outer
      fold (the inner-CV winner refit on that fold's training rows).
    - param_searches[pipeline_key]["test_indices"]: the held-out row indices per fold.
    - X reconstructed from the same LIWC CSV with the same drop/select_dtypes logic as the
      training script has the same column count and order (verified: 118 columns for
      d1_illness_beliefs, matching n_features_in_ on the fitted estimator).
This means per-dimension feature importance can be computed directly from existing
checkpoints via permutation importance on each outer fold's held-out rows, using each
fold's own fitted "best" pipeline (the same pipeline selected in best_model_per_dimension.csv).
No additional training run is required. The one limitation: several winning models are
SVR_RBF, which has no coef_/feature_importances_, so permutation importance (model-agnostic)
is used uniformly instead of pipeline-native importances.

Outputs (written to results/approach1/feature_importance/en_mae/):
    - <dimension>_permutation_importance.csv     (per-feature mean/std importance, all folds)
    - combined_top_features.csv                  (top 15 features per dimension, one table)
    - <dimension>_top15.png                       (one bar plot per dimension, 5 files)
    - cross_dimension_feature_overlap.csv         (features ranking top-15 in >=2 dimensions,
                                                     based on each dimension's full importance
                                                     ranking; shows the feature's importance
                                                     value in every dimension plus a count of
                                                     how many dimensions it cracked top-15 in)
    - cross_dimension_feature_overlap.png         (heatmap of those recurring features
                                                     across dimensions; each cell shows
                                                     "<raw importance> (<% of that
                                                     dimension's baseline MAE>)")
    - cross_dimension_feature_overlap_relative_mae.csv
                                                    (the % of baseline MAE values, also
                                                     written out as data, e.g. for tables)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from biopsykit.classification.model_selection import SklearnPipelinePermuter
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore", message=".*should be used with.*sklearn.utils.parallel.Parallel.*")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "approach1" / "en_results" / "mae_as_goal"
OUT_DIR = PROJECT_ROOT / "results" / "approach1" / "feature_importance" / "en_mae"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "average_score"
N_REPEATS = 30
RANDOM_STATE = 42
TOP_N = 15

METADATA_COLS = [
    "id",
    "dimension",
    "text",
    "Text",
    "ColumnID",
    "Segment",
    "Filename",
    "language",
    "rater_one",
    "rater_two",
    "rater_three",
    TARGET_COL,
]


def load_paths_config() -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "configs" / "paths.json").read_text(encoding="utf-8"))


def load_input_csv() -> pd.DataFrame:
    paths = load_paths_config()
    csv_path = (
        PROJECT_ROOT / paths["data_clean"] / paths["liwc_results_subdir"] / paths["liwc_en_results_filename"]
    )
    return pd.read_csv(csv_path)


def safe_filename(value: Any) -> str:
    text = str(value)
    for ch in '<>:"/\\|?* ':
        text = text.replace(ch, "_")
    return text.strip("_")


def get_target_and_features(df: pd.DataFrame, dimension: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Reproduce the exact feature extraction used by 01_train_mae_as_goal_en.py."""
    sub = df[df["dimension"] == dimension].copy()
    sub = sub.dropna(subset=[TARGET_COL])
    y = sub[TARGET_COL].to_numpy(dtype=float)

    drop_cols = list(dict.fromkeys(METADATA_COLS + [TARGET_COL]))
    X = sub.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.reset_index(drop=True)
    return X, y


def best_pipeline_key(best_row: pd.Series) -> tuple[tuple[str, str], ...]:
    return (
        ("imputer", best_row["pipeline_imputer"]),
        ("scaler", best_row["pipeline_scaler"]),
        ("reduce_dim", best_row["pipeline_reduce_dim"]),
        ("reg", best_row["pipeline_reg"]),
    )


def permutation_importance_for_dimension(
    dimension: str,
    X: pd.DataFrame,
    y: np.ndarray,
    best_row: pd.Series,
) -> pd.DataFrame:
    label = safe_filename(dimension)
    permuter_path = RESULTS_DIR / "permuters" / f"permuter_{label}.pkl"
    permuter = SklearnPipelinePermuter.from_pickle(permuter_path)

    key = best_pipeline_key(best_row)
    if key not in permuter.param_searches:
        raise KeyError(f"Best pipeline {key} for {dimension} not found in checkpoint {permuter_path}")
    entry = permuter.param_searches[key]

    fold_importances: list[np.ndarray] = []
    for fold_idx, (estimator, test_idx) in enumerate(zip(entry["best_estimator"], entry["test_indices"])):
        if "reg__n_jobs" in estimator.get_params():
            estimator.set_params(reg__n_jobs=1)

        X_test = X.iloc[test_idx].to_numpy()
        y_test = y[test_idx]

        result = permutation_importance(
            estimator,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE + fold_idx,
            n_jobs=1,
        )
        fold_importances.append(result.importances_mean)

    fold_matrix = np.vstack(fold_importances)  # shape: (n_folds, n_features)
    out = pd.DataFrame(
        {
            "dimension": dimension,
            "feature": X.columns,
            "mean_importance": fold_matrix.mean(axis=0),
            "std_across_folds": fold_matrix.std(axis=0),
            "n_folds": fold_matrix.shape[0],
        }
    )
    return out.sort_values("mean_importance", ascending=False).reset_index(drop=True)


def plot_top_features_per_dimension(
    combined: pd.DataFrame, out_dir: Path, best_per_dimension: pd.DataFrame | None = None
) -> None:
    """One bar-chart png per dimension (5 files total).

    Bars are highlighted green when a feature is "robustly important": its
    importance stays positive even after subtracting one std across folds
    (mean_importance - std_across_folds > 0). A feature whose error bar
    crosses zero isn't reliably better than shuffling noise on every fold,
    so the mean alone overstates how trustworthy it is -- this criterion
    requires both a high mean AND low variance relative to it.

    When best_per_dimension is given, the title also shows that dimension's
    baseline MAE and winning model, e.g. "d1_illness_beliefs (baseline MAE
    0.502, model SVR_RBF)".
    """
    mae_and_model = (
        best_per_dimension.set_index("dimension")[["MAE", "pipeline_reg"]]
        if best_per_dimension is not None
        else None
    )
    for dimension in combined["dimension"].unique():
        top = combined[combined["dimension"] == dimension].head(TOP_N).iloc[::-1]
        robust = (top["mean_importance"] - top["std_across_folds"]) > 0
        colors = robust.map({True: "#2ca02c", False: "#1f77b4"})
        fig, ax = plt.subplots(figsize=(9, 0.18 * len(top) + 1.2))
        ax.barh(top["feature"], top["mean_importance"], xerr=top["std_across_folds"], color=colors)
        title = dimension
        if mae_and_model is not None and dimension in mae_and_model.index:
            mae = mae_and_model.loc[dimension, "MAE"]
            model = mae_and_model.loc[dimension, "pipeline_reg"]
            title = f"{dimension} (baseline MAE {mae:.3f}, model {model})"
        ax.set_title(title)
        ax.set_xlabel("Mean permutation importance (drop in -MAE when feature is shuffled)")
        handles = [
            Rectangle((0, 0), 1, 1, color="#2ca02c"),
            Rectangle((0, 0), 1, 1, color="#1f77b4"),
        ]
        ax.legend(
            handles,
            ["robust (mean - std > 0)", "not robust"],
            loc="lower right",
            fontsize=7,
        )
        fig.tight_layout()
        fig.savefig(str(out_dir / f"{safe_filename(dimension)}_top15.png"), dpi=150)
        plt.close(fig)


def cross_dimension_feature_overlap(combined: pd.DataFrame) -> pd.DataFrame:
    """Features that rank in the top-N importance list for >=2 dimensions.

    Rank (and hence "important in this dimension") is computed from each
    dimension's full, unrestricted importance ranking -- not from a pre-sliced
    top-N table -- so a feature that ranks, say, 17th in one dimension but 3rd
    in another is still correctly counted. The returned table also reports the
    feature's importance value in every dimension (not just the ones where it
    cracked the top N), so its standing elsewhere is visible too.
    """
    ranked = combined.copy()
    ranked["rank"] = ranked.groupby("dimension")["mean_importance"].rank(method="min", ascending=False)
    is_top_n = pd.DataFrame(ranked.pivot(index="feature", columns="dimension", values="rank")) <= TOP_N
    n_dimensions = is_top_n.sum(axis=1)
    mask = n_dimensions >= 2
    importance_pivot = pd.DataFrame(ranked.pivot(index="feature", columns="dimension", values="mean_importance"))
    overlap = importance_pivot.loc[mask].copy()
    overlap.insert(0, "n_dimensions", n_dimensions[mask])
    overlap = overlap.sort_values(by="n_dimensions", ascending=False)
    return overlap.reset_index()


def plot_cross_dimension_overlap(
    overlap: pd.DataFrame,
    path: Path,
    relative_mae: pd.DataFrame | None = None,
    cbar_label: str = "Mean permutation importance",
    title: str = "Features ranking top-15 in >=2 dimensions",
) -> None:
    """Heatmap of importance values; color encodes the raw importance, and
    each cell is annotated as "<raw> (<pct>%)" when relative_mae is given,
    so the absolute magnitude and the MAE-relative magnitude are both
    visible in one figure instead of two separate plots.
    """
    dimension_cols = [c for c in overlap.columns if c not in ("feature", "n_dimensions")]
    matrix = overlap.set_index("feature")[dimension_cols].to_numpy(dtype=float)
    pct_matrix = (
        relative_mae.set_index("feature")[dimension_cols].to_numpy(dtype=float)
        if relative_mae is not None
        else None
    )
    fig, ax = plt.subplots(figsize=(1.6 * len(dimension_cols) + 2, 0.4 * len(overlap) + 1.5))
    im = ax.imshow(np.nan_to_num(matrix), cmap="cividis", aspect="auto")
    ax.set_xticks(range(len(dimension_cols)))
    ax.set_xticklabels(dimension_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(overlap)))
    ax.set_yticklabels(overlap["feature"])
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                norm_value = (matrix[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                text_color = "white" if norm_value < 0.6 else "black"
                label = f"{matrix[i, j]:.3f}"
                if pct_matrix is not None and not np.isnan(pct_matrix[i, j]):
                    label += f"\n({pct_matrix[i, j]:.0f}%)"
                ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def cross_dimension_relative_to_mae(overlap: pd.DataFrame, best_per_dimension: pd.DataFrame) -> pd.DataFrame:
    """Re-express the overlap heatmap's importance values as a percentage of
    each dimension's own baseline MAE, so dimensions with very different MAE
    scales (e.g. d4 at 0.482 vs d3 at 0.754) become comparable: a feature
    worth 0.02 importance matters far more for a low-MAE dimension than a
    high-MAE one.
    """
    dimension_cols = [c for c in overlap.columns if c not in ("feature", "n_dimensions")]
    mae_per_dimension = best_per_dimension.set_index("dimension")["MAE"]
    relative = overlap.copy()
    for col in dimension_cols:
        relative[col] = overlap[col] / mae_per_dimension[col] * 100
    return relative


COMBINED_TOP_FEATURES_PATH = OUT_DIR / "combined_top_features.csv"
CROSS_DIMENSION_OVERLAP_PATH = OUT_DIR / "cross_dimension_feature_overlap.csv"
CROSS_DIMENSION_RELATIVE_MAE_PATH = OUT_DIR / "cross_dimension_feature_overlap_relative_mae.csv"


def calculate_importances() -> pd.DataFrame:
    """Compute per-dimension permutation importance and write the importance CSVs.

    Returns the full (unfiltered) combined importances, sorted descending within
    each dimension -- this is what plot_results() needs, and combined_top_features.csv
    is just its head(TOP_N) per dimension.
    """
    df = load_input_csv()
    best_per_dimension = pd.read_csv(RESULTS_DIR / "best_model_per_dimension.csv")

    all_importances: list[pd.DataFrame] = []
    for _, best_row in best_per_dimension.iterrows():
        dimension = best_row["dimension"]
        print(f"Computing permutation importance for {dimension} (model={best_row['pipeline_reg']})...")
        X, y = get_target_and_features(df, dimension)

        permuter_path = RESULTS_DIR / "permuters" / f"permuter_{safe_filename(dimension)}.pkl"
        n_features_checkpoint = SklearnPipelinePermuter.from_pickle(permuter_path).param_searches[
            best_pipeline_key(best_row)
        ]["best_estimator"][0].n_features_in_
        if n_features_checkpoint != X.shape[1]:
            raise ValueError(
                f"Feature count mismatch for {dimension}: rebuilt X has {X.shape[1]} columns, "
                f"checkpoint pipeline expects {n_features_checkpoint}. Permutation importance would "
                "be invalid (wrong column alignment). Re-derive X with the exact column order used "
                "at training time before continuing."
            )

        importances = permutation_importance_for_dimension(dimension, X, y, best_row)
        out_path = OUT_DIR / f"{safe_filename(dimension)}_permutation_importance.csv"
        importances.to_csv(out_path, index=False)
        all_importances.append(importances)
        print(f"  Top 5: {importances.head(5)[['feature', 'mean_importance']].to_dict('records')}")

    combined = pd.concat(all_importances, ignore_index=True)

    top15 = combined.groupby("dimension", group_keys=False).head(TOP_N)
    top15.to_csv(COMBINED_TOP_FEATURES_PATH, index=False)

    overlap = cross_dimension_feature_overlap(combined)
    overlap.to_csv(CROSS_DIMENSION_OVERLAP_PATH, index=False)
    print(f"\nFeatures recurring across >=2 dimensions (all features considered, not just top-15) ({len(overlap)}):")
    print(overlap[["feature", "n_dimensions"]].to_string(index=False))

    return combined


def plot_results() -> None:
    """Generate all plots from the result CSVs (combined_top_features.csv already
    holds each dimension's top TOP_N rows pre-sorted, so it doubles as the input
    for the per-dimension bar charts -- no need to re-read the full, unfiltered
    per-dimension importance CSVs).
    """
    # Already-existing training artifact (MAE, winning model per dimension) -- cheap to
    # read, used both for the per-dimension plot titles and the relative-MAE heatmap.
    best_per_dimension = pd.read_csv(RESULTS_DIR / "best_model_per_dimension.csv")

    combined_top = pd.read_csv(COMBINED_TOP_FEATURES_PATH)
    plot_top_features_per_dimension(combined_top, OUT_DIR, best_per_dimension=best_per_dimension)

    overlap = pd.read_csv(CROSS_DIMENSION_OVERLAP_PATH)
    if overlap.empty:
        return

    relative_mae = cross_dimension_relative_to_mae(overlap, best_per_dimension)
    relative_mae.to_csv(CROSS_DIMENSION_RELATIVE_MAE_PATH, index=False)

    plot_cross_dimension_overlap(
        overlap,
        OUT_DIR / "cross_dimension_feature_overlap.png",
        relative_mae=relative_mae,
        title="Features ranking top-15 in >=2 dimensions (importance, % of baseline MAE)",
    )


def main() -> None:
    results_exist = COMBINED_TOP_FEATURES_PATH.exists() and CROSS_DIMENSION_OVERLAP_PATH.exists()
    if results_exist:
        print(f"Found existing result CSVs in {OUT_DIR}, skipping calculation and plotting from them.")
    else:
        calculate_importances()

    plot_results()

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Per-dimension importance CSVs, combined_top_features.csv, per-dimension plots, and ")
    print(f"cross_dimension_feature_overlap.csv/.png written to: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
