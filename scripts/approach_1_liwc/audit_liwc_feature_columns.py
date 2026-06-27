"""
Audit which columns your LIWC modeling scripts would use as numeric features.

This script does NOT import or modify your training scripts. It independently
replicates only the feature-extraction logic used there:

    drop metadata/target columns -> keep remaining numeric columns

Expected placement:
    Put this file in the same folder as 01_train_mae_as_goal_de.py / _en.py
    e.g. scripts/approach_1_liwc/audit_liwc_feature_columns.py

Example commands:
    python audit_liwc_feature_columns.py --language de
    python audit_liwc_feature_columns.py --language en
    python audit_liwc_feature_columns.py --language both
    python audit_liwc_feature_columns.py --language both --dimensions d1_illness_beliefs

Outputs:
    results/approach1/feature_audit/<language>/<dimension>_feature_columns.csv
    results/approach1/feature_audit/<language>/<dimension>_suspicious_columns.csv
    results/approach1/feature_audit/<language>/<dimension>_dropped_numeric_columns.csv
    results/approach1/feature_audit/feature_audit_summary.csv
    results/approach1/feature_audit/suspicious_feature_columns.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TARGET_COL = "average_score"

# Keep this aligned with your training scripts.
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
    DEFAULT_TARGET_COL,
]

# Heuristic only: these names are suspicious if they remain in X.
# It intentionally flags likely IDs/scores/annotation metadata.
SUSPICIOUS_NAME_PATTERN = re.compile(
    r"(^|_)(id|idx|index|row|participant|subject|person|user|case|scenario|item|"
    r"rater|rating|score|target|label|human|annotat|fold|split|avg|average|mean)(_|$)",
    flags=re.IGNORECASE,
)


def safe_filename(value: Any) -> str:
    text = str(value)
    for ch in '<>:"/\\|?* ':
        text = text.replace(ch, "_")
    return text.strip("_")


def infer_project_root(script_path: Path, explicit_project_root: Path | None = None) -> Path:
    """Find the project root containing configs/paths.json."""
    if explicit_project_root is not None:
        root = explicit_project_root.resolve()
        if not (root / "configs" / "paths.json").exists():
            raise FileNotFoundError(f"configs/paths.json not found under --project-root: {root}")
        return root

    # If this audit script is placed beside your modeling scripts, parents[2]
    # should be the project root, matching your training scripts.
    candidates = [script_path.resolve().parents[i] for i in range(min(4, len(script_path.resolve().parents)))]
    candidates.append(Path.cwd().resolve())

    for candidate in candidates:
        if (candidate / "configs" / "paths.json").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find configs/paths.json. Run this from your project, place the script "
        "beside the training scripts, or pass --project-root D:/Thesis/tcc_thesis."
    )


def load_paths_config(project_root: Path) -> dict[str, Any]:
    paths_file = project_root / "configs" / "paths.json"
    return json.loads(paths_file.read_text(encoding="utf-8"))


def input_csv_for_language(project_root: Path, paths: dict[str, Any], language: str) -> Path:
    if language == "de":
        filename_key = "liwc_de_results_filename"
    elif language == "en":
        filename_key = "liwc_en_results_filename"
    else:
        raise ValueError(f"Unsupported language: {language}")

    return (
        project_root
        / paths["data_clean"]
        / paths["liwc_results_subdir"]
        / paths[filename_key]
    )


def output_root(project_root: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir if output_dir.is_absolute() else project_root / output_dir
    return project_root / "results" / "approach1" / "feature_audit"


def get_feature_matrix_like_training_script(
    df: pd.DataFrame,
    dimension: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Replicate the training scripts' feature extraction without fitting models."""
    sub = df[df["dimension"] == dimension].copy()
    sub = sub.dropna(subset=[target_col])

    drop_cols = list(dict.fromkeys(METADATA_COLS + [target_col]))
    X = sub.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.reset_index(drop=True)

    numeric_cols_before_drop = df.select_dtypes(include=[np.number]).columns.tolist()
    dropped_numeric_cols = [col for col in numeric_cols_before_drop if col in drop_cols]

    return X, sub, drop_cols, dropped_numeric_cols


def most_frequent_ratio(series: pd.Series) -> float:
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return float("nan")
    return float(counts.iloc[0] / len(series))


def summarize_feature_columns(X: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, col in enumerate(X.columns, start=1):
        s = pd.to_numeric(X[col], errors="coerce")
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        mf_ratio = most_frequent_ratio(s)
        zero_variance = bool(n_unique <= 1)
        near_zero_variance = bool(n_unique <= 1 or mf_ratio >= 0.95)
        suspicious_name = bool(SUSPICIOUS_NAME_PATTERN.search(str(col)))

        rows.append(
            {
                "feature_index": i,
                "feature_name": col,
                "dtype": str(X[col].dtype),
                "n_missing": n_missing,
                "missing_rate": round(n_missing / len(X), 4) if len(X) else np.nan,
                "n_unique": n_unique,
                "most_frequent_value_ratio": round(mf_ratio, 4) if not np.isnan(mf_ratio) else np.nan,
                "zero_variance": zero_variance,
                "near_zero_variance_95pct_same_value": near_zero_variance,
                "suspicious_name": suspicious_name,
                "min": s.min(skipna=True),
                "max": s.max(skipna=True),
                "mean": s.mean(skipna=True),
                "std": s.std(skipna=True),
            }
        )
    return pd.DataFrame(rows)


def audit_one_language(
    project_root: Path,
    paths: dict[str, Any],
    language: str,
    dimensions: list[str] | None,
    target_col: str,
    out_root: Path,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    input_csv = input_csv_for_language(project_root, paths, language)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist for language={language}: {input_csv}")

    df = pd.read_csv(input_csv)
    if "dimension" not in df.columns:
        raise KeyError(f"Expected a 'dimension' column in {input_csv}")
    if target_col not in df.columns:
        raise KeyError(f"Expected target column '{target_col}' in {input_csv}")

    available_dimensions = df["dimension"].dropna().unique().tolist()
    selected_dimensions = dimensions if dimensions else available_dimensions
    missing = sorted(set(selected_dimensions) - set(available_dimensions))
    if missing:
        raise ValueError(f"Requested dimensions not found in {language} data: {missing}")

    lang_out = out_root / language
    lang_out.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    suspicious_tables: list[pd.DataFrame] = []

    print("=" * 90)
    print(f"LANGUAGE: {language.upper()}")
    print(f"Input CSV: {input_csv}")
    print(f"Dimensions: {selected_dimensions}")
    print("=" * 90)

    for dimension in selected_dimensions:
        label = safe_filename(dimension)
        X, sub, drop_cols, dropped_numeric_cols = get_feature_matrix_like_training_script(
            df=df,
            dimension=dimension,
            target_col=target_col,
        )

        feature_df = summarize_feature_columns(X)
        feature_df.insert(0, "language", language)
        feature_df.insert(1, "dimension", dimension)

        suspicious_df = feature_df[
            feature_df["suspicious_name"]
            | feature_df["zero_variance"]
            | feature_df["near_zero_variance_95pct_same_value"]
        ].copy()

        dropped_df = pd.DataFrame(
            {
                "language": language,
                "dimension": dimension,
                "dropped_numeric_column": dropped_numeric_cols,
            }
        )

        feature_path = lang_out / f"{label}_feature_columns.csv"
        suspicious_path = lang_out / f"{label}_suspicious_columns.csv"
        dropped_path = lang_out / f"{label}_dropped_numeric_columns.csv"

        feature_df.to_csv(feature_path, index=False)
        suspicious_df.to_csv(suspicious_path, index=False)
        dropped_df.to_csv(dropped_path, index=False)

        summary = {
            "language": language,
            "dimension": dimension,
            "n_rows_after_target_filtering": int(len(sub)),
            "n_numeric_features_used": int(X.shape[1]),
            "n_dropped_numeric_metadata_columns": int(len(dropped_numeric_cols)),
            "n_suspicious_name_features": int(feature_df["suspicious_name"].sum()),
            "n_zero_variance_features": int(feature_df["zero_variance"].sum()),
            "n_near_zero_variance_features_95pct_same_value": int(
                feature_df["near_zero_variance_95pct_same_value"].sum()
            ),
            "feature_columns_file": str(feature_path),
            "suspicious_columns_file": str(suspicious_path),
            "dropped_numeric_columns_file": str(dropped_path),
        }
        summary_rows.append(summary)

        if not suspicious_df.empty:
            suspicious_tables.append(suspicious_df)

        print(f"\nDimension: {dimension}")
        print(f"  Rows after target filtering: {len(sub)}")
        print(f"  Numeric features used:       {X.shape[1]}")
        print(f"  Suspicious-name features:    {summary['n_suspicious_name_features']}")
        print(f"  Zero-variance features:      {summary['n_zero_variance_features']}")
        print(f"  Near-zero-variance features: {summary['n_near_zero_variance_features_95pct_same_value']}")
        print(f"  Saved: {feature_path}")

    return summary_rows, suspicious_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit numeric feature columns used by the LIWC modeling scripts without changing them."
    )
    parser.add_argument(
        "--language",
        choices=["de", "en", "both"],
        default="both",
        help="Which LIWC input to audit. Default: both.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="*",
        default=None,
        help="Optional subset of dimension labels, e.g. d1_illness_beliefs.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_COL,
        help="Regression target column. Default: average_score.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing configs/paths.json. Usually not needed if script is placed beside training scripts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: results/approach1/feature_audit under project root.",
    )
    parser.add_argument(
        "--fail-on-suspicious-name",
        action="store_true",
        help="Exit with an error if any feature column has a suspicious name.",
    )
    args = parser.parse_args()

    script_path = Path(__file__)
    project_root = infer_project_root(script_path, args.project_root)
    paths = load_paths_config(project_root)
    out_root = output_root(project_root, args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    languages = ["de", "en"] if args.language == "both" else [args.language]

    all_summary_rows: list[dict[str, Any]] = []
    all_suspicious_tables: list[pd.DataFrame] = []

    for language in languages:
        summary_rows, suspicious_tables = audit_one_language(
            project_root=project_root,
            paths=paths,
            language=language,
            dimensions=args.dimensions,
            target_col=args.target,
            out_root=out_root,
        )
        all_summary_rows.extend(summary_rows)
        all_suspicious_tables.extend(suspicious_tables)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = out_root / "feature_audit_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if all_suspicious_tables:
        suspicious_all = pd.concat(all_suspicious_tables, ignore_index=True)
    else:
        suspicious_all = pd.DataFrame()
    suspicious_all_path = out_root / "suspicious_feature_columns.csv"
    suspicious_all.to_csv(suspicious_all_path, index=False)

    print("\n" + "=" * 90)
    print("AUDIT COMPLETE")
    print(f"Summary saved to:    {summary_path}")
    print(f"Suspicious saved to: {suspicious_all_path}")
    print("=" * 90)

    if args.fail_on_suspicious_name and not suspicious_all.empty:
        n_suspicious_name = int(suspicious_all["suspicious_name"].sum())
        if n_suspicious_name > 0:
            raise SystemExit(f"Found {n_suspicious_name} suspicious-name feature columns.")


if __name__ == "__main__":
    main()
