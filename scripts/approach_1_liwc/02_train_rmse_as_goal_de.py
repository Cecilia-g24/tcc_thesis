"""
Approach 1 (German): LIWC -> human average_score regression with nested CV.

This script trains and evaluates LIWC-based regression models separately for each
transcultural-competence dimension, using the German LIWC results CSV. The
supervised target is `average_score`; RMSE is the optimization goal used for
model selection.

Input CSV and output folder are both read directly from configs/paths.json
(data_clean / liwc_results_subdir / liwc_de_results_filename, and
results_approach1_de_dir). There is no file search/fallback.

Default output folder:
    results/approach1/de_results/rmse_as_goal

Main outputs, CSV-only by default:
    - detailed_pipeline_metrics.csv
    - best_model_per_dimension.csv
    - best_model_family_per_dimension.csv
    - model_family_summary.csv
    - best_vs_baseline.csv
    - outer_cv_predictions.csv
    - run_metadata.json
    - run_log.txt
    - by_dimension/<dimension>/<dimension>_detailed_pipeline_metrics.csv
    - by_dimension/<dimension>/<dimension>_outer_cv_predictions.csv
    - status/<dimension>_DONE.json completion markers for resume
    - permuters/permuter_<dimension>.pkl checkpoints for resume

The script no longer writes duplicate BioPsyKit summary reports, HTML, TXT,
or LaTeX files.

Install once:
    pip install biopsykit scikit-learn pandas numpy
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from biopsykit.classification.model_selection import SklearnPipelinePermuter
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


import warnings
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
    module="sklearn.utils.parallel",
)



# ---------------------------------------------------------------------------
# Project defaults
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_paths_config(project_root: Path) -> dict[str, Any]:
    paths_file = project_root / "configs" / "paths.json"
    return json.loads(paths_file.read_text(encoding="utf-8"))


_PATHS = load_paths_config(PROJECT_ROOT)
INPUT_CSV = (
    PROJECT_ROOT
    / _PATHS["data_clean"]
    / _PATHS["liwc_results_subdir"]
    / _PATHS["liwc_de_results_filename"]
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / _PATHS["results_approach1_de_dir"] / "rmse_as_goal"
DEFAULT_TARGET_COL = "average_score"
DEFAULT_SCORING = "neg_root_mean_squared_error"

# Export policy: CSV only.
# Keep checkpoints and status markers because they are required for resume/reconnection.
# These constants remain False so helper functions cannot create duplicate formats.
SAVE_HTML = False
SAVE_TXT = False
SAVE_LATEX = False

# BioPsyKit builds sklearn.Pipeline(..., memory=joblib.Memory('cachedir')) by default.
# On Windows this can produce PermissionError messages when several CV jobs read/write
# the same cache. Disable this cache; keep only our own permuter checkpoints for resume.
DISABLE_BIOPSYKIT_PIPELINE_CACHE = True
BIOPSYKIT_VERBOSE = 0
BIOPSYKIT_N_JOBS = -1

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


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

class Tee:
    """Write stdout/stderr both to terminal and a log file."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def safe_filename(value: Any) -> str:
    """Create filesystem-safe names from dimension/model labels."""
    text = str(value)
    for ch in '<>:"/\\|?* ':
        text = text.replace(ch, "_")
    return text.strip("_")


def make_report_dirs(results_dir: Path) -> dict[str, Path]:
    """Create only directories that are actually needed.

    - by_dimension: per-dimension files used for resume and inspection
    - permuters: BioPsyKit checkpoints used for resume/reconnection
    - status: DONE markers used to skip completed dimensions

    Final combined tables are saved directly in the root results directory.
    """
    dirs = {
        "root": results_dir,
        "by_dimension": results_dir / "by_dimension",
        "permuters": results_dir / "permuters",
        "status": results_dir / "status",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def cleanup_legacy_outputs(results_dir: Path) -> None:
    """Remove outputs produced by older, more verbose versions of this script.

    This keeps files needed for resume: by_dimension detailed metrics/predictions,
    status markers, and permuter checkpoints.
    """
    # Keep the old combined predictions file before removing the legacy folder.
    old_predictions = results_dir / "predictions" / "outer_cv_predictions.csv"
    new_predictions = results_dir / "outer_cv_predictions.csv"
    if old_predictions.exists() and not new_predictions.exists():
        shutil.copy2(old_predictions, new_predictions)

    for dirname in ["html", "latex", "tables", "text", "predictions"]:
        shutil.rmtree(results_dir / dirname, ignore_errors=True)

    root_patterns = [
        "nested_cv_metric_summary.*",
        "mean_pipeline_score_results.*",
        "best_hyperparameter_pipeline.*",
        "best_estimator_summary.*",
        "metric_summary_latex.*",
    ]
    dimension_patterns = [
        "*_nested_cv_metric_summary.*",
        "*_mean_pipeline_score_results.*",
        "*_best_hyperparameter_pipeline.*",
        "*_best_estimator_summary.*",
        "*_metric_summary_latex.*",
        "*_ERROR.txt",
    ]

    for pattern in root_patterns:
        for path in results_dir.glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)

    by_dimension = results_dir / "by_dimension"
    if by_dimension.exists():
        for pattern in dimension_patterns:
            for path in by_dimension.glob(f"*/{pattern}"):
                if path.is_file():
                    path.unlink(missing_ok=True)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "__".join(str(part) for part in col if str(part) not in {"", "None"})
            for col in out.columns
        ]
    else:
        out.columns = [str(col) for col in out.columns]
    return out


def dataframe_for_export(df: pd.DataFrame, reset_index: bool = True) -> pd.DataFrame:
    """Convert complex cells to strings so BioPsyKit reports export reliably."""
    out = flatten_columns(df)
    if reset_index:
        out = out.reset_index()
        out = flatten_columns(out)

    def convert_cell(value: Any) -> Any:
        if isinstance(value, (np.ndarray, list, tuple, dict)):
            return repr(value)
        return value

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(convert_cell)
    return out


def save_table(df: pd.DataFrame, base_path: Path, reset_index: bool = True) -> None:
    """Save one canonical CSV file, creating the parent directory if needed."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = dataframe_for_export(df, reset_index=reset_index)
    export_df.to_csv(base_path.with_suffix(".csv"), index=False)


def as_numeric_array(value: Any) -> np.ndarray:
    """Convert BioPsyKit-stored lists/arrays to a clean numeric 1D array."""
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple, pd.Series)):
        arr = np.asarray(value)
    else:
        arr = np.asarray([value])

    arr = arr.reshape(-1)
    numeric = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().to_numpy(dtype=float)
    return numeric


def correlation_or_nan(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan
    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method=method))


def regression_metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    return {
        "n_predictions": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "Pearson_r": correlation_or_nan(y_true, y_pred, method="pearson"),
        "Spearman_rho": correlation_or_nan(y_true, y_pred, method="spearman"),
        "mean_error_pred_minus_true": float(np.mean(errors)),
        "median_absolute_error": float(np.median(abs_errors)),
        "max_absolute_error": float(np.max(abs_errors)),
    }


def round_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    metric_cols = [
        "MAE",
        "RMSE",
        "R2",
        "Pearson_r",
        "Spearman_rho",
        "mean_error_pred_minus_true",
        "median_absolute_error",
        "max_absolute_error",
    ]
    for col in metric_cols:
        if col in out.columns:
            out[col] = out[col].round(4)
    return out


def pipeline_index_to_dict(index_value: Any, index_names: list[Any]) -> dict[str, str]:
    if isinstance(index_value, tuple):
        values = list(index_value)
    else:
        values = [index_value]

    names = [str(name) if name is not None else f"pipeline_step_{i}" for i, name in enumerate(index_names)]
    if len(names) != len(values):
        names = [f"pipeline_step_{i}" for i in range(len(values))]
    return {name: str(value) for name, value in zip(names, values)}


def model_family_from_steps(step_dict: dict[str, str]) -> str:
    for key in ["pipeline_reg", "pipeline_regressor", "pipeline_clf", "reg", "clf"]:
        if key in step_dict:
            return step_dict[key]
    # Fallback: the last pipeline index level is usually the model/regressor.
    return next(reversed(step_dict.values())) if step_dict else "unknown"


def get_target_and_features(
    df: pd.DataFrame,
    dimension: str,
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    sub = df[df["dimension"] == dimension].copy()
    sub = sub.dropna(subset=[target_col])
    y = sub[target_col].to_numpy(dtype=float)

    drop_cols = list(dict.fromkeys(METADATA_COLS + [target_col]))
    X = sub.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.reset_index(drop=True)

    if X.empty:
        raise ValueError(f"No numeric LIWC feature columns found for dimension: {dimension}")
    return X, y, sub


# ---------------------------------------------------------------------------
# Model menu, following the BioPsyKit PipelinePermuter structure
# ---------------------------------------------------------------------------

def build_model_config(random_state: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    model_dict = {
        "imputer": {
            "Median": SimpleImputer(strategy="median"),
        },
        "scaler": {
            "Standard": StandardScaler(),
            "Robust": RobustScaler(),
            "MinMax": MinMaxScaler(),
        },
        "reduce_dim": {
            "SelectKBest": SelectKBest(score_func=f_regression),
            "RFE": RFE(estimator=Ridge()),
            "passthrough": "passthrough",
        },
        "reg": {
            "Baseline": DummyRegressor(strategy="mean"),
            "Ridge": Ridge(),
            "Lasso": Lasso(random_state=random_state, max_iter=10000),
            "ElasticNet": ElasticNet(random_state=random_state, max_iter=10000),
            "BayesianRidge": BayesianRidge(),
            "SVR_Linear": SVR(kernel="linear"),
            "SVR_RBF": SVR(kernel="rbf"),
            "HistGradBoost": HistGradientBoostingRegressor(random_state=random_state),
            "RandomForest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(random_state=random_state),
        },
    }

    # Keys must match the names used in model_dict.
    params_dict = {
        "Median": None,
        "Standard": None,
        "Robust": None,
        "MinMax": None,
        "SelectKBest": {"k": [10, 20, 40, "all"]},
        "RFE": {"n_features_to_select": [10, 20, 40]},
        "passthrough": None,
        "Baseline": None,
        "Ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        "Lasso": {"alpha": [0.001, 0.01, 0.1]},
        "ElasticNet": {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.3, 0.5, 0.7]},
        "BayesianRidge": None,
        "SVR_Linear": {"C": [0.1, 1.0, 10.0]},
        "SVR_RBF": {
            "C": [0.1, 1.0, 10.0],
            "epsilon": [0.1, 0.2],
            "gamma": ["auto", "scale"],
        },
        "HistGradBoost": {"max_leaf_nodes": [7, 15]},
        "RandomForest": {
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 3],
        },
        "GradientBoosting": {
            "n_estimators": [100, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
        "KNN": {"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"]},
        "DecisionTree": {"max_depth": [None, 3, 5, 10], "min_samples_leaf": [1, 3, 5]},
    }

    # Randomized search keeps larger grids computationally manageable.
    hyper_search_dict = {
        "SVR_RBF": {"search_method": "random", "n_iter": 6},
        "RandomForest": {"search_method": "random", "n_iter": 8},
        "GradientBoosting": {"search_method": "random", "n_iter": 6},
    }
    return model_dict, params_dict, hyper_search_dict


# ---------------------------------------------------------------------------
# Fitting and exporting
# ---------------------------------------------------------------------------

def _n_completed_searches(permuter: SklearnPipelinePermuter) -> int:
    """Return the number of already stored BioPsyKit parameter searches."""
    param_searches = getattr(permuter, "param_searches", None)
    if param_searches is None:
        return 0
    try:
        return len(param_searches)
    except TypeError:
        return 0


def _pickle_permuter(permuter: SklearnPipelinePermuter, path: Path) -> None:
    """Atomically save a permuter checkpoint.

    BioPsyKit's ``to_pickle()`` validates that the file path ends in ``.pkl``.
    Therefore the temporary file must also keep a ``.pkl`` suffix; a name like
    ``permuter_d1.pkl.tmp`` fails BioPsyKit's extension check.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        permuter.to_pickle(tmp_path)
    except OSError:
        import pickle as _pickle

        with open(str(tmp_path), "wb") as f:
            _pickle.dump(permuter, f)
    tmp_path.replace(path)


def _load_permuter_checkpoint(
    checkpoint_path: Path,
    model_dict: dict[str, dict[str, Any]],
    params_dict: dict[str, Any],
    hyper_search_dict: dict[str, Any],
    random_state: int,
    resume: bool = True,
) -> SklearnPipelinePermuter:
    """Load an existing checkpoint if possible; otherwise create a fresh permuter."""
    if resume and checkpoint_path.exists():
        try:
            permuter = SklearnPipelinePermuter.from_pickle(checkpoint_path)
            print(
                f"Loaded checkpoint ({_n_completed_searches(permuter)} combinations already fitted): "
                f"{checkpoint_path}"
            )
            return permuter
        except Exception as load_err:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            corrupt_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.corrupt_{timestamp}{checkpoint_path.suffix}")
            try:
                checkpoint_path.replace(corrupt_path)
                print(f"Could not load checkpoint ({load_err}). Moved corrupt file to: {corrupt_path}")
            except OSError:
                print(f"Could not load checkpoint ({load_err}). Starting fresh.")

    return SklearnPipelinePermuter(
        model_dict, params_dict, hyper_search_dict=hyper_search_dict, random_state=random_state
    )


def dimension_output_paths(dirs: dict[str, Path], label: str) -> dict[str, Path]:
    """Return the files that indicate a finished dimension-level run."""
    dim_dir = dirs["by_dimension"] / label
    return {
        "dim_dir": dim_dir,
        "done_marker": dirs["status"] / f"{label}_DONE.json",
        "detailed_metrics": dim_dir / f"{label}_detailed_pipeline_metrics.csv",
        "predictions": dim_dir / f"{label}_outer_cv_predictions.csv",
    }


def load_completed_dimension_outputs(
    dirs: dict[str, Path], label: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]] | None:
    """Load completed per-dimension outputs so a rerun can skip that dimension."""
    paths = dimension_output_paths(dirs, label)
    if not (paths["done_marker"].exists() and paths["detailed_metrics"].exists()):
        return None

    detailed_df = pd.read_csv(paths["detailed_metrics"])
    predictions_df = pd.read_csv(paths["predictions"]) if paths["predictions"].exists() else pd.DataFrame()
    try:
        marker = json.loads(paths["done_marker"].read_text(encoding="utf-8"))
        dimension_name = str(marker.get("dimension", label))
    except Exception:
        dimension_name = label

    report_tables: dict[str, pd.DataFrame] = {}
    for report_name in [
        "nested_cv_metric_summary",
        "mean_pipeline_score_results",
        "best_hyperparameter_pipeline",
        "best_estimator_summary",
    ]:
        report_path = paths["dim_dir"] / f"{label}_{report_name}.csv"
        if report_path.exists():
            table = pd.read_csv(report_path)
            if "dimension" not in table.columns:
                table.insert(0, "dimension", dimension_name)
            report_tables[report_name] = table

    return detailed_df, predictions_df, report_tables


def write_dimension_done_marker(
    dirs: dict[str, Path],
    label: str,
    dimension: str,
    checkpoint_path: Path,
    n_rows: int,
    n_features: int,
) -> None:
    """Write a small marker file only after all exports for the dimension succeeded."""
    paths = dimension_output_paths(dirs, label)
    marker = {
        "dimension": dimension,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(checkpoint_path),
        "n_rows": int(n_rows),
        "n_features": int(n_features),
    }
    paths["done_marker"].parent.mkdir(parents=True, exist_ok=True)
    paths["done_marker"].write_text(json.dumps(marker, indent=2), encoding="utf-8")


def fit_dimension_permuter(
    X: pd.DataFrame,
    y: np.ndarray,
    model_dict: dict[str, dict[str, Any]],
    params_dict: dict[str, Any],
    hyper_search_dict: dict[str, Any],
    outer_cv: KFold,
    inner_cv: KFold,
    scoring: str,
    random_state: int,
    intermediate_path: Path,
    save_intermediate: bool = True,
    resume: bool = True,
) -> SklearnPipelinePermuter:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    permuter = _load_permuter_checkpoint(
        checkpoint_path=intermediate_path,
        model_dict=model_dict,
        params_dict=params_dict,
        hyper_search_dict=hyper_search_dict,
        random_state=random_state,
        resume=resume,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if save_intermediate and hasattr(permuter, "fit_and_save_intermediate"):
                try:
                    permuter.fit_and_save_intermediate(
                        X=X,
                        y=y,
                        outer_cv=outer_cv,
                        inner_cv=inner_cv,
                        file_path=intermediate_path,
                        scoring=scoring,
                        use_cache=not DISABLE_BIOPSYKIT_PIPELINE_CACHE,
                        verbose=BIOPSYKIT_VERBOSE,
                        n_jobs=BIOPSYKIT_N_JOBS,
                    )
                except (TypeError, OSError) as exc:
                    # TypeError: older BioPsyKit versions may not expose all fit() kwargs.
                    # OSError [Errno 22]: Windows EINVAL during intermediate pickle saves.
                    # Fallback to normal fit() and still save a final checkpoint.
                    print(
                        "Warning: intermediate checkpoint saving failed "
                        f"({exc}). Continuing with normal fit() and final checkpoint saving."
                    )
                    permuter.fit(
                        X=X,
                        y=y,
                        outer_cv=outer_cv,
                        inner_cv=inner_cv,
                        scoring=scoring,
                        use_cache=not DISABLE_BIOPSYKIT_PIPELINE_CACHE,
                        verbose=BIOPSYKIT_VERBOSE,
                        n_jobs=BIOPSYKIT_N_JOBS,
                    )
                    _pickle_permuter(permuter, intermediate_path)
            else:
                permuter.fit(
                    X=X,
                    y=y,
                    outer_cv=outer_cv,
                    inner_cv=inner_cv,
                    scoring=scoring,
                    use_cache=not DISABLE_BIOPSYKIT_PIPELINE_CACHE,
                    verbose=BIOPSYKIT_VERBOSE,
                    n_jobs=BIOPSYKIT_N_JOBS,
                )
                _pickle_permuter(permuter, intermediate_path)
        except BaseException as exc:
            # Best-effort emergency checkpoint. This is useful if the user interrupts
            # the run or if an estimator crashes after some combinations have finished.
            try:
                _pickle_permuter(permuter, intermediate_path)
                print(
                    f"Emergency checkpoint saved after interruption/error "
                    f"({_n_completed_searches(permuter)} combinations stored): {intermediate_path}"
                )
            except Exception as save_err:
                print(f"Could not save emergency checkpoint: {save_err}")
            raise exc

    return permuter


def biopsykit_report_getters(permuter: SklearnPipelinePermuter) -> list[tuple[str, Callable[[], Any]]]:
    """Reports shown in the BioPsyKit Display Results / Further Functions sections."""
    report_getters: list[tuple[str, Callable[[], Any]]] = [
        ("nested_cv_metric_summary", permuter.metric_summary),
        ("mean_pipeline_score_results", permuter.mean_pipeline_score_results),
        ("best_hyperparameter_pipeline", permuter.best_hyperparameter_pipeline),
    ]
    if hasattr(permuter, "best_estimator_summary"):
        report_getters.append(("best_estimator_summary", permuter.best_estimator_summary))
    return report_getters


def collect_biopsykit_report_tables(permuter: SklearnPipelinePermuter) -> dict[str, pd.DataFrame]:
    """Return BioPsyKit report tables that can be combined across dimensions."""
    tables: dict[str, pd.DataFrame] = {}
    for report_name, getter in biopsykit_report_getters(permuter):
        try:
            report = getter()
        except Exception:
            continue
        if isinstance(report, pd.DataFrame):
            tables[report_name] = report.copy()
    return tables


def save_biopsykit_reports(permuter: SklearnPipelinePermuter, out_dir: Path, label: str) -> None:
    """Save the BioPsyKit display-result reports for one dimension."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for report_name, getter in biopsykit_report_getters(permuter):
        try:
            report = getter()
        except Exception as exc:  # keep the training run from failing only due to reporting
            (out_dir / f"{label}_{report_name}_ERROR.txt").write_text(str(exc), encoding="utf-8")
            continue

        base_path = out_dir / f"{label}_{report_name}"
        if isinstance(report, pd.DataFrame):
            save_table(report, base_path, reset_index=True)
        else:
            if SAVE_TXT:
                base_path.with_suffix(".txt").write_text(str(report), encoding="utf-8")

    if hasattr(permuter, "metric_summary_to_latex"):
        try:
            latex = permuter.metric_summary_to_latex()
            if SAVE_LATEX:
                (out_dir / f"{label}_metric_summary_latex.tex").write_text(latex, encoding="utf-8")
        except Exception as exc:
            (out_dir / f"{label}_metric_summary_latex_ERROR.txt").write_text(str(exc), encoding="utf-8")


def detailed_metrics_from_permuter(
    permuter: SklearnPipelinePermuter,
    dimension: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute MAE/RMSE/R2/etc. from outer-fold predictions in metric_summary()."""
    metric_summary = permuter.metric_summary()
    index_names = list(metric_summary.index.names)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for index_value, row in metric_summary.iterrows():
        y_true = as_numeric_array(row.get("true_labels"))
        y_pred = as_numeric_array(row.get("predicted_labels"))
        n = min(len(y_true), len(y_pred))
        if n == 0:
            continue
        y_true = y_true[:n]
        y_pred = y_pred[:n]

        step_dict = pipeline_index_to_dict(index_value, index_names)
        model_family = model_family_from_steps(step_dict)
        pipeline_str = " | ".join(f"{k}={v}" for k, v in step_dict.items())

        metrics = regression_metric_dict(y_true, y_pred)
        metric_entry: dict[str, Any] = {
            "dimension": dimension,
            "model_family": model_family,
            "pipeline": pipeline_str,
            **step_dict,
            **metrics,
        }

        # Preserve BioPsyKit's outer-CV score columns as reference columns.
        for col in metric_summary.columns:
            if str(col).startswith(("mean_test_", "std_test_", "test_")):
                metric_entry[f"biopsykit_{col}"] = row[col]

        metric_rows.append(metric_entry)

        test_indices = as_numeric_array(row.get("test_indices"))
        if len(test_indices) < n:
            test_indices = np.arange(n)
        for i in range(n):
            prediction_rows.append(
                {
                    "dimension": dimension,
                    "model_family": model_family,
                    "pipeline": pipeline_str,
                    **step_dict,
                    "row_position_within_dimension": int(test_indices[i]),
                    "true_score": float(y_true[i]),
                    "predicted_score": float(y_pred[i]),
                    "error_pred_minus_true": float(y_pred[i] - y_true[i]),
                    "absolute_error": float(abs(y_pred[i] - y_true[i])),
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    if not metrics_df.empty:
        metrics_df = round_metric_columns(metrics_df.sort_values(["dimension", "RMSE", "MAE"]).reset_index(drop=True))
    if not predictions_df.empty:
        predictions_df = predictions_df.sort_values(
            ["dimension", "model_family", "pipeline", "row_position_within_dimension"]
        ).reset_index(drop=True)
    return metrics_df, predictions_df


def export_combined_reports(
    detailed_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    """Save combined metric/prediction reports across all dimensions."""
    if detailed_df.empty:
        raise ValueError("No detailed metrics were produced.")

    save_table(detailed_df, dirs["root"] / "detailed_pipeline_metrics", reset_index=False)

    if not predictions_df.empty:
        predictions_df.to_csv(dirs["root"] / "outer_cv_predictions.csv", index=False)

    best_per_dimension = (
        detailed_df.sort_values(["dimension", "RMSE", "MAE", "R2"], ascending=[True, True, True, False])
        .groupby("dimension", as_index=False)
        .first()
    )
    save_table(best_per_dimension, dirs["root"] / "best_model_per_dimension", reset_index=False)

    best_family = (
        detailed_df.sort_values(["dimension", "model_family", "RMSE", "MAE", "R2"], ascending=[True, True, True, True, False])
        .groupby(["dimension", "model_family"], as_index=False)
        .first()
    )
    save_table(best_family, dirs["root"] / "best_model_family_per_dimension", reset_index=False)

    family_summary = (
        detailed_df.groupby(["dimension", "model_family"], as_index=False)
        .agg(
            n_pipeline_variants=("pipeline", "nunique"),
            best_RMSE=("RMSE", "min"),
            mean_RMSE=("RMSE", "mean"),
            best_MAE=("MAE", "min"),
            mean_MAE=("MAE", "mean"),
            best_R2=("R2", "max"),
            mean_R2=("R2", "mean"),
            best_Pearson_r=("Pearson_r", "max"),
            best_Spearman_rho=("Spearman_rho", "max"),
        )
        .sort_values(["dimension", "best_RMSE"])
        .reset_index(drop=True)
    )
    family_summary = round_metric_columns(
        family_summary.rename(
            columns={
                "best_RMSE": "RMSE",
                "best_MAE": "MAE",
                "best_R2": "R2",
                "best_Pearson_r": "Pearson_r",
                "best_Spearman_rho": "Spearman_rho",
            }
        )
    ).rename(
        columns={
            "RMSE": "best_RMSE",
            "MAE": "best_MAE",
            "R2": "best_R2",
            "Pearson_r": "best_Pearson_r",
            "Spearman_rho": "best_Spearman_rho",
        }
    )
    save_table(family_summary, dirs["root"] / "model_family_summary", reset_index=False)

    baseline_rows = best_family[best_family["model_family"].eq("Baseline")][["dimension", "MAE", "RMSE", "R2"]]
    if not baseline_rows.empty:
        baseline_rows = baseline_rows.rename(
            columns={"MAE": "baseline_MAE", "RMSE": "baseline_RMSE", "R2": "baseline_R2"}
        )
        comparison = best_per_dimension.merge(baseline_rows, on="dimension", how="left")
        comparison["MAE_improvement_vs_baseline"] = comparison["baseline_MAE"] - comparison["MAE"]
        comparison["RMSE_improvement_vs_baseline"] = comparison["baseline_RMSE"] - comparison["RMSE"]
        comparison = round_metric_columns(comparison)
        save_table(comparison, dirs["root"] / "best_vs_baseline", reset_index=False)


def save_metadata(
    dirs: dict[str, Path],
    input_csv: Path,
    results_dir: Path,
    target_col: str,
    scoring: str,
    dimensions: list[str],
    model_dict: dict[str, dict[str, Any]],
    params_dict: dict[str, Any],
    outer_splits: int,
    inner_splits: int,
    random_state: int,
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "input_csv": str(input_csv),
        "results_dir": str(results_dir),
        "target_column": target_col,
        "optimization_scoring": scoring,
        "selection_rule": "lowest RMSE computed from outer-CV predictions",
        "outer_cv": {"type": "KFold", "n_splits": outer_splits, "shuffle": True, "random_state": random_state},
        "inner_cv": {"type": "KFold", "n_splits": inner_splits, "shuffle": True, "random_state": 0},
        "dimensions": dimensions,
        "pipeline_steps": {step: list(options.keys()) for step, options in model_dict.items()},
        "params_dict_keys": list(params_dict.keys()),
        "biopsykit_pipeline_cache_enabled": not DISABLE_BIOPSYKIT_PIPELINE_CACHE,
        "biopsykit_verbose": BIOPSYKIT_VERBOSE,
        "biopsykit_n_jobs": BIOPSYKIT_N_JOBS,
    }
    (dirs["root"] / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested-CV LIWC (German) regression optimized for RMSE.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory. Default: results/approach1/de_results/rmse_as_goal",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET_COL, help="Regression target column.")
    parser.add_argument("--dimensions", nargs="*", default=None, help="Optional subset of dimension labels to run.")
    parser.add_argument("--outer-splits", type=int, default=5, help="Outer CV folds.")
    parser.add_argument("--inner-splits", type=int, default=3, help="Inner CV folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for model search and outer CV.")
    parser.add_argument(
        "--no-intermediate-save",
        action="store_true",
        help="Disable BioPsyKit intermediate permuter saving during fitting.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing completed dimension outputs and permuter checkpoints.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    results_dir = args.results_dir if args.results_dir.is_absolute() else PROJECT_ROOT / args.results_dir
    dirs = make_report_dirs(results_dir)
    cleanup_legacy_outputs(results_dir)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {INPUT_CSV}")
    input_csv = INPUT_CSV
    df = pd.read_csv(input_csv)

    if "dimension" not in df.columns:
        raise KeyError("Expected a 'dimension' column in the LIWC CSV.")
    if args.target not in df.columns:
        raise KeyError(f"Expected target column '{args.target}' in the LIWC CSV.")

    all_dimensions = df["dimension"].dropna().unique().tolist()
    dimensions = args.dimensions if args.dimensions else all_dimensions
    missing_dimensions = sorted(set(dimensions) - set(all_dimensions))
    if missing_dimensions:
        raise ValueError(f"Requested dimensions not found in data: {missing_dimensions}")

    print("=" * 80)
    print("APPROACH 1 (GERMAN): LIWC REGRESSION, RMSE AS OPTIMIZATION GOAL")
    print("=" * 80)
    print(f"Input CSV:   {input_csv}")
    print(f"Results dir: {results_dir}")
    print(f"Target:      {args.target}")
    print(f"Scoring:     {DEFAULT_SCORING}")
    print(f"Dimensions:  {dimensions}")

    model_dict, params_dict, hyper_search_dict = build_model_config(random_state=args.random_state)
    outer_cv = KFold(n_splits=args.outer_splits, shuffle=True, random_state=args.random_state)
    inner_cv = KFold(n_splits=args.inner_splits, shuffle=True, random_state=0)

    all_detailed: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    for dimension in dimensions:
        label = safe_filename(dimension)
        print("\n" + "=" * 80)
        print(f"DIMENSION: {dimension}")
        print("=" * 80)

        X, y, sub = get_target_and_features(df, dimension, args.target)
        print(f"Rows after target filtering: {len(sub)}")
        print(f"X shape: {X.shape}")
        print(f"y mean={np.mean(y):.3f}, std={np.std(y):.3f}, min={np.min(y):.3f}, max={np.max(y):.3f}")

        intermediate_path = dirs["permuters"] / f"permuter_{label}.pkl"
        resume = not args.no_resume
        if resume:
            completed_outputs = load_completed_dimension_outputs(dirs, label)
            if completed_outputs is not None:
                detailed_df, predictions_df, _report_tables = completed_outputs
                print(f"Completed outputs found for {dimension}; skipping training and reusing exported files.")
                if not detailed_df.empty:
                    all_detailed.append(detailed_df)
                if not predictions_df.empty:
                    all_predictions.append(predictions_df)
                continue

        permuter = fit_dimension_permuter(
            X=X,
            y=y,
            model_dict=model_dict,
            params_dict=params_dict,
            hyper_search_dict=hyper_search_dict,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            scoring=DEFAULT_SCORING,
            random_state=args.random_state,
            intermediate_path=intermediate_path,
            save_intermediate=not args.no_intermediate_save,
            resume=not args.no_resume,
        )

        # Always save a final permuter snapshot.
        _pickle_permuter(permuter, intermediate_path)
        print(f"Saved PipelinePermuter: {intermediate_path}")

        dim_dir = dirs["by_dimension"] / label
        dim_dir.mkdir(parents=True, exist_ok=True)

        detailed_df, predictions_df = detailed_metrics_from_permuter(permuter, dimension)
        if not detailed_df.empty:
            save_table(detailed_df, dim_dir / f"{label}_detailed_pipeline_metrics", reset_index=False)
            all_detailed.append(detailed_df)
        if not predictions_df.empty:
            predictions_df.to_csv(dim_dir / f"{label}_outer_cv_predictions.csv", index=False)
            all_predictions.append(predictions_df)

        if detailed_df.empty:
            raise ValueError(f"No detailed metrics were produced for dimension: {dimension}")

        write_dimension_done_marker(
            dirs=dirs,
            label=label,
            dimension=dimension,
            checkpoint_path=intermediate_path,
            n_rows=len(sub),
            n_features=X.shape[1],
        )

        best = detailed_df.sort_values(["RMSE", "MAE", "R2"], ascending=[True, True, False]).iloc[0]
        print(
            f"Best by RMSE: {best['model_family']} | "
            f"RMSE={best['RMSE']:.4f}, MAE={best['MAE']:.4f}, R2={best['R2']:.4f}"
        )

    combined_detailed = pd.concat(all_detailed, ignore_index=True) if all_detailed else pd.DataFrame()
    combined_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    export_combined_reports(combined_detailed, combined_predictions, dirs)

    save_metadata(
        dirs=dirs,
        input_csv=input_csv,
        results_dir=results_dir,
        target_col=args.target,
        scoring=DEFAULT_SCORING,
        dimensions=dimensions,
        model_dict=model_dict,
        params_dict=params_dict,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        random_state=args.random_state,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Main results folder: {results_dir}")
    print("Key files:")
    print(f"  - {results_dir / 'best_model_per_dimension.csv'}")
    print(f"  - {results_dir / 'best_model_family_per_dimension.csv'}")
    print(f"  - {results_dir / 'detailed_pipeline_metrics.csv'}")
    print(f"  - {results_dir / 'model_family_summary.csv'}")
    print(f"  - {results_dir / 'outer_cv_predictions.csv'}")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir if args.results_dir.is_absolute() else PROJECT_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_log.txt"
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            run(args)


if __name__ == "__main__":
    main()
