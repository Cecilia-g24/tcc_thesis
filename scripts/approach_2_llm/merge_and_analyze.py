"""
Merge LLM assessment results back onto the full dataset and compute diagnostics.

This version addresses the main validity issues found in the previous version:
- uses id + dimension as the merge key, not id alone
- treats a successful LLM call as: no error + integer score in {0, 1, 2, 3, 4}
- reports success/failure both overall and by dimension
- computes accuracy metrics on the common set of items scored successfully by all variants
- reports MAE, RMSE, bias, R2, Pearson r, Spearman r, and score means/stds
- avoids Python round() for human average score distributions
- adds an anchor-overlap audit for the prompt variant that includes anchor examples

Run from the repo root or from this script directory:
    python merge_and_analyze.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from prompt_construction import DIMENSIONS
except Exception:  # allows the script to run even if prompt_construction.py is elsewhere
    DIMENSIONS = {}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL_DATASET_CSV = REPO_ROOT / "data" / "data_clean" / "01_csvs_for_liwc_manual_input" / "full_dataset_en.csv"
DEFAULT_LLM_RESULTS_CSV = (
    REPO_ROOT / "results" / "approach2" / "en_text_en_prompt_results" / "dev_results_qwen3-30b-a3b-instruct-2507.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "approach2"
DEFAULT_TEXT_COL = "text_en"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and analyze LLM-as-judge results.")
    parser.add_argument("--full-dataset-csv", type=Path, default=DEFAULT_FULL_DATASET_CSV)
    parser.add_argument("--llm-results-csv", type=Path, default=DEFAULT_LLM_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL)
    parser.add_argument("--anchor-threshold", type=float, default=0.80,
                        help="SequenceMatcher similarity threshold for flagging anchor overlap.")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    return list(header), rows


def write_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["id"]), str(row["dimension"])


def result_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["id"]), str(row["dimension"]), str(row["variant_id"])


def valid_integer_score(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    if re.fullmatch(r"[0-4]", text):
        return int(text)
    return None


def is_valid_score(row: dict[str, Any], variant_id: str) -> bool:
    error = str(row.get(f"llm_error_{variant_id}", "") or "").strip()
    if error:
        return False
    return valid_integer_score(row.get(f"llm_score_{variant_id}", "")) is not None


def load_llm_results(path: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, str]], list[dict[str, str]]]:
    """Pivot long LLM result rows into per-(id, dimension) columns."""
    _, rows = read_csv(path)
    if not rows:
        return [], {}, []

    variant_ids = sorted({row["variant_id"] for row in rows})

    # If duplicates exist, keep the last occurrence for each id + dimension + variant.
    deduped: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        deduped[result_key(row)] = row

    by_item: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in deduped.values():
        key = row_key(row)
        variant_id = row["variant_id"]
        by_item[key][f"llm_score_{variant_id}"] = row.get("llm_score", "")
        by_item[key][f"llm_error_{variant_id}"] = row.get("error", "")
        by_item[key][f"llm_attempts_{variant_id}"] = row.get("attempts", "")

    return variant_ids, by_item, list(deduped.values())


def merge(full_dataset_csv: Path, llm_results_csv: Path) -> tuple[list[str], list[dict[str, str]], list[str], list[dict[str, str]]]:
    full_header, full_rows = read_csv(full_dataset_csv)
    variant_ids, llm_by_item, long_results = load_llm_results(llm_results_csv)

    score_cols = [f"llm_score_{v}" for v in variant_ids]
    error_cols = [f"llm_error_{v}" for v in variant_ids]
    attempt_cols = [f"llm_attempts_{v}" for v in variant_ids]

    merged_rows: list[dict[str, str]] = []
    for row in full_rows:
        key = row_key(row)
        if key not in llm_by_item:
            continue
        merged = dict(row)
        merged.update(llm_by_item[key])
        merged_rows.append(merged)

    merged_header = full_header + score_cols + error_cols + attempt_cols
    return merged_header, merged_rows, variant_ids, long_results


def summarize_success_failure(merged_rows: list[dict[str, str]], variant_ids: list[str]) -> list[dict[str, Any]]:
    """Success/failure rows for ALL and for each dimension."""
    rows: list[dict[str, Any]] = []
    dimensions = ["ALL"] + sorted({row["dimension"] for row in merged_rows})

    for variant_id in variant_ids:
        for dim in dimensions:
            dim_rows = merged_rows if dim == "ALL" else [r for r in merged_rows if r["dimension"] == dim]
            total = len(dim_rows)
            successes = sum(1 for row in dim_rows if is_valid_score(row, variant_id))
            failures = total - successes
            rows.append({
                "variant_id": variant_id,
                "dimension": dim,
                "total": total,
                "successes": successes,
                "failures": failures,
                "success_rate": successes / total if total else float("nan"),
            })
    return rows


def common_evaluated_keys(merged_rows: list[dict[str, str]], variant_ids: list[str]) -> set[tuple[str, str]]:
    common = {row_key(row) for row in merged_rows}
    for variant_id in variant_ids:
        valid_keys = {row_key(row) for row in merged_rows if is_valid_score(row, variant_id)}
        common &= valid_keys
    return common


def pearson_r(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return float("nan")
    preds = [p for p, _ in pairs]
    actuals = [a for _, a in pairs]
    mean_pred = sum(preds) / len(preds)
    mean_actual = sum(actuals) / len(actuals)
    cov = sum((p - mean_pred) * (a - mean_actual) for p, a in pairs)
    std_pred = math.sqrt(sum((p - mean_pred) ** 2 for p in preds))
    std_actual = math.sqrt(sum((a - mean_actual) ** 2 for a in actuals))
    if std_pred == 0 or std_actual == 0:
        return float("nan")
    return cov / (std_pred * std_actual)


def spearman_r(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return float("nan")

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    pred_ranks = rank([p for p, _ in pairs])
    actual_ranks = rank([a for _, a in pairs])
    return pearson_r(list(zip(pred_ranks, actual_ranks)))


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(variance)


def compute_metrics(pairs: list[tuple[float, float]]) -> dict[str, float]:
    if not pairs:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "llm_mean": float("nan"),
            "llm_std": float("nan"),
            "human_mean": float("nan"),
            "human_std": float("nan"),
        }

    errors = [p - a for p, a in pairs]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    bias = sum(errors) / len(errors)

    actuals = [a for _, a in pairs]
    preds = [p for p, _ in pairs]
    mean_actual = sum(actuals) / len(actuals)
    ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
    ss_res = sum(e * e for e in errors)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    llm_mean, llm_std = mean_std(preds)
    human_mean, human_std = mean_std(actuals)

    return {
        "n": len(pairs),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "pearson_r": pearson_r(pairs),
        "spearman_r": spearman_r(pairs),
        "llm_mean": llm_mean,
        "llm_std": llm_std,
        "human_mean": human_mean,
        "human_std": human_std,
    }


def get_pairs(rows: list[dict[str, str]], variant_id: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        llm_score = valid_integer_score(row.get(f"llm_score_{variant_id}", ""))
        avg_str = str(row.get("average_score", "") or "").strip()
        if llm_score is None or not avg_str:
            continue
        try:
            human_avg = float(avg_str)
        except ValueError:
            continue
        pairs.append((float(llm_score), human_avg))
    return pairs


def analyze(merged_rows: list[dict[str, str]], variant_ids: list[str], common_keys: set[tuple[str, str]]) -> list[dict[str, Any]]:
    common_rows = [row for row in merged_rows if row_key(row) in common_keys]
    dimensions = sorted({row["dimension"] for row in common_rows})

    metric_rows: list[dict[str, Any]] = []
    for variant_id in variant_ids:
        overall_pairs = get_pairs(common_rows, variant_id)
        metric_rows.append({"variant_id": variant_id, "dimension": "ALL", **compute_metrics(overall_pairs)})

        for dim in dimensions:
            dim_rows = [row for row in common_rows if row["dimension"] == dim]
            metric_rows.append({"variant_id": variant_id, "dimension": dim, **compute_metrics(get_pairs(dim_rows, variant_id))})

    return metric_rows


def score_distribution(merged_rows: list[dict[str, str]], variant_ids: list[str], common_keys: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """Distribution diagnostics without rounding human average scores."""
    common_rows = [row for row in merged_rows if row_key(row) in common_keys]
    rows: list[dict[str, Any]] = []

    for variant_id in variant_ids:
        pairs = get_pairs(common_rows, variant_id)
        llm_counts = Counter(str(int(p)) for p, _ in pairs)
        human_counts = Counter(f"{a:.3f}" for _, a in pairs)

        for score in sorted(llm_counts, key=float):
            rows.append({
                "variant_id": variant_id,
                "score_source": "llm_integer_score",
                "score_value": score,
                "count": llm_counts[score],
            })
        for score in sorted(human_counts, key=float):
            rows.append({
                "variant_id": variant_id,
                "score_source": "human_average_score_exact_3dp",
                "score_value": score,
                "count": human_counts[score],
            })

    return rows


def normalize_for_similarity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def anchor_overlap_audit(
    merged_rows: list[dict[str, str]],
    text_col: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Flag assessed transcripts that are highly similar to prompt anchor examples."""
    if not DIMENSIONS:
        return []

    audit_rows: list[dict[str, Any]] = []
    for row in merged_rows:
        dim_code = row.get("dimension", "")
        spec = DIMENSIONS.get(dim_code)
        if spec is None:
            continue

        transcript = normalize_for_similarity(row.get(text_col, ""))
        if not transcript:
            continue

        best_score = ""
        best_anchor = ""
        best_similarity = -1.0
        best_contains = False

        for anchor_score, examples in spec.anchor_examples.items():
            for anchor in examples:
                norm_anchor = normalize_for_similarity(anchor)
                if not norm_anchor:
                    continue
                similarity = SequenceMatcher(None, transcript, norm_anchor).ratio()
                contains = norm_anchor in transcript or transcript in norm_anchor
                if similarity > best_similarity:
                    best_score = str(anchor_score)
                    best_anchor = anchor
                    best_similarity = similarity
                    best_contains = contains

        flagged = best_similarity >= threshold or best_contains
        audit_rows.append({
            "id": row.get("id", ""),
            "dimension": dim_code,
            "best_anchor_score": best_score,
            "best_similarity": best_similarity,
            "contains_or_contained": best_contains,
            "flagged": flagged,
            "transcript": row.get(text_col, ""),
            "best_matching_anchor": best_anchor,
        })

    return audit_rows


def main() -> None:
    args = parse_args()

    merged_csv = args.output_dir / "full_dataset_en_with_llm_scores.csv"
    summary_csv = args.output_dir / "llm_success_failure_summary.csv"
    metrics_csv = args.output_dir / "llm_accuracy_metrics.csv"
    distribution_csv = args.output_dir / "llm_score_distribution.csv"
    anchor_audit_csv = args.output_dir / "llm_anchor_overlap_audit.csv"

    merged_header, merged_rows, variant_ids, _ = merge(args.full_dataset_csv, args.llm_results_csv)
    write_csv(merged_csv, merged_header, merged_rows)
    print(f"Assessed transcript rows merged: {len(merged_rows)}")
    print(f"Saved merged dataset: {merged_csv}\n")

    summary_rows = summarize_success_failure(merged_rows, variant_ids)
    write_csv(summary_csv, ["variant_id", "dimension", "total", "successes", "failures", "success_rate"], summary_rows)
    print(f"Saved success/failure summary: {summary_csv}")

    print("\nSuccess/failure by prompt variant, overall:")
    print(f"{'variant_id':35s} {'total':>6s} {'success':>8s} {'failure':>8s} {'rate':>7s}")
    for row in summary_rows:
        if row["dimension"] != "ALL":
            continue
        print(
            f"{row['variant_id']:35s} {row['total']:6d} {row['successes']:8d} "
            f"{row['failures']:8d} {row['success_rate']:7.1%}"
        )

    all_keys = {row_key(row) for row in merged_rows}
    common_keys = common_evaluated_keys(merged_rows, variant_ids)
    print(f"\nItems with valid scores across all {len(variant_ids)} variants: {len(common_keys)} / {len(all_keys)}\n")

    metric_rows = analyze(merged_rows, variant_ids, common_keys)
    metric_header = [
        "variant_id", "dimension", "n", "mae", "rmse", "bias", "r2", "pearson_r", "spearman_r",
        "llm_mean", "llm_std", "human_mean", "human_std",
    ]
    write_csv(metrics_csv, metric_header, metric_rows)
    print(f"Saved accuracy metrics on common evaluated set: {metrics_csv}")

    print("\nOverall metrics per prompt variant, common evaluated set:")
    print(f"{'variant_id':35s} {'n':>4s} {'MAE':>6s} {'RMSE':>6s} {'Bias':>7s} {'R2':>7s} {'Pearson':>8s} {'Spearman':>9s}")
    for row in metric_rows:
        if row["dimension"] != "ALL":
            continue
        print(
            f"{row['variant_id']:35s} {row['n']:4d} {row['mae']:6.3f} {row['rmse']:6.3f} "
            f"{row['bias']:7.3f} {row['r2']:7.3f} {row['pearson_r']:8.3f} {row['spearman_r']:9.3f}"
        )

    dist_rows = score_distribution(merged_rows, variant_ids, common_keys)
    write_csv(distribution_csv, ["variant_id", "score_source", "score_value", "count"], dist_rows)
    print(f"\nSaved score distribution diagnostics: {distribution_csv}")

    audit_rows = anchor_overlap_audit(merged_rows, args.text_col, args.anchor_threshold)
    write_csv(
        anchor_audit_csv,
        ["id", "dimension", "best_anchor_score", "best_similarity", "contains_or_contained", "flagged", "transcript", "best_matching_anchor"],
        audit_rows,
    )
    flagged_count = sum(1 for row in audit_rows if row.get("flagged"))
    print(f"Saved anchor-overlap audit: {anchor_audit_csv}")
    print(f"Anchor-overlap flags: {flagged_count} / {len(audit_rows)}")


if __name__ == "__main__":
    main()
