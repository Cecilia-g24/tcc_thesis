"""
Run LLM-as-judge assessment on transcripts using prompts built by prompt_construction.py.

This version is designed for resumable, auditable development runs:
- random fixed-seed dev sampling instead of taking the first rows
- id + dimension + variant_id as the unique call key
- immediate result saving after every call
- resume support: successful calls are skipped on rerun
- failed/missing calls are retried by default
- exponential backoff for rate limits and transient API errors
- strict parsing/validation of integer scores in {0, 1, 2, 3, 4}

Default condition:
    English transcript + English prompt

Run from the repo root or from this script directory:
    python llm_assessment.py

Useful options:
    python llm_assessment.py --n-per-dimension 20 --sample-mode random --random-state 42
    python llm_assessment.py --n-per-dimension 0     # run all rows
    python llm_assessment.py --overwrite            # ignore previous results
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from prompt_construction import PROMPT_TEMPLATES, build_prompt

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "data_clean" / "01_csvs_for_liwc_manual_input" / "full_dataset_en.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "approach2" / "en_text_en_prompt_results"

DEFAULT_TEXT_COL = "text_en"
DEFAULT_N_PER_DIMENSION = 20
DEFAULT_SAMPLE_MODE = "random"  # random | first | all
DEFAULT_RANDOM_STATE = 42

DEFAULT_MODEL = "qwen3-30b-a3b-instruct-2507"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024

MAX_RETRIES = 8
BASE_RETRY_DELAY_S = 10
MAX_RETRY_DELAY_S = 300
STOP_AFTER_CONSECUTIVE_RATE_LIMIT_FAILURES = 8

RESULT_COLUMNS = [
    "id",
    "dimension",
    "variant_id",
    "model",
    "llm_score",
    "raw_response",
    "error",
    "attempts",
    "average_score",
    "timestamp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable LLM-as-judge assessment.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL)
    parser.add_argument("--n-per-dimension", type=int, default=DEFAULT_N_PER_DIMENSION,
                        help="Rows per dimension for dev runs. Use 0 or a negative value for all rows.")
    parser.add_argument("--sample-mode", choices=["random", "first", "all"], default=DEFAULT_SAMPLE_MODE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing result files and start a fresh run.")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Do not retry existing failed/unparseable calls. By default, failed calls are retried.")
    return parser.parse_args()


def normalize_text_value(value: Any) -> str:
    """Convert a dataframe cell to safe prompt text."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def load_subset(input_csv: Path, n_per_dimension: int, sample_mode: str, random_state: int) -> pd.DataFrame:
    """Load the dataset and return either all rows or a fixed-size per-dimension dev subset."""
    df = pd.read_csv(input_csv)

    required_cols = {"id", "dimension"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    if sample_mode == "all" or n_per_dimension <= 0:
        return df.reset_index(drop=True)

    if sample_mode == "first":
        return df.groupby("dimension", group_keys=False).head(n_per_dimension).reset_index(drop=True)

    # Fixed-seed random sample. If a dimension has fewer rows than requested, keep all rows.
    return (
        df.groupby("dimension", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_dimension, len(g)), random_state=random_state))
        .reset_index(drop=True)
    )


def result_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (str(record["id"]), str(record["dimension"]), str(record["variant_id"]))


def valid_integer_score(value: Any) -> int | None:
    """Return an int score if value is exactly 0, 1, 2, 3, or 4; otherwise None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in {0, 1, 2, 3, 4} else None
    if isinstance(value, float):
        if value.is_integer() and int(value) in {0, 1, 2, 3, 4}:
            return int(value)
        return None

    text = str(value).strip()
    if re.fullmatch(r"[0-4]", text):
        return int(text)
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from the model response, tolerating markdown fences."""
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Last-resort extraction if the provider wraps the JSON in extra text.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    return None


def parse_and_validate_response(raw_response: str | None) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """Return parsed JSON, validated integer score, and parse/validation error."""
    if not raw_response:
        return None, None, "Empty response"

    parsed = extract_json_object(raw_response)
    if parsed is None:
        return None, None, "JSON parse error: response did not contain a valid JSON object"

    score = valid_integer_score(parsed.get("score"))
    if score is None:
        return parsed, None, "Invalid score: expected integer 0, 1, 2, 3, or 4"

    return parsed, score, None


def is_rate_limit_error(error_text: str | None) -> bool:
    if not error_text:
        return False
    lowered = error_text.lower()
    return "429" in lowered or "rate limit" in lowered or "too many requests" in lowered


def backoff_delay(attempt: int, is_rate_limit: bool) -> float:
    """Exponential backoff with jitter; rate limits start with a longer delay."""
    base = BASE_RETRY_DELAY_S * (2 if is_rate_limit else 1)
    delay = min(MAX_RETRY_DELAY_S, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, min(5, delay * 0.1))
    return delay + jitter


def call_model(client: OpenAI, prompt: str, model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    """Send one prompt to the model and return raw response plus parse/validation status."""
    result: dict[str, Any] = {
        "raw_response": None,
        "parsed_json": None,
        "llm_score": None,
        "error": None,
        "attempts": 0,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        result["attempts"] = attempt
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            raw_response = content.strip()
            parsed, score, parse_error = parse_and_validate_response(raw_response)
            result.update({
                "raw_response": raw_response,
                "parsed_json": parsed,
                "llm_score": score,
                "error": parse_error,
            })
            return result

        except Exception as exc:  # provider-specific exceptions vary, so keep broad handling here
            error = str(exc)
            result["error"] = error
            if attempt < MAX_RETRIES:
                time.sleep(backoff_delay(attempt, is_rate_limit_error(error)))

    return result


def load_existing_results(csv_path: Path, overwrite: bool) -> list[dict[str, Any]]:
    if overwrite or not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    records = df.to_dict(orient="records")
    # Keep the last occurrence of duplicated keys, if any.
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rec in records:
        by_key[result_key(rec)] = rec
    return list(by_key.values())


def is_successful_record(record: dict[str, Any]) -> bool:
    if str(record.get("error", "")).strip():
        return False
    return valid_integer_score(record.get("llm_score")) is not None


def write_results(csv_path: Path, jsonl_path: Path, records: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for rec in records:
        normalized.append({col: rec.get(col, "") for col in RESULT_COLUMNS})

    tmp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    pd.DataFrame(normalized, columns=RESULT_COLUMNS).to_csv(tmp_csv, index=False)
    tmp_csv.replace(csv_path)

    tmp_jsonl = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with tmp_jsonl.open("w", encoding="utf-8") as f:
        for rec in normalized:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_jsonl.replace(jsonl_path)


def should_skip_existing(record: dict[str, Any], keep_failed: bool) -> bool:
    if is_successful_record(record):
        return True
    return keep_failed


def run_assessment(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    api_key = os.getenv("GWDG_API_KEY")
    base_url = os.getenv("GWDG_BASE_URL")
    if not api_key:
        raise RuntimeError("Missing GWDG_API_KEY in environment or .env file.")
    if not base_url:
        raise RuntimeError("Missing GWDG_BASE_URL in environment or .env file.")

    if args.text_col not in pd.read_csv(args.input_csv, nrows=1).columns:
        raise ValueError(f"Text column {args.text_col!r} was not found in {args.input_csv}")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    df = load_subset(args.input_csv, args.n_per_dimension, args.sample_mode, args.random_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"dev_results_{args.model}.csv"
    jsonl_path = args.output_dir / f"dev_results_{args.model}.jsonl"
    log_path = args.output_dir / f"dev_interaction_log_{args.model}.jsonl"

    calls: list[tuple[pd.Series, str]] = [
        (row, variant_id)
        for _, row in df.iterrows()
        for variant_id in PROMPT_TEMPLATES
    ]
    planned_keys = {
        (str(row["id"]), str(row["dimension"]), str(variant_id))
        for row, variant_id in calls
    }

    # Load previous results, but keep only rows that belong to the currently planned
    # sample. This prevents accidentally mixing an old first-20 dev run with a new
    # random dev sample under the same output filename.
    records = [
        rec for rec in load_existing_results(csv_path, overwrite=args.overwrite)
        if result_key(rec) in planned_keys
    ]
    records_by_key = {result_key(rec): rec for rec in records}

    pending_calls = []
    for row, variant_id in calls:
        key = (str(row["id"]), str(row["dimension"]), str(variant_id))
        existing = records_by_key.get(key)
        if existing and should_skip_existing(existing, keep_failed=args.keep_failed):
            continue
        pending_calls.append((row, variant_id))

    print(f"Total planned calls: {len(calls)}")
    print(f"Existing result rows for this planned sample: {len(records)}")
    print(f"Pending calls to run: {len(pending_calls)}")

    consecutive_rate_limit_failures = 0

    with log_path.open("a", encoding="utf-8") as log_f:
        for row, variant_id in tqdm(pending_calls, desc="LLM assessment", unit="call"):
            transcript = normalize_text_value(row[args.text_col])
            prompt = build_prompt(
                dimension_code=str(row["dimension"]),
                transcript=transcript,
                variant_id=variant_id,
            )

            result = call_model(
                client=client,
                prompt=prompt,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            timestamp = dt.datetime.now().isoformat(timespec="seconds")
            error = result.get("error")
            if is_rate_limit_error(error):
                consecutive_rate_limit_failures += 1
            elif error is None:
                consecutive_rate_limit_failures = 0

            record = {
                "id": str(row["id"]),
                "dimension": str(row["dimension"]),
                "variant_id": str(variant_id),
                "model": args.model,
                "llm_score": "" if result.get("llm_score") is None else result.get("llm_score"),
                "raw_response": result.get("raw_response"),
                "error": "" if error is None else error,
                "attempts": result.get("attempts"),
                "average_score": row.get("average_score", ""),
                "timestamp": timestamp,
            }

            key = result_key(record)
            records_by_key[key] = record
            records = list(records_by_key.values())
            write_results(csv_path, jsonl_path, records)

            log_f.write(json.dumps({
                "timestamp": timestamp,
                "id": record["id"],
                "dimension": record["dimension"],
                "variant_id": record["variant_id"],
                "model": args.model,
                "attempts": record["attempts"],
                "prompt": prompt,
                "raw_response": result.get("raw_response"),
                "parsed_json": result.get("parsed_json"),
                "validated_score": result.get("llm_score"),
                "error": error,
            }, ensure_ascii=False) + "\n")
            log_f.flush()

            if consecutive_rate_limit_failures >= STOP_AFTER_CONSECUTIVE_RATE_LIMIT_FAILURES:
                print(
                    "\nStopping early because too many consecutive calls ended with rate-limit errors. "
                    "Rerun the script later; successful calls will be skipped automatically."
                )
                break

    write_results(csv_path, jsonl_path, list(records_by_key.values()))
    return csv_path, jsonl_path, log_path


if __name__ == "__main__":
    output_csv, output_jsonl, output_log = run_assessment(parse_args())
    print(f"\nSaved results:         {output_csv}")
    print(f"Saved results:         {output_jsonl}")
    print(f"Saved interaction log: {output_log}")
