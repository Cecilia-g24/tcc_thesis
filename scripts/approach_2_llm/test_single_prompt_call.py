"""
Send exactly one real (transcript + prompt-variant) pair to the API in a single call.

This is a debug/sanity-check tool, not a batch run: it picks one transcript row
from a German dataset CSV, builds the matching prompt with de_prompt_construction.build_prompt()
(the same renderer that produced the preview files under
data/assets/de_assets/de_prompts/<dimension_code>/<variant_id>.txt), sends it to the model
in one API call, and writes:
  - a one-row CSV with the variant's required output (score, and brief_rationale if the
    variant includes it)
  - a detailed JSON log with runtime, the full prompt, the full raw response, every retry
    attempt, and the parsed/validated result

Default condition: dimension d1_illness_beliefs, variant V1_full_manual_baseline,
first matching transcript in data/data_clean/01_csvs_for_liwc_manual_input/full_dataset_de.csv.

Run from the repo root or from this script directory:
    python test_single_prompt_call.py
    python test_single_prompt_call.py --transcript-id AU01.000847
    python test_single_prompt_call.py --dimension d2_lack_of_knowledge --variant V5_structured_checklist
    python test_single_prompt_call.py --model glm-4.7
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

from de_prompt_construction import DIMENSIONS, VARIANTS, build_prompt

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "data_clean" / "01_csvs_for_liwc_manual_input" / "full_dataset_de.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "approach2" / "single_call_test"
MODELS_CONFIG_PATH = REPO_ROOT / "configs" / "api_models.json"

DEFAULT_TEXT_COL = "text"
DEFAULT_DIMENSION = "d1_illness_beliefs"
DEFAULT_VARIANT = "V1_full_manual_baseline"
DEFAULT_MODEL = "qwen3-30b-a3b-instruct-2507"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024

# Maps a provider name (as used in configs/api_models.json) to its env var names.
PROVIDER_ENV = {
    "nhr": {"api_key": "NHR_API_KEY", "base_url": "NHR_BASE_URL"},
    "gwdg": {"api_key": "GWDG_API_KEY", "base_url": "GWDG_BASE_URL"},
}
DEFAULT_PROVIDER = "gwdg"

MAX_RETRIES = 8
BASE_RETRY_DELAY_S = 10
MAX_RETRY_DELAY_S = 300

CSV_COLUMNS = [
    "id",
    "dimension",
    "variant_id",
    "model",
    "average_score",
    "llm_score",
    "brief_rationale",
    "error",
    "attempts",
    "runtime_seconds",
    "timestamp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one real transcript + prompt-variant pair to the API.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL)
    parser.add_argument("--dimension", type=str, default=DEFAULT_DIMENSION, choices=sorted(DIMENSIONS))
    parser.add_argument("--variant", type=str, default=DEFAULT_VARIANT, choices=sorted(VARIANTS))
    parser.add_argument("--transcript-id", type=str, default=None,
                        help="Row id to use. Defaults to the first row in the CSV matching --dimension.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args()


def load_model_provider_map(config_path: Path = MODELS_CONFIG_PATH) -> dict[str, str]:
    """Return {model_id: provider_name} for every model listed in configs/api_models.json."""
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    model_provider: dict[str, str] = {}
    for provider_name, models in config.items():
        for model_id in models:
            if model_id.startswith("_"):
                continue
            model_provider[model_id] = provider_name
    return model_provider


def make_client(provider_name: str) -> OpenAI:
    env = PROVIDER_ENV.get(provider_name, PROVIDER_ENV[DEFAULT_PROVIDER])
    api_key = os.getenv(env["api_key"])
    base_url = os.getenv(env["base_url"])
    if not api_key:
        raise RuntimeError(f"Missing {env['api_key']} in environment or .env file.")
    if not base_url:
        raise RuntimeError(f"Missing {env['base_url']} in environment or .env file.")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)


def normalize_text_value(value: Any) -> str:
    """Convert a dataframe cell to safe prompt text."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def select_transcript_row(df: pd.DataFrame, dimension: str, transcript_id: str | None) -> pd.Series:
    """Pick the transcript row to send: a specific id, or the first row matching dimension."""
    if transcript_id is not None:
        matches = df[df["id"].astype(str) == str(transcript_id)]
        if matches.empty:
            raise ValueError(f"No row with id {transcript_id!r} found in the input CSV.")
        row = matches.iloc[0]
        if str(row["dimension"]) != dimension:
            raise ValueError(
                f"Row {transcript_id!r} has dimension {row['dimension']!r}, "
                f"which does not match --dimension {dimension!r}. The prompt is dimension-specific, "
                "so the transcript's dimension must match."
            )
        return row

    matches = df[df["dimension"] == dimension].sort_values("id")
    if matches.empty:
        raise ValueError(f"No rows with dimension {dimension!r} found in the input CSV.")
    return matches.iloc[0]


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
    """Send one prompt to the model and return raw response, parse status, and a per-attempt log."""
    result: dict[str, Any] = {
        "raw_response": None,
        "parsed_json": None,
        "llm_score": None,
        "error": None,
        "attempts": 0,
        "attempt_log": [],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        result["attempts"] = attempt
        attempt_started_at = dt.datetime.now().isoformat(timespec="seconds")
        attempt_t0 = time.perf_counter()
        attempt_record: dict[str, Any] = {
            "attempt": attempt,
            "started_at": attempt_started_at,
            "duration_seconds": None,
            "error": None,
            "retry_delay_seconds": None,
        }
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            attempt_record["duration_seconds"] = round(time.perf_counter() - attempt_t0, 3)
            result["attempt_log"].append(attempt_record)

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
            attempt_record["duration_seconds"] = round(time.perf_counter() - attempt_t0, 3)
            attempt_record["error"] = error
            result["error"] = error
            if attempt < MAX_RETRIES:
                delay = backoff_delay(attempt, is_rate_limit_error(error))
                attempt_record["retry_delay_seconds"] = round(delay, 1)
                result["attempt_log"].append(attempt_record)
                time.sleep(delay)
            else:
                result["attempt_log"].append(attempt_record)

    return result


def sanitize_for_filename(name: str) -> str:
    """Provider-qualified model ids (e.g. ibm-granite/granite-4.1-3b) can't be used as-is in a filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name)


def run_single_call(args: argparse.Namespace) -> tuple[Path, Path]:
    provider_name = load_model_provider_map().get(args.model, DEFAULT_PROVIDER)
    client = make_client(provider_name)

    df = pd.read_csv(args.input_csv)
    required_cols = {"id", "dimension", args.text_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    row = select_transcript_row(df, args.dimension, args.transcript_id)
    transcript = normalize_text_value(row[args.text_col])

    prompt = build_prompt(
        dimension_code=args.dimension,
        transcript=transcript,
        variant_id=args.variant,
    )

    run_started_at = dt.datetime.now()
    t0 = time.perf_counter()
    result = call_model(
        client=client,
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    runtime_seconds = round(time.perf_counter() - t0, 3)
    run_finished_at = dt.datetime.now()

    variant_spec = VARIANTS[args.variant]
    parsed_json = result.get("parsed_json") or {}
    brief_rationale = parsed_json.get("brief_rationale") if variant_spec.include_rationale else ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"{args.dimension}_{args.variant}_{sanitize_for_filename(args.model)}_"
        f"{run_started_at.strftime('%Y%m%d_%H%M%S')}"
    )
    csv_path = args.output_dir / f"single_call_result_{run_id}.csv"
    log_path = args.output_dir / f"single_call_log_{run_id}.json"

    csv_row = {
        "id": str(row["id"]),
        "dimension": args.dimension,
        "variant_id": args.variant,
        "model": args.model,
        "provider": provider_name,
        "llm_score": "" if result.get("llm_score") is None else result.get("llm_score"),
        "brief_rationale": "" if brief_rationale is None else brief_rationale,
        "average_score": row.get("average_score", ""),
        "error": "" if result.get("error") is None else result.get("error"),
        "attempts": result.get("attempts"),
        "runtime_seconds": runtime_seconds,
        "timestamp": run_finished_at.isoformat(timespec="seconds"),
    }
    pd.DataFrame([csv_row], columns=CSV_COLUMNS).to_csv(csv_path, index=False)

    log_record = {
        "id": str(row["id"]),
        "dimension": args.dimension,
        "variant_id": args.variant,
        "model": args.model,
        "provider": provider_name,
        "input_csv": str(args.input_csv),
        "text_col": args.text_col,
        "request_params": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "started_at": run_started_at.isoformat(timespec="seconds"),
        "finished_at": run_finished_at.isoformat(timespec="seconds"),
        "runtime_seconds": runtime_seconds,
        "attempts": result.get("attempts"),
        "attempt_log": result.get("attempt_log"),
        "prompt": prompt,
        "raw_response": result.get("raw_response"),
        "parsed_json": result.get("parsed_json"),
        "validated_score": result.get("llm_score"),
        "average_score": row.get("average_score", ""),
        "error": result.get("error"),
    }
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log_record, f, ensure_ascii=False, indent=2)

    return csv_path, log_path


if __name__ == "__main__":
    output_csv, output_log = run_single_call(parse_args())
    print(f"Saved result: {output_csv}")
    print(f"Saved log:    {output_log}")
