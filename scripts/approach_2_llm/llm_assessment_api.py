"""
Run LLM-as-judge assessment on transcripts using prompts built by de_prompt_construction.py.

This script mirrors llm_assessment_local.py, but replaces local generation with an
OpenAI-compatible API call. It runs one model per invocation, across every transcript
and every prompt variant (no dev subsampling).

Designed for resumable, auditable runs:
- id + dimension + variant_id as the unique call key
- immediate result saving after every call
- resume support: successful calls are skipped on rerun
- failed/missing calls are retried by default
- exponential backoff for rate limits and transient API errors
- responses are parsed and validated against integer scores in {0, 1, 2, 3, 4}
- explicit, neutral decoding parameters and a default generation seed of 42
- full interaction log (prompt, raw response, retry attempts, inference settings)

Each --model gets its own subfolder under --output-dir (results.csv, results.jsonl,
interaction_log.jsonl), so multiple API models can be run into the same output-dir
without colliding.

Default condition:
    German transcript + German prompt, all rows, all 5 dimensions, all prompt variants

Run from the repo root or from this script directory:
    python llm_assessment_api.py

Useful options:
    python llm_assessment_api.py --model glm-4.7                       # run just this one model
    python llm_assessment_api.py -v V1_full_manual_baseline            # run just this one variant
    python llm_assessment_api.py --overwrite                           # ignore previous results
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import random
import re
import time
from pathlib import Path
from typing import Any

from importlib import metadata as importlib_metadata

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from de_prompt_construction import VARIANTS, build_prompt
from results_schema import CORE_RESULT_COLUMNS

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "data_clean" / "01_csvs_for_liwc_manual_input" / "full_dataset_de.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "approach2" / "de_text_de_prompt_results" / "api_results"
MODELS_CONFIG_PATH = REPO_ROOT / "configs" / "api_models.json"

# Maps a provider name (as used in configs/api_models.json) to its env var names.
PROVIDER_ENV = {
    "nhr": {"api_key": "NHR_API_KEY", "base_url": "NHR_BASE_URL"},
    "gwdg": {"api_key": "GWDG_API_KEY", "base_url": "GWDG_BASE_URL"},
}
DEFAULT_PROVIDER = "gwdg"

DEFAULT_TEXT_COL = "text"

DEFAULT_MODEL = "qwen3-30b-a3b-instruct-2507"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_N = 1
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_GENERATION_SEED = 42

MAX_RETRIES = 8
BASE_RETRY_DELAY_S = 10
MAX_RETRY_DELAY_S = 300
STOP_AFTER_CONSECUTIVE_RATE_LIMIT_FAILURES = 8

# Core columns (shared name/semantics with llm_assessment_local.py's RESULT_COLUMNS) plus
# this script's own API-specific columns, appended after.
RESULT_COLUMNS = CORE_RESULT_COLUMNS + [
    "provider",
    "n",
    "presence_penalty",
    "frequency_penalty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable LLM-as-judge assessment.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL)
    parser.add_argument(
        "-v", "--variants",
        type=str,
        default="all",
        help="Comma-separated list of prompt variant ids to run (e.g. V1_full_manual_baseline), "
             f"or 'all' (default) to run every variant. Choices: {sorted(VARIANTS)}.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Run only this model.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Requested API temperature. The default 0.0 requests deterministic/greedy-like decoding, "
             "but exact behavior remains provider-dependent.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus-sampling cutoff. Kept at the neutral value 1.0 for temperature-zero benchmark runs.",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_GENERATION_SEED,
        help="Generation seed sent to the provider when supported (default: 42).",
    )
    parser.add_argument(
        "--no-seed",
        dest="seed",
        action="store_const",
        const=None,
        help="Do not send a seed, for providers or models that reject the seed parameter.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=["provider-default", "disabled", "enabled"],
        default="disabled",
        help="Thinking/reasoning control. 'disabled' (default) or 'enabled' sends the common "
             "vLLM-compatible chat_template_kwargs.enable_thinking flag, matching "
             "llm_assessment_local.py's --disable-thinking default; use provider-default if the "
             "model/provider rejects that field.",
    )
    parser.add_argument(
        "--extra-body-json",
        type=str,
        default=None,
        help="Optional JSON object merged into the OpenAI-compatible request's extra_body. "
             "Use this for provider/model-specific controls.",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing result files and start a fresh run.")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Do not retry existing failed/unparseable calls. By default, failed calls are retried.")

    args = parser.parse_args()

    if args.variants.strip().lower() == "all":
        args.variants = sorted(VARIANTS)
    else:
        requested = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in requested if v not in VARIANTS]
        if unknown:
            parser.error(f"Unknown --variants {unknown}; choices are {sorted(VARIANTS)}.")
        args.variants = requested

    if args.temperature < 0:
        parser.error("--temperature must be >= 0.")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the interval (0, 1].")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive.")

    # top_p has no useful role in a temperature-zero benchmark. Normalizing it makes the
    # intended decoding configuration explicit and comparable across local/API scripts.
    if args.temperature == 0.0 and args.top_p != 1.0:
        print("temperature=0: overriding top_p to the neutral value 1.0.")
        args.top_p = 1.0

    if args.extra_body_json:
        try:
            args.extra_body = json.loads(args.extra_body_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--extra-body-json is not valid JSON: {exc}")
        if not isinstance(args.extra_body, dict):
            parser.error("--extra-body-json must decode to a JSON object.")
    else:
        args.extra_body = {}

    return args


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


def package_version(package_name: str) -> str:
    """Return an installed package version without making the run fail if unavailable."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def build_api_request_parameters(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """Build explicit neutral decoding parameters plus optional provider-specific controls."""
    request: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": DEFAULT_N,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
    }
    if args.seed is not None:
        request["seed"] = args.seed

    extra_body = dict(args.extra_body)
    if args.thinking_mode != "provider-default":
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
        chat_template_kwargs["enable_thinking"] = args.thinking_mode == "enabled"
        extra_body["chat_template_kwargs"] = chat_template_kwargs

    if extra_body:
        request["extra_body"] = extra_body

    thinking_label = (
        "provider_default_no_explicit_control"
        if args.thinking_mode == "provider-default"
        else f"{args.thinking_mode}_requested"
    )
    return request, thinking_label


def api_runtime_metadata() -> dict[str, str]:
    return {
        "backend": "openai_compatible_rest",
        "openai_sdk_version": package_version("openai"),
        "python_version": platform.python_version(),
    }


def normalize_text_value(value: Any) -> str:
    """Convert a dataframe cell to safe prompt text."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def load_subset(input_csv: Path) -> pd.DataFrame:
    """Load the full dataset (every row, every dimension)."""
    df = pd.read_csv(input_csv)

    required_cols = {"id", "dimension"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    return df.reset_index(drop=True)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "api_model"


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

    # Last-resort extraction if the provider wraps the JSON in extra text, or -- as small
    # local models sometimes do -- keeps decoding past the first object and repeats the
    # template several times (e.g. "{...}\n{...}\n{...}"). Parse only the first complete
    # JSON object starting at the first "{" and ignore anything that follows it, rather
    # than joining first-"{"..last-"}" into one (invalid) blob.
    start = cleaned.find("{")
    if start == -1:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned, start)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
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


def call_model(client: OpenAI, prompt: str, model: str, request_parameters: dict[str, Any]) -> dict[str, Any]:
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
            # Every retry uses the identical model, prompt, and inference parameters.
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **request_parameters,
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
                tqdm.write(f"[{model}] attempt {attempt}/{MAX_RETRIES} failed: {error} — retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                result["attempt_log"].append(attempt_record)
                tqdm.write(f"[{model}] attempt {attempt}/{MAX_RETRIES} failed: {error} — giving up on this call")

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


def resolve_output_paths(args: argparse.Namespace, model: str) -> tuple[Path, Path, Path]:
    safe_model_name = sanitize_filename(model)
    model_dir = args.output_dir / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    csv_path = model_dir / "results.csv"
    jsonl_path = model_dir / "results.jsonl"
    log_path = model_dir / "interaction_log.jsonl"
    return csv_path, jsonl_path, log_path


def prepare_pending_calls(
    df: pd.DataFrame,
    args: argparse.Namespace,
    csv_path: Path,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[tuple[pd.Series, str]]]:
    """Plan all (row, variant) calls, load resumable prior results, and return what's left to run."""
    calls: list[tuple[pd.Series, str]] = [(row, variant_id) for _, row in df.iterrows() for variant_id in args.variants]
    planned_keys = {(str(row["id"]), str(row["dimension"]), str(variant_id)) for row, variant_id in calls}

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

    return records_by_key, pending_calls


def run_assessment(args: argparse.Namespace, model: str) -> tuple[Path, Path, Path]:
    provider_name = load_model_provider_map().get(model, DEFAULT_PROVIDER)
    client = make_client(provider_name)
    request_parameters, thinking_mode = build_api_request_parameters(args)
    runtime_metadata = api_runtime_metadata()

    if args.text_col not in pd.read_csv(args.input_csv, nrows=1).columns:
        raise ValueError(f"Text column {args.text_col!r} was not found in {args.input_csv}")

    df = load_subset(args.input_csv)
    csv_path, jsonl_path, log_path = resolve_output_paths(args, model)
    records_by_key, pending_calls = prepare_pending_calls(df, args, csv_path)

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
                model=model,
                request_parameters=request_parameters,
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
                "model": model,
                "provider": provider_name,
                "backend": runtime_metadata["backend"],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_output_tokens": args.max_tokens,
                "n": DEFAULT_N,
                "presence_penalty": DEFAULT_PRESENCE_PENALTY,
                "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
                "seed": "" if args.seed is None else args.seed,
                "decoding_mode": "temperature_zero_requested" if args.temperature == 0.0 else "sampling_requested",
                "thinking_mode": thinking_mode,
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
                "model": model,
                "provider": provider_name,
                "runtime_metadata": runtime_metadata,
                "inference_parameters": request_parameters,
                "decoding_mode": record["decoding_mode"],
                "thinking_mode": thinking_mode,
                "attempts": record["attempts"],
                "attempt_log": result.get("attempt_log"),
                "prompt": prompt,
                "raw_response": result.get("raw_response"),
                "parsed_json": result.get("parsed_json"),
                "validated_score": result.get("llm_score"),
                "average_score": record["average_score"],
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


def main() -> None:
    args = parse_args()

    output_csv, output_jsonl, output_log = run_assessment(args, args.model)
    print(f"\nSaved results:         {output_csv}")
    print(f"Saved results:         {output_jsonl}")
    print(f"Saved interaction log: {output_log}")


if __name__ == "__main__":
    main()
