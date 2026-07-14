"""
Run LLM-as-judge assessment on transcripts using a locally downloaded Hugging Face model.

This script mirrors llm_assessment.py, but replaces the OpenAI/GWDG API call with local
Transformers generation.

Designed for resumable, auditable development runs:
- fixed-seed dev sampling or full-run mode
- id + dimension + variant_id as the unique call key
- immediate result saving after every call
- resume support: successful calls are skipped on rerun
- failed/missing calls are retried by default on rerun
- strict parsing/validation of integer scores in {0, 1, 2, 3, 4}

Default condition:
    English transcript + English prompt

Run from the repo root or from this script directory:
    python scripts/approach_2_llm/llm_assessment_local.py

Useful options:
    python scripts/approach_2_llm/llm_assessment_local.py --n-per-dimension 20 --sample-mode random --random-state 42
    python scripts/approach_2_llm/llm_assessment_local.py --n-per-dimension 0     # run all rows
    python scripts/approach_2_llm/llm_assessment_local.py --model-path ../models/qwen3_30b_instruct
    python scripts/approach_2_llm/llm_assessment_local.py --overwrite            # ignore previous results
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # Works when this file is placed next to prompt_construction.py.
    from prompt_construction import VARIANTS, build_prompt
except ModuleNotFoundError:
    # Works when running from unusual working directories.
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from prompt_construction import VARIANTS, build_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "data_clean" / "01_csvs_for_liwc_manual_input" / "full_dataset_en.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "approach2" / "en_text_en_prompt_results_local"

# In the VS Code server view, models/ and tcc/ appear to be sibling folders.
# Override this with --model-path if your model is elsewhere.
DEFAULT_MODEL_PATH = REPO_ROOT.parent / "models" / "qwen3_30b_instruct"
DEFAULT_MODEL_ALIAS = "qwen3_30b_instruct_local"

DEFAULT_TEXT_COL = "text_en"
DEFAULT_N_PER_DIMENSION = 20
DEFAULT_SAMPLE_MODE = "random"  # random | first | all
DEFAULT_RANDOM_STATE = 42

DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_NEW_TOKENS = 1024

MAX_RETRIES = 3
BASE_RETRY_DELAY_S = 10
MAX_RETRY_DELAY_S = 120
STOP_AFTER_CONSECUTIVE_RUNTIME_FAILURES = 3

RESULT_COLUMNS = [
    "id",
    "dimension",
    "variant_id",
    "model",
    "model_path",
    "llm_score",
    "raw_response",
    "error",
    "attempts",
    "average_score",
    "timestamp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable local LLM-as-judge assessment.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL)
    parser.add_argument(
        "--n-per-dimension",
        type=int,
        default=DEFAULT_N_PER_DIMENSION,
        help="Rows per dimension for dev runs. Use 0 or a negative value for all rows.",
    )
    parser.add_argument("--sample-mode", choices=["random", "first", "all"], default=DEFAULT_SAMPLE_MODE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv("LOCAL_MODEL_PATH", DEFAULT_MODEL_PATH)),
        help="Path to the local Hugging Face model directory.",
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default=os.getenv("LOCAL_MODEL_ALIAS", DEFAULT_MODEL_ALIAS),
        help="Short name used in output filenames and result rows.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype. bfloat16 is often best on supported GPUs; float16 is safer on V100.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Transformers device_map. Use 'auto' for GPU dispatch. Use 'cpu' only for tiny tests.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", dest="local_files_only", action="store_false")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit quantization. Requires bitsandbytes and a compatible GPU setup.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Use bitsandbytes 8-bit quantization. Requires bitsandbytes and a compatible GPU setup.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional: e.g., flash_attention_2, sdpa, or eager. Leave unset if unsure.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=True,
        help="For Qwen3-style chat templates, request non-thinking mode where supported.",
    )
    parser.add_argument("--allow-thinking", dest="disable_thinking", action="store_false")
    parser.add_argument(
        "--empty-cache-every-call",
        action="store_true",
        help="Call torch.cuda.empty_cache() after every generation. Slower, but can help fragmented VRAM.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing result files and start a fresh run.")
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Do not retry existing failed/unparseable calls. By default, failed calls are retried.",
    )
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
    """Parse a JSON object from the model response, tolerating markdown fences and wrapper text."""
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

    # Last-resort extraction if the model wraps the JSON in extra text.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
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


def is_runtime_error(error_text: str | None) -> bool:
    if not error_text:
        return False
    lowered = error_text.lower()
    runtime_markers = [
        "cuda out of memory",
        "outofmemoryerror",
        "device-side assert",
        "cublas",
        "cudnn",
        "nccl",
        "runtimeerror",
    ]
    return any(marker in lowered for marker in runtime_markers)


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter for transient local generation failures."""
    delay = min(MAX_RETRY_DELAY_S, BASE_RETRY_DELAY_S * (2 ** (attempt - 1)))
    jitter = random.uniform(0, min(5, delay * 0.1))
    return delay + jitter


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "local_model"


def resolve_torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def build_quantization_config(args: argparse.Namespace) -> Any | None:
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load-in-4bit or --load-in-8bit.")

    if not args.load_in_4bit and not args.load_in_8bit:
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes quantization was requested, but BitsAndBytesConfig could not be imported. "
            "Install a compatible transformers/bitsandbytes setup, or run without quantization."
        ) from exc

    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    return BitsAndBytesConfig(load_in_8bit=True)


def load_local_model(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load tokenizer and local causal language model."""
    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {model_path}\n"
            "Pass the correct directory with --model-path, for example: --model-path ../models/qwen3_30b_instruct"
        )

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    quantization_config = build_quantization_config(args)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = resolve_torch_dtype(args.dtype)

    if args.device_map and args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def render_chat_prompt(tokenizer: Any, prompt: str, disable_thinking: bool) -> str:
    """Apply chat template where available; otherwise fall back to the raw prompt."""
    messages = [{"role": "user", "content": prompt}]

    if getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if disable_thinking:
            # Qwen3 chat templates support this; non-Qwen templates usually ignore/accept kwargs,
            # but some template implementations may reject it, so we retry below without it.
            kwargs["enable_thinking"] = False
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)

    return prompt


def model_input_device(model: Any) -> torch.device:
    """Choose a safe input device for normal and accelerate-dispatched models."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_local_response(
    tokenizer: Any,
    model: Any,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    disable_thinking: bool,
) -> str:
    """Generate one local model response."""
    rendered_prompt = render_chat_prompt(tokenizer, prompt, disable_thinking=disable_thinking)
    inputs = tokenizer(rendered_prompt, return_tensors="pt")

    target_device = model_input_device(model)
    inputs = {key: value.to(target_device) for key, value in inputs.items()}

    do_sample = temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def call_model(
    tokenizer: Any,
    model: Any,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    disable_thinking: bool,
    empty_cache_every_call: bool,
) -> dict[str, Any]:
    """Send one prompt to the local model and return raw response plus parse/validation status."""
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
            raw_response = generate_local_response(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                disable_thinking=disable_thinking,
            )
            parsed, score, parse_error = parse_and_validate_response(raw_response)
            result.update(
                {
                    "raw_response": raw_response,
                    "parsed_json": parsed,
                    "llm_score": score,
                    "error": parse_error,
                }
            )
            return result

        except Exception as exc:  # local CUDA/Transformers errors vary by setup
            error = str(exc)
            result["error"] = error

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if attempt < MAX_RETRIES:
                time.sleep(backoff_delay(attempt))

    if empty_cache_every_call and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
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
    if args.text_col not in pd.read_csv(args.input_csv, nrows=1).columns:
        raise ValueError(f"Text column {args.text_col!r} was not found in {args.input_csv}")

    df = load_subset(args.input_csv, args.n_per_dimension, args.sample_mode, args.random_state)
    tokenizer, model = load_local_model(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = sanitize_filename(args.model_alias)
    csv_path = args.output_dir / f"dev_results_{safe_model_name}.csv"
    jsonl_path = args.output_dir / f"dev_results_{safe_model_name}.jsonl"
    log_path = args.output_dir / f"dev_interaction_log_{safe_model_name}.jsonl"

    calls: list[tuple[pd.Series, str]] = [
        (row, variant_id)
        for _, row in df.iterrows()
        for variant_id in VARIANTS
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

    consecutive_runtime_failures = 0
    resolved_model_path = str(args.model_path.expanduser().resolve())

    with log_path.open("a", encoding="utf-8") as log_f:
        for row, variant_id in tqdm(pending_calls, desc="Local LLM assessment", unit="call"):
            transcript = normalize_text_value(row[args.text_col])
            prompt = build_prompt(
                dimension_code=str(row["dimension"]),
                transcript=transcript,
                variant_id=variant_id,
            )

            result = call_model(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                disable_thinking=args.disable_thinking,
                empty_cache_every_call=args.empty_cache_every_call,
            )

            timestamp = dt.datetime.now().isoformat(timespec="seconds")
            error = result.get("error")
            if is_runtime_error(error):
                consecutive_runtime_failures += 1
            elif error is None:
                consecutive_runtime_failures = 0

            record = {
                "id": str(row["id"]),
                "dimension": str(row["dimension"]),
                "variant_id": str(variant_id),
                "model": args.model_alias,
                "model_path": resolved_model_path,
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

            log_f.write(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "id": record["id"],
                        "dimension": record["dimension"],
                        "variant_id": record["variant_id"],
                        "model": args.model_alias,
                        "model_path": resolved_model_path,
                        "attempts": record["attempts"],
                        "prompt": prompt,
                        "raw_response": result.get("raw_response"),
                        "parsed_json": result.get("parsed_json"),
                        "validated_score": result.get("llm_score"),
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_f.flush()

            if args.empty_cache_every_call and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if consecutive_runtime_failures >= STOP_AFTER_CONSECUTIVE_RUNTIME_FAILURES:
                print(
                    "\nStopping early because too many consecutive local runtime errors occurred. "
                    "Check GPU memory/model loading, then rerun; successful calls will be skipped automatically."
                )
                break

    write_results(csv_path, jsonl_path, list(records_by_key.values()))
    return csv_path, jsonl_path, log_path


if __name__ == "__main__":
    output_csv, output_jsonl, output_log = run_assessment(parse_args())
    print(f"\nSaved results:         {output_csv}")
    print(f"Saved results:         {output_jsonl}")
    print(f"Saved interaction log: {output_log}")
