"""
Run LLM-as-judge assessment on transcripts using a locally downloaded Hugging Face model.

This script mirrors llm_assessment_api.py, but replaces the OpenAI/GWDG API call with
local generation via the local_models/ package. Two backends are supported:

    --backend vllm          (default) Batched generation via vLLM, with automatic
                             multi-GPU tensor-parallel sharding for large models.
                             Requires `pip install vllm`. Best for well-supported
                             architectures (Qwen, Gemma, Mistral, ...) and for anything
                             too large to run one call at a time.
    --backend transformers  Plain AutoModelForCausalLM.generate(), one call at a
                             time. Slower and not batched, but works with any
                             HF-compatible checkpoint, including small/custom
                             research models that vLLM may not support, and it is
                             the only backend that supports bitsandbytes 4/8-bit
                             quantization (--load-in-4bit / --load-in-8bit).

Model loading and generation live in local_models/ (one file per model family, e.g.
qwen.py, llama.py), each exposing a `.chat(messages)` / `.batch_chat(messages_list)`
class per backend -- see local_models/base.py. This file only does the assessment task:
data loading/sampling, prompt building, resumable result bookkeeping, and retries.
--model-path/--model-alias are matched against local_models/registry.py to pick the
right family automatically.

Designed for resumable, auditable development runs:
- fixed-seed dev sampling or full-run mode
- id + dimension + variant_id as the unique call key
- immediate result saving after every call (transformers) or every batch (vLLM)
- resume support: successful calls are skipped on rerun
- failed/missing calls are retried by default on rerun
- responses are parsed and validated against integer scores in {0, 1, 2, 3, 4}
- greedy decoding enforced inside local_models/base.py for temperature-zero runs
- inference/runtime metadata recorded for auditability

Each --model-alias gets its own subfolder under --output-dir (results.csv, results.jsonl,
interaction_log.jsonl), so multiple local models can be run into the same output-dir
without colliding.

Default condition:
    German transcript + German prompt (override --input-csv/--text-col for English),
    all rows, all 5 dimensions, all prompt variants -- same coverage as
    llm_assessment_api.py by default. Pass --n-per-dimension/--sample-mode for a
    fixed-size dev subsample instead.

Run from the repo root or from this script directory:
    python scripts/approach_2_llm/llm_assessment_local.py

-m/--model-path accepts a bare model folder name (resolved against the sibling models/
folder) and -a/--model-alias defaults to that folder name, so you don't need to type
either "../models/" or repeat the model name as an alias.

Useful options:
    python scripts/approach_2_llm/llm_assessment_local.py --n-per-dimension 20 --sample-mode random --random-state 42  # dev subsample
    python scripts/approach_2_llm/llm_assessment_local.py -m qwen3_30b_instruct -v V1_full_manual_baseline
    python scripts/approach_2_llm/llm_assessment_local.py -m qwen3_235b_instruct --tensor-parallel-size 8
    python scripts/approach_2_llm/llm_assessment_local.py --backend transformers -m llammlein_1b
    python scripts/approach_2_llm/llm_assessment_local.py --overwrite            # ignore previous results
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
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
import torch
from tqdm import tqdm

try:
    # Works when this file is placed next to de_prompt_construction.py.
    from de_prompt_construction import VARIANTS, build_prompt
    from results_schema import CORE_RESULT_COLUMNS
except ModuleNotFoundError:
    # Works when running from unusual working directories.
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from de_prompt_construction import VARIANTS, build_prompt
    from results_schema import CORE_RESULT_COLUMNS

try:
    from local_models import (
        BaseLocalModel,
        BaseVLLMModel,
        check_cuda_compatibility,
        get_transformers_class,
        get_vllm_class,
        resolve_family,
    )
except ModuleNotFoundError:
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from local_models import (
        BaseLocalModel,
        BaseVLLMModel,
        check_cuda_compatibility,
        get_transformers_class,
        get_vllm_class,
        resolve_family,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "data_clean" / "01_csvs_for_liwc_manual_input" / "full_dataset_de.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "approach2" / "de_text_de_prompt_results" / "local_results"

# In the VS Code server view, models/ and tcc/ appear to be sibling folders.
# Override this with --model-path if your model is elsewhere.
DEFAULT_MODEL_PATH = REPO_ROOT.parent / "models" / "qwen3_30b_instruct"

DEFAULT_TEXT_COL = "text"
DEFAULT_N_PER_DIMENSION = 0  # 0 = all rows, matching llm_assessment_api.py's uncapped default
DEFAULT_SAMPLE_MODE = "all"  # random | first | all
DEFAULT_RANDOM_STATE = 42
DEFAULT_GENERATION_SEED = 42

DEFAULT_BACKEND = "vllm"  # vllm | transformers
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_VLLM_BATCH_SIZE = 16
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90

MAX_RETRIES = 3
BASE_RETRY_DELAY_S = 10
MAX_RETRY_DELAY_S = 120
STOP_AFTER_CONSECUTIVE_RUNTIME_FAILURES = 3

# Core columns (shared name/semantics with llm_assessment_api.py's RESULT_COLUMNS) plus
# this script's own local-backend-specific columns, appended after.
RESULT_COLUMNS = CORE_RESULT_COLUMNS + [
    "model_path",
    "backend_version",
    "torch_version",
    "transformers_version",
    "python_version",
    "gpu",
    "dtype",
    "quantization",
    "do_sample",
    "num_beams",
    "num_return_sequences",
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
        help="Rows per dimension for dev runs. Default 0 (or any value <= 0) runs every row, "
             "matching llm_assessment_api.py. Pass a positive number for a fixed-size dev subsample.",
    )
    parser.add_argument("--sample-mode", choices=["random", "first", "all"], default=DEFAULT_SAMPLE_MODE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=DEFAULT_GENERATION_SEED,
        help="Seed Python/PyTorch and pass the same seed into the local backend. Greedy decoding does not "
             "sample, but a fixed seed improves auditability and supports reproducible sampling runs.",
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help="Request deterministic PyTorch algorithms where available (warn-only). This can reduce speed.",
    )
    parser.add_argument(
        "-v", "--variants",
        type=str,
        default="all",
        help="Comma-separated list of prompt variant ids to run (e.g. V1_full_manual_baseline), "
             f"or 'all' (default) to run every variant. Choices: {sorted(VARIANTS)}.",
    )

    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default=os.getenv("LOCAL_BACKEND", DEFAULT_BACKEND),
        help="'vllm' (default) batches all pending calls and shards large models across GPUs "
             "automatically; requires `pip install vllm` and a vLLM-supported architecture. "
             "'transformers' runs one call at a time via AutoModelForCausalLM.generate(); slower "
             "and unbatched, but works with any HF-compatible checkpoint and supports "
             "--load-in-4bit/--load-in-8bit.",
    )
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        default=os.getenv("LOCAL_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        help="Path to the local Hugging Face model directory. A bare name with no slash "
             "(e.g. qwen3_32b) is resolved against the sibling models/ folder "
             f"({DEFAULT_MODEL_PATH.parent}/<name>).",
    )
    parser.add_argument(
        "-a", "--model-alias",
        type=str,
        default=os.getenv("LOCAL_MODEL_ALIAS"),
        help="Short name used for the output subfolder and result rows, and (together "
             "with --model-path) to auto-detect the model family in local_models/. "
             "Defaults to the --model-path folder name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Generation temperature. At 0.0 this script explicitly requests greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus-sampling cutoff. Normalized to the neutral value 1.0 when temperature=0.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype. bfloat16 is often best on supported GPUs; float16 is safer on V100.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", dest="local_files_only", action="store_false")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=True,
        help="For chat templates that support a thinking-mode toggle (e.g. Qwen3), request "
             "non-thinking mode. Families without such a toggle ignore this.",
    )
    parser.add_argument("--allow-thinking", dest="disable_thinking", action="store_false")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing result files and start a fresh run.")
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Do not retry existing failed/unparseable calls. By default, failed calls are retried.",
    )

    transformers_group = parser.add_argument_group("transformers backend only")
    transformers_group.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Transformers device_map. Use 'auto' for GPU dispatch. Use 'cpu' only for tiny tests.",
    )
    transformers_group.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit quantization. Requires bitsandbytes and a compatible GPU setup.",
    )
    transformers_group.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Use bitsandbytes 8-bit quantization. Requires bitsandbytes and a compatible GPU setup.",
    )
    transformers_group.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional: e.g., flash_attention_2, sdpa, or eager. Leave unset if unsure.",
    )
    transformers_group.add_argument(
        "--empty-cache-every-call",
        action="store_true",
        help="Call torch.cuda.empty_cache() after every generation. Slower, but can help fragmented VRAM.",
    )

    vllm_group = parser.add_argument_group("vllm backend only")
    vllm_group.add_argument(
        "--vllm-batch-size",
        type=int,
        default=DEFAULT_VLLM_BATCH_SIZE,
        help="How many prompts to generate per vLLM batch, and how often results are checkpointed.",
    )
    vllm_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs to shard the model across. Defaults to all visible GPUs (torch.cuda.device_count()).",
    )
    vllm_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory vLLM is allowed to reserve for weights + KV cache.",
    )
    vllm_group.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional cap on vLLM's max context length, e.g. to fit KV cache in memory for long transcripts.",
    )
    vllm_group.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Optional vLLM quantization method, e.g. fp8, awq, gptq, bitsandbytes. Leave unset for full precision.",
    )

    args = parser.parse_args()
    if args.backend == "vllm" and (args.load_in_4bit or args.load_in_8bit):
        parser.error("--load-in-4bit/--load-in-8bit are transformers-only; use --quantization for the vllm backend.")

    # A bare name with no path separator (e.g. "qwen3_32b") is shorthand for the sibling
    # models/ folder, so --model-path doesn't need the "../models/" prefix every time.
    model_path = Path(args.model_path)
    if os.sep not in args.model_path and "/" not in args.model_path and not model_path.exists():
        model_path = DEFAULT_MODEL_PATH.parent / args.model_path
    args.model_path = model_path

    if args.model_alias is None:
        args.model_alias = args.model_path.name

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
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive.")

    # For greedy decoding, top_p is unused. Setting it to 1.0 avoids ambiguous or
    # backend-specific treatment of a leftover sampling parameter.
    if args.temperature == 0.0 and args.top_p != 1.0:
        print("temperature=0: overriding top_p to the neutral value 1.0.")
        args.top_p = 1.0

    return args


def package_version(package_name: str) -> str:
    """Return an installed package version without failing when the package is absent."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def configure_reproducibility(seed: int, deterministic_algorithms: bool) -> None:
    """Seed available RNGs and optionally request deterministic PyTorch algorithms."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)



def gpu_description() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return " | ".join(names)


def local_runtime_metadata(args: argparse.Namespace) -> dict[str, Any]:
    backend_package = "vllm" if args.backend == "vllm" else "transformers"
    if args.backend == "vllm":
        quantization = args.quantization or "none"
    elif args.load_in_4bit:
        quantization = "bitsandbytes_4bit"
    elif args.load_in_8bit:
        quantization = "bitsandbytes_8bit"
    else:
        quantization = "none"

    return {
        "backend": args.backend,
        "backend_version": package_version(backend_package),
        "torch_version": torch.__version__,
        "transformers_version": package_version("transformers"),
        "python_version": platform.python_version(),
        "gpu": gpu_description(),
        "dtype": args.dtype,
        "quantization": quantization,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": False if args.temperature == 0.0 else True,
        "num_beams": 1,
        "num_return_sequences": 1,
        "max_output_tokens": args.max_new_tokens,
        "seed": args.generation_seed,
        "decoding_mode": "temperature_zero_requested" if args.temperature == 0.0 else "sampling_requested",
        "thinking_mode": "disabled_requested" if args.disable_thinking else "enabled_or_model_default",
    }


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
    # Iterates the groupby directly (rather than groupby(...).apply(...)) so every group
    # keeps all of its original columns, including "dimension" itself, regardless of
    # pandas version (recent pandas can exclude the grouping column from what's passed
    # into an apply() callable, dropping "dimension" from the result).
    parts = [
        group.sample(n=min(n_per_dimension, len(group)), random_state=random_state)
        for _, group in df.groupby("dimension")
    ]
    return pd.concat(parts, ignore_index=True)


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

    # Last-resort extraction if the model wraps the JSON in extra text, or -- as small
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
        "engine core",  # vLLM's async engine wraps worker crashes with this phrase
    ]
    return any(marker in lowered for marker in runtime_markers)


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter for transient local generation failures."""
    delay = min(MAX_RETRY_DELAY_S, BASE_RETRY_DELAY_S * (2 ** (attempt - 1)))
    jitter = random.uniform(0, min(5, delay * 0.1))
    return delay + jitter


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "local_model"


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


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str]:
    safe_model_name = sanitize_filename(args.model_alias)
    model_dir = args.output_dir / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    csv_path = model_dir / "results.csv"
    jsonl_path = model_dir / "results.jsonl"
    log_path = model_dir / "interaction_log.jsonl"
    resolved_model_path = str(args.model_path.expanduser().resolve())
    return csv_path, jsonl_path, log_path, resolved_model_path


def prepare_pending_calls(
    df: pd.DataFrame,
    args: argparse.Namespace,
    csv_path: Path,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[tuple[pd.Series, str]]]:
    """Plan all (row, variant) calls, load resumable prior results, and return what's left to run."""
    calls: list[tuple[pd.Series, str]] = [(row, variant_id) for _, row in df.iterrows() for variant_id in args.variants]
    planned_keys = {(str(row["id"]), str(row["dimension"]), str(variant_id)) for row, variant_id in calls}

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

    return records_by_key, pending_calls


def build_transformers_model(args: argparse.Namespace) -> BaseLocalModel:
    """Resolve this model's family from --model-path/--model-alias (local_models/registry.py)
    and instantiate the matching transformers-backend class."""
    family = resolve_family(args.model_path, args.model_alias)
    print(f"Resolved model family: {family} (backend=transformers)")

    model_cls = get_transformers_class(family)
    model = model_cls(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_seed=args.generation_seed,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        disable_thinking=args.disable_thinking,
    )
    return model


def build_vllm_model(args: argparse.Namespace) -> BaseVLLMModel:
    """Resolve this model's family from --model-path/--model-alias (local_models/registry.py)
    and instantiate the matching vLLM-backend class."""
    family = resolve_family(args.model_path, args.model_alias)
    print(f"Resolved model family: {family} (backend=vllm)")

    model_cls = get_vllm_class(family)
    model = model_cls(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_seed=args.generation_seed,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        quantization=args.quantization,
        disable_thinking=args.disable_thinking,
    )
    return model


# --------------------------------------------------------------------------- #
# transformers backend
# --------------------------------------------------------------------------- #


def call_model(model: BaseLocalModel, prompt: str) -> dict[str, Any]:
    """Send one prompt to the local model and return raw response plus parse/validation status."""
    result: dict[str, Any] = {
        "raw_response": None,
        "parsed_json": None,
        "llm_score": None,
        "error": None,
        "attempts": 0,
    }
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(1, MAX_RETRIES + 1):
        result["attempts"] = attempt
        try:
            raw_response = model.chat(messages)
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

    return result


def run_transformers(
    args: argparse.Namespace,
    csv_path: Path,
    jsonl_path: Path,
    log_path: Path,
    resolved_model_path: str,
    records_by_key: dict[tuple[str, str, str], dict[str, Any]],
    pending_calls: list[tuple[pd.Series, str]],
    runtime_metadata: dict[str, Any],
) -> None:
    model = build_transformers_model(args)
    consecutive_runtime_failures = 0

    with log_path.open("a", encoding="utf-8") as log_f:
        for row, variant_id in tqdm(pending_calls, desc="Local LLM assessment", unit="call"):
            transcript = normalize_text_value(row[args.text_col])
            prompt = build_prompt(
                dimension_code=str(row["dimension"]),
                transcript=transcript,
                variant_id=variant_id,
            )

            result = call_model(model, prompt)

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
                **runtime_metadata,
                "llm_score": "" if result.get("llm_score") is None else result.get("llm_score"),
                "raw_response": result.get("raw_response"),
                "error": "" if error is None else error,
                "attempts": result.get("attempts"),
                "average_score": row.get("average_score", ""),
                "timestamp": timestamp,
            }

            records_by_key[result_key(record)] = record
            write_results(csv_path, jsonl_path, list(records_by_key.values()))

            log_f.write(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "id": record["id"],
                        "dimension": record["dimension"],
                        "variant_id": record["variant_id"],
                        "model": args.model_alias,
                        "model_path": resolved_model_path,
                        "runtime_metadata": runtime_metadata,
                        "inference_parameters": {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "do_sample": False if args.temperature == 0.0 else True,
                            "num_beams": 1,
                            "max_new_tokens": args.max_new_tokens,
                            "seed": args.generation_seed,
                            "thinking_mode": runtime_metadata["thinking_mode"],
                        },
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


# --------------------------------------------------------------------------- #
# vllm backend
# --------------------------------------------------------------------------- #


def call_vllm_chunk(model: BaseVLLMModel, prompts: list[str]) -> list[dict[str, Any]]:
    """Generate a chunk of prompts with vLLM.

    Batch-generates the whole chunk first. Items that fail to parse/validate are retried
    one at a time (cheap: vLLM handles size-1 batches fine) up to MAX_RETRIES total
    attempts. If the batch call itself raises (e.g. a transient CUDA error), the whole
    chunk is retried together with backoff before being marked failed.
    """
    n = len(prompts)
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    results: list[dict[str, Any]] = [
        {"raw_response": None, "parsed_json": None, "llm_score": None, "error": None, "attempts": 0}
        for _ in range(n)
    ]
    pending_idx = list(range(n))

    for attempt in range(1, MAX_RETRIES + 1):
        if not pending_idx:
            break
        batch_messages = [messages_list[i] for i in pending_idx]
        try:
            raw_responses = model.batch_chat(batch_messages)
        except Exception as exc:  # vLLM/CUDA errors vary by setup
            error = str(exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            for i in pending_idx:
                results[i]["attempts"] += 1
                results[i]["error"] = error
            if attempt < MAX_RETRIES:
                time.sleep(backoff_delay(attempt))
            continue

        still_pending = []
        for i, raw_response in zip(pending_idx, raw_responses):
            results[i]["attempts"] += 1
            parsed, score, parse_error = parse_and_validate_response(raw_response)
            results[i].update(
                {
                    "raw_response": raw_response,
                    "parsed_json": parsed,
                    "llm_score": score,
                    "error": parse_error,
                }
            )
            if parse_error is not None and attempt < MAX_RETRIES:
                still_pending.append(i)
        pending_idx = still_pending

    return results


def run_vllm(
    args: argparse.Namespace,
    csv_path: Path,
    jsonl_path: Path,
    log_path: Path,
    resolved_model_path: str,
    records_by_key: dict[tuple[str, str, str], dict[str, Any]],
    pending_calls: list[tuple[pd.Series, str]],
    runtime_metadata: dict[str, Any],
) -> None:
    model = build_vllm_model(args)
    consecutive_runtime_failures = 0
    batch_size = max(1, args.vllm_batch_size)

    chunks = [pending_calls[i : i + batch_size] for i in range(0, len(pending_calls), batch_size)]

    with log_path.open("a", encoding="utf-8") as log_f:
        for chunk in tqdm(chunks, desc="Local vLLM assessment", unit="batch"):
            prompts = [
                build_prompt(
                    dimension_code=str(row["dimension"]),
                    transcript=normalize_text_value(row[args.text_col]),
                    variant_id=variant_id,
                )
                for row, variant_id in chunk
            ]
            chunk_results = call_vllm_chunk(model, prompts)

            for (row, variant_id), prompt, result in zip(chunk, prompts, chunk_results):
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
                    **runtime_metadata,
                    "llm_score": "" if result.get("llm_score") is None else result.get("llm_score"),
                    "raw_response": result.get("raw_response"),
                    "error": "" if error is None else error,
                    "attempts": result.get("attempts"),
                    "average_score": row.get("average_score", ""),
                    "timestamp": timestamp,
                }
                records_by_key[result_key(record)] = record

                log_f.write(
                    json.dumps(
                        {
                            "timestamp": timestamp,
                            "id": record["id"],
                            "dimension": record["dimension"],
                            "variant_id": record["variant_id"],
                            "model": args.model_alias,
                            "model_path": resolved_model_path,
                            "runtime_metadata": runtime_metadata,
                            "inference_parameters": {
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "do_sample": False if args.temperature == 0.0 else True,
                                "num_beams": 1,
                                "max_new_tokens": args.max_new_tokens,
                                "seed": args.generation_seed,
                                "thinking_mode": runtime_metadata["thinking_mode"],
                            },
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
            write_results(csv_path, jsonl_path, list(records_by_key.values()))

            if consecutive_runtime_failures >= STOP_AFTER_CONSECUTIVE_RUNTIME_FAILURES:
                print(
                    "\nStopping early because too many consecutive local runtime errors occurred. "
                    "Check GPU memory/model loading, then rerun; successful calls will be skipped automatically."
                )
                break


def run_assessment(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    check_cuda_compatibility()
    configure_reproducibility(args.generation_seed, args.deterministic_algorithms)
    runtime_metadata = local_runtime_metadata(args)

    if args.text_col not in pd.read_csv(args.input_csv, nrows=1).columns:
        raise ValueError(f"Text column {args.text_col!r} was not found in {args.input_csv}")

    df = load_subset(args.input_csv, args.n_per_dimension, args.sample_mode, args.random_state)
    csv_path, jsonl_path, log_path, resolved_model_path = resolve_output_paths(args)
    records_by_key, pending_calls = prepare_pending_calls(df, args, csv_path)

    if args.backend == "vllm":
        run_vllm(
            args, csv_path, jsonl_path, log_path, resolved_model_path,
            records_by_key, pending_calls, runtime_metadata,
        )
    else:
        run_transformers(
            args, csv_path, jsonl_path, log_path, resolved_model_path,
            records_by_key, pending_calls, runtime_metadata,
        )

    write_results(csv_path, jsonl_path, list(records_by_key.values()))
    return csv_path, jsonl_path, log_path


if __name__ == "__main__":
    output_csv, output_jsonl, output_log = run_assessment(parse_args())
    print(f"\nSaved results:         {output_csv}")
    print(f"Saved results:         {output_jsonl}")
    print(f"Saved interaction log: {output_log}")
