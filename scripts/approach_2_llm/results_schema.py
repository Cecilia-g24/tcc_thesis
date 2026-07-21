"""Shared results-CSV schema for the LLM-as-judge assessment scripts.

llm_assessment_api.py and llm_assessment_local.py write results for the same
(id, dimension, variant_id) call key, but through different backends, so each
records its own backend-specific metadata (e.g. provider/rate-limit knobs for
the API script; gpu/dtype/quantization for the local script). CORE_RESULT_COLUMNS
lists the fields that mean the same thing in both -- same name, same semantics --
so results from the two scripts can be concatenated or compared without a
renaming step. Each script's RESULT_COLUMNS is this list plus its own
backend-specific columns appended after it.
"""

from __future__ import annotations

CORE_RESULT_COLUMNS: list[str] = [
    "id",
    "dimension",
    "variant_id",
    "model",
    "backend",
    "temperature",
    "top_p",
    "max_output_tokens",
    "seed",
    "decoding_mode",
    "thinking_mode",
    "llm_score",
    "raw_response",
    "error",
    "attempts",
    "average_score",
    "timestamp",
]
