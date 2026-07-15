"""Picks the right model family (and therefore chat-template quirks) from a model path
or alias, so the assessment script doesn't need to know which family a given checkpoint
belongs to.

To add a new family with its own quirks: create `local_models/<family>.py` with
`<Family>LocalModel(BaseLocalModel)` / `<Family>VLLMModel(BaseVLLMModel)`, then register
it in FAMILY_NAME_PATTERNS + TRANSFORMERS_CLASSES/VLLM_CLASSES below. A family with no
quirks needs no file at all; it just falls back to the generic base classes.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseLocalModel, BaseVLLMModel
from .llama import LlamaLocalModel, LlamaVLLMModel
from .qwen import QwenLocalModel, QwenVLLMModel

# Substring match against the lowercased model path + alias.
FAMILY_NAME_PATTERNS: dict[str, tuple[str, ...]] = {
    "qwen": ("qwen",),
    "llama": ("llama", "llammlein", "leolm", "leo-lm"),
}

TRANSFORMERS_CLASSES: dict[str, type[BaseLocalModel]] = {
    "qwen": QwenLocalModel,
    "llama": LlamaLocalModel,
}
VLLM_CLASSES: dict[str, type[BaseVLLMModel]] = {
    "qwen": QwenVLLMModel,
    "llama": LlamaVLLMModel,
}


def resolve_family(model_path: str | Path, model_alias: str | None = None) -> str:
    """Return a family key (e.g. "qwen") by matching known substrings against the model
    path and alias; falls back to "generic" (plain base classes, no special quirks)."""
    haystack = f"{model_path} {model_alias or ''}".lower()
    for family, patterns in FAMILY_NAME_PATTERNS.items():
        if any(pattern in haystack for pattern in patterns):
            return family
    return "generic"


def get_transformers_class(family: str) -> type[BaseLocalModel]:
    return TRANSFORMERS_CLASSES.get(family, BaseLocalModel)


def get_vllm_class(family: str) -> type[BaseVLLMModel]:
    return VLLM_CLASSES.get(family, BaseVLLMModel)
