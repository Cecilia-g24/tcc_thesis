"""Helpers shared by every local model family: dtype/quantization resolution and a
CUDA-capability preflight check. Kept separate from base.py so family files only pull in
what they need, without depending on argparse-shaped config objects.
"""

from __future__ import annotations

from typing import Any

import torch


def render_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Render a chat messages list to a prompt string via the tokenizer's chat template,
    falling back to the raw user content when no chat template is defined (e.g. base/
    completion checkpoints that were never fine-tuned into an instruct chat format).

    Shared by both the transformers and vLLM backends so a model without a chat template
    behaves the same way (plain prompt, no template) regardless of backend, instead of
    vLLM's own `.chat()` raising because it always requires one.
    """
    if not getattr(tokenizer, "chat_template", None):
        return messages[-1]["content"]

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        **(chat_template_kwargs or {}),
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Some template implementations reject unknown kwargs (e.g. enable_thinking on
        # a non-Qwen3 template); retry with the plain call.
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resolve_torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def build_quantization_config(load_in_4bit: bool, load_in_8bit: bool) -> Any | None:
    """Build a transformers BitsAndBytesConfig, or None for full precision.

    Transformers-only: vLLM has its own --quantization flag instead.
    """
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one of --load-in-4bit or --load-in-8bit.")
    if not load_in_4bit and not load_in_8bit:
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes quantization was requested, but BitsAndBytesConfig could not be imported. "
            "Install a compatible transformers/bitsandbytes setup, or run without quantization."
        ) from exc

    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def model_input_device(model: Any) -> torch.device:
    """Choose a safe input device for normal and accelerate-dispatched models."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_cuda_compatibility() -> None:
    """Fail fast if this PyTorch build has no CUDA kernels for the visible GPU(s).

    vLLM's V1 engine runs GPU work in a subprocess, so a hardware/kernel mismatch there
    (e.g. an old GPU like a V100 that a recent torch wheel no longer ships kernels for)
    surfaces to the caller as a generic "Engine core initialization failed" error with
    none of the actual CUDA error text. Checking compute capability against
    torch.cuda.get_arch_list() upfront avoids waiting through model download/engine
    startup just to hit that opaque failure, and applies to the transformers backend too
    since it's the same torch install running on the same GPU.
    """
    if not torch.cuda.is_available():
        return

    supported_archs = set(torch.cuda.get_arch_list())
    unsupported = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        if f"sm_{major}{minor}" not in supported_archs:
            unsupported.append(f"GPU {i}: {torch.cuda.get_device_name(i)} (compute capability {major}.{minor})")

    if unsupported:
        raise RuntimeError(
            "This PyTorch build has no CUDA kernels for: " + "; ".join(unsupported) + ". "
            f"It only supports compute capabilities matching: {sorted(supported_archs)}. "
            "Both the vllm and transformers backends will fail on this GPU with the current "
            "torch install. Request a node with a supported GPU, or reinstall a PyTorch build that "
            "includes this GPU's architecture."
        )
