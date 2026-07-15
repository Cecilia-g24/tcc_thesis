"""Backend base classes shared by every model family.

Mirrors the shape of tmp/Qwen_llm.py: a transformers-backend class that loads the model
once and exposes `.chat(messages)`, and a vLLM-backend class that loads the engine once
and exposes `.batch_chat(messages_list)` / `.chat(messages)`. Decoding params (max
tokens, temperature, top_p) are fixed at construction time, same as tmp/Qwen_llm.py, so
callers just pass chat messages in and get text back.

Family files (qwen.py, llama.py, ...) subclass these and override `chat_template_kwargs()`
only where a model's chat template needs family-specific options (e.g. Qwen3's
`enable_thinking`). A family with no quirks needs no file at all — registry.py falls back
to these base classes directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import build_quantization_config, model_input_device, resolve_torch_dtype


class BaseLocalModel:
    """One HF checkpoint loaded via AutoModelForCausalLM, called one prompt at a time."""

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        dtype: str = "auto",
        device_map: str | None = "auto",
        trust_remote_code: bool = True,
        local_files_only: bool = True,
        attn_implementation: str | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        disable_thinking: bool = True,
    ) -> None:
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model path does not exist: {model_path}\n"
                "Pass the correct directory with --model-path, for example: --model-path ../models/qwen3_30b_instruct"
            )
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.disable_thinking = disable_thinking

        print(f"Loading tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

        quantization_config = build_quantization_config(load_in_4bit, load_in_8bit)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = resolve_torch_dtype(dtype)
        if device_map and device_map.lower() != "none":
            model_kwargs["device_map"] = device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        print(f"Loading model from: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def chat_template_kwargs(self) -> dict[str, Any]:
        """Family-specific `apply_chat_template()` kwargs. Empty for plain models."""
        return {}

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return messages[-1]["content"]

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            **self.chat_template_kwargs(),
        }
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Some template implementations reject unknown kwargs (e.g. enable_thinking on
            # a non-Qwen3 template); retry with the plain call.
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate one response for one conversation."""
        rendered_prompt = self._render_prompt(messages)
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt")

        target_device = model_input_device(self.model)
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

        do_sample = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class BaseVLLMModel:
    """One vLLM engine, batched across every prompt handed to `batch_chat()`."""

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        local_files_only: bool = True,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        quantization: str | None = None,
        disable_thinking: bool = True,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "The vllm backend was requested, but the `vllm` package is not installed. "
                "Run `pip install vllm`, or use --backend transformers instead."
            ) from exc

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model path does not exist: {model_path}\n"
                "Pass the correct directory with --model-path, for example: --model-path ../models/qwen3_30b_instruct"
            )
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self.disable_thinking = disable_thinking

        do_sample = temperature > 0
        self.sampling_params = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            max_tokens=max_new_tokens,
        )

        if tensor_parallel_size is None:
            n_gpus = torch.cuda.device_count()
            tensor_parallel_size = n_gpus if n_gpus > 1 else 1

        engine_kwargs: dict[str, Any] = {
            "model": str(model_path),
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
        }
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len
        if quantization:
            engine_kwargs["quantization"] = quantization

        print(f"Loading vLLM engine from: {model_path} (tensor_parallel_size={tensor_parallel_size})")
        try:
            self.llm = LLM(**engine_kwargs)
        except Exception as exc:
            if "no kernel image is available" in str(exc).lower():
                raise RuntimeError(
                    "vLLM engine startup failed with a CUDA 'no kernel image' error. This means the "
                    "installed PyTorch build has no kernels for this GPU's compute capability (e.g. "
                    "PyTorch dropped Volta/V100 sm_70 support in recent releases). This also affects "
                    "--backend transformers on the same GPU, since it's the same torch install. Fix by "
                    "requesting a node with a newer GPU (compute capability >= 7.5), or by reinstalling "
                    "a PyTorch build that supports this GPU's compute capability."
                ) from exc
            raise

    def chat_template_kwargs(self) -> dict[str, Any]:
        """Family-specific `chat_template_kwargs` passed to `LLM.chat()`. Empty for plain models."""
        return {}

    def batch_chat(self, messages_list: list[list[dict[str, str]]]) -> list[str]:
        chat_kwargs: dict[str, Any] = {}
        extra = self.chat_template_kwargs()
        if extra:
            chat_kwargs["chat_template_kwargs"] = extra
        try:
            outputs = self.llm.chat(messages_list, self.sampling_params, **chat_kwargs)
        except TypeError:
            outputs = self.llm.chat(messages_list, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self.batch_chat([messages])[0]
