"""Qwen model family.

Adds Qwen3's `enable_thinking` chat-template toggle on top of the generic backends.
Everything else (loading, generation loop, batching) is inherited unchanged from base.py.
"""

from __future__ import annotations

from typing import Any

from .base import BaseLocalModel, BaseVLLMModel


class QwenLocalModel(BaseLocalModel):
    def chat_template_kwargs(self) -> dict[str, Any]:
        return {"enable_thinking": False} if self.disable_thinking else {}


class QwenVLLMModel(BaseVLLMModel):
    def chat_template_kwargs(self) -> dict[str, Any]:
        return {"enable_thinking": False} if self.disable_thinking else {}
