"""Llama model family (Llama, LLaMmlein, LeoLM, ...).

Their chat templates take no extra kwargs, so these classes are currently thin
pass-throughs over base.py. They exist as an explicit, documented home for
Llama-specific quirks if/when one shows up, rather than silently relying on the
generic fallback in registry.py.
"""

from .base import BaseLocalModel, BaseVLLMModel


class LlamaLocalModel(BaseLocalModel):
    pass


class LlamaVLLMModel(BaseVLLMModel):
    pass
