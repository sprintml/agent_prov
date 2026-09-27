"""ProbeAdapter registry.

Adapters implement per-family invocation: chat template assembly, thinking
dial control, synthetic history serialization, tool-call extraction. Each
adapter presents a single `run_sample(spec, gen_config, rng) -> ProbeSample`
entry point so probe code is family-agnostic.

Registration happens at import time from each concrete module. To add a
new family (e.g. Anthropic once access is available), create a subclass of
ProbeAdapter and call `register(family_name, cls)`.
"""

from __future__ import annotations

from .base import ProbeAdapter, ProbeSample, ProbeSpec, ThinkingLevel

_REGISTRY: dict[str, type[ProbeAdapter]] = {}


def register(family: str, cls: type[ProbeAdapter]) -> None:
    """Register a ProbeAdapter subclass for a family name."""
    _REGISTRY[family] = cls


def list_families() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_adapter(model_cfg: dict) -> ProbeAdapter:
    """Select and instantiate a ProbeAdapter for a model config dict.

    Priority:
      1. explicit ``adapter:`` key in model_cfg
      2. ``backend: openai_api``     -> openai adapter
      3. ``backend: openrouter_api`` -> openrouter adapter
      4. HF family name              -> hf adapter (always; family routes parser)

    Args:
        model_cfg: entry from config/models.yaml (with keys like ``id``,
                   ``family``, ``short_name``, optional ``backend``,
                   ``gen_overrides``, ``adapter``).

    Returns:
        Concrete ProbeAdapter instance. The adapter's own constructor handles
        loading models/clients lazily.
    """
    adapter_key = model_cfg.get("adapter")
    if adapter_key is None:
        backend = model_cfg.get("backend")
        if backend == "openai_api":
            adapter_key = "openai"
        elif backend == "openrouter_api":
            adapter_key = "openrouter"
        else:
            adapter_key = "hf"

    if adapter_key not in _REGISTRY:
        raise KeyError(
            f"No adapter registered for '{adapter_key}'. "
            f"Registered: {list_families()}"
        )
    return _REGISTRY[adapter_key](model_cfg)


# Register concrete adapters at import time. Each import is side-effectful
# (calls register). Import last so the registry is populated when we
# `from src.adapters import get_adapter`.
from . import hf_adapter  # noqa: F401,E402  (local HuggingFace reference)
from . import openai_adapter  # noqa: F401,E402  (OpenAI-compatible API)
from . import openrouter_adapter  # noqa: F401,E402  (third-party API endpoints)

__all__ = [
    "ProbeAdapter",
    "ProbeSample",
    "ProbeSpec",
    "ThinkingLevel",
    "register",
    "list_families",
    "get_adapter",
]
