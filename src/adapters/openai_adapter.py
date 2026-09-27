"""OpenAI API adapter.

Implements the ProbeAdapter interface over the OpenAI chat-completions API:
  - Synthetic history injected as prior chat messages.
  - ``thinking`` mapped to ``reasoning_effort`` for gpt-5 / o-series.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from .base import ProbeAdapter, ProbeSample, ProbeSpec, ThinkingLevel


def _history_to_openai_messages(history: list[dict]) -> list[dict]:
    """Convert neutral history to OpenAI chat messages."""
    out: list[dict] = []
    for turn in history:
        role = turn.get("role")
        if role == "user":
            out.append({"role": "user", "content": turn.get("content", "")})
        elif role == "assistant":
            tcs = turn.get("tool_calls") or []
            if tcs:
                native = []
                for i, tc in enumerate(tcs):
                    args = tc.get("arguments", {})
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    native.append({
                        "id": tc.get("id", f"call_synth_{i}"),
                        "type": "function",
                        "function": {"name": tc.get("name", ""), "arguments": args},
                    })
                out.append({
                    "role": "assistant",
                    "content": turn.get("content", None),
                    "tool_calls": native,
                })
            else:
                out.append({"role": "assistant", "content": turn.get("content", "")})
        elif role == "tool":
            # OpenAI tool messages must reference the call's `tool_call_id`.
            # Use the adjacent assistant tool_call id (deterministic name) if
            # the caller didn't supply one.
            out.append({
                "role": "tool",
                "tool_call_id": turn.get("tool_call_id", "call_synth_0"),
                "content": turn.get("content", ""),
            })
        else:
            out.append(dict(turn))
    return out


class OpenAIAdapter(ProbeAdapter):
    family = "openai"

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.model_id = model_cfg["id"]
        self._base_url = model_cfg.get("base_url")
        # When a custom base_url is set, an api_key_provider is mandatory
        # because load_api_key needs a provider name. Infer "openrouter"
        # from the URL host when not supplied so the common case works
        # without forcing extra config; explicit values still win.
        api_key_provider = model_cfg.get("api_key_provider")
        if self._base_url and not api_key_provider:
            if "openrouter.ai" in self._base_url:
                api_key_provider = "openrouter"
            else:
                raise ValueError(
                    f"OpenAIAdapter '{self.model_id}': base_url={self._base_url!r} "
                    "set without api_key_provider. Add 'api_key_provider' to "
                    "models.yaml (e.g., 'openrouter')."
                )
        self._api_key_provider = api_key_provider
        self._client = None

    def supported_dial_levels(self) -> list[ThinkingLevel]:
        if self.model_id.startswith("gpt-5") or self.model_id.startswith("o"):
            return ["low", "medium", "high", "default"]
        return ["default"]  # gpt-4o etc.: dial unsupported

    def _ensure_client(self):
        if self._client is not None:
            return
        if self._base_url:
            from openai import OpenAI
            from .api_keys import load_api_key
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=load_api_key(self._api_key_provider),
            )
        else:
            from src.api_runner import create_client
            self._client = create_client()

    def run_sample(self, spec: ProbeSpec, gen_config: dict, rng: Any) -> ProbeSample:
        self._ensure_client()

        from src.tool_formatter import shuffle_tools

        if spec.shuffle_tools_seed is not None:
            local_rng = random.Random(spec.shuffle_tools_seed)
        else:
            local_rng = rng
        shuffled = shuffle_tools(spec.tools, local_rng)
        tool_order = [t["function"]["name"] for t in shuffled]

        messages: list[dict] = [{"role": "system", "content": spec.system_prompt}]
        if spec.history:
            messages.extend(_history_to_openai_messages(spec.history))
        if spec.user_turn:
            messages.append({"role": "user", "content": spec.user_turn})

        is_reasoning_family = (
            self.model_id.startswith("gpt-5") or self.model_id.startswith("o")
        )
        uses_reasoning = (
            spec.thinking in ("low", "medium", "high")
            and is_reasoning_family
        )
        thinking_active = bool(uses_reasoning)

        kwargs: dict = {"model": self.model_id, "messages": messages}
        tok_key = "max_completion_tokens" if is_reasoning_family else "max_tokens"
        kwargs[tok_key] = gen_config.get("max_new_tokens", 256)

        if is_reasoning_family:
            # o-series / gpt-5 reject custom temperature and top_p.
            if uses_reasoning:
                kwargs["reasoning_effort"] = spec.thinking
        else:
            kwargs["temperature"] = gen_config.get("temperature", 0.7)
            kwargs["top_p"] = gen_config.get("top_p", 0.9)

        if shuffled:
            kwargs["tools"] = shuffled

        try:
            response = self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message
            tool_calls: list[dict] = []
            is_malformed = False
            raw_output = message.content or ""
            has_tool_call = False
            direct_answer = True
            if getattr(message, "tool_calls", None):
                direct_answer = False
                has_tool_call = True
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"_raw": tc.function.arguments}
                        is_malformed = True
                    tool_calls.append({
                        "name": tc.function.name,
                        "arguments": arguments,
                        "raw": tc.function.arguments,
                    })
            return ProbeSample(
                raw_output=raw_output,
                tool_calls=tool_calls,
                has_tool_call=has_tool_call,
                is_malformed=is_malformed,
                direct_answer=direct_answer,
                error=None,
                tool_order=tool_order,
                thinking_active=thinking_active,
                api_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )
        except Exception as e:  # surface as malformed sample; keep pipeline moving
            return ProbeSample(
                raw_output="",
                tool_calls=[],
                has_tool_call=False,
                is_malformed=True,
                direct_answer=False,
                error=str(e),
                tool_order=tool_order,
                thinking_active=thinking_active,
                api_usage=None,
            )


from . import register  # noqa: E402

register("openai", OpenAIAdapter)
