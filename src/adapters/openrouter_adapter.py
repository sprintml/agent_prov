"""OpenRouter API adapter (tokenizer-formatted completions mode).

Formats prompts locally using the model's HuggingFace tokenizer - the exact
same ``format_prompt()`` call the HF adapter uses - then sends the formatted
string to OpenRouter's ``/v1/completions`` endpoint.  This guarantees the
model sees identical tokens whether running locally or via API.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict
from typing import Any

from .api_keys import load_api_key
from .base import ProbeAdapter, ProbeSample, ProbeSpec, ThinkingLevel


class OpenRouterAdapter(ProbeAdapter):
    family = "openrouter"

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.model_id = model_cfg["id"]
        self.model_family = model_cfg.get("family", "")
        self.parser_family = model_cfg.get("parser_family", self.model_family)
        # HuggingFace model ID for loading the tokenizer (may differ from
        # the OpenRouter model ID).
        self.tokenizer_id = model_cfg.get("tokenizer_id", "")
        # Optional OpenRouter provider pin. When set, the request includes
        # extra_body={"provider": {"order": [<provider>], "allow_fallbacks": False}}
        # so all calls land on the same backend (reproducibility / audit).
        self.provider: str | None = model_cfg.get("provider")
        self._client = None
        self._tokenizer = None

    def supported_dial_levels(self) -> list[ThinkingLevel]:
        if self.model_family in ("qwen3", "smollm3"):
            return ["off", "default", "on"]
        return ["default"]

    def _resolve_enable_thinking(self, thinking: ThinkingLevel) -> bool:
        if self.model_family in ("qwen3", "smollm3"):
            if thinking == "off":
                return False
            if thinking in ("on", "default", "low", "medium", "high"):
                return True
        return False

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=load_api_key("openrouter"),
        )

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id, trust_remote_code=True,
        )

    def _history_to_messages(self, history: list[dict]) -> list[dict]:
        """Convert neutral history to OpenAI-style messages.

        Mirrors HFAdapter._history_to_messages so the tokenizer renders
        identical history turns.
        """
        import hashlib
        import json

        def _tool_call_id(name: str, args_str: str) -> str:
            return hashlib.sha1(f"{name}|{args_str}".encode()).hexdigest()[:9]

        out: list[dict] = []
        last_tool_call_id: str | None = None
        for turn in history:
            role = turn.get("role")
            if role == "user":
                out.append({"role": "user", "content": turn.get("content", "")})
            elif role == "assistant":
                tcs = turn.get("tool_calls") or []
                if tcs:
                    native_tcs = []
                    for tc in tcs:
                        args = tc.get("arguments", {})
                        if not isinstance(args, str):
                            args = json.dumps(args)
                        name = tc.get("name", "")
                        tc_id = _tool_call_id(name, args)
                        native_tcs.append({
                            "type": "function",
                            "id": tc_id,
                            "function": {"name": name, "arguments": args},
                        })
                    last_tool_call_id = native_tcs[-1]["id"]
                    out.append({
                        "role": "assistant",
                        "content": turn.get("content", ""),
                        "tool_calls": native_tcs,
                    })
                else:
                    out.append({"role": "assistant", "content": turn.get("content", "")})
            elif role == "tool":
                tool_msg: dict = {
                    "role": "tool",
                    "name": turn.get("name", ""),
                    "content": turn.get("content", ""),
                }
                if last_tool_call_id is not None:
                    tool_msg["tool_call_id"] = last_tool_call_id
                out.append(tool_msg)
            else:
                out.append(dict(turn))
        return out

    def run_sample(self, spec: ProbeSpec, gen_config: dict, rng: Any) -> ProbeSample:
        self._ensure_client()
        self._ensure_tokenizer()

        from src.output_parser import parse_output
        from src.tool_formatter import format_prompt, shuffle_tools

        if spec.shuffle_tools_seed is not None:
            local_rng = random.Random(spec.shuffle_tools_seed)
        else:
            local_rng = rng
        shuffled = shuffle_tools(spec.tools, local_rng)
        tool_order = [t["function"]["name"] for t in shuffled]

        use_prefill = (
            (not spec.user_turn or not spec.user_turn.strip())
            and bool(spec.history)
            and spec.history[-1].get("role") == "assistant"
        )
        extra_messages = (
            self._history_to_messages(spec.history)
            if spec.history else None
        )

        enable_thinking = self._resolve_enable_thinking(spec.thinking)

        formatted = format_prompt(
            self._tokenizer,
            user_message=spec.user_turn,
            tools=shuffled,
            system_prompt=spec.system_prompt,
            model_family=self.model_family,
            enable_thinking=enable_thinking,
            extra_messages=extra_messages,
            prefill=use_prefill,
        )

        max_tokens = gen_config.get("max_new_tokens", 256)

        request: dict = {
            "model": self.model_id,
            "prompt": formatted,
            "max_tokens": max_tokens,
            "temperature": gen_config.get("temperature", 0.7),
            "top_p": gen_config.get("top_p", 0.9),
        }
        if self.provider:
            request["extra_body"] = {
                "provider": {
                    "order": [self.provider],
                    "allow_fallbacks": False,
                }
            }

        try:
            response = self._call_with_retry(request)
            choice = response.choices[0]
            raw_output = choice.text or ""
            was_truncated = getattr(choice, "finish_reason", None) == "length"

            result = parse_output(raw_output, self.parser_family)
            tool_calls = [asdict(tc) for tc in result.tool_calls]
            has_tool_call = len(tool_calls) > 0

            api_usage = None
            if response.usage:
                api_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            # Record actual provider used (returned by OpenRouter in metadata)
            actual_provider = getattr(response, "provider", None)
            if actual_provider is None:
                actual_provider = getattr(response, "_provider", None)
            if api_usage is not None:
                api_usage["pinned_provider"] = self.provider
                api_usage["actual_provider"] = actual_provider

            return ProbeSample(
                raw_output=raw_output,
                tool_calls=tool_calls,
                has_tool_call=has_tool_call,
                is_malformed=result.is_malformed,
                direct_answer=not has_tool_call and not result.is_malformed,
                error=None,
                tool_order=tool_order,
                thinking_active=False,
                api_usage=api_usage,
                was_truncated=was_truncated,
            )
        except Exception as e:
            return ProbeSample(
                raw_output="",
                tool_calls=[],
                has_tool_call=False,
                is_malformed=True,
                direct_answer=False,
                error=str(e),
                tool_order=tool_order,
                thinking_active=False,
                api_usage=None,
            )

    def _call_with_retry(self, kwargs: dict, max_retries: int = 5, base_delay: float = 2.0):
        # Retry on the SDK's typed rate-limit / transient exceptions instead
        # of substring-matching error strings.
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self._client.completions.create(**kwargs)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_err = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                raise
            except Exception:
                # Permanent error (4xx other than 429, auth, etc.) - surface.
                raise
        raise last_err  # type: ignore[misc]


from . import register  # noqa: E402

register("openrouter", OpenRouterAdapter)
