"""HuggingFace local-inference adapter.

Wraps the existing ``src.model_loader``, ``src.tool_formatter``, and
``src.output_parser`` into the ProbeAdapter interface. Adds:
  - Controllable ``thinking`` dial via ``enable_thinking`` (Qwen3).
  - Synthetic history serialization via ``extra_messages``.

The model and tokenizer are loaded lazily on the first ``run_sample`` call,
so constructing a bunch of HFAdapter instances (e.g. to list them) is cheap.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

from .base import ProbeAdapter, ProbeSample, ProbeSpec, ThinkingLevel


class HFAdapter(ProbeAdapter):
    family = "hf"

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.model_family = model_cfg.get("family", "")
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Thinking dial
    # ------------------------------------------------------------------

    def supported_dial_levels(self) -> list[ThinkingLevel]:
        # Hybrid-dial families: Qwen3 and SmolLM3 expose a boolean
        # enable_thinking knob through apply_chat_template.
        if self.model_family in ("qwen3", "smollm3"):
            return ["off", "default", "on"]
        # Always-on reasoning families: the thinking block is emitted
        # regardless of what the caller requests; dial requests are no-ops.
        if self.model_family in ("deepseek-r1", "glm-z1", "phi-reasoning"):
            return ["default"]
        return ["default"]

    def _resolve_enable_thinking(self, thinking: ThinkingLevel) -> bool:
        if self.model_family in ("qwen3", "smollm3"):
            if thinking == "off":
                return False
            if thinking in ("on", "default", "low", "medium", "high"):
                return True
        # For all other families, always False (no thinking dial).
        return False

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return
        from src.model_loader import load_model

        device = self.model_cfg.get("device", "auto")
        model_id = self.model_cfg["id"]
        dtype_override = self.model_cfg.get("dtype")
        load_in_8bit = self.model_cfg.get("load_in_8bit", False)
        load_in_4bit = self.model_cfg.get("load_in_4bit", False)
        self._model, self._tokenizer = load_model(
            model_id, device, dtype_override=dtype_override,
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
        )

    # ------------------------------------------------------------------
    # History -> OpenAI-style messages expected by apply_chat_template
    # ------------------------------------------------------------------

    def _history_to_messages(
        self, history: list[dict], prefill: bool = False
    ) -> list[dict]:
        """Convert neutral history into OpenAI-style chat messages.

        Neutral format:
          {"role": "user", "content": "..."},
          {"role": "assistant", "tool_calls": [{"name", "arguments"}]} or
          {"role": "assistant", "content": "..."},
          {"role": "tool", "name": "...", "content": "..."}

        OpenAI chat template format (accepted by most HF chat templates):
          {"role": "user", "content": "..."},
          {"role": "assistant",
           "tool_calls": [{"type": "function",
                           "function": {"name": "...", "arguments": "..."}}]},
          {"role": "tool", "name": "...", "content": "..."}

        For R1-distill families we wrap assistant content in an empty
        ``<think></think>`` block when none is present, since those models
        destabilize on assistant turns that lack one. EXCEPTION: when
        ``prefill=True``, we leave the LAST assistant message's content
        untouched, because R1's chat template strips ``<think>...</think>``
        prefixes from assistant turns; stripping would make the rendered
        text differ from the raw content and break HF's
        ``continue_final_message=True`` validator.

        Every native tool_call is assigned a 9-character alphanumeric
        ``id`` - Mistral-family chat templates (Ministral, Mistral-Nemo,
        v0.3) explicitly validate tool-call IDs and raise
        ``TemplateError("Tool call IDs should be alphanumeric strings
        with length 9!")`` if missing. The ID is deterministic (first 9
        chars of a SHA-1 hex digest of the serialized call) so the same
        input produces the same rendered prompt - essential for the
        fingerprint reproducibility guarantee. Other families ignore
        the ``id`` field, so this is harmless across the adapter cohort.
        """
        import hashlib
        import json

        def _tool_call_id(name: str, args_str: str) -> str:
            """9-char alphanumeric id derived from the call contents."""
            h = hashlib.sha1(f"{name}|{args_str}".encode()).hexdigest()
            # hex is already alphanumeric; take first 9.
            return h[:9]

        out: list[dict] = []
        # Track the most recently emitted tool_call id so subsequent `tool`
        # turns can reference it via tool_call_id - Mistral templates
        # validate that each tool result links back to an outstanding call.
        last_tool_call_id: str | None = None
        n = len(history)
        for i, turn in enumerate(history):
            role = turn.get("role")
            is_last = (i == n - 1)
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
                            "function": {
                                "name": name,
                                "arguments": args,
                            },
                        })
                    last_tool_call_id = native_tcs[-1]["id"]
                    msg = {"role": "assistant", "content": turn.get("content", ""),
                           "tool_calls": native_tcs}
                else:
                    content = turn.get("content", "")
                    skip_wrap_for_prefill = prefill and is_last
                    if (
                        self.model_family == "deepseek-r1"
                        and "<think>" not in content
                        and not skip_wrap_for_prefill
                    ):
                        content = "<think></think>\n" + content
                    msg = {"role": "assistant", "content": content}
                out.append(msg)
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
                # Pass-through for unknown roles; apply_chat_template will
                # complain if the family doesn't accept them.
                out.append(dict(turn))
        return out

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_sample(self, spec: ProbeSpec, gen_config: dict, rng: Any) -> ProbeSample:
        import torch

        from src.output_parser import parse_output
        from src.tool_formatter import format_prompt, shuffle_tools

        self._ensure_loaded()

        # Shuffle tool order with the caller-supplied rng (or a seed-derived
        # one) for deterministic per-sample reproducibility.
        if spec.shuffle_tools_seed is not None:
            local_rng = random.Random(spec.shuffle_tools_seed)
        else:
            local_rng = rng
        shuffled = shuffle_tools(spec.tools, local_rng)
        tool_order = [t["function"]["name"] for t in shuffled]

        enable_thinking = self._resolve_enable_thinking(spec.thinking)

        # Prefill / continuation mode: empty user_turn + assistant-last history.
        # Determine it from spec.history first (the raw neutral format) so we
        # can pass the flag into _history_to_messages, which needs to know
        # whether to skip R1's <think></think> wrap on the last assistant
        # turn (the wrap would break continue_final_message=True).
        use_prefill = (
            (not spec.user_turn or not spec.user_turn.strip())
            and bool(spec.history)
            and spec.history[-1].get("role") == "assistant"
        )
        extra_messages = (
            self._history_to_messages(spec.history, prefill=use_prefill)
            if spec.history else None
        )

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

        inputs = self._tokenizer(formatted, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self._model.device)

        max_new_tokens = gen_config.get("max_new_tokens", 256)
        extra_gen_kwargs = self.model_cfg.get("extra_generate_kwargs", {})
        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p", 0.9),
                do_sample=gen_config.get("do_sample", True),
                pad_token_id=self._tokenizer.eos_token_id,
                **extra_gen_kwargs,
            )

        generated_ids = output_ids[0][input_ids.shape[1]:]
        raw_output = self._tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Truncation detection: the model hit max_new_tokens without emitting
        # any configured EOS / stop token. Without this flag, long-reasoning
        # traces that get cut off mid-self-verify would be classified as if
        # their partial-output final number were the intended answer -
        # differentially biasing AGAINST the reasoning models we want to
        # detect.
        eos_ids: set[int] = set()
        tok_eos = getattr(self._tokenizer, "eos_token_id", None)
        if isinstance(tok_eos, int):
            eos_ids.add(tok_eos)
        elif isinstance(tok_eos, (list, tuple)):
            eos_ids.update(int(x) for x in tok_eos if x is not None)
        # Model-config-level generation-stop tokens (e.g. Qwen2.5 lists both
        # <|im_end|> and <|endoftext|>).
        model_gen_cfg = getattr(self._model, "generation_config", None)
        cfg_eos = getattr(model_gen_cfg, "eos_token_id", None) if model_gen_cfg else None
        if isinstance(cfg_eos, int):
            eos_ids.add(cfg_eos)
        elif isinstance(cfg_eos, (list, tuple)):
            eos_ids.update(int(x) for x in cfg_eos if x is not None)

        last_tok = int(generated_ids[-1].item()) if generated_ids.numel() > 0 else -1
        was_truncated = (
            generated_ids.numel() >= max_new_tokens
            and last_tok not in eos_ids
        )

        result = parse_output(raw_output, self.model_family)
        # Thinking-tag detection - the ground truth for "did the model actually
        # engage thinking on this turn" is the presence of reasoning-block
        # tokens in its output, NOT what we *requested* via `enable_thinking`.
        # Using `enable_thinking or <tag-present>` would mean Qwen3 always
        # reports True whenever the dial is on, masking "the dial is on but
        # the model chose to skip thinking on a trivial prompt" - which is
        # precisely the signal s_think_gate trivial tier tries to measure
        # for the Stage-1 Dial Mechanism Diagnostic.
        #
        # R1-distill's chat template injects `<think>\n` into the generation
        # prompt, so `raw_output` only contains the CLOSING `</think>` tag.
        # Checking only for `<think>` would misclassify every R1 sample as
        # did_not_think. We accept either tag (opener OR closer) to cover
        # Qwen3 (emits both), R1 (emits only closer), and any future family
        # that emits only an opener on truncation.
        thinking_tag_present = ("<think>" in raw_output) or ("</think>" in raw_output)
        return ProbeSample(
            raw_output=raw_output,
            tool_calls=[asdict(tc) for tc in result.tool_calls],
            has_tool_call=result.has_tool_call,
            is_malformed=result.is_malformed,
            direct_answer=result.direct_answer,
            error=result.error or None,
            tool_order=tool_order,
            thinking_active=thinking_tag_present,
            api_usage=None,
            was_truncated=was_truncated,
        )

    def close(self) -> None:
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Register under both "hf" and every HF family name - get_adapter will route
# to the hf entry; the extra family aliases make it possible to look up by
# family name directly if needed.
from . import register  # noqa: E402

register("hf", HFAdapter)
