"""ProbeAdapter abstract base class + probe I/O dataclasses.

The adapter abstracts every family-specific concern: tokenizer/chat-template
quirks, thinking dial invocation, tool-call parsing, synthetic-history
serialization. Probes construct a `ProbeSpec` (the neutral 5-tuple) and
receive a `ProbeSample` back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

ThinkingLevel = Literal["off", "low", "medium", "high", "default"]


@dataclass
class ProbeSpec:
    """Neutral 5-tuple input to an adapter."""

    system_prompt: str
    tools: list[dict]  # OpenAI-style function schemas
    history: list[dict] = field(default_factory=list)
    """Prior conversation, in neutral format:
        {role: 'user'|'assistant'|'tool', content?: str,
         tool_calls?: list[{name, arguments}], name?: str (for tool role)}.
    Adapters serialize this into the family's native format.
    """
    user_turn: str = ""
    """The elicitation turn. Can be empty for pure continuation probes where
    the model is asked to act on the injected history alone; adapters handle
    this gracefully (some families require at least a non-empty user turn)."""
    thinking: ThinkingLevel = "default"
    shuffle_tools_seed: int | None = None
    """When set, the adapter shuffles the tool list with this seed for
    deterministic per-sample reproducibility. When None, the adapter uses
    a caller-supplied rng."""


@dataclass
class ProbeSample:
    """Normalized output from one adapter call."""

    raw_output: str
    tool_calls: list[dict]  # list of {"name", "arguments", "raw"}
    has_tool_call: bool
    is_malformed: bool
    direct_answer: bool
    error: str | None = None
    tool_order: list[str] = field(default_factory=list)
    thinking_active: bool = False
    """True iff the thinking dial engaged for this call (Qwen3 /think,
    Claude thinking block present, OpenAI reasoning_effort non-zero, etc.)."""
    api_usage: dict | None = None
    was_truncated: bool = False
    """True when the adapter hit the max_new_tokens (or equivalent) budget
    without the model emitting an EOS/stop token - i.e., the reasoning trace
    was cut off mid-stream. Differential-bias hedge: reasoning models emit
    longer traces and are more likely to truncate, which would systematically
    under-count their self-verification events if treated as `direct_wrong`.
    Probes whose option_set includes `_truncated` automatically route these
    samples there via the StateProbe.classify wrapper."""

    def to_dict(self) -> dict:
        """JSON-friendly representation for persisting in results/."""
        return {
            "raw_output": self.raw_output,
            "tool_calls": self.tool_calls,
            "has_tool_call": self.has_tool_call,
            "is_malformed": self.is_malformed,
            "direct_answer": self.direct_answer,
            "error": self.error,
            "tool_order": self.tool_order,
            "thinking_active": self.thinking_active,
            "api_usage": self.api_usage,
            "was_truncated": self.was_truncated,
        }


class ProbeAdapter(ABC):
    """Per-family invocation wrapper.

    Subclasses must implement `run_sample` and `supported_dial_levels`. The
    default `extract_tool_call` delegates to `src.output_parser.parse_output`
    using the family name; override when a family needs custom extraction
    beyond what the existing parser supports.
    """

    family: str = "abstract"

    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self.short_name = model_cfg.get("short_name", model_cfg.get("id", "unknown"))

    @abstractmethod
    def supported_dial_levels(self) -> list[ThinkingLevel]: ...

    @abstractmethod
    def run_sample(self, spec: ProbeSpec, gen_config: dict, rng: Any) -> ProbeSample: ...

    def extract_tool_call(self, raw: str) -> list[dict]:
        """Default parser - delegates to output_parser.parse_output with family.

        Returns a list of `{name, arguments, raw}` dicts or [] on parse failure.
        """
        from dataclasses import asdict

        from src.output_parser import parse_output

        result = parse_output(raw, self.family)
        return [asdict(tc) for tc in result.tool_calls]

    def close(self) -> None:
        """Release any resources (GPU memory, clients). Default: no-op."""
        pass
