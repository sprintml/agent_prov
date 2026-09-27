"""StateProbe ABC + shared dataclasses.

A probe owns everything state-specific: the template library, the mapping
from ProbeSample -> option label, and (optionally) a MockEnv. Probes are
decoupled from adapters - the probe defines ``what to ask``, the adapter
defines ``how to render the ask for a family``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.adapters.base import ProbeSample


@dataclass
class Template:
    """A single state probe template - the doc's 5-tuple + metadata."""

    template_id: str
    system_prompt: str
    tool_schemas: list[dict]
    synthetic_history: list[dict]
    elicitation_turn: str
    option_set: list[str]
    source: str = "unknown"
    pattern: str = "A"  # "A" = natural tool-schema selection, "B" = post-hoc classification
    # Free-form extras consumed by probe subclasses (e.g. action_map for
    # S_post_err, mock_env_responses for S_exec_feedback).
    extras: dict = field(default_factory=dict)


class StateProbe(ABC):
    """Base class for state-conditional probes."""

    state: str = "abstract"
    pattern: str = "A"

    def __init__(self, templates: list[Template]):
        self._templates = templates

    @classmethod
    @abstractmethod
    def from_yaml(cls, path: str) -> "StateProbe": ...

    def templates(self) -> list[Template]:
        return list(self._templates)

    def classify(self, sample: ProbeSample, template: Template) -> str:
        """Map one ProbeSample to an option label. Public entry point.

        Handles cross-cutting concerns (truncation) before delegating to the
        probe-specific :meth:`_classify_impl`. Subclasses implement
        :meth:`_classify_impl`, not this method.

        Truncation: when the adapter flags ``sample.was_truncated=True`` AND
        the template's ``option_set`` includes ``_truncated``, returns
        ``_truncated`` without running classification - since the reasoning
        trace was cut off, answer-extraction is unreliable and treating the
        sample as "direct_wrong" would bias against reasoning models.
        """
        if getattr(sample, "was_truncated", False) and "_truncated" in template.option_set:
            return "_truncated"
        return self._classify_impl(sample, template)

    @abstractmethod
    def _classify_impl(self, sample: ProbeSample, template: Template) -> str:
        """Map one ProbeSample to an option label in the template's option_set.

        Must always return a label that appears in ``template.option_set``.
        Use ``off_option`` as a catch-all for noncompliant outputs.
        """

    def mock_env(self, template: Template):  # -> "MockEnv | None"
        """Optional: provide a MockEnv for multi-turn probes. Default: None."""
        return None

    # ------------------------------------------------------------------
    # Helpers used by concrete subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _load_tool_schemas(
        tool_sets: dict, categories: list[str]
    ) -> list[dict]:
        """Build OpenAI-style tool schemas from a list of category names."""
        out: list[dict] = []
        for cat in categories:
            spec = tool_sets["categories"][cat]
            for t in spec["tools"]:
                out.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                })
        return out
