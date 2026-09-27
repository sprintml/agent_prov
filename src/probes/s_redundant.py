"""S_redundant: tool-preference prior on equivalence-class tools.

Templates span the tool categories in ``config/tool_sets.yaml``, each with a
redundant tool set (functionally identical tools with different names).
Pattern A; ``synthetic_history=[]``. Each template's option_set is
``tools + [_no_call, _malformed, _other]``.

Classification reuses ``src.classifier.trial_to_choice``.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

from src.adapters.base import ProbeSample

from . import register
from .base import StateProbe, Template


class SRedundantProbe(StateProbe):
    state = "s_redundant"
    pattern = "A"

    @classmethod
    def from_yaml(cls, path: str) -> "SRedundantProbe":
        with open(path) as f:
            data = yaml.safe_load(f)
        templates = [
            Template(
                template_id=t["template_id"],
                system_prompt=t["system_prompt"],
                tool_schemas=t["tool_schemas"],
                synthetic_history=t.get("synthetic_history", []),
                elicitation_turn=t["elicitation_turn"],
                option_set=t["option_set"],
                source=t.get("source", "config/tool_sets.yaml + config/prompts.yaml"),
                pattern="A",
                extras=t.get("extras", {}),
            )
            for t in data["templates"]
        ]
        return cls(templates)

    def _classify_impl(self, sample: ProbeSample, template: Template) -> str:
        from src.classifier import trial_to_choice

        tools_in_option_set = [
            o for o in template.option_set
            if o not in ("_no_call", "_malformed", "_other")
        ]
        trial = {
            "is_malformed": sample.is_malformed,
            "direct_answer": sample.direct_answer,
            "tool_calls": sample.tool_calls,
        }
        choice = trial_to_choice(trial, tools_in_option_set)
        return choice if choice in template.option_set else "_other"


register("s_redundant", SRedundantProbe)
