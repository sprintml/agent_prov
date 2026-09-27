"""S_init - MCP-registry tool-selection prior.

Sibling measurement to S_redundant but the tool *names* come from MCP server
schemas rather than provider documentation. Measures exposure to agent-SFT
mixtures (Kimi K2's 3000+ MCP tools, xLAM, ToolBench).

Same classification mechanism as S_redundant - delegate to the existing
``classifier.trial_to_choice`` logic; option_set is tools + _no_call +
_malformed + _other.
"""

from __future__ import annotations

import yaml

from src.adapters.base import ProbeSample

from . import register
from .base import StateProbe, Template


class SInitProbe(StateProbe):
    state = "s_init"
    pattern = "A"

    @classmethod
    def from_yaml(cls, path: str) -> "SInitProbe":
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
                source=t.get("source", "modelcontextprotocol/servers"),
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


register("s_init", SInitProbe)
