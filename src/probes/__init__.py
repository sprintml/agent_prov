"""StateProbe registry + YAML loader.

Each probe module (``s_redundant``, ``s_parallel``, ``s_post_err``, ...) is
imported at package-load time and registers itself. New states add a single
YAML under ``config/probes/`` plus a ~50-line Python file - no framework
changes.
"""

from __future__ import annotations

from .base import StateProbe, Template

_REGISTRY: dict[str, type[StateProbe]] = {}


def register(name: str, cls: type[StateProbe]) -> None:
    _REGISTRY[name] = cls


def list_states() -> list[str]:
    return sorted(_REGISTRY.keys())


def load_probe(state_name: str, config_dir: str = "config") -> StateProbe:
    """Instantiate a registered probe from its YAML under config_dir/probes/."""
    import os

    if state_name not in _REGISTRY:
        raise KeyError(f"Unknown state '{state_name}'. Registered: {list_states()}")
    path = os.path.join(config_dir, "probes", f"{state_name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Probe YAML not found: {path}")
    return _REGISTRY[state_name].from_yaml(path)


# The paper's tool-selection probe is the union of two template families,
# merged round-robin into the K=20 headline configuration:
#   s_redundant - general-tool redundancy (provider-doc / benchmark names)
#   s_init      - equivalent operations across MCP server families
from . import s_redundant  # noqa: F401,E402
from . import s_init  # noqa: F401,E402

__all__ = [
    "StateProbe", "Template",
    "register", "list_states", "load_probe",
]
