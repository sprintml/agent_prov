"""Centralized API key loading for external providers."""

from __future__ import annotations

import configparser
import os
from pathlib import Path

_ENV_VARS = {
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Default INI path under XDG_CONFIG_HOME (or ~/.config). Override with
# API_KEY_INI_PATH env var. No HPC-specific path is baked in.
_DEFAULT_INI_PATH = str(
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    / "agentic_fp"
    / "api_keys.ini"
)


def load_api_key(provider: str) -> str:
    """Load API key for *provider* (``'google'`` or ``'openrouter'``).

    Priority:
      1. Environment variable (GOOGLE_API_KEY / OPENROUTER_API_KEY)
      2. INI file at $API_KEY_INI_PATH if set, else
         $XDG_CONFIG_HOME/agentic_fp/api_keys.ini (default ~/.config/...)
    """
    env_var = _ENV_VARS.get(provider)
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val.strip()

    ini_path = os.environ.get("API_KEY_INI_PATH", _DEFAULT_INI_PATH)
    if os.path.isfile(ini_path):
        cfg = configparser.ConfigParser()
        cfg.read(ini_path)
        if cfg.has_option(provider, "api_key"):
            return cfg.get(provider, "api_key").strip()

    raise ValueError(
        f"No API key for '{provider}'. "
        f"Set {env_var} or add [{provider}] api_key=... to {ini_path}"
    )
