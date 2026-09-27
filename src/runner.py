"""Configuration fingerprinting and resume bookkeeping.

A config fingerprint is a SHA-256 hash of everything that affects sample
results (tool schemas, prompts, generation parameters, seed). Runs with the
same fingerprint accumulate into the same directory; a changed fingerprint
opens a new timestamped directory so incompatible runs are never merged.
"""

import hashlib
import json
import os
from datetime import datetime

import yaml


def load_config(config_dir: str = "config") -> dict:
    """Load all YAML configuration files."""
    config = {}
    for name in ["models", "tool_sets", "prompts"]:
        with open(os.path.join(config_dir, f"{name}.yaml")) as f:
            config[name] = yaml.safe_load(f)
    return config


def compute_config_fingerprint(tool_sets: dict, prompts_config: dict, gen_config: dict, seed: int) -> str:
    """Compute a SHA-256 hash of the experiment configuration.

    Captures everything that affects trial results: tool schemas, prompt texts,
    generation parameters, and RNG seed. If any of these change, results from
    different runs are not comparable and must not be merged.
    """
    canonical = json.dumps({
        "tool_sets": tool_sets,
        "prompts": prompts_config,
        "generation": gen_config,
        "seed": seed,
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def check_or_create_fingerprint(output_dir: str, fingerprint: str) -> str:
    """Check if existing results are compatible with current config.

    Returns:
        The output_dir to use (may be a new timestamped dir if config changed).
    """
    fp_path = os.path.join(output_dir, "config_fingerprint.json")

    if os.path.exists(fp_path):
        with open(fp_path) as f:
            stored = json.load(f)
        if stored["fingerprint"] == fingerprint:
            return output_dir

        # Config changed: create a new directory.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir = f"{output_dir}_{timestamp}"
        os.makedirs(new_dir, exist_ok=True)
        print(f"\n*** Config changed since last run ***")
        print(f"  Previous config hash: {stored['fingerprint'][:16]}...")
        print(f"  Current  config hash: {fingerprint[:16]}...")
        print(f"  Saving to new directory: {new_dir}")
        _save_fingerprint(new_dir, fingerprint)
        return new_dir

    # No existing fingerprint: save it.
    os.makedirs(output_dir, exist_ok=True)
    _save_fingerprint(output_dir, fingerprint)
    return output_dir


def _save_fingerprint(output_dir: str, fingerprint: str):
    """Save the config fingerprint to the output directory."""
    fp_path = os.path.join(output_dir, "config_fingerprint.json")
    # Use a PID-unique temp file to avoid a race when multiple processes run in parallel.
    tmp_path = fp_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump({
            "fingerprint": fingerprint,
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)
    os.replace(tmp_path, fp_path)
