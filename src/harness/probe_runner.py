"""State-aware, resume-safe probe runner.

Iterates (template x sample) for a single model & probe, persisting each
sample as a JSON row. Reuses ``src.runner``'s config-fingerprint machinery
so re-runs with changed configs land in a new output directory automatically.
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.adapters import get_adapter
from src.adapters.base import ProbeAdapter, ProbeSpec
from src.probes.base import StateProbe, Template
from src.runner import check_or_create_fingerprint, compute_config_fingerprint


_FINGERPRINT_GEN_KEYS = ("temperature", "top_p", "do_sample")
"""Only these gen_config keys participate in the fingerprint.

Rationale: ``max_new_tokens`` differs per-model via ``gen_overrides``
(gpt-5.4 needs 4096 for reasoning, gpt-4o-mini needs 1024, Qwen3 needs
768, etc.), but we don't want those per-model knobs to fragment a probe's
data across directories. The fingerprint should identify
"probe-config-shape" (template semantics + sampling temperature), not
per-model generation budgets. Seed is also excluded (see below).

Note: seed is deliberately excluded. Per-model / per-run seeds should
select which random draws we get, not which fingerprint directory we
write to. Including seed would fragment samples across directories
whenever different models used different seeds."""


def _probe_config_fingerprint(state: str, templates: list[Template],
                               gen_config: dict,
                               extra_system_prompt: str = "") -> str:
    """Per-state fingerprint - tweaking one probe's YAML doesn't invalidate
    other states' data, and per-model ``gen_overrides`` don't fragment
    samples across timestamped subdirs.
    """
    # Capture template content deterministically - template_id + option_set +
    # tools (by name) + history. Full schemas would change fingerprints on
    # cosmetic edits; names are the stable part we care about.
    compact_templates = [
        {
            "template_id": t.template_id,
            "option_set": t.option_set,
            "tool_names": sorted(f["function"]["name"] for f in t.tool_schemas),
            "history_len": len(t.synthetic_history),
            "source": t.source,
        }
        for t in templates
    ]
    # Filter gen_config to only the keys that define sampling semantics.
    gen_for_fingerprint = {k: gen_config[k] for k in _FINGERPRINT_GEN_KEYS if k in gen_config}
    # If an extra system prompt is being injected (e.g., for calibration
    # experiments simulating provider-side prompt injection), include it in
    # the fingerprint so different calibration prompts write to distinct
    # directories instead of overwriting each other.
    prompts_config = (
        {"extra_system_prompt": extra_system_prompt}
        if extra_system_prompt else {}
    )
    return compute_config_fingerprint(
        tool_sets={"state": state, "templates": compact_templates},
        prompts_config=prompts_config,
        gen_config=gen_for_fingerprint,
        seed=0,  # fingerprint-canonical seed; actual RNG is driven per-sample
    )


def run_probe(
    probe: StateProbe,
    model_cfg: dict,
    n_samples_per_template: int,
    out_dir: str,
    gen_config: dict,
    seed: int = 42,
    thinking: str = "default",
    template_ids: list[str] | None = None,
    max_workers: int = 1,
    extra_system_prompt: str = "",
) -> dict:
    """Run a state probe against a single model.

    Resumes from existing per-template sample files. Output layout::

        out_dir/<state>/<config-hash>/
            fingerprint.json
            <short_name>.json   # {template_id -> list of sample dicts}

    Args:
        template_ids: optional subset to run (default: all probe templates).
        max_workers: when >1, run samples within a template concurrently.
            Safe for API adapters (they derive a local rng from
            spec.shuffle_tools_seed and ignore the shared rng arg).
            Local-GPU adapters should keep max_workers=1.

    Returns:
        dict summary: {state, model, samples_written, per_template_counts}.
    """
    adapter: ProbeAdapter = get_adapter(model_cfg)
    short_name = model_cfg.get("short_name", model_cfg.get("id", "unknown"))
    selected_templates = [
        t for t in probe.templates()
        if template_ids is None or t.template_id in set(template_ids)
    ]

    # Fingerprint over the FULL probe.templates() - not the filtered subset.
    # Rationale: filtering by --templates is a runtime convenience (which
    # subset to sample this invocation); it should NOT create a new
    # fingerprint directory. Only YAML edits + gen_config changes should.
    fingerprint = _probe_config_fingerprint(
        probe.state, probe.templates(), gen_config,
        extra_system_prompt=extra_system_prompt,
    )
    state_dir = os.path.join(out_dir, probe.state)
    os.makedirs(state_dir, exist_ok=True)
    # Use the shared fingerprint-directory helper to pick where to write.
    state_dir = check_or_create_fingerprint(state_dir, fingerprint)
    out_path = os.path.join(state_dir, f"{short_name}.json")

    # Load existing results for resume.
    existing: dict[str, list[dict]] = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    t_start = time.time()
    per_template_counts: dict[str, int] = {}
    samples_written = 0
    model_results = dict(existing)

    try:
        for tpl in selected_templates:
            done = len(existing.get(tpl.template_id, []))
            needed = max(0, n_samples_per_template - done)
            per_template_counts[tpl.template_id] = done + needed
            if needed == 0:
                continue

            rng = random.Random(seed + abs(hash(tpl.template_id)) % 10_000)
            # Advance RNG past existing samples so shuffles are deterministic.
            from src.tool_formatter import shuffle_tools as _sh
            for _ in range(done):
                _sh(tpl.tool_schemas, rng)

            # If --extra-system-prompt is set, prepend it to the template's
            # natural system prompt (simulates provider-side prompt injection
            # ahead of the user-visible prompt).
            effective_system_prompt = (
                (extra_system_prompt + "\n\n" + tpl.system_prompt)
                if extra_system_prompt else tpl.system_prompt
            )

            def _run_one(sample_idx: int) -> dict:
                spec = ProbeSpec(
                    system_prompt=effective_system_prompt,
                    tools=tpl.tool_schemas,
                    history=tpl.synthetic_history,
                    user_turn=tpl.elicitation_turn,
                    thinking=thinking,  # type: ignore
                    shuffle_tools_seed=seed + sample_idx,
                )
                t0 = time.perf_counter()
                sample = adapter.run_sample(spec, gen_config, rng)
                inference_seconds = round(time.perf_counter() - t0, 3)
                option = probe.classify(sample, tpl)
                row = sample.to_dict()
                row.update({
                    "template_id": tpl.template_id,
                    "state": probe.state,
                    "sample_idx": sample_idx,
                    "model": short_name,
                    "option": option,
                    "inference_seconds": inference_seconds,
                })
                return row

            indices = [done + i for i in range(needed)]
            if max_workers > 1 and len(indices) >= 2:
                # Warm up serially so the adapter's one-time setup (tokenizer
                # load, HTTP client init) doesn't race across worker threads.
                first = _run_one(indices[0])
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    rest = list(pool.map(_run_one, indices[1:]))
                new_rows = [first] + rest
            else:
                new_rows = [_run_one(i) for i in indices]
            samples_written += len(new_rows)

            model_results[tpl.template_id] = existing.get(tpl.template_id, []) + new_rows

            # Read-modify-write the per-(model, probe) JSON file under an
            # advisory lock so two concurrent writers (e.g. parallel jobs
            # sharded by template) can't clobber each other's templates.
            # The tmp filename is unique per writer to avoid os.replace races.
            tmp = "{}.tmp.{}.{}".format(out_path, os.getpid(), threading.get_ident())
            lock_path = out_path + ".lock"
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                try:
                    if os.path.exists(out_path):
                        with open(out_path) as f:
                            on_disk = json.load(f)
                        on_disk[tpl.template_id] = model_results[tpl.template_id]
                        model_results = on_disk
                    with open(tmp, "w") as f:
                        json.dump(model_results, f, indent=2, default=str)
                    os.replace(tmp, out_path)
                finally:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    finally:
        # Always release GPU memory / API clients, even on exception.
        try:
            adapter.close()
        except Exception as e:
            print(f"[probe_runner] adapter.close() raised: {e!r} - continuing", flush=True)

    return {
        "state": probe.state,
        "model": short_name,
        "out_path": out_path,
        "samples_written": samples_written,
        "elapsed_seconds": time.time() - t_start,
        "per_template_counts": per_template_counts,
    }


def load_probe_samples(state: str, out_dir: str, short_name: str | None = None):
    """Load all per-model probe-sample files for ``state``.

    If ``short_name`` is given, load only that one. Handles both layouts
    produced by ``check_or_create_fingerprint``:
      - fresh run: files live directly under ``out_dir/<state>/``
      - config-change run: files live under ``out_dir/<state>/<timestamp>/``
    """
    state_root = os.path.join(out_dir, state)
    if not os.path.isdir(state_root):
        return {}

    # Collect candidate directories to read from:
    #   1. state_root itself if it has a config_fingerprint.json
    #   2. any subdirectories with a config_fingerprint.json
    candidates: list[str] = []
    if os.path.exists(os.path.join(state_root, "config_fingerprint.json")):
        candidates.append(state_root)
    for item in os.listdir(state_root):
        full = os.path.join(state_root, item)
        if os.path.isdir(full) and os.path.exists(
            os.path.join(full, "config_fingerprint.json")
        ):
            candidates.append(full)

    if not candidates:
        return {}
    # Prefer the non-subdir (most recent canonical) layout; otherwise the
    # most recently modified timestamped one.
    if state_root in candidates:
        chosen = state_root
    else:
        chosen = max(candidates, key=os.path.getmtime)

    out = {}
    # Skip framework-artifact JSONs so only per-model sample files land in
    # the returned dict.
    _FRAMEWORK_ARTIFACTS = {
        "config_fingerprint.json", "metadata.json",
    }
    for fname in os.listdir(chosen):
        if not fname.endswith(".json") or fname in _FRAMEWORK_ARTIFACTS:
            continue
        if short_name and fname != f"{short_name}.json":
            continue
        with open(os.path.join(chosen, fname)) as f:
            out[fname[:-5]] = json.load(f)
    return out
