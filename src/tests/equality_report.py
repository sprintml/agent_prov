"""3-level equality report: global -> per-state -> per-block.

Given probe samples for two models (or one model split in half for
self-test), compute:
  - Global MMD^2 across all states and templates  -> headline decision
  - Per-state MMD^2 (each probe independently)    -> interpretability
  - Per-block ||d_a - d_b||^2 with raw counts     -> finest grain
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from src.tests.mmd import (
    decision,
    delta_kernel_mmd2,
    permutation_null,
    samples_to_probs,
)


# -- Helpers -----------------------------------------------------------------

def canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Alphabetically sorted pair -- ensures A-vs-B == B-vs-A."""
    return (a, b) if a <= b else (b, a)


def pair_filename(a: str, b: str) -> str:
    ca, cb = canonical_pair(a, b)
    return f"{ca}__vs__{cb}.json"


def _state_split_seed(base_seed: int, state: str) -> int:
    """Deterministic per-state seed for self-test splits."""
    h = hashlib.sha256(f"{base_seed}:{state}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def split_samples(
    samples_by_tpl: dict, seed: int,
) -> tuple[dict, dict]:
    """Split each template's samples randomly in half for self-test."""
    rng = np.random.default_rng(seed)
    half_a: dict = {}
    half_b: dict = {}
    for tpl_id, rows in samples_by_tpl.items():
        n = len(rows)
        idx = rng.permutation(n)
        mid = n // 2
        half_a[tpl_id] = [rows[int(i)] for i in idx[:mid]]
        half_b[tpl_id] = [rows[int(i)] for i in idx[mid:]]
    return half_a, half_b


# -- Matrix building ---------------------------------------------------------

def build_onehot(
    samples_by_tpl: dict,
    template_blocks: list[tuple[str, list[str]]],
    drop_truncated: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Build (N, D) one-hot matrix.

    Args:
        samples_by_tpl: {template_id: [sample_dicts]}
        template_blocks: [(template_id, option_set), ...]
        drop_truncated: if True, skip samples that were truncated
            (``was_truncated=True`` OR ``option == "_truncated"``).
            This removes truncation-rate artifacts caused by
            max_new_tokens mismatches between local and API inference.

    Returns:
        (N, D) float32 matrix, list of per-block sizes.
    """
    block_sizes = [len(opts) for _, opts in template_blocks]
    D = sum(block_sizes)

    offsets: list[int] = []
    off = 0
    for bs in block_sizes:
        offsets.append(off)
        off += bs

    rows: list[np.ndarray] = []
    for (tid, option_set), offset in zip(template_blocks, offsets):
        for r in samples_by_tpl.get(tid, []):
            if drop_truncated and (
                r.get("was_truncated", False)
                or r.get("option") == "_truncated"
            ):
                continue
            row = np.zeros(D, dtype=np.float32)
            lab = r.get("option", "off_option")
            if lab in option_set:
                row[offset + option_set.index(lab)] = 1.0
            else:
                for rescue in ("off_option", "_other", "_malformed"):
                    if rescue in option_set:
                        row[offset + option_set.index(rescue)] = 1.0
                        break
            rows.append(row)

    mat = np.stack(rows) if rows else np.zeros((0, D), dtype=np.float32)
    return mat, block_sizes


# -- Per-block standalone metric ---------------------------------------------

def block_l2_sq(
    counts_a: dict, counts_b: dict, option_set: list[str],
) -> float:
    """Standalone ||d_a - d_b||^2 for one block from raw counts."""
    n_a = sum(counts_a.get(o, 0) for o in option_set)
    n_b = sum(counts_b.get(o, 0) for o in option_set)
    if n_a == 0 or n_b == 0:
        return 0.0
    return sum(
        (counts_a.get(o, 0) / n_a - counts_b.get(o, 0) / n_b) ** 2
        for o in option_set
    )


# -- MMD^2 test wrapper -------------------------------------------------------

def _run_test(mat_a, mat_b, block_sizes, alpha, B, seed):
    """Run MMD^2 + permutation null, return result dict."""
    p_a = samples_to_probs(mat_a, block_sizes)
    p_b = samples_to_probs(mat_b, block_sizes)
    observed = delta_kernel_mmd2(p_a, p_b, block_sizes)
    null = permutation_null(mat_a, mat_b, block_sizes, B=B, seed=seed)
    dec = decision(observed, null, alpha=alpha)
    return {
        "observed_mmd2": float(observed),
        "crit": float(dec["crit"]),
        "pvalue": float(dec["pvalue"]),
        "reject": bool(dec["reject"]),
    }


# -- Main entry point --------------------------------------------------------

def compute_report(
    model_a: str,
    model_b: str,
    state_inputs: list[tuple],
    alpha: float = 0.05,
    B: int = 1000,
    seed: int = 42,
    is_self_test: bool = False,
    split_seed: int | None = None,
    drop_truncated: bool = False,
    min_effect_size: float = 0.0,
) -> dict:
    """Compute 3-level equality report.

    Args:
        model_a, model_b: short names (will be canonically sorted).
        state_inputs: list of (state_name, probe, samples_a_by_tpl,
            samples_b_by_tpl) tuples. For self-test callers must
            pre-split; this function does NOT split internally.
        alpha: significance level.
        B: number of permutations.
        seed: RNG seed for permutation null.
        is_self_test: flag for self-test metadata.
        split_seed: recorded in output when is_self_test is True.
        drop_truncated: remap ``_truncated`` to the block's rescue
            category before building the one-hot matrix.
        min_effect_size: practical significance floor for MMD^2. Reject
            only when MMD^2 exceeds BOTH the permutation critical value
            AND this floor. 0.0 disables (original behavior).

    Returns:
        Structured dict with global / per_state / per_block.
    """
    ca, cb = canonical_pair(model_a, model_b)

    # -- Gather blocks per state ---------------------------------------------
    all_blocks: list[tuple[str, list[str]]] = []
    block_to_state: dict[str, str] = {}
    state_data: list[tuple] = []  # (state, tpl_blocks, sa, sb)

    for state_name, probe, sa, sb in state_inputs:
        tpl_blocks: list[tuple[str, list[str]]] = []
        for tpl in probe.templates():
            tid = tpl.template_id
            if not sa.get(tid) and not sb.get(tid):
                continue
            tpl_blocks.append((tid, list(tpl.option_set)))
            block_to_state[tid] = state_name
        if tpl_blocks:
            state_data.append((state_name, tpl_blocks, sa, sb))
            all_blocks.extend(tpl_blocks)

    if not all_blocks:
        raise ValueError("No samples found for any state/model combination")

    # Merge samples across states (template IDs are state-prefixed, no clash)
    merged_a: dict = {}
    merged_b: dict = {}
    for _, _, sa, sb in state_data:
        merged_a.update(sa)
        merged_b.update(sb)

    # -- Global --------------------------------------------------------------
    mat_a, block_sizes = build_onehot(merged_a, all_blocks, drop_truncated)
    mat_b, _ = build_onehot(merged_b, all_blocks, drop_truncated)

    global_result = _run_test(mat_a, mat_b, block_sizes, alpha, B, seed)
    if min_effect_size > 0.0 and global_result["observed_mmd2"] < min_effect_size:
        global_result["reject"] = False
        global_result["reject_reason"] = "below_min_effect_size"
    global_result.update({
        "n_a": int(mat_a.shape[0]),
        "n_b": int(mat_b.shape[0]),
        "D_total": int(sum(block_sizes)),
        "n_states": len(state_data),
        "n_blocks": len(all_blocks),
    })

    # -- Per-state (Bonferroni-corrected) ------------------------------------
    n_states_tested = len(state_data)
    alpha_bonf = alpha / max(n_states_tested, 1)

    per_state_report: dict = {}
    for state_name, tpl_blocks, sa, sb in state_data:
        smat_a, sbs = build_onehot(sa, tpl_blocks, drop_truncated)
        smat_b, _ = build_onehot(sb, tpl_blocks, drop_truncated)
        if smat_a.shape[0] == 0 or smat_b.shape[0] == 0:
            continue
        sr = _run_test(smat_a, smat_b, sbs, alpha_bonf, B, seed)
        sr["alpha_bonferroni"] = alpha_bonf
        if min_effect_size > 0.0 and sr["observed_mmd2"] < min_effect_size:
            sr["reject"] = False
            sr["reject_reason"] = "below_min_effect_size"
        sr.update({
            "n_a": int(smat_a.shape[0]),
            "n_b": int(smat_b.shape[0]),
            "n_blocks": len(tpl_blocks),
        })
        per_state_report[state_name] = sr

    # -- Per-block -----------------------------------------------------------
    per_block_report: dict = {}
    for tid, option_set in all_blocks:
        sa_rows = merged_a.get(tid, [])
        sb_rows = merged_b.get(tid, [])
        counts_a = Counter(r.get("option", "off_option") for r in sa_rows)
        counts_b = Counter(r.get("option", "off_option") for r in sb_rows)
        n_a = len(sa_rows)
        n_b = len(sb_rows)
        per_block_report[tid] = {
            "state": block_to_state.get(tid, "unknown"),
            "block_mmd2": block_l2_sq(counts_a, counts_b, option_set),
            "block_size": len(option_set),
            "n_a": n_a,
            "n_b": n_b,
            "option_set": option_set,
            "counts_a": {o: counts_a.get(o, 0) for o in option_set},
            "counts_b": {o: counts_b.get(o, 0) for o in option_set},
            "dist_a": {o: counts_a.get(o, 0) / max(n_a, 1)
                       for o in option_set},
            "dist_b": {o: counts_b.get(o, 0) / max(n_b, 1)
                       for o in option_set},
        }

    # -- Assemble ------------------------------------------------------------
    report: dict = {
        "model_a": ca,
        "model_b": cb,
        "is_self_test": is_self_test,
        "alpha": alpha,
        "B": B,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global": global_result,
        "per_state": per_state_report,
        "per_block": per_block_report,
    }
    if is_self_test:
        report["split_seed"] = split_seed
    return report


def compute_report_from_dir(
    model_a: str,
    model_b: str,
    states: list[str],
    out_dir: str,
    config_dir: str = "config",
    alpha: float = 0.05,
    B: int = 1000,
    seed: int = 42,
    split_seed: int = 42,
    drop_truncated: bool = False,
    min_effect_size: float = 0.0,
    out_dir_b: str | None = None,
) -> dict:
    """Load probes + samples from disk, then compute report.

    If ``out_dir_b`` is given, ``model_a`` is loaded from ``out_dir`` and
    ``model_b`` from ``out_dir_b``, and the comparison is always cross-sample
    (never a self-test split) even when the two model names are identical.
    This is how the hidden-prompt experiment compares a clean run against an
    injected run of the same model (each lives in its own output directory).
    """
    from src.harness.probe_runner import load_probe_samples
    from src.probes import load_probe

    if out_dir_b is not None:
        is_self = False
    else:
        ca, cb = canonical_pair(model_a, model_b)
        is_self = ca == cb

    state_inputs: list[tuple] = []
    for state in states:
        probe = load_probe(state, config_dir=config_dir)

        if out_dir_b is not None:
            sa = load_probe_samples(state, out_dir).get(model_a, {})
            sb = load_probe_samples(state, out_dir_b).get(model_b, {})
            if not sa and not sb:
                continue
        else:
            all_samples = load_probe_samples(state, out_dir)
            if is_self:
                if ca not in all_samples:
                    continue
                sseed = _state_split_seed(split_seed, state)
                sa, sb = split_samples(all_samples[ca], sseed)
            else:
                sa = all_samples.get(ca, {})
                sb = all_samples.get(cb, {})
                if not sa and not sb:
                    continue

        state_inputs.append((state, probe, sa, sb))

    return compute_report(
        model_a=model_a,
        model_b=model_b,
        state_inputs=state_inputs,
        alpha=alpha,
        B=B,
        seed=seed,
        is_self_test=is_self,
        split_seed=split_seed if is_self else None,
        drop_truncated=drop_truncated,
        min_effect_size=min_effect_size,
    )


def save_report(report: dict, out_dir: str) -> str:
    """Persist report as JSON under out_dir/tests/. Returns file path."""
    tests_dir = os.path.join(out_dir, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    fname = pair_filename(report["model_a"], report["model_b"])
    path = os.path.join(tests_dir, fname)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, path)
    return path
