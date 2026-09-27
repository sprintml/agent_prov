"""Unified CLI for the AgentProv tool-selection equality test.

Subcommands:
    list  - show registered probes + adapters
    smoke - run N samples of one probe against one model (for wiring checks)
    run   - collect probe samples for one configured model
    test  - MMD permutation equality test (global / per-state / per-block)
            between two endpoints, with JSON output

Example:
    # 1. Collect a fingerprint from a local reference and a suspect endpoint
    python run_probe.py run --state s_redundant --model qwen2.5-7b --n 50
    python run_probe.py run --state s_init      --model qwen2.5-7b --n 50
    python run_probe.py run --state s_redundant --model qwen2.5-7b-api --n 50
    python run_probe.py run --state s_init      --model qwen2.5-7b-api --n 50

    # 2. Run the equality test over the merged K=20 probe
    python run_probe.py test --model-a qwen2.5-7b --model-b qwen2.5-7b-api \
        --states s_redundant,s_init --B 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the bundled `src` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

from src.adapters import list_families
from src.harness.probe_runner import run_probe
from src.probes import list_states, load_probe


def _load_models_cfg(config_dir: str) -> dict:
    with open(os.path.join(config_dir, "models.yaml")) as f:
        return yaml.safe_load(f)


def _find_model_cfg(models_cfg: dict, short_name: str) -> dict:
    for m in models_cfg["models"]:
        if m["short_name"] == short_name:
            return m
    raise KeyError(f"Model '{short_name}' not found in models.yaml")


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------

def cmd_list(args):
    print("Registered adapters:", list_families())
    print("Registered probes:  ", list_states())
    for state in list_states():
        try:
            probe = load_probe(state, config_dir=args.config_dir)
            print(f"  {state}: {len(probe.templates())} templates")
        except FileNotFoundError:
            print(f"  {state}: (no YAML at config/probes/{state}.yaml)")


# --------------------------------------------------------------------------
# smoke / run
# --------------------------------------------------------------------------

def cmd_smoke(args):
    cmd_run(args, smoke=True)


def cmd_run(args, smoke: bool = False):
    models_cfg = _load_models_cfg(args.config_dir)
    probe = load_probe(args.state, config_dir=args.config_dir)
    model_cfg = _find_model_cfg(models_cfg, args.model)
    gen_config = dict(models_cfg.get("generation", {}))
    if "gen_overrides" in model_cfg:
        gen_config.update(model_cfg["gen_overrides"])

    n = args.n if not smoke else max(args.n or 0, 3)
    tpl_ids = args.templates.split(",") if args.templates else None

    summary = run_probe(
        probe=probe,
        model_cfg=model_cfg,
        n_samples_per_template=n,
        out_dir=args.out,
        gen_config=gen_config,
        seed=args.seed,
        thinking=args.thinking,
        template_ids=tpl_ids,
        max_workers=args.workers,
        extra_system_prompt=args.extra_system_prompt,
    )
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------
# test (3-level equality test)
# --------------------------------------------------------------------------

def cmd_test(args):
    from src.tests.equality_report import (
        compute_report_from_dir, pair_filename, save_report,
    )

    if args.states == "all":
        states = list_states()
    else:
        states = [s.strip() for s in args.states.split(",")]

    if args.exclude_states:
        excluded = {s.strip() for s in args.exclude_states.split(",")}
        states = [s for s in states if s not in excluded]

    # When comparing two output roots (e.g. clean vs injected), save the
    # report alongside the second (suspect) directory so each condition's
    # report lives with its own samples.
    report_dir = args.out_b or args.out

    # Check for existing report (skip unless --force)
    fname = pair_filename(args.model_a, args.model_b)
    existing = os.path.join(report_dir, "tests", fname)
    if os.path.exists(existing) and not args.force:
        print(f"Report already exists: {existing}", file=sys.stderr)
        print("Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    report = compute_report_from_dir(
        model_a=args.model_a,
        model_b=args.model_b,
        states=states,
        out_dir=args.out,
        out_dir_b=args.out_b,
        config_dir=args.config_dir,
        alpha=args.alpha,
        B=args.B,
        seed=args.seed,
        split_seed=args.split_seed,
        drop_truncated=args.drop_truncated,
        min_effect_size=args.min_effect_size,
    )

    path = save_report(report, report_dir)

    # Print summary
    g = report["global"]
    label = "SELF-TEST" if report["is_self_test"] else "EQUALITY TEST"
    print(f"{label}")
    print(f"model_a: {report['model_a']}  (N={g['n_a']})")
    print(f"model_b: {report['model_b']}  (N={g['n_b']})")
    print(f"states:  {g['n_states']}  blocks: {g['n_blocks']}  D: {g['D_total']}")
    if args.drop_truncated:
        print("  [truncated samples dropped]")
    if args.min_effect_size > 0:
        print(f"  [min effect size: {args.min_effect_size}]")
    print()
    print(f"GLOBAL  MMD2={g['observed_mmd2']:.6f}  crit={g['crit']:.6f}  "
          f"p={g['pvalue']:.4f}  "
          f"{'REJECT' if g['reject'] else 'fail to reject'}")
    print()
    for state, sr in sorted(report["per_state"].items()):
        tag = "REJECT" if sr["reject"] else "fail  "
        reason = f"  ({sr['reject_reason']})" if sr.get("reject_reason") else ""
        print(f"  {state:30s}  MMD2={sr['observed_mmd2']:.6f}  "
              f"p={sr['pvalue']:.4f}  {tag}{reason}")
    print(f"\nSaved: {path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(prog="run_probe")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--out", default="results/equality")
    ap.add_argument("--seed", type=int, default=42)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    for name in ("smoke", "run"):
        p = sub.add_parser(name)
        p.add_argument("--state", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--n", type=int, default=50)
        p.add_argument("--thinking", default="default",
                       choices=["off", "low", "medium", "high", "default"])
        p.add_argument("--templates", default=None,
                       help="Comma-separated subset of template_ids.")
        p.add_argument("--workers", type=int, default=1,
                       help="Concurrent samples within a template (API only; keep 1 for local GPU).")
        p.add_argument("--extra-system-prompt", default="",
                       help="String prepended to every template's system_prompt. "
                            "Simulates provider-side hidden prompt injection for "
                            "calibration / robustness experiments. The extra prompt is "
                            "part of the config fingerprint so different prompts write "
                            "to distinct directories.")

    pt = sub.add_parser("test")
    pt.add_argument("--model-a", required=True)
    pt.add_argument("--model-b", required=True)
    pt.add_argument("--out-b", default=None,
                    help="Second output root. If set, model-a is read from "
                         "--out and model-b from --out-b, always as a "
                         "cross-sample comparison (used for the hidden-prompt "
                         "experiment: clean run vs injected run of the same model).")
    pt.add_argument("--states", default="all",
                    help="Comma-separated state names, or 'all'.")
    pt.add_argument("--exclude-states", default=None,
                    help="Comma-separated states to exclude.")
    pt.add_argument("--B", type=int, default=1000)
    pt.add_argument("--alpha", type=float, default=0.05)
    pt.add_argument("--split-seed", type=int, default=42,
                    help="RNG seed for self-test sample splitting.")
    pt.add_argument("--drop-truncated", action="store_true",
                    help="Remove truncated samples before testing (fixes max_new_tokens mismatch).")
    pt.add_argument("--min-effect-size", type=float, default=0.0,
                    help="Practical significance floor for MMD^2. Reject only above this AND crit.")
    pt.add_argument("--force", action="store_true",
                    help="Overwrite existing report.")

    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "test":
        cmd_test(args)


if __name__ == "__main__":
    main()
