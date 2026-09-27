# AgentProv: Auditing Agentic LLM API Providers via Tool-use Policy Probes

This repository contains the implementation for **AgentProv** (Agentic
Provenance), a tool-use policy based audit for agentic LLM APIs.

## Abstract

Commercial LLM APIs advertise a specific foundation model, but the served
backbone may be silently substituted, quantized, or wrapped, for example to save
deployment costs. All existing audits decide backbone identity from the
text-output channel, which is structurally fragile for agentic APIs because
modern serving stacks discard text and expose only structured actions when the
model calls a tool, and any provider-injected system prompt distorts text
distributions enough that text-channel tests falsely accuse honest providers of
substituting the claimed model. We observe that recent agentic post-training
internalizes tool-use directly into the weights, opening a new audit channel
that the serving stack still exposes and that is invariant to deployment
context. We introduce Agentic Provenance (AgentProv), the first action-based
identity audit for agentic LLM APIs: AgentProv fingerprints a deployed model
through its categorical tool-call distribution and decides identity via an MMD
permutation test. AgentProv catches every substituted model (100% on 630
evaluated checkpoint pairs), while holding the false-positive rate under
system-prompt injection at 7% (vs. 67% for MET and 53% for RUT). On third-party
API endpoints, AgentProv's disagreements with MET are consistent with an
independent token-count side-channel that detects provider-injected hidden
system prompts.

## How it works

Given black-box query access to a **suspect** endpoint claiming to serve model
`M`, and **reference** access to an authoritative instance of `M` (open weights
run locally, or the vendor's official API), AgentProv:

1. **Probes** both endpoints on `K` templates, each offering functionally
   equivalent tools with different names (drawn from real provider docs,
   benchmarks, and MCP server families). Tool order is shuffled per sample with
   a deterministic seed; descriptions are length-matched.
2. **Fingerprints** each endpoint as the empirical categorical distribution over
   tool selections per template (a one-hot mean), with `_no_call` and
   `_malformed` as explicit options.
3. **Tests** the two fingerprints with a delta-kernel MMD^2 statistic, calibrated
   by a `B`-shuffle permutation null, and rejects identity at level `alpha`.

The probe is the union of two template families, merged round-robin into the
headline `K = 20` configuration:

- `s_redundant` - general-tool redundancy (provider-doc / benchmark names)
- `s_init` - equivalent operations across MCP server families

The headline configuration is `K = 20`, `N = 50` samples per template (1,000
queries per endpoint).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is required to run local (HuggingFace) reference models. API endpoints
need an API key, supplied via environment variable (`OPENROUTER_API_KEY`) or an
INI file (see `src/adapters/api_keys.py`). On HPC systems where the module stack
leaks a `PYTHONPATH`, run `unset PYTHONPATH` before activating your environment.

## Usage

Models are registered in `config/models.yaml` by `short_name`. A local model
uses the HuggingFace `id`; an API endpoint adds a `backend:` key.

```bash
# 1. Collect a fingerprint from each endpoint (both probe families each).
python run_probe.py run --state s_redundant --model qwen2.5-7b  --n 50
python run_probe.py run --state s_init      --model qwen2.5-7b  --n 50
python run_probe.py run --state s_redundant --model llama3.1-8b --n 50
python run_probe.py run --state s_init      --model llama3.1-8b --n 50

# 2. Run the equality test over the merged probe.
python run_probe.py test \
    --model-a qwen2.5-7b --model-b llama3.1-8b \
    --states s_redundant,s_init --B 1000 --alpha 0.05
```

The `test` command writes a JSON report under `results/equality/tests/` with a
global decision, per-state breakdown, and per-template (per-block) diagnostics.

For API provider verification, point the suspect at a third-party endpoint:
`config/models.yaml` entries with a `backend:` key (e.g. `llama3.1-8b-api`) are
queried over HTTP and compared against their local reference the same way.


### Hidden-prompt robustness experiment

To simulate a provider-injected hidden system prompt, pass
`--extra-system-prompt "..."` to `run`; the string is prepended to every
template's system prompt and folded into the config fingerprint, so each
condition writes to its own output directory and never collides with the clean
run. The injected conditions used in the paper are in
`config/hidden_prompts.yaml` (empty / short / long). Each condition is tested
against the clean run (no injected prompt), so every rejection is a false
positive on identity:

```bash
# clean control run
python run_probe.py run --state s_redundant --model gemma3-1b --n 50 \
    --out results/clean
# long-injection condition (separate output root)
python run_probe.py run --state s_redundant --model gemma3-1b --n 50 \
    --out results/inject_long \
    --extra-system-prompt "You are an AI assistant deployed by Acme Corp for customer support. Always respond in a friendly, professional tone. Refuse requests unrelated to Acme products."
# test the injected run against the clean run (same model, two output roots)
python run_probe.py test --model-a gemma3-1b --model-b gemma3-1b \
    --states s_redundant --B 1000 \
    --out results/clean --out-b results/inject_long
```

Because the weights and decoding are identical, any rejection here is a false
positive on identity. `--out-b` forces a cross-sample comparison (not a
self-test split) even though both sides carry the same model name.

## Repository layout

```
AgentProv/
├── run_probe.py            CLI entry point (list / smoke / run / test)
├── config/
│   ├── models.yaml         model registry (short_name -> HF id / API backend)
│   ├── tool_sets.yaml      tool schemas per category
│   ├── prompts.yaml        neutral prompts
│   ├── hidden_prompts.yaml injected system-prompt conditions (robustness exp.)
│   └── probes/
│       ├── s_redundant.yaml general-tool templates
│       └── s_init.yaml      MCP-server templates
└── src/
    ├── adapters/           model backends (HF local, OpenAI, OpenRouter)
    ├── harness/            sample collection + fingerprinting (probe_runner)
    ├── probes/             probe definitions (s_redundant, s_init)
    ├── tests/              delta-kernel MMD + permutation null + 3-level report
    ├── tool_formatter.py   tool-schema assembly + per-sample shuffling
    ├── output_parser.py    tool-call extraction
    ├── classifier.py       map a tool call to a categorical option
    ├── model_loader.py     HF model/tokenizer loading
    ├── api_runner.py       OpenAI-compatible client factory
    └── runner.py           config fingerprinting / resume bookkeeping
```

## Reproducibility

Results accumulate across runs and are keyed by a config fingerprint (a hash of
the tool schemas, prompts, generation config, and seed). If the config changes
between runs, a new output directory is created automatically. The RNG state is
advanced deterministically so tool-order shuffles are consistent regardless of
batch size.

## Intended use

AgentProv is an accountability tool to help API customers and third-party
auditors check whether a deployed endpoint serves the model it claims. A reject
decision at `alpha = 0.05` is a statistical statement, not a finding of fraud,
and should be one input to a broader investigation. Auditors should respect
endpoint terms of service and rate limits.
