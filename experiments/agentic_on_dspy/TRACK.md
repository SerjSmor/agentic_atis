# Agentic On DSPy Track

## Purpose

This track measures an agent-assisted DSPy workflow.

The intended interpretation is:

- DSPy provides the optimization substrate
- the agent is allowed to inspect DSPy outputs and modify the DSPy system

It should be compared against:

- `experiments/agentic`
- `experiments/dspy`

## Status

Current status: `scratch`

This track is intentionally adaptive.

Unlike `experiments/dspy`, it is not restricted to a fixed hyperparameter plan. Iterations may be driven by error analysis, artifact inspection, and redesign decisions made after each run.

## Canonical Data

This track consumes the frozen split in `shared/data/` directly.

Frozen source files:

- `shared/data/train_subset.jsonl`
- `shared/data/val_subset.jsonl`
- `shared/data/test_subset.jsonl`

Rules:

- iterate on `val`
- do not use `test` during development
- use `test` only for a final locked run

## Allowed Changes

Allowed:

- all DSPy-level changes allowed in the pure DSPy track
- agent-authored prompt guidance
- signature changes
- label descriptions
- optimizer strategy changes
- choosing, replacing, or redesigning the DSPy optimizer
- inspection of per-example DSPy outputs
- targeted modification of the DSPy program after analysis

Not allowed:

- changing the canonical split
- tuning directly on `test`
- using test predictions to drive prompt or program changes
- changing the main task model away from `gpt-4.1-nano`

## Model Constraint

The main task model for this track must be:

- `gpt-4.1-nano`

Teacher setting:

- allowed
- must be logged explicitly when used

## Optimizer Guidance

This track is explicitly allowed to explore optimizer choice as part of the agentic search space.

Optimizer choice, signature design, prompt or guidance structure, and program shape may all change across runs when justified by error analysis.

Recommended optimizers to consider:

- `MIPROv2`
- `GEPA`
- `BootstrapFewShotWithRandomSearch`

Requirement:

- always log `optimizer_name` and relevant optimizer settings in local results and W&B

## Baseline Command

Run from repo root:

```bash
set -a && source .env && set +a
cd experiments/agentic_on_dspy
../../venv/bin/inv run
```

Final test run:

```bash
set -a && source .env && set +a
cd experiments/agentic_on_dspy
../../venv/bin/inv run --eval-split=test --wandb-job-type=final-test
```

## Logging

Primary local outputs:

- `experiments/agentic_on_dspy/results.tsv`
- `experiments/agentic_on_dspy/runs/`
- `experiments/agentic_on_dspy/programs/`
- `experiments/agentic_on_dspy/prompts/`
- `experiments/agentic_on_dspy/dev_sessions.tsv`

W&B project:

- `agentic-atis-compare-agentic-on-dspy`

Recommended W&B job types:

- `val-baseline`
- `val-iteration`
- `final-test`

## Effort Tracking

Use `experiments/agentic_on_dspy/dev_sessions.tsv` to record development effort.

Suggested row fields:

- `session_id`
- `track`
- `started_at`
- `ended_at`
- `minutes`
- `files_touched`
- `runs_triggered`
- `notes`

## Success Metric

Primary metric:

- `macro_f1`

Secondary metrics:

- `weighted_f1`
- `micro_f1`

Also track:

- tokens
- estimated cost
- compile runtime
- eval runtime
- development effort recorded in `dev_sessions.tsv`

## Data Preparation

There is one canonical prepare path for the whole repo:

- `prepare.py`

This track must not create or use a separate train/val split.
