# DSPy Track

## Purpose

This track measures a pure DSPy ATIS intent-classification workflow using DSPy optimization.

It should be compared against:

- `experiments/agentic`
- `experiments/agentic_on_dspy`

## Status

Current status: `scratch`

Run-budget rule for this track:

- exactly `5` validation runs
- all `5` runs use `MIPROv2`
- runs differ only by fixed hyperparameter settings
- no adaptive agent-style redesign between runs

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

- DSPy module shape
- optimizer settings
- switching among approved DSPy optimizers when explicitly logged
- teacher model choice
- label guidance or signature design
- number of demos or trials

Not allowed:

- changing the canonical split
- tuning directly on `test`
- importing the prompt-only agentic workflow as the optimization method
- changing the main task model away from `gpt-4.1-nano`

## Model Constraint

The main task model for this track must be:

- `gpt-4.1-nano`

Teacher setting:

- allowed
- must be logged explicitly when used

## Optimizer Guidance

Fixed optimizer for the clean comparison:

- `MIPROv2`

This track is intentionally narrower than `agentic_on_dspy`.

For the clean comparison phase:

- do not switch optimizer families
- use the fixed five-run plan in `experiments/dspy/run_plan.json`
- vary only deterministic hyperparameters for `MIPROv2`

Requirement:

- always log `optimizer_name` and relevant optimizer settings in local results and W&B

## Baseline Command

Run from repo root:

```bash
set -a && source .env && set +a
cd experiments/dspy
../../venv/bin/inv run
```

Deterministic five-run plan:

```bash
cd experiments/dspy
../../venv/bin/inv run-plan
```

Final test run:

```bash
set -a && source .env && set +a
cd experiments/dspy
../../venv/bin/inv run --eval-split=test --wandb-job-type=final-test
```

## Logging

Primary local outputs:

- `experiments/dspy/results.tsv`
- `experiments/dspy/runs/`
- `experiments/dspy/programs/`
- `experiments/dspy/dev_sessions.tsv`

W&B project:

- `agentic-atis-compare-dspy`

Recommended W&B job types:

- `val-baseline`
- `val-iteration`
- `final-test`

## Effort Tracking

Use `experiments/dspy/dev_sessions.tsv` to record development effort.

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

## Data Preparation

There is one canonical prepare path for the whole repo:

- `prepare.py`

This track must not create or use a separate train/val split.
