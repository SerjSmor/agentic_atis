# Agentic Track

## Purpose

This track measures a prompt-only, agentic ATIS intent-classification workflow.

It should be compared against:

- `experiments/dspy`
- `experiments/agentic_on_dspy`

## Status

Current status: `scratch`

## Canonical Data

Use the frozen split in `shared/data/`:

- `shared/data/train_subset.jsonl`
- `shared/data/val_subset.jsonl`
- `shared/data/test_subset.jsonl`

Rules:

- iterate on `val`
- do not use `test` during development
- use `test` only for a final locked run

## Allowed Changes

Allowed:

- prompt wording
- prompt structure
- label guidance text
- train-example selection strategy
- light orchestration in `train.py`

Not allowed:

- changing the canonical split
- tuning directly on `test`
- using DSPy optimization in this track
- changing the main model away from `gpt-4.1-nano`

## Model Constraint

The main model for this track must be:

- `gpt-4.1-nano`

No teacher model is used in this track.

## Baseline Command

Run from repo root:

```bash
set -a && source .env && set +a
cd experiments/agentic
../../venv/bin/inv run
```

Final test run:

```bash
set -a && source .env && set +a
venv/bin/python train.py --eval-path shared/data/test_subset.jsonl --eval-split-name test --wandb-job-type final-test
```

## Logging

Primary local outputs:

- `experiments/agentic/results.tsv`
- `experiments/agentic/runs/`
- `experiments/agentic/dev_sessions.tsv`

W&B project:

- `agentic-atis-compare-agentic`

Recommended W&B job types:

- `val-baseline`
- `val-iteration`
- `final-test`

## Effort Tracking

Use `experiments/agentic/dev_sessions.tsv` to record development effort.

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
- runtime
