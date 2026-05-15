# Agentic ATIS

This repo compares three workflows for ATIS intent classification under a shared protocol:

- `agentic`: prompt-only iterative engineering
- `dspy`: plain DSPy optimization
- `agentic_on_dspy`: agent-guided improvement of a DSPy system

The point is not only to compare scores, but to compare workflows: how different systems spend model calls, optimization budget, and engineering effort on the same small classification task.

## Current Results

Best validation runs currently recorded:

| Track | Run | Model / Optimizer | Macro F1 | Weighted F1 | Micro F1 | Estimated Total Run Cost |
|---|---|---|---:|---:|---:|---:|
| `agentic` | `gpt-4.1-nano-iter3-20260501T155404Z` | `gpt-4.1-nano` | `0.574187` | `0.910407` | `0.880000` | `$0.004983` |
| `dspy` | `miprov2-light-search` | `gpt-4.1-nano` / `MIPROv2` | `0.532653` | `0.927714` | `0.880000` | `$0.008119` |
| `agentic_on_dspy` | `gepa-light-weighted-macro-reflect-4-1-v3-costfix` | `gpt-4.1-nano` / `GEPA` | `0.771895` | `0.942980` | `0.940000` | `$0.040111` |

For DSPy-based tracks, `Estimated Total Run Cost` now includes both compilation and evaluation. The plain DSPy cost shown for `miprov2-light-search` comes from a cost-accounted rerun of the same configuration.

## Overview

The repo uses a shared frozen split and a common evaluation setup so the three tracks can be compared directly.

All tracks use:

- the same ATIS subset
- the same train / validation / test split
- the same primary metric: `macro_f1`
- the same secondary metrics: `weighted_f1`, `micro_f1`
- token, cost, and runtime logging

The current clean-comparison task model is:

- `gpt-4.1-nano`

DSPy-based tracks may use teacher or reflection models, but those must be logged explicitly.

## Methodology

There is one canonical data-preparation path for the whole repo:

- [prepare.py](/home/serj/dev/DSPyPR/agentic_atis/prepare.py)

That script produces a frozen split under [shared/data](/home/serj/dev/DSPyPR/agentic_atis/shared/data):

- `50` train examples
- `50` validation examples
- `30` test examples

Rules:

- iterate on `val`
- do not tune on `test`
- use `test` only for a final locked run

This matters because ATIS is imbalanced. `macro_f1` is the main metric here because it is much more sensitive to rare-label behavior than `weighted_f1` or `micro_f1`.

## Repo Layout

```text
shared/
  data/
  constants.py
  results.py

experiments/
  agentic/
    TRACK.md
    tasks.py
    results.tsv
    runs/

  dspy/
    TRACK.md
    tasks.py
    run_plan.json
    results.tsv
    runs/
    programs/

  agentic_on_dspy/
    TRACK.md
    tasks.py
    results.tsv
    runs/
    programs/

prepare.py
train.py
```

Each experiment folder has its own `TRACK.md` with local rules and constraints.

## Tracks

### Agentic

Prompt-only workflow using [train.py](/home/serj/dev/DSPyPR/agentic_atis/train.py).

Allowed changes include:

- prompt wording
- prompt structure
- label guidance
- train-example selection strategy

This track does not use DSPy optimization.

### DSPy

Structured DSPy optimization using [experiments/dspy/train_dspy.py](/home/serj/dev/DSPyPR/agentic_atis/experiments/dspy/train_dspy.py).

For the clean comparison phase, this track is intentionally constrained:

- exactly `5` validation runs
- fixed optimizer family: `MIPROv2`
- deterministic plan in [experiments/dspy/run_plan.json](/home/serj/dev/DSPyPR/agentic_atis/experiments/dspy/run_plan.json)

### Agentic on DSPy

DSPy as the substrate, but with adaptive agent-guided redesign between runs.

This track may:

- inspect errors and predictions
- rewrite task instructions
- change signature and guidance structure
- alter demo prioritization
- switch optimizer strategy

Recommended optimizers explored here include:

- `MIPROv2`
- `GEPA`
- `BootstrapFewShotWithRandomSearch`

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Load environment variables:

```bash
set -a
source .env
set +a
```

Expected environment variables:

- `OPENAI_API_KEY`
- `WANDB_API_KEY` if you want W&B logging

## Prepare Data

```bash
venv/bin/python prepare.py
```

No track should create its own split.

## Run Experiments

### Agentic

```bash
cd experiments/agentic
../../venv/bin/inv run
```

Final test:

```bash
cd /home/serj/dev/DSPyPR/agentic_atis
venv/bin/python train.py \
  --eval-path shared/data/test_subset.jsonl \
  --eval-split-name test \
  --wandb-job-type final-test
```

### DSPy

Single validation run:

```bash
cd experiments/dspy
../../venv/bin/inv run
```

Deterministic five-run plan:

```bash
cd experiments/dspy
../../venv/bin/inv run-plan
```

Final test:

```bash
cd experiments/dspy
../../venv/bin/inv run --eval-split=test --wandb-job-type=final-test
```

### Agentic on DSPy

```bash
cd experiments/agentic_on_dspy
../../venv/bin/inv run
```

Final test:

```bash
cd experiments/agentic_on_dspy
../../venv/bin/inv run --eval-split=test --wandb-job-type=final-test
```

## Logging

Each track logs:

- `results.tsv`
- per-example predictions in `runs/`
- compiled programs for DSPy-based tracks in `programs/`

W&B projects:

- `agentic-atis-compare-agentic`
- `agentic-atis-compare-dspy`
- `agentic-atis-compare-agentic-on-dspy`

There are also per-track `dev_sessions.tsv` placeholders for development-effort tracking.

## Main Files

Start here:

- [prepare.py](/home/serj/dev/DSPyPR/agentic_atis/prepare.py)
- [train.py](/home/serj/dev/DSPyPR/agentic_atis/train.py)
- [experiments/dspy/train_dspy.py](/home/serj/dev/DSPyPR/agentic_atis/experiments/dspy/train_dspy.py)
- [experiments/agentic_on_dspy/train_dspy.py](/home/serj/dev/DSPyPR/agentic_atis/experiments/agentic_on_dspy/train_dspy.py)
- [experiments/agentic/tasks.py](/home/serj/dev/DSPyPR/agentic_atis/experiments/agentic/tasks.py)
- [experiments/dspy/tasks.py](/home/serj/dev/DSPyPR/agentic_atis/experiments/dspy/tasks.py)
- [experiments/agentic_on_dspy/tasks.py](/home/serj/dev/DSPyPR/agentic_atis/experiments/agentic_on_dspy/tasks.py)
- [experiments/agentic/TRACK.md](/home/serj/dev/DSPyPR/agentic_atis/experiments/agentic/TRACK.md)
- [experiments/dspy/TRACK.md](/home/serj/dev/DSPyPR/agentic_atis/experiments/dspy/TRACK.md)
- [experiments/agentic_on_dspy/TRACK.md](/home/serj/dev/DSPyPR/agentic_atis/experiments/agentic_on_dspy/TRACK.md)
