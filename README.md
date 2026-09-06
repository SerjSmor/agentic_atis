# Agentic ATIS

Which harness do you trust to improve a prompt: a **compiler** (DSPy), or a **coding
agent**?

This repo runs that comparison on ATIS intent classification. Three tracks, one
frozen split, one model, one metric. The only variable is what is allowed to edit the
prompt.

| Track | Harness under test |
|---|---|
| [`experiments/agentic`](experiments/agentic/AGENTS.md) | A coding agent editing prompts directly. No DSPy. |
| [`experiments/dspy`](experiments/dspy/AGENTS.md) | DSPy optimization on a fixed plan. No agent redesign. |
| [`experiments/agentic_on_dspy`](experiments/agentic_on_dspy/AGENTS.md) | A coding agent redesigning a DSPy program between runs. |

The point is not only to compare scores, but to compare workflows: how each harness
spends model calls, optimization budget, and engineering effort on the same small
classification task.

## Each track is an agent workspace

The three `AGENTS.md` files under `experiments/` **are** the experiment. Each one
tells a coding agent what it is and is not allowed to change. Everything else is
pinned.

Agent instruction files are discovered by walking up from the working directory, so
starting an agent inside a track folder gives it exactly two files: the shared
protocol at the root, and its own track's rules.

```
AGENTS.md                              # shared protocol — inherited by all tracks
experiments/agentic/AGENTS.md          # ── the three prompts under test ──
experiments/dspy/AGENTS.md
experiments/agentic_on_dspy/AGENTS.md
```

To run a track, start your agent with that folder as its working directory:

```bash
cd experiments/dspy && codex          # or claude, or your agent of choice
```

Then tell it to run its track. Nothing else to configure — and the three can run in
parallel, in three terminals, against one clone.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Nothing else — uv brings its own
Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone https://github.com/SerjSmor/agentic_atis.git
cd agentic_atis

uvx --from invoke inv bootstrap   # venv + requirements + frozen split

cp .env.example .env              # add your OPENAI_API_KEY
set -a && source .env && set +a

venv/bin/inv doctor               # confirms the clone is ready
```

`uvx` runs `inv` without installing it globally; after bootstrap the venv has its
own copy at `venv/bin/inv`, which is what every later command uses.

`inv bootstrap` is idempotent. `inv doctor` checks uv, the venv, dependencies, the
split, the API key, that `import shared` resolves from any directory, and that each
track's `inv run` is registered — run it before a live demo.

## Running a track by hand

Every track is driven by `invoke` from inside its own folder. The task runner changes
to the repo root first, so relative paths resolve correctly:

```bash
cd experiments/agentic         && ../../venv/bin/inv run
cd experiments/dspy            && ../../venv/bin/inv run-plan
cd experiments/agentic_on_dspy && ../../venv/bin/inv run --optimizer=gepa
```

Use `../../venv/bin/inv --list` to see a track's tasks. **Do not call the Python
entrypoints directly** — paths are relative to the repo root and will break.

Final locked test run (only when you mean it):

```bash
cd experiments/<track> && ../../venv/bin/inv run --eval-split=test --wandb-job-type=final-test
```

## Shared protocol

All three tracks use:

- the same ATIS subset and the same frozen split
- the same task model: `gpt-4.1-nano`
- the same primary metric: `macro_f1`
- the same secondary metrics: `weighted_f1`, `micro_f1`
- token, cost, and runtime logging

`macro_f1` is primary because ATIS is imbalanced — `flight` dominates while
`capacity`, `restriction`, and `city` are rare. Weighted and micro F1 will report
~0.9 while a model never once gets a rare class right, so they are recorded but not
optimized against.

DSPy-based tracks may use teacher or reflection models, but those must be logged
explicitly.

## Data

One canonical preparation path for the whole repo: [`prepare.py`](prepare.py), which
writes a frozen split to `shared/data/`:

- `50` train examples
- `50` validation examples
- `30` test examples

Rules: iterate on `val`, do not tune on `test`, use `test` only for a final locked
run. No track creates its own split.

The split is generated rather than committed, so run `inv bootstrap` after cloning.

## Repo layout

```text
AGENTS.md                 shared protocol, inherited by every track
tasks.py                  inv bootstrap, inv doctor
prepare.py                canonical split generation
shared/
  data/                   frozen split (generated)
  constants.py            labels, guidance, model pricing
  results.py              TSV logging

experiments/
  agentic/
    AGENTS.md             track rules — prompt-only, no DSPy
    train.py              entrypoint
    tasks.py
    train_iter1..5.py     archived prompt iterations
  dspy/
    AGENTS.md             track rules — fixed MIPROv2 plan, no redesign
    train_dspy.py         entrypoint (frozen — the control arm)
    run_plan.json         the deterministic five-run plan
    tasks.py
  agentic_on_dspy/
    AGENTS.md             track rules — adaptive, agent may redesign
    train_dspy.py         entrypoint (starts identical to dspy/, meant to diverge)
    tasks.py
```

`experiments/dspy/train_dspy.py` and `experiments/agentic_on_dspy/train_dspy.py`
start out byte-identical **on purpose**. The `agentic_on_dspy` agent is expected to
rewrite its copy; the `dspy` copy stays frozen so there is a control to compare
against. Do not deduplicate them into a shared module — the duplication is the
experiment.

## Logging

Each track writes `results.tsv`, per-example predictions to `runs/`, and compiled
programs to `programs/` (DSPy tracks). These are gitignored — they are per-clone
experiment output, not repo content.

W&B projects, used when `WANDB_API_KEY` is set:

- `agentic-atis-compare-agentic`
- `agentic-atis-compare-dspy`
- `agentic-atis-compare-agentic-on-dspy`

## Background

The task definition and its original constraints are in
[`agentic-atis-task.md`](agentic-atis-task.md). The loop shape is modeled on
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch): small reviewable
repo surface, fixed evaluation protocol, fixed per-experiment budget, autonomous
keep/discard over experiments.

Dataset: [`tuetschek/atis`](https://huggingface.co/datasets/tuetschek/atis).
