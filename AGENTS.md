# Agentic ATIS — Shared Protocol

This repo compares **four harnesses** for improving an ATIS intent-classification
prompt. Each harness is a separate agent workspace under `experiments/`.

This file holds the rules **every** track inherits. Your track's own `AGENTS.md`
holds the rules specific to you, and it is the one that says what you are allowed
to change.

## You are working in one track

| Workspace | Harness under test |
|---|---|
| `experiments/agentic/` | A coding agent editing prompts directly. No DSPy. |
| `experiments/dspy/` | DSPy optimization on a fixed plan. No agent redesign. |
| `experiments/agentic_on_dspy/` | A coding agent redesigning a DSPy program between runs. |
| `experiments/multi_agent/` | An orchestrator fanning out to ten agents in parallel. |

You were started with one of these as your working directory. **That folder is your
track.** Read its `AGENTS.md` before doing anything else.

Do not read, run, or modify another track's folder. The comparison is only
meaningful if the four tracks stay independent. In particular, never copy a prompt,
signature, or result from a sibling track into yours.

## First command, always

If `shared/data/` is missing or `venv/` does not exist, run this from the repo root:

```bash
inv bootstrap
```

That creates the venv, installs requirements, and generates the frozen split. It is
idempotent — safe to run twice. Do not hand-roll these steps.

You also need a `.env` with an OpenAI key. Copy `.env.example` to `.env` and fill it
in, then:

```bash
set -a && source .env && set +a
```

If `OPENAI_API_KEY` is unset, **stop and say so.** Do not try to work around it.

## How to run things

Every track is driven by `invoke` from inside its own folder:

```bash
cd experiments/<your-track>
../../venv/bin/inv run
```

**Always go through `inv`. Never call `python train.py` or `python train_dspy.py`
directly.** The task runner changes directory to the repo root before executing, so
all the relative paths (`shared/data/...`, `experiments/...`) resolve correctly. If
you invoke Python yourself from inside a track folder, those paths silently break.

Run `../../venv/bin/inv --list` to see the tasks your track exposes.

## Invariants — the same for all four tracks

These are what make the comparison valid. **Changing any of them invalidates the
experiment.** If you believe one of them is wrong, say so in your final summary; do
not change it.

- **Frozen split.** `shared/data/` is generated once by `prepare.py`: 50 train /
  50 validation / 30 test. Never regenerate it, never resample it, never create your
  own split.
- **Task model is pinned** to `gpt-4.1-nano`. Do not swap it for a stronger model.
  (Some tracks allow a separate teacher or reflection model — your track's
  `AGENTS.md` says whether yours does, and it must be logged.)
- **Primary metric is `macro_f1`.** Secondary: `weighted_f1`, `micro_f1`. ATIS is
  heavily imbalanced, so macro F1 is the only one of the three that is sensitive to
  rare-label behavior. Optimize macro F1. Do not report a win on the basis of
  weighted or micro F1.
- **Iterate on validation. `test` is locked.** Never evaluate on
  `shared/data/test_subset.jsonl` during development, and never let anything you
  learned from test predictions influence a prompt or program change. Test is for a
  single final run, only when explicitly asked.

## Logging — required for every run

Runs log themselves. Do not write results by hand, and do not edit `results.tsv`
after the fact. Each run must produce:

- a row in your track's `results.tsv`
- per-example predictions in your track's `runs/`
- a saved program in `programs/` (DSPy-based tracks only)

Every row records macro/weighted/micro F1, model, tokens, estimated cost, and
runtime. W&B logging is on by default and is skipped automatically when
`WANDB_API_KEY` is absent — that is fine, keep going.

## Cost discipline

- Keep a single run under **$1**. The defaults are far below this; a run costing more
  than a few cents means something is misconfigured — stop and investigate rather
  than paying for it.
- Your track's `AGENTS.md` sets a **run budget**. Respect it. When you hit it, stop
  and report, even if you think one more run would help.

## Keep / discard

This is a bounded experiment loop, not an open-ended optimization:

1. The first run of a session establishes the baseline.
2. Change one thing.
3. Rerun and compare macro F1 on validation.
4. Keep the change if macro F1 improved. Revert it if it did not.
5. Repeat until the run budget is spent.

Record honestly. A run that made things worse is a result, not a failure — leave its
row in `results.tsv` and say what you learned.

## When you finish

Report, in this order:

1. The baseline macro F1 and the final macro F1.
2. Every change you made, and whether it was kept or discarded.
3. How many runs you used and the total estimated cost.
4. What you would try next, and anything about the protocol that got in your way.

Do not claim an improvement you did not measure on the validation set.
