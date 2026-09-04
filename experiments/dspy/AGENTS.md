# Track: `dspy` — the optimizer, alone

DSPy is the harness here. **You are not.** Your job is to execute a fixed plan and
report what the optimizer achieved without your help.

Read the repo-root `AGENTS.md` first — it holds the shared protocol (frozen split,
pinned model, macro F1, locked test set, logging, keep/discard loop). This file only
covers what is specific to your track.

This track is deliberately constrained. It is the control arm: it isolates what
`MIPROv2` contributes on its own, so that `agentic_on_dspy` can be read as "what did
adding an agent buy." A more clever run here would destroy that comparison.

## Your entrypoint

`experiments/dspy/train_dspy.py` — signature, module, metric, and optimizer wiring.

Read it to understand the program. **Do not modify it.**

```bash
cd experiments/dspy
../../venv/bin/inv run          # single run
../../venv/bin/inv run-plan     # the full five-run plan
../../venv/bin/inv --list       # all available tasks
```

## Run budget

**Exactly five validation runs**, defined in `run_plan.json`. Not four, not six.

The five configurations vary only `optimizer_auto`, demo counts, and seed. They are
already written down. `inv run-plan` executes all five in order and appends five rows
to `results.tsv` — that is the intended way to run this track.

## You may change

- nothing in the program
- nothing in the plan

You may pass the documented `inv run` flags for a one-off diagnostic run, but the
five recorded runs must come from `run_plan.json` as written.

## You may not

- edit `train_dspy.py` — not the signature, not the docstring
  (`ATIS_DISAMBIGUATION_RULES`), not the metric, not the module
- edit `run_plan.json`
- switch optimizer family — `MIPROv2` only, no GEPA, no
  `BootstrapFewShotWithRandomSearch`
- change the metric objective away from `exact_match`
- do error analysis and feed it back into the program. **This is the specific thing
  this track exists to not do.** You may look at `runs/` to report what failed; you
  may not act on it.
- change the task model away from `gpt-4.1-nano`
- touch `shared/`, `prepare.py`, or the split
- read from or write to another track's folder
- evaluate on `test`

## If the plan underperforms

It probably will, particularly on rare labels — the metric is `exact_match`, which
gives the optimizer no reason to care about them.

**That is a finding, not a bug.** Report it. Do not fix it. The fix is what
`agentic_on_dspy` is for, and if you apply it here there is nothing left to compare
against.

## Definition of done

- Five rows in `experiments/dspy/results.tsv`, one per plan entry, named as in
  `run_plan.json`.
- Compiled programs in `experiments/dspy/programs/`, per-example predictions in
  `experiments/dspy/runs/`.
- A summary giving macro F1 for all five runs, which configuration won, and the
  total estimated cost **including compilation**, not just evaluation.
- `git status` clean apart from generated artifacts. If you edited a tracked source
  file in this track, you did the wrong thing — revert it.

## Local outputs

- `results.tsv` — one row per run
- `runs/` — per-example predictions
- `programs/` — saved compiled DSPy programs
- `dev_sessions.tsv` — optional effort tracking
- W&B project: `agentic-atis-compare-dspy`
