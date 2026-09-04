# Track: `agentic_on_dspy` — the agent on top of the optimizer

DSPy is your substrate. **You are the redesign loop on top of it.**

Read the repo-root `AGENTS.md` first — it holds the shared protocol (frozen split,
pinned model, macro F1, locked test set, logging, keep/discard loop). This file only
covers what is specific to your track.

The `dspy` track runs a fixed plan and cannot adapt. You can. Everything DSPy searches
over, it searches over *within the program it was given* — and you are allowed to
change that program between runs, based on what you learn from the errors.

## Your entrypoint

`experiments/agentic_on_dspy/train_dspy.py` — your own copy of the DSPy program.

It starts out identical to the one in `experiments/dspy/`. **That is intentional: it
is your starting point, and you are expected to change it.** The other copy stays
frozen so the comparison has a control. Never edit the other copy.

```bash
cd experiments/agentic_on_dspy
../../venv/bin/inv run
../../venv/bin/inv run --optimizer=gepa --optimizer-auto=light --reflection-model=openai/gpt-4.1
../../venv/bin/inv --list
```

## Run budget

**Eight validation runs per session**, including the baseline.

Run 1 is the baseline: run it unmodified, so you have the same starting point the
`dspy` track has. Then adapt.

Eight is enough to change optimizer family once and still iterate. It is not enough
to brute-force. Choose changes by error analysis, not by sweeping.

Stop at eight and report.

## You may change

Everything the `dspy` track may change, plus the things it may not:

- **the signature docstring** (`ATIS_DISAMBIGUATION_RULES`) — this is the instruction
  block the optimizer starts from, and rewriting it after reading errors is the most
  direct lever you have
- label descriptions and guidance
- signature and module shape (`dspy.Predict` vs `dspy.ChainOfThought`, extra fields)
- demo prioritization — e.g. deliberately surfacing rare-label and boundary examples
- **optimizer family** — `MIPROv2`, `GEPA`, `BootstrapFewShotWithRandomSearch`
- **the metric objective** — `exact_match` vs `weighted_macro`, which up-weights rare
  labels inside the metric itself
- the GEPA feedback text in `make_gepa_metric` — GEPA consumes natural-language
  feedback, not just a scalar, so what you write there is instruction to the optimizer
- a teacher or reflection model, **if you log it**

## You may not

- change the **task** model away from `gpt-4.1-nano` — a reflection or teacher model
  may differ, but the model doing the classifying may not
- touch `shared/`, `prepare.py`, or the split
- read from or write to another track's folder, including
  `experiments/dspy/train_dspy.py`
- evaluate on `test`, or let anything learned from test predictions influence a change

## Log what you changed

Because your search space is wide, the results are only interpretable if each run
says what was different. `optimizer_name`, `optimizer_auto`, `reflection_model`, and
`teacher_model` are already written to `results.tsv` and W&B — use `--run-name` to
give each run a name that states the change:

```bash
../../venv/bin/inv run --optimizer=gepa --run-name=gepa-weighted-macro-v1
```

**Change one thing per run.** If a run switches optimizer *and* changes the objective
*and* adds a reflection model, and it wins, you have learned nothing about which one
mattered. If you do bundle changes deliberately — to get a strong result quickly —
say so explicitly in your summary and name the confound.

## Where the wins probably are

The `dspy` track's weakness is structural: `exact_match` gives the optimizer no
gradient toward rare labels, and macro F1 is dominated by rare labels. Look at:

1. The objective. `weighted_macro` aligns what the optimizer maximizes with what is
   being scored.
2. The instruction block. The confusions cluster around `flight` vs `airline` /
   `flight_time` / `flight+airfare`, and `ground_service` vs `ground_fare`. Explicit
   boundary rules help more than generic guidance.
3. The optimizer. GEPA can consume the feedback you write; MIPROv2 cannot.

Check `runs/<run_name>.jsonl` after every run. That file is your error analysis.

## Definition of done

- Up to eight rows in `experiments/agentic_on_dspy/results.tsv`, each with a run name
  that identifies its change.
- Compiled programs in `programs/`, per-example predictions in `runs/`.
- A summary listing baseline macro F1, best macro F1, what changed between them,
  total estimated cost **including compilation**, and any confounded runs.
- The winning configuration left in place in `train_dspy.py`; discarded experiments
  reverted.

## Local outputs

- `results.tsv` — one row per run
- `runs/` — per-example predictions
- `programs/` — saved compiled DSPy programs
- `prompts/` — optional agent-authored prompt notes
- `dev_sessions.tsv` — optional effort tracking
- W&B project: `agentic-atis-compare-agentic-on-dspy`
