# Track: `agentic` — the coding agent, alone

You are the harness. There is no optimizer in this track: every improvement comes
from you reading errors and rewriting the prompt.

Read the repo-root `AGENTS.md` first — it holds the shared protocol (frozen split,
pinned model, macro F1, locked test set, logging, keep/discard loop). This file only
covers what is specific to your track.

## Your entrypoint

`experiments/agentic/train.py` — a prompt-only classifier built directly on the
OpenAI API. The prompt is assembled in `build_system_prompt()`; the few-shot
selection lives in `choose_few_shot_examples()` and `choose_boundary_examples()`.

Run it:

```bash
cd experiments/agentic
../../venv/bin/inv run
```

Useful knobs without editing code:

```bash
../../venv/bin/inv run --max-train-examples-in-prompt=16
```

## Run budget

**Five validation runs per session**, including the baseline.

Run 1 is the baseline — run it before changing anything, so you have a number to
compare against. That leaves four changes. Spend them on the biggest errors, not on
wording polish.

Stop at five and report, even mid-idea.

## You may change

- prompt wording and structure in `build_system_prompt()`
- the disambiguation rules block
- label guidance text
- which training examples go into the prompt, and in what order
- `choose_few_shot_examples` / `choose_boundary_examples` selection strategy
- light orchestration inside `train.py`

## You may not

- use DSPy, or any optimizer — if you find yourself writing a search loop over
  prompt candidates, you have left this track
- change the task model away from `gpt-4.1-nano`
- use a teacher or reflection model; this track has no second model
- touch `shared/`, `prepare.py`, or the split
- read from or write to another track's folder
- evaluate on `test`

## How to actually improve here

The predictions in `runs/<run_name>.jsonl` are the whole point. Each line has
`gold_label`, `predicted_label`, and the raw model output. After a run:

1. Build the confusion pairs — which gold label is being predicted as what.
2. Find the **systematic** confusion, not the one-off. ATIS failure modes cluster:
   `flight` vs `airline`, `flight` vs `flight_time`, `flight` vs `flight+airfare`,
   `airfare` vs `flight+airfare`, `ground_service` vs `ground_fare`,
   `abbreviation` vs everything containing a code.
3. Write a boundary rule that separates the pair explicitly, or put a demonstrating
   example into the prompt.
4. Rerun and check macro F1.

Watch macro F1, not accuracy. Micro F1 barely moves on this task because `flight`
dominates the label distribution; if micro F1 goes up and macro F1 goes down, you
have made the model worse at exactly the classes that matter.

## Definition of done

- Five runs recorded in `experiments/agentic/results.tsv`.
- Per-example predictions for each run in `experiments/agentic/runs/`.
- A summary naming the baseline macro F1, the best macro F1, and which prompt change
  produced it.
- Every kept change is still in `train.py`; every discarded change is reverted.

## Local outputs

- `results.tsv` — one row per run
- `runs/` — per-example predictions
- `dev_sessions.tsv` — optional effort tracking
- W&B project: `agentic-atis-compare-agentic`
