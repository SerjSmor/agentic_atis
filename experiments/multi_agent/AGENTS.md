# Track: `multi_agent` — ten agents, one generation

You are an **orchestrator**. You do not write prompts yourself. You spawn ten
sub-agents, give them the same error analysis, let them each write a different
prompt, evaluate all ten, and keep the best one.

Read the repo-root `AGENTS.md` first — it holds the shared protocol (frozen split,
pinned model, macro F1, locked test set, logging, keep/discard loop). This file only
covers what is specific to your track.

This track exists to answer one question: **does breadth beat depth?** The `agentic`
track gets five sequential runs, and each change is informed by the last. You get ten
prompts at once, all written from the *same* starting information, with no chance to
learn between them. Those are two genuinely different ways to spend a budget. If ten
uninformed parallel attempts do not beat four informed sequential ones, that is the
finding.

## Your entrypoint

`experiments/multi_agent/train.py` — a prompt-only classifier on the OpenAI API. It
is a fork of the `agentic` track's, with one addition: `--variant`.

**Do not edit `train.py`.** Workers do not edit it either. Every prompt lives in its
own file under `variants/`, which is what lets ten workers write at once without
colliding.

```bash
cd experiments/multi_agent
../../venv/bin/inv run                        # baseline, no variant
../../venv/bin/inv run --variant=variant_03   # one worker's prompt
../../venv/bin/inv run-wave                   # evaluate every variant in variants/
../../venv/bin/inv leaderboard                # rank all recorded runs by macro F1
```

## How a variant works

`variants/<name>.py` defines any subset of these three functions, with the same
signatures as the defaults in `train.py`:

- `build_system_prompt(labels, prompt_examples, boundary_examples) -> str`
- `choose_few_shot_examples(train_examples, limit) -> list[Example]`
- `choose_boundary_examples(train_examples) -> list[Example]`

Anything a variant omits falls through to the default. A variant that defines none of
them is rejected, since it would just be the baseline again.

`variants/_example.py` is a working reference. It is skipped by `run-wave` because of
the leading underscore — copy it, do not evaluate it.

## The loop

1. **Baseline.** `../../venv/bin/inv run --run-name=baseline`. One run, no variant.
   This is the number all ten workers have to beat, and it is identical to the
   `agentic` track's starting point.

2. **Error analysis, once, by you.** Read `runs/baseline.jsonl` and build the
   confusion pairs — which gold label is being predicted as what. This analysis is
   the *shared input* to all ten workers. Do not do per-worker analysis; that would
   make this a sequential track with extra steps.

3. **Spawn ten workers in parallel.** Each gets: the confusion table, the contents of
   `train.py`'s default prompt functions, the `variants/_example.py` reference, and a
   distinct assigned angle (see below). Each writes exactly one file,
   `variants/variant_NN.py`. Workers do not run anything, do not read each other's
   files, and do not read `results.tsv`.

4. **Evaluate the wave.** `../../venv/bin/inv run-wave`. Ten rows appear in
   `results.tsv`.

5. **Select and report.** `../../venv/bin/inv leaderboard`. The winner is the highest
   macro F1. Leave every variant file in place — the losers are data.

One generation. Do not spawn a second wave.

## Assigning angles

Ten workers all told "improve the prompt" will write ten near-identical prompts, and
you will have bought nothing over one worker. Give each a distinct, *stated* angle so
the wave actually covers ground. Suggested split:

| worker | angle |
|---|---|
| 1 | explicit boundary rules for `flight` vs `airline` |
| 2 | explicit boundary rules for `flight` vs `flight_time` |
| 3 | explicit boundary rules for `airfare` vs `flight+airfare` |
| 4 | explicit boundary rules for `ground_service` vs `ground_fare` |
| 5 | rare-label recall: `capacity`, `restriction`, `city`, `abbreviation` |
| 6 | few-shot *selection* only — one example per label, prompt text untouched |
| 7 | few-shot ordering and count, no new rules |
| 8 | restructure the output format / label list presentation |
| 9 | terse prompt — strip guidance rather than add it |
| 10 | free choice, informed by the confusion table |

Worker 9 is not a joke entry. Adding rules is the obvious move and every other worker
is doing it; the cheapest way to learn whether the baseline prompt is *overloaded* is
to have someone cut it. Record what it scores.

Record the angle in your summary next to each variant's score. A wave where nine
workers converge on the same idea is itself a result about multi-agent fan-out.

## Run budget

**Eleven validation runs**: one baseline plus ten variants. Not twenty-two, not two
generations.

Each run is a prompt-only evaluation over 50 validation examples on `gpt-4.1-nano` —
a fraction of a cent. Eleven runs is well under the $1 ceiling. `run-wave` evaluates
five at a time to stay inside rate limits; raise it with `--max-parallel` only if you
are sure the account tolerates it.

## You may change

- anything inside `variants/*.py` — that is the whole search space
- which angles you assign, if the confusion table suggests better ones
- `--max-parallel` on `run-wave`

## You may not

- edit `train.py`, including its default prompt functions — the baseline must stay
  the baseline, or the ten variants have nothing to be measured against
- use DSPy or any optimizer; this is the prompt-only substrate
- let workers read each other's variants, `results.tsv`, or the baseline predictions
  of a *previous* wave — the whole point is that they write blind and simultaneously
- run a second generation, or feed a wave's results back into a new wave
- use a teacher or reflection model for the classification itself; the task model is
  `gpt-4.1-nano`. (The sub-agents are of course models, and their cost is part of
  what this track spends — see below.)
- touch `shared/`, `prepare.py`, or the split
- read from or write to another track's folder
- evaluate on `test`

## Report the orchestration cost honestly

This is the part that makes the comparison fair, and the part easiest to skip.
`results.tsv` records what the **classifier** cost — eleven cheap evaluations. It
does not record what the **ten sub-agents** cost to run, and they are the expensive
part of this track by a wide margin.

Your summary must state both:

- classifier cost from `results.tsv` (comparable to the other three tracks)
- orchestration cost: ten sub-agent sessions, and roughly what they consumed

A track that wins on macro F1 while costing 50x in agent tokens has not obviously
won. Say so if that is what happened.

## Definition of done

- Eleven rows in `experiments/multi_agent/results.tsv`: `baseline` plus ten variants.
- Ten files in `variants/`, one per worker, all left in place including the losers.
- Per-example predictions in `runs/`.
- A summary giving: baseline macro F1; the leaderboard; which angle won and by how
  much; how many of the ten beat the baseline; how much the ten converged; classifier
  cost and orchestration cost separately.
- `git status` clean apart from generated artifacts and the new `variants/` files.

## Local outputs

- `results.tsv` — one row per run
- `runs/` — per-example predictions
- `variants/` — the ten worker prompts (committed, not gitignored — they are the
  experiment's record)
- `dev_sessions.tsv` — optional effort tracking
- W&B project: `agentic-atis-compare-multi-agent`
