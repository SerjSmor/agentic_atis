# Agentic ATIS Task

## Objective

Build a Codex-based, prompt-only ATIS intent classification system in this repo that follows an `autoresearch`-style workflow:

- the human defines the experiment program in markdown
- the agent runs a bounded batch of experiments
- each experiment is evaluated against a fixed metric
- results are logged and compared
- only winning changes are kept

## Reference

This task should conceptually resemble `karpathy/autoresearch`:
- small, reviewable repo surface
- fixed evaluation protocol
- fixed per-experiment budget
- autonomous keep/discard loop over experiments

Reference repo:
- https://github.com/karpathy/autoresearch

Dataset:
- https://huggingface.co/datasets/tuetschek/atis

## Success Metric

- Primary metric: macro F1

## Approach

- Prompt-only classification
- No model training or fine-tuning
- The implementation should create a local `train.py`
- The agent is allowed to iterate on:
  - prompts
  - label names
  - model choice among GPT models

## Dataset Policy

- Do not use the full train/test dataset for each experiment
- Use a smaller subset for bounded experiments
- The evaluation loop should operate on a small subset rather than the full corpus
- Use a random subset of `100` samples from `train`
- Use a random subset of `30` samples from `test`
- Subset selection must be reproducible via fixed seeds

## Experiment Budget

- Per-experiment budget: `$1` maximum
- This budget constraint should shape subset size and evaluation scope
- Experiments should be designed so that a single run stays within the dollar cap

## Logging Requirements

Every experiment should log to both:

- local experiment logs, including a `results.tsv`-style artifact
- Weights & Biases (`wandb`)

At minimum, log:

- macro F1
- model name
- prompt or prompt version
- label-set version
- estimated or measured API cost
- runtime
- keep/discard status

## Loop Style

- Use a bounded experiment batch, not an unbounded overnight loop
- Each batch should run a finite number of experiments and then stop
- The first run in a batch should establish a baseline

## Evaluation Rules

- Evaluation should be fixed and comparable across experiments
- The agent should not silently change the subset definition between experiments in the same batch
- Winning experiments are kept
- Non-improving experiments are discarded

## Deliverable Direction

The repo should evolve toward an `autoresearch`-style setup for prompt experimentation on ATIS, centered around:

- `train.py` as the main runnable experiment entrypoint
- a fixed evaluation slice for comparison
- bounded-cost prompt experiments
- reproducible experiment logging

## Open Details To Finalize

These details still need to be specified later during implementation:

- exact GPT model shortlist allowed for experiments
- exact `results.tsv` schema
- exact definition of cost tracking for Codex versus model inference cost
- batch size in number of experiments

## Notes

- This file records the original task definition for the repo, expanded with
  additional requirements during implementation. It is kept for provenance; the
  operational rules live in `AGENTS.md` at the repo root and in each track folder.
