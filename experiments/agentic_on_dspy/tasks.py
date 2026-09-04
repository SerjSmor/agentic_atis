from __future__ import annotations

from pathlib import Path

from invoke import task

REPO_ROOT = Path(__file__).resolve().parents[2]


@task
def prepare(c):
    """Regenerate the canonical frozen split. Prefer `inv bootstrap` from the repo root."""
    with c.cd(str(REPO_ROOT)):
        c.run("venv/bin/python prepare.py", pty=True)


@task
def run(
    c,
    model="openai/gpt-4.1-nano",
    prompt_model="openai/gpt-4.1-nano",
    optimizer="miprov2",
    reflection_model="",
    teacher_model="",
    eval_split="val",
    optimizer_auto="medium",
    num_trials="",
    gepa_max_full_evals="",
    gepa_max_metric_calls="",
    gepa_reflection_minibatch_size=3,
    gepa_candidate_selection_strategy="pareto",
    gepa_disable_merge=False,
    max_bootstrapped_demos=2,
    max_labeled_demos=4,
    num_threads=4,
    seed=20260501,
    wandb_job_type="val-baseline",
    verbose=False,
):
    eval_path = "shared/data/val_subset.jsonl" if eval_split == "val" else "shared/data/test_subset.jsonl"
    parts = [
        "venv/bin/python experiments/agentic_on_dspy/train_dspy.py",
        "--train-path shared/data/train_subset.jsonl",
        "--val-path shared/data/val_subset.jsonl",
        "--results-path experiments/agentic_on_dspy/results.tsv",
        "--predictions-dir experiments/agentic_on_dspy/runs",
        "--programs-dir experiments/agentic_on_dspy/programs",
        "--wandb-project agentic-atis-compare-agentic-on-dspy",
        f"--model {model}",
        f"--prompt-model {prompt_model}",
        f"--optimizer {optimizer}",
        f"--eval-path {eval_path}",
        f"--eval-split-name {eval_split}",
        f"--wandb-job-type {wandb_job_type}",
        f"--optimizer-auto {optimizer_auto}",
        f"--max-bootstrapped-demos {int(max_bootstrapped_demos)}",
        f"--max-labeled-demos {int(max_labeled_demos)}",
        f"--num-threads {int(num_threads)}",
        f"--seed {int(seed)}",
    ]
    if reflection_model:
        parts.append(f"--reflection-model {reflection_model}")
    if teacher_model:
        parts.append(f"--teacher-model {teacher_model}")
    if num_trials:
        parts.append(f"--num-trials {int(num_trials)}")
    if gepa_max_full_evals:
        parts.append(f"--gepa-max-full-evals {int(gepa_max_full_evals)}")
    if gepa_max_metric_calls:
        parts.append(f"--gepa-max-metric-calls {int(gepa_max_metric_calls)}")
    if int(gepa_reflection_minibatch_size) != 3:
        parts.append(f"--gepa-reflection-minibatch-size {int(gepa_reflection_minibatch_size)}")
    if gepa_candidate_selection_strategy != "pareto":
        parts.append(f"--gepa-candidate-selection-strategy {gepa_candidate_selection_strategy}")
    if gepa_disable_merge:
        parts.append("--gepa-disable-merge")
    if verbose:
        parts.append("--verbose")
    with c.cd(str(REPO_ROOT)):
        c.run(" ".join(parts), pty=True)
