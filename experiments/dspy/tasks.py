from __future__ import annotations

from pathlib import Path

from invoke import task

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACK_DIR = Path(__file__).resolve().parent
RUN_PLAN_PATH = TRACK_DIR / "run_plan.json"


@task
def prepare(c, dspy_train_size=100, val_size=50, split_seed=20260501):
    cmd = (
        "venv/bin/python dspy_prepare.py "
        f"--dspy-train-size {int(dspy_train_size)} "
        f"--val-size {int(val_size)} "
        f"--split-seed {int(split_seed)}"
    )
    with c.cd(str(REPO_ROOT)):
        c.run(cmd, pty=True)


@task
def run(
    c,
    model="openai/gpt-4.1-nano",
    prompt_model="openai/gpt-4.1-nano",
    teacher_model="",
    eval_split="val",
    optimizer_auto="medium",
    num_trials="",
    max_bootstrapped_demos=2,
    max_labeled_demos=4,
    num_threads=4,
    seed=20260501,
    wandb_job_type="val-baseline",
    verbose=False,
):
    eval_path = "shared/data/val_subset.jsonl" if eval_split == "val" else "shared/data/test_subset.jsonl"
    parts = [
        "venv/bin/python experiments/dspy/train_dspy.py",
        f"--model {model}",
        f"--prompt-model {prompt_model}",
        "--train-path shared/data/train_subset.jsonl",
        "--val-path shared/data/val_subset.jsonl",
        f"--eval-path {eval_path}",
        f"--eval-split-name {eval_split}",
        f"--wandb-job-type {wandb_job_type}",
        f"--optimizer-auto {optimizer_auto}",
        f"--max-bootstrapped-demos {int(max_bootstrapped_demos)}",
        f"--max-labeled-demos {int(max_labeled_demos)}",
        f"--num-threads {int(num_threads)}",
        f"--seed {int(seed)}",
    ]
    if teacher_model:
        parts.append(f"--teacher-model {teacher_model}")
    if num_trials:
        parts.append(f"--num-trials {int(num_trials)}")
    if verbose:
        parts.append("--verbose")
    with c.cd(str(REPO_ROOT)):
        c.run(" ".join(parts), pty=True)


@task
def run_plan(c, eval_split="val", model="openai/gpt-4.1-nano", prompt_model="openai/gpt-4.1-nano"):
    cmd = (
        "venv/bin/python experiments/dspy/run_plan.py "
        f"--model {model} "
        f"--prompt-model {prompt_model} "
        f"--eval-split {eval_split}"
    )
    with c.cd(str(REPO_ROOT)):
        c.run(cmd, pty=True)
