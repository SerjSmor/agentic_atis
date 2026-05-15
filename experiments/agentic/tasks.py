from __future__ import annotations

from pathlib import Path

from invoke import task

REPO_ROOT = Path(__file__).resolve().parents[2]


@task
def run(
    c,
    model="gpt-4.1-nano",
    eval_split="val",
    max_train_examples_in_prompt=12,
    wandb_job_type="val-baseline",
):
    eval_path = "shared/data/val_subset.jsonl" if eval_split == "val" else "shared/data/test_subset.jsonl"
    cmd = (
        "venv/bin/python train.py "
        f"--model {model} "
        f"--eval-path {eval_path} "
        f"--eval-split-name {eval_split} "
        f"--wandb-job-type {wandb_job_type} "
        f"--max-train-examples-in-prompt {int(max_train_examples_in_prompt)}"
    )
    with c.cd(str(REPO_ROOT)):
        c.run(cmd, pty=True)
