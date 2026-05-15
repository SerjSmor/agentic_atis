from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PLAN_PATH = Path(__file__).with_name("run_plan.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed DSPy MIPROv2 plan.")
    parser.add_argument("--model", default="openai/gpt-4.1-nano")
    parser.add_argument("--prompt-model", default="openai/gpt-4.1-nano")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(RUN_PLAN_PATH.read_text(encoding="utf-8"))
    eval_path = "shared/data/val_subset.jsonl" if args.eval_split == "val" else "shared/data/test_subset.jsonl"

    for index, run_config in enumerate(plan, start=1):
        cmd = [
            "venv/bin/python",
            "experiments/dspy/train_dspy.py",
            "--model",
            args.model,
            "--prompt-model",
            args.prompt_model,
            "--wandb-project",
            "agentic-atis-compare-dspy",
            "--wandb-job-type",
            f"plan-run-{index}",
            "--run-name",
            str(run_config["name"]),
            "--optimizer-auto",
            str(run_config["optimizer_auto"]),
            "--max-bootstrapped-demos",
            str(int(run_config["max_bootstrapped_demos"])),
            "--max-labeled-demos",
            str(int(run_config["max_labeled_demos"])),
            "--num-threads",
            str(int(run_config["num_threads"])),
            "--seed",
            str(int(run_config["seed"])),
            "--train-path",
            "shared/data/train_subset.jsonl",
            "--val-path",
            "shared/data/val_subset.jsonl",
            "--eval-path",
            eval_path,
            "--eval-split-name",
            args.eval_split,
            "--results-path",
            "experiments/dspy/results.tsv",
            "--predictions-dir",
            "experiments/dspy/runs",
            "--programs-dir",
            "experiments/dspy/programs",
        ]
        if "num_trials" in run_config and run_config["num_trials"] is not None:
            cmd.extend(["--num-trials", str(int(run_config["num_trials"]))])

        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
