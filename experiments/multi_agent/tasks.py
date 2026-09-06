from __future__ import annotations

from pathlib import Path

from invoke import task

from shared.results import append_tsv_row

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACK_DIR = Path(__file__).resolve().parent
VARIANTS_DIR = TRACK_DIR / "variants"


@task
def run(
    c,
    model="gpt-4.1-nano",
    eval_split="val",
    max_train_examples_in_prompt=12,
    variant="",
    run_name="",
    wandb_job_type="val-baseline",
):
    """Evaluate one prompt. Omit --variant for the baseline, pass it for a worker."""
    eval_path = "shared/data/val_subset.jsonl" if eval_split == "val" else "shared/data/test_subset.jsonl"
    parts = [
        "venv/bin/python experiments/multi_agent/train.py",
        f"--model {model}",
        f"--eval-path {eval_path}",
        f"--eval-split-name {eval_split}",
        f"--wandb-job-type {wandb_job_type}",
        f"--max-train-examples-in-prompt {int(max_train_examples_in_prompt)}",
    ]
    if variant:
        parts.append(f"--variant {variant}")
        parts.append(f"--run-name {run_name or variant}")
    elif run_name:
        parts.append(f"--run-name {run_name}")
    with c.cd(str(REPO_ROOT)):
        c.run(" ".join(parts), pty=True)


@task(name="run-wave")
def run_wave(c, model="gpt-4.1-nano", eval_split="val", max_parallel=5):
    """Evaluate every variant in variants/, up to max_parallel at a time.

    Each variant writes to its own results file, which this task then merges into
    results.tsv serially. That is not fussiness: shared.results.append_tsv_row
    rewrites the whole TSV on every append, so two concurrent writers silently
    lose a row. Predictions in runs/ are already per-run-name and safe.
    """
    import csv
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    variants = sorted(p.stem for p in VARIANTS_DIR.glob("*.py") if not p.stem.startswith("_"))
    if not variants:
        print("no variants in variants/ - nothing to evaluate")
        return

    eval_path = "shared/data/val_subset.jsonl" if eval_split == "val" else "shared/data/test_subset.jsonl"
    partial_dir = TRACK_DIR / ".wave"
    partial_dir.mkdir(exist_ok=True)
    logs_dir = partial_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    def evaluate(name: str) -> tuple[str, int, Path]:
        partial = partial_dir / f"{name}.tsv"
        partial.unlink(missing_ok=True)
        cmd = [
            "venv/bin/python", "experiments/multi_agent/train.py",
            "--model", model,
            "--eval-path", eval_path,
            "--eval-split-name", eval_split,
            "--wandb-job-type", "wave",
            "--variant", name,
            "--run-name", name,
            "--results-path", str(partial),
        ]
        log = logs_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        return name, proc.returncode, log

    print(f"==> evaluating {len(variants)} variants, {max_parallel} at a time")
    with ThreadPoolExecutor(max_workers=int(max_parallel)) as pool:
        outcomes = list(pool.map(evaluate, variants))

    failed = []
    merged = 0
    for name, code, log in outcomes:
        partial = partial_dir / f"{name}.tsv"
        if code != 0 or not partial.exists():
            failed.append((name, log))
            print(f"  [XX] {name}  (exit {code}, see {log})")
            continue
        with partial.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                append_tsv_row(TRACK_DIR / "results.tsv", row)
                merged += 1
        print(f"  [ok] {name}")

    print(f"\n==> merged {merged} row(s) into results.tsv")
    if failed:
        print(f"==> {len(failed)} variant(s) FAILED: {', '.join(n for n, _ in failed)}")
        print("    a wave with failures is not a complete generation - fix and rerun those")


@task
def leaderboard(c, top=0):
    """Print every recorded run sorted by macro F1 - the selection step."""
    import csv

    path = TRACK_DIR / "results.tsv"
    if not path.exists():
        print("no results.tsv yet - run the baseline first")
        return
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    rows.sort(key=lambda r: float(r["macro_f1"]), reverse=True)
    if top:
        rows = rows[: int(top)]
    print(f"{'run_name':32}{'macro_f1':>10}{'weighted':>10}{'micro':>8}")
    for r in rows:
        print(
            f"{r['run_name'][:31]:32}{float(r['macro_f1']):10.4f}"
            f"{float(r['weighted_f1']):10.4f}{float(r['micro_f1']):8.4f}"
        )
