from __future__ import annotations

import os
import sys
from pathlib import Path

from invoke import task

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
SHARED_DATA_DIR = REPO_ROOT / "shared" / "data"
SPLIT_FILES = ("train_subset.jsonl", "val_subset.jsonl", "test_subset.jsonl")
TRACKS = ("agentic", "dspy", "agentic_on_dspy")


def _split_exists() -> bool:
    return all((SHARED_DATA_DIR / name).exists() for name in SPLIT_FILES)


@task
def bootstrap(c, force=False):
    """Create the venv, install requirements, and generate the frozen split.

    Idempotent: safe to run repeatedly. This is the first command to run after
    cloning. Pass --force to regenerate the split even if it already exists.
    """
    with c.cd(str(REPO_ROOT)):
        if not VENV_PYTHON.exists():
            print("==> creating venv")
            c.run(f"{sys.executable} -m venv venv", pty=True)
        else:
            print("==> venv already exists, skipping")

        print("==> installing requirements")
        c.run("venv/bin/python -m pip install --upgrade pip --quiet", pty=True)
        c.run("venv/bin/python -m pip install -r requirements.txt --quiet", pty=True)

        if _split_exists() and not force:
            print(f"==> frozen split already present in {SHARED_DATA_DIR}, skipping")
            print("    (pass --force to regenerate)")
        else:
            print("==> generating frozen split")
            c.run("venv/bin/python prepare.py", pty=True)

    print("\nBootstrap complete. Next:")
    print("  cp .env.example .env   # then add your OPENAI_API_KEY")
    print("  set -a && source .env && set +a")
    print("  cd experiments/<track> && ../../venv/bin/inv run")


@task
def doctor(c):
    """Check whether this clone is ready to run an experiment."""
    ok = True

    def check(label: str, passed: bool, hint: str = "") -> None:
        nonlocal ok
        print(f"  [{'ok' if passed else 'XX'}] {label}")
        if not passed:
            ok = False
            if hint:
                print(f"         -> {hint}")

    print("\nagentic-atis doctor\n")

    check("venv exists", VENV_PYTHON.exists(), "run: inv bootstrap")

    if VENV_PYTHON.exists():
        result = c.run(
            f"{VENV_PYTHON} -c 'import dspy, sklearn, openai, invoke'",
            warn=True,
            hide=True,
        )
        check("dependencies importable", result.ok, "run: inv bootstrap")

    check(
        "frozen split present",
        _split_exists(),
        f"run: inv bootstrap  (expected {', '.join(SPLIT_FILES)} in shared/data/)",
    )

    check(
        "OPENAI_API_KEY set",
        bool(os.environ.get("OPENAI_API_KEY")),
        "cp .env.example .env, add your key, then: set -a && source .env && set +a",
    )

    if not os.environ.get("WANDB_API_KEY"):
        print("  [--] WANDB_API_KEY not set (optional; W&B logging will be skipped)")

    for track in TRACKS:
        agents_file = REPO_ROOT / "experiments" / track / "AGENTS.md"
        check(f"experiments/{track}/AGENTS.md", agents_file.exists())

    if VENV_PYTHON.exists():
        for track in TRACKS:
            result = c.run(
                f"cd {REPO_ROOT / 'experiments' / track} && {VENV_DIR / 'bin' / 'inv'} --list",
                warn=True,
                hide=True,
            )
            has_run = result.ok and "run" in result.stdout
            check(f"experiments/{track}: inv run is registered", has_run)

    print()
    if ok:
        print("Ready. Start an agent with one of the track folders as its "
              "working directory:\n")
        print("  cd experiments/dspy && codex        # or your agent of choice")
    else:
        print("Not ready — see the hints above.")
    print()
