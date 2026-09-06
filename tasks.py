from __future__ import annotations

import os
import shutil
from pathlib import Path

from invoke import Exit, task

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
SHARED_DATA_DIR = REPO_ROOT / "shared" / "data"
SPLIT_FILES = ("train_subset.jsonl", "val_subset.jsonl", "test_subset.jsonl")
TRACKS = ("agentic", "dspy", "agentic_on_dspy")


def _split_exists() -> bool:
    return all((SHARED_DATA_DIR / name).exists() for name in SPLIT_FILES)


def _require_uv() -> str:
    """This repo standardises on uv: it brings its own Python and its own
    installer, so the venv never depends on the host python having ensurepip."""
    uv = shutil.which("uv")
    if uv is None:
        raise Exit(
            "uv is required but not on PATH.\n"
            "  install: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "  docs:    https://docs.astral.sh/uv/"
        )
    return uv


def _site_packages() -> Path:
    matches = sorted(VENV_DIR.glob("lib/python*/site-packages"))
    if not matches:
        raise Exit(f"no site-packages under {VENV_DIR}; run: inv bootstrap")
    return matches[0]


def _write_repo_root_pth() -> None:
    """Make `import shared` work from any cwd, for any script under the repo.

    The tracks run `venv/bin/python experiments/<track>/train*.py` from the repo
    root. Python puts the *script's* directory on sys.path, not the cwd, so the
    repo root is never importable without help. A .pth in site-packages is the
    least intrusive way to add it: it applies to every process that uses this
    venv, including subprocesses, and needs no PYTHONPATH in the environment.
    """
    pth = _site_packages() / "agentic_atis_repo_root.pth"
    pth.write_text(f"{REPO_ROOT}\n", encoding="utf-8")
    print(f"    wrote {pth}")


@task
def bootstrap(c, force=False):
    """Create the venv, install requirements, and generate the frozen split.

    Idempotent: safe to run repeatedly. This is the first command to run after
    cloning. Pass --force to regenerate the split even if it already exists.
    """
    with c.cd(str(REPO_ROOT)):
        _require_uv()

        if not VENV_PYTHON.exists():
            print("==> creating venv")
            # Deliberately `venv/`, not uv's default `.venv/`: the path is
            # hardcoded across every track's tasks.py and AGENTS.md.
            c.run("uv venv venv", pty=True)
        else:
            print("==> venv already exists, skipping")

        print("==> installing requirements")
        # No pip inside the venv, and none needed - uv does the installing.
        c.run(
            "uv pip install --python venv/bin/python -r requirements.txt --quiet",
            pty=True,
        )

        print("==> putting the repo root on the venv's import path")
        _write_repo_root_pth()

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


@task(name="wandb-check")
def wandb_check(c, project="agentic-atis-preflight", keep=False):
    """Prove W&B logging works end to end before spending a real run on it.

    Starts a throwaway run, logs a metric, and finishes. If this passes, the three
    tracks will log correctly; if it fails, they will skip W&B and keep going.
    """
    script = '''
import os, sys
try:
    import wandb
except ImportError:
    print("FAIL: wandb is not installed. Run: inv bootstrap")
    sys.exit(1)

print(f"wandb version      : {wandb.__version__}")

key_source = None
if os.environ.get("WANDB_API_KEY"):
    key_source = "WANDB_API_KEY env var"
else:
    try:
        if wandb.api.api_key:
            key_source = "prior `wandb login` (~/.netrc)"
    except Exception:
        pass

if key_source is None:
    print("FAIL: no W&B credentials found.")
    print("      Add WANDB_API_KEY to .env and `set -a && source .env && set +a`,")
    print("      or run `venv/bin/wandb login`.")
    print("      Runs still work without W&B - they just skip it.")
    sys.exit(1)

print(f"credentials from   : {key_source}")
print("starting throwaway run ...")

try:
    run = wandb.init(
        project=os.environ["PREFLIGHT_PROJECT"],
        job_type="preflight",
        name="preflight-check",
        config={"purpose": "verify wandb connectivity"},
        settings=wandb.Settings(init_timeout=60),
    )
    wandb.log({"preflight_metric": 1.0})
    url = run.url
    run.finish()
except Exception as exc:
    print(f"FAIL: wandb.init/log raised: {exc}")
    sys.exit(1)

print("")
print("PASS: W&B logging is working.")
print(f"run URL: {url}")
'''
    script_path = REPO_ROOT / ".wandb_check.py"
    script_path.write_text(script, encoding="utf-8")
    try:
        with c.cd(str(REPO_ROOT)):
            result = c.run(
                f"PREFLIGHT_PROJECT={project} venv/bin/python .wandb_check.py",
                pty=True,
                warn=True,
            )
    finally:
        if not keep:
            script_path.unlink(missing_ok=True)

    if not result.ok:
        print("\nW&B is not logging. The tracks will still run and still write "
              "results.tsv and runs/ - W&B is the only thing you lose.")


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

    check(
        "uv installed",
        shutil.which("uv") is not None,
        "install: curl -LsSf https://astral.sh/uv/install.sh | sh",
    )

    check("venv exists", VENV_PYTHON.exists(), "run: inv bootstrap")

    if VENV_PYTHON.exists():
        result = c.run(
            f"{VENV_PYTHON} -c 'import dspy, sklearn, openai, invoke'",
            warn=True,
            hide=True,
        )
        check("dependencies importable", result.ok, "run: inv bootstrap")

        result = c.run(
            f"cd / && {VENV_PYTHON} -c 'import shared.constants, shared.results'",
            warn=True,
            hide=True,
        )
        check(
            "repo root importable from any cwd",
            result.ok,
            "run: inv bootstrap  (it writes the .pth that puts the repo on sys.path)",
        )

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

    if VENV_PYTHON.exists():
        result = c.run(
            f"{VENV_PYTHON} -c \"import os,wandb;"
            f"print(bool(os.environ.get('WANDB_API_KEY') or wandb.api.api_key))\"",
            warn=True,
            hide=True,
        )
        if result.ok and "True" in result.stdout:
            print("  [ok] W&B credentials found")
            print("       -> run `inv wandb-check` to confirm they actually work")
        else:
            print("  [--] no W&B credentials (optional; runs skip W&B and still "
                  "write results.tsv)")
            print("       -> to enable: add WANDB_API_KEY to .env, or `wandb login`")

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
