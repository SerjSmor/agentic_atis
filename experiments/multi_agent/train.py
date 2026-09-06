from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import OpenAI
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from shared.constants import KNOWN_ATIS_LABELS, LABEL_GUIDANCE, MODEL_PRICING_PER_1M_TOKENS, MULTI_AGENT_DIR, SHARED_DATA_DIR
from shared.results import append_tsv_row

# Variants import the defaults from here by name. Registering the *running* module
# means they reuse it rather than re-executing this file as a second module object,
# which would break dataclass resolution.
sys.modules.setdefault("multi_agent_train", sys.modules[__name__])

DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_TRAIN_PATH = SHARED_DATA_DIR / "train_subset.jsonl"
DEFAULT_EVAL_PATH = SHARED_DATA_DIR / "val_subset.jsonl"
DEFAULT_RESULTS_PATH = MULTI_AGENT_DIR / "results.tsv"
DEFAULT_PREDICTIONS_DIR = MULTI_AGENT_DIR / "runs"
DEFAULT_MAX_TRAIN_EXAMPLES_IN_PROMPT = 12
DEFAULT_WANDB_PROJECT = "agentic-atis-compare-multi-agent"


@dataclass
class Example:
    id: int
    text: str
    label: str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a prompt-only ATIS intent classification experiment."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval-path", "--test-path", dest="eval_path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--eval-split-name", choices=["val", "test"], default="val")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-train-examples-in-prompt",
        type=int,
        default=DEFAULT_MAX_TRAIN_EXAMPLES_IN_PROMPT,
    )
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-job-type", default="agentic")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--variant",
        default=None,
        help="Name of a module in variants/ whose prompt hooks override the "
             "defaults below. This is how ten workers edit prompts in "
             "parallel without touching the same file.",
    )
    return parser.parse_args()


def load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            examples.append(Example(id=int(row["id"]), text=row["text"], label=str(row["label"])))
    return examples


def choose_few_shot_examples(
    train_examples: list[Example],
    max_examples: int,
) -> list[Example]:
    by_label: dict[str, list[Example]] = {}
    for example in train_examples:
        by_label.setdefault(example.label, []).append(example)

    selected: list[Example] = []
    labels = sorted(by_label)

    # Round-robin selection gives better label coverage than taking the first N rows.
    while len(selected) < max_examples:
        made_progress = False
        for label in labels:
            bucket = by_label[label]
            if bucket:
                selected.append(bucket.pop(0))
                made_progress = True
                if len(selected) >= max_examples:
                    break
        if not made_progress:
            break

    return selected


def choose_boundary_examples(train_examples: list[Example]) -> list[Example]:
    preferred_texts = [
        "does any airline have an afternoon flight from atlanta to boston",
        "please list all airline flights between denver and boston",
        "when does continental fly from philadelphia to denver on sundays",
        "on tuesday i 'd like to find a flight from detroit to st. petersburg that arrives before 10 pm",
        "please give me a flight leaving boston going to washington arriving in washington at 5 o'clock in the afternoon",
        "what flight from boston to atlanta arrives earliest in atlanta",
    ]
    by_text = {example.text: example for example in train_examples}
    return [by_text[text] for text in preferred_texts if text in by_text]


def build_system_prompt(
    labels: list[str],
    prompt_examples: list[Example],
    boundary_examples: list[Example],
) -> str:
    label_list = ", ".join(labels)
    label_guidance = "\n".join(
        f"- {label}: {LABEL_GUIDANCE.get(label, 'ATIS intent label')}" for label in labels
    )
    examples_block = "\n".join(
        f'Text: "{example.text}"\nIntent: {example.label}'
        for example in prompt_examples
    )
    boundary_examples_block = "\n".join(
        f'Text: "{example.text}"\nIntent: {example.label}'
        for example in boundary_examples
    )
    return (
        "You are classifying airline travel utterances into ATIS intent labels.\n"
        f"Valid labels: {label_list}\n"
        "Return exactly one label from the valid label list and no extra text.\n"
        "Choose the single best intent for the user's main request.\n"
        "Intent guidance:\n"
        f"{label_guidance}\n\n"
        "Disambiguation rules:\n"
        "- Use flight when the user wants flights to be found or listed, even if they mention cheapest, nonstop, round trip, class, or meal constraints.\n"
        "- Use airfare only when the user is asking about fare or ticket price rather than asking to list flights.\n"
        "- Use flight+airfare only when the user explicitly asks for both fares and flights together.\n"
        "- Use ground_service for availability or type of ground transportation.\n"
        "- Use ground_fare for the cost of ground transportation such as taxi, limousine, or rental car.\n"
        "- Use distance when the user asks for miles or distance between airport and city or between two places.\n"
        "- Use flight_time only when the user is asking for timing information itself, such as departure time, arrival time, or duration of a flight.\n"
        "- If the user is asking to find flights and mentions arrival or departure times only as search constraints, use flight.\n"
        "Pay extra attention to these boundary examples from the training subset:\n\n"
        f"{boundary_examples_block}\n\n"
        "Use the examples below as additional guidance.\n\n"
        f"{examples_block}"
    )


def assert_prompt_examples_from_train(
    train_examples: list[Example],
    prompt_examples: list[Example],
) -> None:
    train_keys = {(example.id, example.text, example.label) for example in train_examples}
    prompt_keys = {(example.id, example.text, example.label) for example in prompt_examples}
    invalid_examples = prompt_keys - train_keys
    if invalid_examples:
        raise ValueError(
            "All prompt examples must come from the training subset. "
            f"Found non-train examples: {sorted(invalid_examples)}"
        )


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = MODEL_PRICING_PER_1M_TOKENS.get(model)
    if pricing is None:
        return None
    return (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )


def extract_text(response: Any) -> str:
    text = getattr(response, "output_text", "")
    if text:
        return text.strip()
    raise ValueError("Model response did not contain output_text.")


def normalize_prediction(raw_prediction: str, valid_labels: set[str]) -> str:
    prediction = raw_prediction.strip().splitlines()[0].strip()
    if prediction in valid_labels:
        return prediction

    lowered = prediction.lower()
    for label in valid_labels:
        if lowered == label.lower():
            return label

    return prediction


def wandb_credentials_available() -> bool:
    """True if wandb can start without prompting for interactive input.

    Without this check, `wandb.init()` on an unauthenticated machine prompts on a
    TTY (invoke runs with pty=True, so it hangs forever) or raises in a headless
    context. Either way an unattended agent stalls, so we skip W&B instead.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True
    if os.environ.get("WANDB_MODE", "").lower() in {"offline", "disabled", "dryrun"}:
        return True
    try:
        return bool(wandb.api.api_key)  # picks up a previous `wandb login`
    except Exception:  # pragma: no cover - defensive
        return False


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    if args.disable_wandb or wandb is None:
        return None

    if not wandb_credentials_available():
        print(
            "wandb: no credentials found - set WANDB_API_KEY or run `wandb login`. "
            "Continuing without W&B logging; results.tsv and runs/ are unaffected."
        )
        return None

    try:
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            job_type=args.wandb_job_type,
            config=config,
            settings=wandb.Settings(init_timeout=60),
        )
    except Exception as exc:
        print(f"wandb: init failed ({exc}). Continuing without W&B logging.")
        return None


VARIANTS_DIR = MULTI_AGENT_DIR / "variants"
OVERRIDABLE = ("build_system_prompt", "choose_few_shot_examples", "choose_boundary_examples")


def load_variant(name: str) -> dict[str, Any]:
    """Load prompt hooks from variants/<name>.py.

    A variant defines any subset of OVERRIDABLE with the same signatures as the
    module-level defaults; anything it omits falls through to the default. Ten
    workers therefore write ten separate files and never touch train.py itself,
    so they can run against one checkout with no merge conflicts.
    """
    path = VARIANTS_DIR / f"{name}.py"
    if not path.exists():
        available = sorted(p.stem for p in VARIANTS_DIR.glob("*.py") if p.stem != "__init__")
        raise SystemExit(
            f"no such variant: {path}\n"
            f"available: {', '.join(available) if available else '(none yet)'}"
        )
    spec = importlib.util.spec_from_file_location(f"variants.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    hooks = {fn: getattr(module, fn) for fn in OVERRIDABLE if hasattr(module, fn)}
    if not hooks:
        raise SystemExit(
            f"{path} defines none of {OVERRIDABLE} - it would be identical to the baseline"
        )
    print(f"variant {name}: overriding {', '.join(sorted(hooks))}")
    return hooks


def main() -> None:
    args = parse_args()
    client = OpenAI()

    train_examples = load_examples(args.train_path)
    eval_examples = load_examples(args.eval_path)
    labels = sorted(set(KNOWN_ATIS_LABELS) | {example.label for example in train_examples})
    valid_labels = set(labels)
    hooks = load_variant(args.variant) if args.variant else {}
    _choose_few_shot = hooks.get("choose_few_shot_examples", choose_few_shot_examples)
    _choose_boundary = hooks.get("choose_boundary_examples", choose_boundary_examples)
    _build_prompt = hooks.get("build_system_prompt", build_system_prompt)

    few_shot_examples = _choose_few_shot(
        train_examples, args.max_train_examples_in_prompt
    )
    boundary_examples = _choose_boundary(train_examples)
    boundary_texts = {example.text for example in boundary_examples}
    supplemental_examples = [
        example for example in few_shot_examples if example.text not in boundary_texts
    ]
    remaining_slots = max(0, args.max_train_examples_in_prompt - len(boundary_examples))
    prompt_examples = boundary_examples + supplemental_examples[:remaining_slots]
    assert_prompt_examples_from_train(train_examples, prompt_examples)
    system_prompt = _build_prompt(labels, prompt_examples, boundary_examples)

    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.model}-{run_id}"
    predictions_path = args.predictions_dir / f"{run_name}.jsonl"
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "train_path": str(args.train_path),
        "eval_path": str(args.eval_path),
        "eval_split_name": args.eval_split_name,
        "max_train_examples_in_prompt": args.max_train_examples_in_prompt,
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "label_count": len(labels),
        "labels": labels,
        "boundary_example_count": len(boundary_examples),
    }
    wandb_run = maybe_init_wandb(args, config)

    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    started_timer = perf_counter()

    with predictions_path.open("w", encoding="utf-8") as predictions_handle:
        for example in eval_examples:
            response = client.responses.create(
                model=args.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example.text},
                ],
            )
            raw_prediction = extract_text(response)
            prediction = normalize_prediction(raw_prediction, valid_labels)

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            gold_labels.append(example.label)
            predicted_labels.append(prediction)

            predictions_handle.write(
                json.dumps(
                    {
                        "id": example.id,
                        "text": example.text,
                        "gold_label": example.label,
                        "predicted_label": prediction,
                        "raw_prediction": raw_prediction,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    runtime_seconds = perf_counter() - started_timer
    macro_f1 = f1_score(gold_labels, predicted_labels, average="macro")
    weighted_f1 = f1_score(gold_labels, predicted_labels, average="weighted")
    micro_f1 = f1_score(gold_labels, predicted_labels, average="micro")
    estimated_cost_usd = estimate_cost_usd(
        args.model, total_input_tokens, total_output_tokens
    )

    label_distribution = Counter(predicted_labels)
    keep = True
    result_row = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp_utc": started_at.isoformat(),
        "model": args.model,
        "macro_f1": f"{macro_f1:.6f}",
        "weighted_f1": f"{weighted_f1:.6f}",
        "micro_f1": f"{micro_f1:.6f}",
        "eval_split_name": args.eval_split_name,
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "label_count": len(labels),
        "prompt_examples": len(prompt_examples),
        "boundary_example_count": len(boundary_examples),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost_usd": (
            f"{estimated_cost_usd:.6f}" if estimated_cost_usd is not None else ""
        ),
        "runtime_seconds": f"{runtime_seconds:.3f}",
        "keep": str(keep).lower(),
        "predictions_path": str(predictions_path),
    }
    append_tsv_row(args.results_path, result_row)

    if wandb_run is not None:
        wandb.log(
            {
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "micro_f1": micro_f1,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "runtime_seconds": runtime_seconds,
                "eval_size": len(eval_examples),
                "eval_split_name": args.eval_split_name,
                "label_count": len(labels),
                "prompt_examples": len(prompt_examples),
                "boundary_example_count": len(boundary_examples),
            }
        )
        wandb_run.summary["predicted_label_distribution"] = dict(label_distribution)
        wandb_run.summary["predictions_path"] = str(predictions_path)
        wandb_run.finish()

    print(json.dumps(result_row, indent=2))


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY must be set before running train.py.")
    main()
