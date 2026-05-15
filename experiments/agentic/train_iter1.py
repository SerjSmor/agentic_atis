from __future__ import annotations

import argparse
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

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.constants import AGENTIC_DIR, KNOWN_ATIS_LABELS, LABEL_GUIDANCE, MODEL_PRICING_PER_1M_TOKENS, SHARED_DATA_DIR
from shared.results import append_tsv_row

DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_TRAIN_PATH = SHARED_DATA_DIR / "train_subset.jsonl"
DEFAULT_EVAL_PATH = SHARED_DATA_DIR / "val_subset.jsonl"
DEFAULT_RESULTS_PATH = AGENTIC_DIR / "results.tsv"
DEFAULT_PREDICTIONS_DIR = AGENTIC_DIR / "runs"
DEFAULT_MAX_TRAIN_EXAMPLES_IN_PROMPT = 16
DEFAULT_WANDB_PROJECT = "agentic-atis-compare-agentic"

CORE_BOUNDARY_TEXTS = [
    "does any airline have an afternoon flight from atlanta to boston",
    "please list all airline flights between denver and boston",
    "when does continental fly from philadelphia to denver on sundays",
    "all flights and fares from atlanta to dallas round trip after 12 pm less than 1100 dollars",
    "what kind of aircraft does delta use before 8 am on august second from boston to denver",
]

FLIGHT_COUNTEREXAMPLE_TEXTS = [
    "what is the latest flight leaving newark for los angeles wednesday",
    "show me the earliest flight from san jose to pittsburgh that serves a snack",
    "what flight from boston to atlanta arrives earliest in atlanta",
    "i need the earliest flight from denver to boston that serves dinner",
    "what flights from houston to milwaukee on friday on american airlines",
    "information on american airlines flight from washington to philadelphia",
    "on united airlines flying from denver to san francisco before 10 am what type of aircraft is used",
    "what is the cheapest flight from pittsburgh to atlanta one way",
    "what are the cheapest one way flights from denver to atlanta",
    "does midwest express serve indianapolis",
]


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
    parser.add_argument("--wandb-job-type", default="val-iteration")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
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
    by_text = {example.text: example for example in train_examples}
    selected: list[Example] = []
    seen_texts: set[str] = set()

    def add_texts(texts: list[str]) -> None:
        for text in texts:
            example = by_text.get(text)
            if example is None or example.text in seen_texts:
                continue
            selected.append(example)
            seen_texts.add(example.text)
            if len(selected) >= max_examples:
                return

    add_texts(CORE_BOUNDARY_TEXTS)
    if len(selected) < max_examples:
        add_texts(FLIGHT_COUNTEREXAMPLE_TEXTS)

    if len(selected) >= max_examples:
        return selected[:max_examples]

    by_label: dict[str, list[Example]] = {}
    for example in train_examples:
        if example.text in seen_texts:
            continue
        by_label.setdefault(example.label, []).append(example)

    labels = sorted(by_label)
    while len(selected) < max_examples:
        made_progress = False
        for label in labels:
            bucket = by_label[label]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            seen_texts.add(selected[-1].text)
            made_progress = True
            if len(selected) >= max_examples:
                break
        if not made_progress:
            break

    return selected


def build_system_prompt(
    labels: list[str],
    prompt_examples: list[Example],
) -> str:
    label_list = ", ".join(labels)
    label_guidance = "\n".join(
        f"- {label}: {LABEL_GUIDANCE.get(label, 'ATIS intent label')}" for label in labels
    )
    examples_block = "\n\n".join(
        f'Text: "{example.text}"\nIntent: {example.label}'
        for example in prompt_examples
    )
    return (
        "You are classifying airline travel utterances into ATIS intent labels.\n"
        "Follow ATIS annotation conventions, even when the wording sounds ambiguous.\n"
        f"Valid labels: {label_list}\n"
        "Return exactly one label from the valid label list and no extra text.\n"
        "Choose the label that matches the main information request.\n\n"
        "Intent guidance:\n"
        f"{label_guidance}\n\n"
        "High-priority decision rules:\n"
        "- Use flight when the user is trying to find, list, compare, or check availability of flights.\n"
        "- Still use flight when the request includes airline preferences, class, stops, meals, price caps, cheapest/earliest/latest wording, or arrival/departure constraints as filters.\n"
        "- Use airline only when the core question is which airline serves a route or whether any airline serves it.\n"
        "- Use airfare only when the core request is the fare or ticket price itself.\n"
        "- Use flight+airfare only when the user explicitly asks for both flights and fares together.\n"
        "- Use flight_time only when the user asks for schedule information itself, such as when a carrier flies or what the arrival/departure time is, rather than asking to find a flight option.\n"
        "- Use aircraft only when the core question is the aircraft type itself; however, some aircraft wording in this subset is still labeled flight when the utterance is framed as flight search.\n"
        "- If multiple concepts appear, prefer the label supported by ATIS-style examples below over the most literal keyword match.\n\n"
        "ATIS-style boundary examples from the training subset:\n\n"
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


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    if args.disable_wandb or wandb is None:
        return None

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        job_type=args.wandb_job_type,
        config=config,
    )


def main() -> None:
    args = parse_args()
    client = OpenAI()

    train_examples = load_examples(args.train_path)
    eval_examples = load_examples(args.eval_path)
    labels = sorted(set(KNOWN_ATIS_LABELS) | {example.label for example in train_examples})
    valid_labels = set(labels)
    prompt_examples = choose_few_shot_examples(
        train_examples, args.max_train_examples_in_prompt
    )
    assert_prompt_examples_from_train(train_examples, prompt_examples)
    system_prompt = build_system_prompt(labels, prompt_examples)

    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.model}-iter1-{run_id}"
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
        "prompt_examples": [example.text for example in prompt_examples],
        "runner": "experiments/agentic/train_iter1.py",
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
        "boundary_example_count": len(CORE_BOUNDARY_TEXTS),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost_usd": (
            f"{estimated_cost_usd:.6f}" if estimated_cost_usd is not None else ""
        ),
        "runtime_seconds": f"{runtime_seconds:.3f}",
        "keep": "true",
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
                "boundary_example_count": len(CORE_BOUNDARY_TEXTS),
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
