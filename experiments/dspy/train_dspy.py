from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import dspy
from dspy.teleprompt import GEPA
from dspy.teleprompt import MIPROv2
from sklearn.metrics import f1_score

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    from wandb.integration.dspy import WandbDSPyCallback
except ImportError:  # pragma: no cover
    WandbDSPyCallback = None

from shared.constants import (
    DSPY_DIR,
    KNOWN_ATIS_LABELS,
    LABEL_GUIDANCE,
    MODEL_PRICING_PER_1M_TOKENS,
    SHARED_DATA_DIR,
)
from shared.results import append_tsv_row

DEFAULT_MODEL = "openai/gpt-4.1-nano"
DEFAULT_PROMPT_MODEL = "openai/gpt-4.1-nano"
DEFAULT_OPTIMIZER = "miprov2"
DEFAULT_TRAIN_PATH = SHARED_DATA_DIR / "train_subset.jsonl"
DEFAULT_VAL_PATH = SHARED_DATA_DIR / "val_subset.jsonl"
DEFAULT_EVAL_PATH = SHARED_DATA_DIR / "val_subset.jsonl"
DEFAULT_TEST_PATH = SHARED_DATA_DIR / "test_subset.jsonl"
DEFAULT_RESULTS_PATH = DSPY_DIR / "results.tsv"
DEFAULT_PREDICTIONS_DIR = DSPY_DIR / "runs"
DEFAULT_PROGRAMS_DIR = DSPY_DIR / "programs"
DEFAULT_WANDB_PROJECT = "agentic-atis-compare-dspy"
DEFAULT_METRIC_OBJECTIVE = "exact_match"
RARE_LABEL_MAX_COUNT = 2
FLIGHT_BOUNDARY_TEXTS = [
    "on tuesday i 'd like to find a flight from detroit to st. petersburg that arrives before 10 pm",
    "please give me a flight leaving boston going to washington arriving in washington at 5 o'clock in the afternoon",
    "what flight from boston to atlanta arrives earliest in atlanta",
    "does any airline have an afternoon flight from atlanta to boston",
    "please list all airline flights between denver and boston",
    "when does continental fly from philadelphia to denver on sundays",
]
FLIGHT_PRICE_CONSTRAINT_TEXTS = [
    "what are the cheapest one way flights from denver to atlanta",
    "give me the cheapest flight from dallas to baltimore on saturday",
    "what is the cheapest flight from pittsburgh to atlanta one way",
    "please show me all round trip flights from new york to miami",
]
FLIGHT_ATTRIBUTE_CONSTRAINT_TEXTS = [
    "is there a flight on united airlines from boston to denver",
    "what northwest airlines flights leave denver before noon",
    "what flights from houston to milwaukee on friday on american airlines",
    "information on american airlines flight from washington to philadelphia",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a DSPy ATIS intent classifier with MIPROv2."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--eval-path", "--test-path", dest="eval_path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--eval-split-name", choices=["val", "test"], default="val")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-model", default=DEFAULT_PROMPT_MODEL)
    parser.add_argument("--optimizer", choices=["miprov2", "gepa"], default=DEFAULT_OPTIMIZER)
    parser.add_argument("--reflection-model", default=None)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--programs-dir", type=Path, default=DEFAULT_PROGRAMS_DIR)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-job-type", default="dspy")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--optimizer-auto", choices=["light", "medium", "heavy"], default="medium")
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--max-bootstrapped-demos", type=int, default=2)
    parser.add_argument("--max-labeled-demos", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--metric-objective",
        choices=["weighted_macro", "exact_match"],
        default=DEFAULT_METRIC_OBJECTIVE,
    )
    parser.add_argument("--gepa-max-full-evals", type=int, default=None)
    parser.add_argument("--gepa-max-metric-calls", type=int, default=24)
    parser.add_argument("--gepa-reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--gepa-tracking-val-size", type=int, default=None)
    parser.add_argument(
        "--gepa-candidate-selection-strategy",
        choices=["pareto", "current_best"],
        default="pareto",
    )
    parser.add_argument("--gepa-disable-merge", action="store_true")
    return parser.parse_args()


def json_row_to_example(row: dict[str, object]) -> dspy.Example:
    return dspy.Example(
        id=int(row["id"]),
        text=str(row["text"]),
        label=str(row["label"]),
    ).with_inputs("text")


def load_examples(path: Path) -> list[dspy.Example]:
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            return pickle.load(handle)

    if path.suffix == ".jsonl":
        examples: list[dspy.Example] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                examples.append(json_row_to_example(json.loads(line)))
        return examples

    raise ValueError(f"Unsupported example file format: {path}")


def normalize_label(raw_label: str, valid_labels: list[str]) -> str:
    prediction = raw_label.strip()
    if prediction in valid_labels:
        return prediction

    lowered = prediction.lower()
    for label in valid_labels:
        if lowered == label.lower():
            return label

    return prediction


def build_label_weights(examples: list[dspy.Example]) -> dict[str, float]:
    counts = Counter(example.label for example in examples)
    inverse = {label: 1.0 / count for label, count in counts.items()}
    max_inverse = max(inverse.values())
    return {label: value / max_inverse for label, value in inverse.items()}


def prioritize_demo_candidates(examples: list[dspy.Example]) -> list[dspy.Example]:
    counts = Counter(example.label for example in examples)
    rare_examples = [
        example for example in examples if counts[example.label] <= RARE_LABEL_MAX_COUNT
    ]
    by_text = {example.text: example for example in examples}
    prioritized_flight_examples = [
        by_text[text]
        for text in (
            FLIGHT_BOUNDARY_TEXTS
            + FLIGHT_PRICE_CONSTRAINT_TEXTS
            + FLIGHT_ATTRIBUTE_CONSTRAINT_TEXTS
        )
        if text in by_text and by_text[text] not in rare_examples
    ]

    selected_ids = {
        (getattr(example, "id", None), example.text, example.label)
        for example in rare_examples + prioritized_flight_examples
    }
    remaining = [
        example
        for example in examples
        if (getattr(example, "id", None), example.text, example.label) not in selected_ids
    ]
    return rare_examples + prioritized_flight_examples + remaining


def select_gepa_tracking_valset(
    examples: list[dspy.Example],
    max_size: int | None,
    seed: int,
) -> list[dspy.Example]:
    if max_size is None or max_size <= 0 or max_size >= len(examples):
        return examples

    indexed = list(enumerate(examples))
    counts = Counter(example.label for example in examples)
    rare_pairs = [
        (index, example)
        for index, example in indexed
        if counts[example.label] <= RARE_LABEL_MAX_COUNT
    ]
    common_pairs = [
        (index, example)
        for index, example in indexed
        if counts[example.label] > RARE_LABEL_MAX_COUNT
    ]

    chosen_indices = {index for index, _ in rare_pairs[:max_size]}
    remaining_slots = max_size - len(chosen_indices)
    if remaining_slots > 0:
        rng = __import__("random").Random(seed)
        shuffled_common = common_pairs[:]
        rng.shuffle(shuffled_common)
        for index, _ in shuffled_common[:remaining_slots]:
            chosen_indices.add(index)

    return [example for index, example in indexed if index in chosen_indices]


def usage_totals_from_prediction(prediction: dspy.Prediction) -> dict[str, int]:
    return aggregate_usage_totals(usage_by_lm_from_prediction(prediction))


def usage_by_lm_from_prediction(prediction: dspy.Prediction) -> dict[str, dict[str, int]]:
    usage = {}
    if hasattr(prediction, "get_lm_usage"):
        usage = prediction.get_lm_usage() or {}

    normalized_usage: dict[str, dict[str, int]] = {}
    for lm_name, stats in usage.items():
        if not isinstance(stats, dict):
            continue
        normalized_usage[lm_name] = {
            "prompt_tokens": int(stats.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(stats.get("completion_tokens", 0) or 0),
            "total_tokens": int(stats.get("total_tokens", 0) or 0),
        }
    return normalized_usage


def merge_usage_by_lm(
    totals_by_lm: dict[str, dict[str, int]],
    usage_by_lm: dict[str, dict[str, int]],
) -> None:
    for lm_name, stats in usage_by_lm.items():
        current = totals_by_lm.setdefault(
            lm_name,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        current["prompt_tokens"] += int(stats.get("prompt_tokens", 0) or 0)
        current["completion_tokens"] += int(stats.get("completion_tokens", 0) or 0)
        current["total_tokens"] += int(stats.get("total_tokens", 0) or 0)


def aggregate_usage_totals(usage_by_lm: dict[str, dict[str, int]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for stats in usage_by_lm.values():
        totals["prompt_tokens"] += int(stats.get("prompt_tokens", 0) or 0)
        totals["completion_tokens"] += int(stats.get("completion_tokens", 0) or 0)
        totals["total_tokens"] += int(stats.get("total_tokens", 0) or 0)
    return totals


def pricing_for_model(model: str) -> dict[str, float] | None:
    pricing = MODEL_PRICING_PER_1M_TOKENS.get(model)
    if pricing is not None:
        return pricing
    if "/" in model:
        _, stripped_model = model.split("/", 1)
        return MODEL_PRICING_PER_1M_TOKENS.get(stripped_model)
    return None


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    pricing = pricing_for_model(model)
    if pricing is None:
        return None
    return (
        (prompt_tokens / 1_000_000) * pricing["input"]
        + (completion_tokens / 1_000_000) * pricing["output"]
    )


def estimate_usage_cost_usd(usage_by_lm: dict[str, dict[str, int]]) -> float | None:
    estimated_cost = 0.0
    found_priced_usage = False
    for lm_name, stats in usage_by_lm.items():
        lm_cost = estimate_cost_usd(
            lm_name,
            int(stats.get("prompt_tokens", 0) or 0),
            int(stats.get("completion_tokens", 0) or 0),
        )
        if lm_cost is None:
            continue
        found_priced_usage = True
        estimated_cost += lm_cost
    return estimated_cost if found_priced_usage else None


def sum_optional_costs(*costs: float | None) -> float | None:
    present_costs = [cost for cost in costs if cost is not None]
    if not present_costs:
        return None
    return sum(present_costs)


def optimizer_display_name(optimizer: str) -> str:
    return {
        "miprov2": "MIPROv2",
        "gepa": "GEPA",
    }[optimizer]


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


ATIS_DISAMBIGUATION_RULES = (
    "Choose the single best ATIS intent for the user's main request.\n"
    "- Use flight when the user wants flights to be found or listed, even if they mention "
    "cheapest, nonstop, round trip, class, airline, aircraft, or meal constraints.\n"
    "- Price constraints such as cheap, cheapest, discount, coach, one way, round trip, or less "
    "than a dollar amount are still flight when the user is searching for flights rather than "
    "asking for fare information itself.\n"
    "- Use airfare only when the user is asking about fare or ticket price rather than asking "
    "to list or find flights.\n"
    "- Use flight+airfare only when the user explicitly asks for both fares and flights together; "
    "a price constraint alone does not make the label flight+airfare.\n"
    "- Use airline only when the main intent is to identify which airline operates or serves a route.\n"
    "- Use aircraft only when the main intent is the aircraft type itself, not when aircraft is "
    "just an attribute of a flight search.\n"
    "- Mentions of airport names, airline names, aircraft types, or meals do not override flight "
    "when the user is still asking to find or list flights.\n"
    "- Use flight_time only when the user is asking for arrival time, departure time, or duration "
    "itself; if times are only search constraints, use flight.\n"
    "- Use abbreviation when the user asks what a code, shorthand, or term means, even if the term "
    "contains a word like restriction.\n"
    "Return exactly one label from the valid label list."
)


def build_label_guidance(labels: list[str]) -> str:
    return "\n".join(
        f"- {label}: {LABEL_GUIDANCE.get(label, 'ATIS intent label')}"
        for label in labels
    )


class ATISIntentSignature(dspy.Signature):
    __doc__ = ATIS_DISAMBIGUATION_RULES

    text = dspy.InputField(desc="ATIS user query to classify")
    valid_labels = dspy.InputField(desc="Comma-separated valid ATIS labels")
    label_guidance = dspy.InputField(desc="Short description of each ATIS label")
    label = dspy.OutputField(desc="Exactly one ATIS label from valid_labels")


class ATISIntentClassifier(dspy.Module):
    def __init__(self, labels: list[str]):
        super().__init__()
        self.labels = labels
        self.valid_labels = ", ".join(labels)
        self.label_guidance = build_label_guidance(labels)
        self.classify = dspy.Predict(ATISIntentSignature)

    def forward(self, text: str) -> dspy.Prediction:
        prediction = self.classify(
            text=text,
            valid_labels=self.valid_labels,
            label_guidance=self.label_guidance,
        )
        normalized_label = normalize_label(getattr(prediction, "label", ""), self.labels)
        reasoning = getattr(prediction, "reasoning", "")
        rationale = getattr(prediction, "rationale", "")
        return dspy.Prediction(
            label=normalized_label,
            raw_label=getattr(prediction, "label", ""),
            reasoning=reasoning or rationale,
        )


def make_metric(
    objective: str,
    valid_labels: list[str],
    label_weights: dict[str, float],
):
    def metric(
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any = None,
    ) -> float:
        gold_label = normalize_label(example.label, valid_labels)
        predicted_label = normalize_label(getattr(prediction, "label", ""), valid_labels)
        matched = gold_label == predicted_label
        if objective == "exact_match":
            return 1.0 if matched else 0.0
        return float(label_weights.get(gold_label, 1.0)) if matched else 0.0

    return metric


def make_gepa_metric(
    objective: str,
    valid_labels: list[str],
    label_weights: dict[str, float],
):
    def metric(
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> float | dspy.Prediction:
        gold_label = normalize_label(gold.label, valid_labels)
        predicted_label = normalize_label(getattr(pred, "label", ""), valid_labels)
        matched = gold_label == predicted_label
        score = 1.0 if matched else 0.0
        if objective == "weighted_macro":
            score = float(label_weights.get(gold_label, 1.0)) if matched else 0.0

        if pred_name is None:
            return score

        if matched:
            feedback = (
                f"Correctly predicted {predicted_label}. Preserve the rule that maps this query "
                "to the user's main ATIS intent."
            )
        else:
            feedback = (
                f"Incorrect label. Gold label: {gold_label}. Predicted label: {predicted_label}. "
                "Focus on the user's main ATIS intent instead of salient surface words. "
                "Use ATIS boundary rules carefully, especially for flight versus airline, aircraft, "
                "flight+airfare, abbreviation, and flight_time. Rare classes matter more in this objective."
            )

        return dspy.Prediction(score=score, feedback=feedback)

    return metric


def compile_program(
    args: argparse.Namespace,
    program: dspy.Module,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    student_lm: dspy.LM,
    prompt_lm: dspy.LM,
    teacher_lm: dspy.LM | None,
    reflection_lm: dspy.LM | None,
    run_name: str,
    optimizer_name: str,
    programs_dir: Path,
    metric: Any,
    gepa_metric: Any,
) -> dspy.Module:
    if args.optimizer == "miprov2":
        optimizer = MIPROv2(
            metric=metric,
            auto=args.optimizer_auto,
            prompt_model=prompt_lm,
            task_model=student_lm,
            teacher_settings={"lm": teacher_lm} if teacher_lm is not None else None,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            num_threads=args.num_threads,
            seed=args.seed,
            verbose=args.verbose,
            track_stats=True,
        )

        compile_kwargs: dict[str, Any] = {
            "student": program,
            "trainset": trainset,
            "valset": valset,
            "requires_permission_to_run": False,
        }
        if args.num_trials is not None:
            compile_kwargs["num_trials"] = args.num_trials

        return optimizer.compile(**compile_kwargs)

    if teacher_lm is not None:
        raise ValueError("GEPA does not support teacher_model in this codepath.")
    if reflection_lm is None:
        raise ValueError("GEPA requires --reflection-model or a prompt model fallback.")

    optimizer = GEPA(
        metric=gepa_metric,
        auto=args.optimizer_auto if args.gepa_max_full_evals is None and args.gepa_max_metric_calls is None else None,
        max_full_evals=args.gepa_max_full_evals,
        max_metric_calls=args.gepa_max_metric_calls,
        reflection_minibatch_size=args.gepa_reflection_minibatch_size,
        candidate_selection_strategy=args.gepa_candidate_selection_strategy,
        reflection_lm=reflection_lm,
        use_merge=not args.gepa_disable_merge,
        num_threads=args.num_threads,
        seed=args.seed,
        track_stats=True,
        log_dir=str(programs_dir / f"{run_name}-{optimizer_name.lower()}-logs"),
    )
    return optimizer.compile(student=program, trainset=trainset, valset=valset)


def evaluate_program(
    program: dspy.Module,
    evalset: list[dspy.Example],
    valid_labels: list[str],
    predictions_path: Path,
) -> dict[str, Any]:
    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    usage_by_lm: dict[str, dict[str, int]] = {}

    with predictions_path.open("w", encoding="utf-8") as handle:
        for example in evalset:
            prediction = program(text=example.text)
            normalized_label = normalize_label(getattr(prediction, "label", ""), valid_labels)
            prediction_usage_by_lm = usage_by_lm_from_prediction(prediction)
            merge_usage_by_lm(usage_by_lm, prediction_usage_by_lm)
            usage_totals = aggregate_usage_totals(prediction_usage_by_lm)
            gold_labels.append(example.label)
            predicted_labels.append(normalized_label)

            handle.write(
                json.dumps(
                    {
                        "id": getattr(example, "id", None),
                        "text": example.text,
                        "gold_label": example.label,
                        "predicted_label": normalized_label,
                        "raw_label": getattr(prediction, "raw_label", normalized_label),
                        "reasoning": getattr(prediction, "reasoning", ""),
                        "prompt_tokens": usage_totals["prompt_tokens"],
                        "completion_tokens": usage_totals["completion_tokens"],
                        "total_tokens": usage_totals["total_tokens"],
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    usage_totals = aggregate_usage_totals(usage_by_lm)
    return {
        "macro_f1": f1_score(gold_labels, predicted_labels, average="macro"),
        "weighted_f1": f1_score(gold_labels, predicted_labels, average="weighted"),
        "micro_f1": f1_score(gold_labels, predicted_labels, average="micro"),
        "prompt_tokens": usage_totals["prompt_tokens"],
        "completion_tokens": usage_totals["completion_tokens"],
        "total_tokens": usage_totals["total_tokens"],
        "usage_by_lm": usage_by_lm,
        "predicted_label_distribution": dict(Counter(predicted_labels)),
    }


def main() -> None:
    args = parse_args()
    raw_trainset = load_examples(args.train_path)
    train_label_counts = Counter(example.label for example in raw_trainset)
    rare_train_example_count = sum(
        1 for example in raw_trainset if train_label_counts[example.label] <= RARE_LABEL_MAX_COUNT
    )
    trainset = prioritize_demo_candidates(raw_trainset)
    valset = load_examples(args.val_path)
    evalset = load_examples(args.eval_path)
    gepa_valset = (
        select_gepa_tracking_valset(valset, args.gepa_tracking_val_size, args.seed)
        if args.optimizer == "gepa"
        else valset
    )

    valid_labels = sorted(set(KNOWN_ATIS_LABELS) | {example.label for example in trainset + valset})
    label_weights = build_label_weights(trainset)
    metric = make_metric(args.metric_objective, valid_labels, label_weights)
    gepa_metric = make_gepa_metric(args.metric_objective, valid_labels, label_weights)
    student_lm = dspy.LM(args.model, cache=False)
    prompt_lm = dspy.LM(args.prompt_model, cache=False)
    reflection_model = args.reflection_model or args.prompt_model
    reflection_lm = dspy.LM(reflection_model, cache=False) if args.optimizer == "gepa" else None
    teacher_lm = dspy.LM(args.teacher_model, cache=False) if args.teacher_model else None
    dspy.configure(lm=student_lm, track_usage=True)

    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    optimizer_name = optimizer_display_name(args.optimizer)
    run_name = args.run_name or f"dspy-{args.optimizer}-{run_id}"
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    args.programs_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.predictions_dir / f"{run_name}.jsonl"
    program_path = args.programs_dir / f"{run_name}.json"

    config = {
        "model": args.model,
        "prompt_model": args.prompt_model,
        "reflection_model": reflection_model if args.optimizer == "gepa" else "",
        "teacher_model": args.teacher_model,
        "optimizer": optimizer_name,
        "optimizer_auto": args.optimizer_auto,
        "num_trials": args.num_trials,
        "max_bootstrapped_demos": args.max_bootstrapped_demos,
        "max_labeled_demos": args.max_labeled_demos,
        "gepa_max_full_evals": args.gepa_max_full_evals,
        "gepa_max_metric_calls": args.gepa_max_metric_calls,
        "gepa_reflection_minibatch_size": args.gepa_reflection_minibatch_size,
        "gepa_tracking_val_size": args.gepa_tracking_val_size,
        "gepa_candidate_selection_strategy": args.gepa_candidate_selection_strategy,
        "gepa_use_merge": not args.gepa_disable_merge,
        "metric_objective": args.metric_objective,
        "label_weights": label_weights,
        "seed": args.seed,
        "train_size": len(trainset),
        "val_size": len(valset),
        "gepa_tracking_val_size": len(gepa_valset) if args.optimizer == "gepa" else "",
        "eval_size": len(evalset),
        "eval_split_name": args.eval_split_name,
        "label_count": len(valid_labels),
        "labels": valid_labels,
        "rare_label_max_count": RARE_LABEL_MAX_COUNT,
        "rare_train_example_count": rare_train_example_count,
    }
    wandb_run = maybe_init_wandb(args, config)

    added_callback = None
    callbacks = getattr(dspy.settings, "callbacks", None)
    if wandb_run is not None and callbacks is not None and WandbDSPyCallback is not None:
        added_callback = WandbDSPyCallback(run=wandb_run)
        callbacks.append(added_callback)

    try:
        compile_started = perf_counter()
        program = ATISIntentClassifier(valid_labels)
        with dspy.track_usage() as compile_usage_tracker:
            optimized_program = compile_program(
                args=args,
                program=program,
                trainset=trainset,
                valset=gepa_valset,
                student_lm=student_lm,
                prompt_lm=prompt_lm,
                teacher_lm=teacher_lm,
                reflection_lm=reflection_lm,
                run_name=run_name,
                optimizer_name=optimizer_name,
                programs_dir=args.programs_dir,
                metric=metric,
                gepa_metric=gepa_metric,
            )
        compile_runtime_seconds = perf_counter() - compile_started
        compile_usage_by_lm = compile_usage_tracker.get_total_tokens()
        compile_usage_totals = aggregate_usage_totals(compile_usage_by_lm)

        optimized_program.save(str(program_path))

        eval_started = perf_counter()
        evaluation = evaluate_program(
            program=optimized_program,
            evalset=evalset,
            valid_labels=valid_labels,
            predictions_path=predictions_path,
        )
        eval_runtime_seconds = perf_counter() - eval_started
        eval_usage_by_lm = evaluation["usage_by_lm"]
        eval_usage_totals = aggregate_usage_totals(eval_usage_by_lm)
        total_prompt_tokens = compile_usage_totals["prompt_tokens"] + eval_usage_totals["prompt_tokens"]
        total_completion_tokens = (
            compile_usage_totals["completion_tokens"] + eval_usage_totals["completion_tokens"]
        )
        total_tokens = compile_usage_totals["total_tokens"] + eval_usage_totals["total_tokens"]
        estimated_compile_cost_usd = estimate_usage_cost_usd(compile_usage_by_lm)
        estimated_eval_cost_usd = estimate_usage_cost_usd(eval_usage_by_lm)
        estimated_total_run_cost_usd = sum_optional_costs(
            estimated_compile_cost_usd,
            estimated_eval_cost_usd,
        )

        result_row = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp_utc": started_at.isoformat(),
            "optimizer": optimizer_name,
            "model": args.model,
            "prompt_model": args.prompt_model,
            "reflection_model": reflection_model if args.optimizer == "gepa" else "",
            "teacher_model": args.teacher_model or "",
            "optimizer_auto": args.optimizer_auto,
            "num_trials": args.num_trials or "",
            "gepa_max_full_evals": args.gepa_max_full_evals or "",
            "gepa_max_metric_calls": args.gepa_max_metric_calls or "",
            "gepa_reflection_minibatch_size": (
                args.gepa_reflection_minibatch_size if args.optimizer == "gepa" else ""
            ),
            "gepa_tracking_val_size": (
                len(gepa_valset) if args.optimizer == "gepa" else ""
            ),
            "gepa_candidate_selection_strategy": (
                args.gepa_candidate_selection_strategy if args.optimizer == "gepa" else ""
            ),
            "gepa_use_merge": (not args.gepa_disable_merge) if args.optimizer == "gepa" else "",
            "metric_objective": args.metric_objective,
            "macro_f1": f"{evaluation['macro_f1']:.6f}",
            "weighted_f1": f"{evaluation['weighted_f1']:.6f}",
            "micro_f1": f"{evaluation['micro_f1']:.6f}",
            "eval_split_name": args.eval_split_name,
            "train_size": len(trainset),
            "val_size": len(valset),
            "eval_size": len(evalset),
            "label_count": len(valid_labels),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "compile_prompt_tokens": compile_usage_totals["prompt_tokens"],
            "compile_completion_tokens": compile_usage_totals["completion_tokens"],
            "compile_total_tokens": compile_usage_totals["total_tokens"],
            "eval_prompt_tokens": eval_usage_totals["prompt_tokens"],
            "eval_completion_tokens": eval_usage_totals["completion_tokens"],
            "eval_total_tokens": eval_usage_totals["total_tokens"],
            "estimated_cost_usd": (
                f"{estimated_total_run_cost_usd:.6f}" if estimated_total_run_cost_usd is not None else ""
            ),
            "estimated_compile_cost_usd": (
                f"{estimated_compile_cost_usd:.6f}" if estimated_compile_cost_usd is not None else ""
            ),
            "estimated_eval_cost_usd": (
                f"{estimated_eval_cost_usd:.6f}" if estimated_eval_cost_usd is not None else ""
            ),
            "estimated_total_run_cost_usd": (
                f"{estimated_total_run_cost_usd:.6f}" if estimated_total_run_cost_usd is not None else ""
            ),
            "compile_runtime_seconds": f"{compile_runtime_seconds:.3f}",
            "eval_runtime_seconds": f"{eval_runtime_seconds:.3f}",
            "program_path": str(program_path),
            "predictions_path": str(predictions_path),
        }
        append_tsv_row(args.results_path, result_row)

        if wandb_run is not None:
            wandb.log(
                {
                    "macro_f1": evaluation["macro_f1"],
                    "weighted_f1": evaluation["weighted_f1"],
                    "micro_f1": evaluation["micro_f1"],
                    "optimizer": optimizer_name,
                    "metric_objective": args.metric_objective,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                    "compile_prompt_tokens": compile_usage_totals["prompt_tokens"],
                    "compile_completion_tokens": compile_usage_totals["completion_tokens"],
                    "compile_total_tokens": compile_usage_totals["total_tokens"],
                    "eval_prompt_tokens": eval_usage_totals["prompt_tokens"],
                    "eval_completion_tokens": eval_usage_totals["completion_tokens"],
                    "eval_total_tokens": eval_usage_totals["total_tokens"],
                    "estimated_cost_usd": estimated_total_run_cost_usd,
                    "estimated_compile_cost_usd": estimated_compile_cost_usd,
                    "estimated_eval_cost_usd": estimated_eval_cost_usd,
                    "estimated_total_run_cost_usd": estimated_total_run_cost_usd,
                    "compile_runtime_seconds": compile_runtime_seconds,
                    "eval_runtime_seconds": eval_runtime_seconds,
                    "eval_size": len(evalset),
                    "eval_split_name": args.eval_split_name,
                    "label_count": len(valid_labels),
                }
            )
            wandb_run.summary["predicted_label_distribution"] = evaluation[
                "predicted_label_distribution"
            ]
            wandb_run.summary["predictions_path"] = str(predictions_path)
            wandb_run.summary["program_path"] = str(program_path)
            wandb_run.summary["compile_usage_by_lm"] = compile_usage_by_lm
            wandb_run.summary["eval_usage_by_lm"] = eval_usage_by_lm

        print(json.dumps(result_row, indent=2))
    finally:
        if added_callback is not None and callbacks is not None and added_callback in callbacks:
            callbacks.remove(added_callback)
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY must be set before running train_dspy.py.")
    main()
