from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import random

import dspy

from shared.constants import DSPY_DIR, SHARED_DATA_DIR


DEFAULT_TRAIN_PATH = SHARED_DATA_DIR / "train_subset.jsonl"
DEFAULT_VAL_PATH = SHARED_DATA_DIR / "val_subset.jsonl"
DEFAULT_TEST_PATH = SHARED_DATA_DIR / "test_subset.jsonl"
DEFAULT_OUTPUT_DIR = DSPY_DIR / "data"
DEFAULT_DSPY_TRAIN_SIZE = 100
DEFAULT_VAL_SIZE = 50
DEFAULT_SPLIT_SEED = 20260501


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare DSPy Example pickles from the ATIS subset files."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dspy-train-size", type=int, default=DEFAULT_DSPY_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def to_example(row: dict[str, object]) -> dspy.Example:
    return dspy.Example(
        id=int(row["id"]),
        text=str(row["text"]),
        label=str(row["label"]),
    ).with_inputs("text")


def save_pickle(path: Path, examples: list[dspy.Example]) -> None:
    with path.open("wb") as handle:
        pickle.dump(examples, handle)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(args.train_path)
    val_rows_from_agentic = load_rows(args.val_path)
    test_rows = load_rows(args.test_path)
    full_train_pool = train_rows + val_rows_from_agentic

    if args.dspy_train_size <= 1 or args.dspy_train_size > len(full_train_pool):
        raise ValueError(
            "dspy_train_size must be between 2 and "
            f"{len(full_train_pool)}, got {args.dspy_train_size}."
        )

    if args.val_size <= 0 or args.val_size >= args.dspy_train_size:
        raise ValueError(
            "val_size must be between 1 and "
            f"{args.dspy_train_size - 1}, got {args.val_size}."
        )

    shuffled_train_rows = list(full_train_pool)
    random.Random(args.split_seed).shuffle(shuffled_train_rows)
    dspy_train_rows = shuffled_train_rows[: args.dspy_train_size]

    val_rows = dspy_train_rows[: args.val_size]
    opt_train_rows = dspy_train_rows[args.val_size :]

    train_examples = [to_example(row) for row in opt_train_rows]
    val_examples = [to_example(row) for row in val_rows]
    test_examples = [to_example(row) for row in test_rows]

    train_pickle_path = args.output_dir / "train_examples.pkl"
    val_pickle_path = args.output_dir / "val_examples.pkl"
    test_pickle_path = args.output_dir / "test_examples.pkl"
    metadata_path = args.output_dir / "metadata.json"

    save_pickle(train_pickle_path, train_examples)
    save_pickle(val_pickle_path, val_examples)
    save_pickle(test_pickle_path, test_examples)

    metadata = {
        "source_train_path": str(args.train_path),
        "source_val_path": str(args.val_path),
        "source_test_path": str(args.test_path),
        "split_seed": args.split_seed,
        "dspy_train_size": args.dspy_train_size,
        "val_size": args.val_size,
        "optimizer_train_size": len(train_examples),
        "val_count": len(val_examples),
        "test_count": len(test_examples),
        "train_pickle_path": str(train_pickle_path),
        "val_pickle_path": str(val_pickle_path),
        "test_pickle_path": str(test_pickle_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote optimizer train examples to {train_pickle_path}")
    print(f"Wrote validation examples to {val_pickle_path}")
    print(f"Wrote test examples to {test_pickle_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
