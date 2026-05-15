from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any
from collections import defaultdict

from datasets import Dataset, load_dataset

from shared.constants import SHARED_DATA_DIR


DATASET_NAME = "tuetschek/atis"
DEFAULT_OUTPUT_DIR = SHARED_DATA_DIR
DEFAULT_TRAIN_SIZE = 100
DEFAULT_VAL_SIZE = 50
DEFAULT_TEST_SIZE = 30
DEFAULT_TRAIN_SEED = 1337
DEFAULT_VAL_SEED = 20260501
DEFAULT_TEST_SEED = 4242


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare reproducible ATIS train/test subsets for bounded experiments."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED)
    parser.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    return parser.parse_args()


def sample_split(dataset: Dataset, sample_size: int, seed: int) -> Dataset:
    if sample_size > len(dataset):
        raise ValueError(
            f"Requested sample_size={sample_size}, but split only has {len(dataset)} rows."
        )
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return dataset.select(indices[:sample_size])


def partition_split(dataset: Dataset, val_size: int, seed: int) -> tuple[Dataset, Dataset]:
    rows = list(dataset)
    rng = random.Random(seed)
    grouped_indices: dict[Any, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_indices[row["intent"]].append(index)

    val_indices: list[int] = []
    train_indices: list[int] = []

    # Keep at least one example of each label in train whenever possible.
    for label_indices in grouped_indices.values():
        shuffled = list(label_indices)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_indices.extend(shuffled)
        else:
            val_take = len(shuffled) // 2
            if val_take == len(shuffled):
                val_take -= 1
            val_indices.extend(shuffled[:val_take])
            train_indices.extend(shuffled[val_take:])

    # Rebalance to the requested val_size without removing the last train example of any label.
    label_train_counts: dict[Any, int] = defaultdict(int)
    index_to_label: dict[int, Any] = {}
    for label, label_indices in grouped_indices.items():
        for index in label_indices:
            index_to_label[index] = label
    for index in train_indices:
        label_train_counts[index_to_label[index]] += 1

    rng.shuffle(val_indices)
    rng.shuffle(train_indices)

    while len(val_indices) > val_size:
        moved = val_indices.pop()
        train_indices.append(moved)
        label_train_counts[index_to_label[moved]] += 1

    while len(val_indices) < val_size:
        moved_index = None
        for i, candidate in enumerate(train_indices):
            label = index_to_label[candidate]
            if label_train_counts[label] > 1:
                moved_index = i
                break
        if moved_index is None:
            break
        moved = train_indices.pop(moved_index)
        label_train_counts[index_to_label[moved]] -= 1
        val_indices.append(moved)

    return dataset.select(train_indices), dataset.select(val_indices)


def detect_text_key(row: dict[str, Any]) -> str:
    candidate_keys = ["text", "sentence", "utterance", "query"]
    for key in candidate_keys:
        if key in row:
            return key
    raise KeyError(
        f"Could not find a supported text field in dataset row. Available keys: {sorted(row.keys())}"
    )


def detect_label_key(row: dict[str, Any], text_key: str) -> str:
    candidate_keys = ["intent", "label"]
    for key in candidate_keys:
        if key in row:
            return key

    remaining_keys = [key for key in row.keys() if key != text_key]
    if len(remaining_keys) == 1:
        return remaining_keys[0]

    raise KeyError(
        "Could not infer label field from dataset row. "
        f"Available keys: {sorted(row.keys())}"
    )


def normalize_records(dataset: Dataset) -> list[dict[str, Any]]:
    first_row = dataset[0]
    text_key = detect_text_key(first_row)
    label_key = detect_label_key(first_row, text_key)
    records: list[dict[str, Any]] = []

    for index, row in enumerate(dataset):
        records.append(
            {
                "id": index,
                "text": row[text_key],
                "label": row[label_key],
                "source_text_key": text_key,
                "source_label_key": label_key,
            }
        )

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def partition_records(
    records: list[dict[str, Any]],
    val_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    grouped_indices: dict[Any, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped_indices[record["label"]].append(index)

    val_indices: list[int] = []
    train_indices: list[int] = []
    for label_indices in grouped_indices.values():
        shuffled = list(label_indices)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_indices.extend(shuffled)
        else:
            val_take = len(shuffled) // 2
            if val_take == len(shuffled):
                val_take -= 1
            val_indices.extend(shuffled[:val_take])
            train_indices.extend(shuffled[val_take:])

    label_train_counts: dict[Any, int] = defaultdict(int)
    index_to_label = {index: record["label"] for index, record in enumerate(records)}
    for index in train_indices:
        label_train_counts[index_to_label[index]] += 1

    rng.shuffle(val_indices)
    rng.shuffle(train_indices)

    while len(val_indices) > val_size:
        moved = val_indices.pop()
        train_indices.append(moved)
        label_train_counts[index_to_label[moved]] += 1

    while len(val_indices) < val_size:
        moved_index = None
        for i, candidate in enumerate(train_indices):
            label = index_to_label[candidate]
            if label_train_counts[label] > 1:
                moved_index = i
                break
        if moved_index is None:
            break
        moved = train_indices.pop(moved_index)
        label_train_counts[index_to_label[moved]] -= 1
        val_indices.append(moved)

    train_records = [records[index] for index in train_indices]
    val_records = [records[index] for index in val_indices]
    return train_records, val_records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.val_size <= 0 or args.val_size >= args.train_size:
        raise ValueError(
            f"val_size must be between 1 and {args.train_size - 1}, got {args.val_size}."
        )

    train_path = args.output_dir / "train_subset.jsonl"
    val_path = args.output_dir / "val_subset.jsonl"
    test_path = args.output_dir / "test_subset.jsonl"
    metadata_path = args.output_dir / "subset_metadata.json"

    try:
        dataset = load_dataset(DATASET_NAME)
        train_subset = sample_split(dataset["train"], args.train_size, args.train_seed)
        train_for_prompt, val_subset = partition_split(train_subset, args.val_size, args.val_seed)
        test_subset = sample_split(dataset["test"], args.test_size, args.test_seed)

        train_records = normalize_records(train_for_prompt)
        val_records = normalize_records(val_subset)
        test_records = normalize_records(test_subset)
        source_mode = "huggingface"
    except OSError:
        if not test_path.exists():
            raise

        existing_train_records: list[dict[str, Any]] = []
        if train_path.exists():
            existing_train_records.extend(read_jsonl(train_path))
        if val_path.exists():
            existing_train_records.extend(read_jsonl(val_path))
        existing_test_records = read_jsonl(test_path)
        if len(existing_train_records) != args.train_size:
            raise ValueError(
                "Fallback split regeneration requires existing local train/val files "
                f"with {args.train_size} rows, found {len(existing_train_records)}."
            )
        if len(existing_test_records) != args.test_size:
            raise ValueError(
                "Fallback split regeneration requires an existing test_subset.jsonl "
                f"with {args.test_size} rows, found {len(existing_test_records)}."
            )

        train_records, val_records = partition_records(
            existing_train_records,
            args.val_size,
            args.val_seed,
        )
        test_records = existing_test_records
        source_mode = "local_fallback"

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_jsonl(test_path, test_records)

    metadata = {
        "dataset_name": DATASET_NAME,
        "source_mode": source_mode,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "train_seed": args.train_seed,
        "val_seed": args.val_seed,
        "test_seed": args.test_seed,
        "train_output": str(train_path),
        "val_output": str(val_path),
        "test_output": str(test_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {len(train_records)} train examples to {train_path}")
    print(f"Wrote {len(val_records)} validation examples to {val_path}")
    print(f"Wrote {len(test_records)} test examples to {test_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
