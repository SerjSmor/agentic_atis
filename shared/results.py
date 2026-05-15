from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def append_tsv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        existing_rows = list(reader)
        existing_fieldnames = reader.fieldnames or []

    merged_fieldnames = list(existing_fieldnames)
    for fieldname in fieldnames:
        if fieldname not in merged_fieldnames:
            merged_fieldnames.append(fieldname)

    existing_rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(existing_rows)

