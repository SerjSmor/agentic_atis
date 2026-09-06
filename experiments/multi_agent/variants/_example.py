"""Reference variant. Copy this to variant_NN.py; do not evaluate it directly.

The leading underscore keeps `inv run-wave` from picking it up.

A variant defines any subset of these three hooks. Whatever you omit falls back to
the default in train.py, so overriding one function does not oblige you to
reimplement the others.

    build_system_prompt(labels, prompt_examples, boundary_examples) -> str
    choose_few_shot_examples(train_examples, max_examples) -> list[Example]
    choose_boundary_examples(train_examples) -> list[Example]

`Example` is a dataclass with `.id`, `.text`, `.label`. Import anything you need
from `multi_agent_train` — that is train.py, already loaded, exposed under a stable
name so you can extend the defaults instead of rewriting them.
"""

from __future__ import annotations

from multi_agent_train import build_system_prompt as default_build_system_prompt


def build_system_prompt(labels, prompt_examples, boundary_examples) -> str:
    """Append one boundary rule to the default prompt.

    This is the smallest useful variant shape: take the default and add the single
    rule your assigned angle is about, so the delta in macro F1 is attributable.
    """
    base = default_build_system_prompt(labels, prompt_examples, boundary_examples)
    return base + (
        "\n\nBoundary rule:\n"
        "- If the question asks *which carrier* operates a route, the label is "
        "`airline`, not `flight` — even when the words 'flight' or 'flights' appear.\n"
    )
