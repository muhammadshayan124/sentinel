"""Loads eval fixtures: a JSON list of {"question", "expected_source", "expected_keywords"}."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_source: str
    expected_keywords: list[str]


def load_eval_dataset(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            question=item["question"],
            expected_source=item["expected_source"],
            expected_keywords=item["expected_keywords"],
        )
        for item in raw
    ]
