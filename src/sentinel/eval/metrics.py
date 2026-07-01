"""Retrieval and answer-quality metrics used by the eval runner.

Deliberately heuristic (substring/keyword based) rather than LLM-as-judge by default, so
the eval suite runs deterministically in CI without an API key or added cost. An
LLM-judge pass can be layered on top for nuance -- see runner.py's `--llm-judge` flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.retrieval.vector_store import RetrievedChunk


def source_hit_at_k(retrieved: list[RetrievedChunk], expected_source: str) -> bool:
    return any(expected_source in chunk.source for chunk in retrieved)


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


@dataclass(frozen=True)
class CaseResult:
    question: str
    retrieval_hit: bool
    keyword_coverage: float
    answer: str


@dataclass(frozen=True)
class EvalReport:
    results: list[CaseResult]

    @property
    def retrieval_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.retrieval_hit for r in self.results) / len(self.results)

    @property
    def mean_keyword_coverage(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.keyword_coverage for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "retrieval_recall": self.retrieval_recall,
            "mean_keyword_coverage": self.mean_keyword_coverage,
            "cases": [
                {
                    "question": r.question,
                    "retrieval_hit": r.retrieval_hit,
                    "keyword_coverage": r.keyword_coverage,
                    "answer": r.answer,
                }
                for r in self.results
            ],
        }
