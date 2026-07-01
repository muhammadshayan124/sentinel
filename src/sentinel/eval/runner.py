"""Runs the eval dataset end-to-end against a live (already-ingested) vector store + agent,
and writes a JSON report. This is what CI runs on every PR against a small fixture corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from sentinel.agent.llm import AnthropicClient
from sentinel.agent.loop import run_agent
from sentinel.agent.tools import build_toolset
from sentinel.eval.dataset import load_eval_dataset
from sentinel.eval.metrics import CaseResult, EvalReport, keyword_coverage, source_hit_at_k
from sentinel.retrieval.retriever import retrieve
from sentinel.retrieval.vector_store import VectorStore

logger = structlog.get_logger(__name__)


def run_eval(dataset_path: Path, store: VectorStore, report_path: Path | None = None) -> EvalReport:
    cases = load_eval_dataset(dataset_path)
    llm = AnthropicClient()
    tools = build_toolset(store)

    results: list[CaseResult] = []
    for case in cases:
        retrieval = retrieve(case.question, store)
        hit = source_hit_at_k(retrieval.text_chunks, case.expected_source)

        agent_result = run_agent(case.question, llm=llm, tools=tools)
        coverage = keyword_coverage(agent_result.answer, case.expected_keywords)

        results.append(
            CaseResult(
                question=case.question,
                retrieval_hit=hit,
                keyword_coverage=coverage,
                answer=agent_result.answer,
            )
        )
        logger.info("eval_case_complete", question=case.question, hit=hit, coverage=coverage)

    report = EvalReport(results=results)
    if report_path is not None:
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report
