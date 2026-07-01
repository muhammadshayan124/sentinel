from sentinel.eval.metrics import CaseResult, EvalReport, keyword_coverage, source_hit_at_k
from sentinel.retrieval.vector_store import RetrievedChunk


def test_source_hit_at_k_true_when_source_present():
    retrieved = [RetrievedChunk(text="x", source="docs/a.pdf", distance=0.1)]
    assert source_hit_at_k(retrieved, "a.pdf") is True


def test_source_hit_at_k_false_when_source_absent():
    retrieved = [RetrievedChunk(text="x", source="docs/b.pdf", distance=0.1)]
    assert source_hit_at_k(retrieved, "a.pdf") is False


def test_keyword_coverage_full_match():
    assert keyword_coverage("the quick brown fox", ["quick", "fox"]) == 1.0


def test_keyword_coverage_partial_match():
    assert keyword_coverage("the quick brown fox", ["quick", "zebra"]) == 0.5


def test_keyword_coverage_empty_expected_is_perfect():
    assert keyword_coverage("anything", []) == 1.0


def test_eval_report_aggregates_correctly():
    report = EvalReport(
        results=[
            CaseResult(question="q1", retrieval_hit=True, keyword_coverage=1.0, answer="a1"),
            CaseResult(question="q2", retrieval_hit=False, keyword_coverage=0.5, answer="a2"),
        ]
    )
    assert report.retrieval_recall == 0.5
    assert report.mean_keyword_coverage == 0.75


def test_eval_report_empty_results():
    report = EvalReport(results=[])
    assert report.retrieval_recall == 0.0
    assert report.mean_keyword_coverage == 0.0
