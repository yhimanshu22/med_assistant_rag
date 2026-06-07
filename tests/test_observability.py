"""Tests for observability metrics and structured logging helpers."""

from med_assistant.core.observability import MetricsRegistry


def test_metrics_registry_records_query_and_hit_rate():
    registry = MetricsRegistry()

    registry.record_rag_query(
        {
            "total_ms": 1200.0,
            "stages_ms": {"rewrite": 10.0, "retrieve": 200.0, "llm": 900.0, "eval": 90.0},
            "retrieval_hit": True,
            "weak_retrieval": False,
            "cache_hit": False,
            "conversational": False,
            "evaluation_enabled": True,
            "faithfulness": 0.8,
            "relevance": 0.6,
        }
    )
    registry.record_rag_query(
        {
            "total_ms": 800.0,
            "stages_ms": {"rewrite": 0.0, "retrieve": 150.0, "llm": 650.0, "eval": 0.0},
            "retrieval_hit": False,
            "weak_retrieval": True,
            "cache_hit": False,
            "conversational": False,
            "evaluation_enabled": False,
        }
    )

    snapshot = registry.snapshot()
    assert snapshot["queries_total"] == 2
    assert snapshot["retrieval_hits"] == 1
    assert snapshot["retrieval_misses"] == 1
    assert snapshot["retrieval_hit_rate"] == 0.5
    assert snapshot["latency_ms"]["count"] == 2
    assert snapshot["stages_ms"]["retrieve"]["count"] == 2
    assert snapshot["evaluation_scores"]["faithfulness"]["count"] == 1


def test_metrics_registry_records_errors():
    registry = MetricsRegistry()
    registry.record_error(event="test.error", error="boom", path="/query")
    snapshot = registry.snapshot()
    assert snapshot["errors_total"] == 1
    assert snapshot["recent_errors"][0]["error"] == "boom"
