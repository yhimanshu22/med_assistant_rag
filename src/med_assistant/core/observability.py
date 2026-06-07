"""Structured logging, request context, and in-process metrics."""

from __future__ import annotations

import json
import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

OBS_LOGGER = logging.getLogger("med_assistant.observability")


def get_request_id() -> str | None:
    return request_id_ctx.get()


def set_request_id(request_id: str):
    return request_id_ctx.set(request_id)


def reset_request_id(token) -> None:
    request_id_ctx.reset(token)


def configure_logging(level: str = "INFO") -> None:
    """Configure JSON structured logs for the med_assistant loggers."""
    root = logging.getLogger()
    if getattr(configure_logging, "_configured", False):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("med_assistant").setLevel(getattr(logging, level.upper(), logging.INFO))
    configure_logging._configured = True  # type: ignore[attr-defined]


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = get_request_id()
        if rid:
            payload["request_id"] = rid
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "fields") and isinstance(record.fields, dict):
            payload.update(record.fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def log_event(level: int, event: str, **fields: Any) -> None:
    OBS_LOGGER.log(level, event, extra={"event": event, "fields": fields})


def init_sentry(dsn: str | None, environment: str = "development") -> None:
    if not dsn:
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            integrations=[
                FastApiIntegration(),
                LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
            ],
            traces_sample_rate=0.1,
        )
        log_event(logging.INFO, "sentry.initialized", environment=environment)
    except ImportError:
        log_event(
            logging.WARNING,
            "sentry.skipped",
            reason="sentry-sdk not installed; pip install sentry-sdk to enable",
        )


@dataclass
class _LatencyBucket:
    count: int = 0
    total_ms: float = 0.0
    samples: list[float] = field(default_factory=list)
    max_samples: int = 200

    def record(self, value_ms: float) -> None:
        self.count += 1
        self.total_ms += value_ms
        self.samples.append(value_ms)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def summary(self) -> dict[str, float | int]:
        if not self.samples:
            return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        ordered = sorted(self.samples)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
        return {
            "count": self.count,
            "avg_ms": round(self.total_ms / self.count, 2),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
        }


class MetricsRegistry:
    """Thread-safe in-process metrics (suitable for single-instance deployments)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.queries_total = 0
        self.errors_total = 0
        self.retrieval_hits = 0
        self.retrieval_misses = 0
        self.cache_hits = 0
        self.conversational_queries = 0
        self.query_latency = _LatencyBucket()
        self.stage_latency: dict[str, _LatencyBucket] = {
            "rewrite": _LatencyBucket(),
            "retrieve": _LatencyBucket(),
            "llm": _LatencyBucket(),
            "eval": _LatencyBucket(),
        }
        self.eval_score_totals = {"faithfulness": 0.0, "relevance": 0.0, "confidence": 0.0}
        self.eval_score_count = 0
        self.recent_errors: list[dict[str, Any]] = []
        self.max_recent_errors = 20

    def record_error(self, *, event: str, error: str, path: str | None = None) -> None:
        with self._lock:
            self.errors_total += 1
            entry = {
                "at": time.time(),
                "event": event,
                "error": error,
                "path": path,
                "request_id": get_request_id(),
            }
            self.recent_errors.append(entry)
            if len(self.recent_errors) > self.max_recent_errors:
                self.recent_errors.pop(0)
        log_event(logging.ERROR, event, error=error, path=path)

    def record_rag_query(self, obs: dict[str, Any]) -> None:
        with self._lock:
            self.queries_total += 1
            if obs.get("cache_hit"):
                self.cache_hits += 1
            if obs.get("conversational"):
                self.conversational_queries += 1
            if obs.get("weak_retrieval"):
                self.retrieval_misses += 1
            elif obs.get("retrieval_hit"):
                self.retrieval_hits += 1

            total_ms = float(obs.get("total_ms", 0.0))
            self.query_latency.record(total_ms)

            for stage, bucket in self.stage_latency.items():
                stage_ms = float(obs.get("stages_ms", {}).get(stage, 0.0))
                if stage_ms > 0:
                    bucket.record(stage_ms)

            if obs.get("evaluation_enabled"):
                self.eval_score_count += 1
                self.eval_score_totals["faithfulness"] += float(obs.get("faithfulness", 0.0))
                self.eval_score_totals["relevance"] += float(obs.get("relevance", 0.0))
                confidence = obs.get("faithfulness", 0.0) * 0.7 + obs.get("relevance", 0.0) * 0.3
                self.eval_score_totals["confidence"] += confidence

        log_event(
            logging.INFO,
            "rag.query.completed",
            total_ms=obs.get("total_ms"),
            stages_ms=obs.get("stages_ms"),
            retrieval_hit=obs.get("retrieval_hit"),
            weak_retrieval=obs.get("weak_retrieval"),
            cache_hit=obs.get("cache_hit"),
            conversational=obs.get("conversational"),
            evaluation_enabled=obs.get("evaluation_enabled"),
            faithfulness=obs.get("faithfulness"),
            relevance=obs.get("relevance"),
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            hit_denominator = self.retrieval_hits + self.retrieval_misses
            hit_rate = (
                round(self.retrieval_hits / hit_denominator, 4) if hit_denominator else 0.0
            )
            return {
                "queries_total": self.queries_total,
                "errors_total": self.errors_total,
                "cache_hits": self.cache_hits,
                "conversational_queries": self.conversational_queries,
                "retrieval_hit_rate": hit_rate,
                "retrieval_hits": self.retrieval_hits,
                "retrieval_misses": self.retrieval_misses,
                "latency_ms": self.query_latency.summary(),
                "stages_ms": {k: v.summary() for k, v in self.stage_latency.items()},
                "evaluation_scores": {
                    k: {
                        "count": self.eval_score_count,
                        "avg": round(self.eval_score_totals[k] / self.eval_score_count, 4)
                        if self.eval_score_count
                        else 0.0,
                    }
                    for k in self.eval_score_totals
                },
                "recent_errors": list(self.recent_errors),
            }


metrics_registry = MetricsRegistry()
