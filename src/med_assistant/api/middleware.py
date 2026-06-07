import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from med_assistant.core.observability import (
    log_event,
    metrics_registry,
    reset_request_id,
    set_request_id,
)

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request ID and log HTTP latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = set_request_id(request_id)
        started = time.perf_counter()

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            response.headers["X-Request-ID"] = request_id
            log_event(
                logging.INFO,
                "http.request.completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            return response
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            metrics_registry.record_error(
                event="http.request.failed",
                error=str(exc),
                path=request.url.path,
            )
            log_event(
                logging.ERROR,
                "http.request.failed",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise
        finally:
            reset_request_id(token)
