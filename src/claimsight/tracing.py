"""Langfuse tracing: one trace per claim, a span per agent, a generation per Claude call.

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and the `llm` extra is
installed; then every call here is a no-op. What lands in Langfuse feeds the dashboards in
dashboards/claimsight.json (README, "Langfuse dashboards"):

* trace `claim`, tagged with loss type, decision and mode (replay or live)
* spans `intake`, `vision`, `coverage`, ... (LangGraph runs the parallel ones in threads
  that copy the context, so they nest under the claim)
* generations `coverage_review` per Claude call, with model, token usage (cache reads
  separate) and cost at list prices (agents/coverage_review.py PRICES)
* scores per claim: decision, escalated, confidence, review verdict and route, cost, and
  when the eval runner knows the label, decision_correct

Claim text (statements, clause text) stays out unless CLAIMSIGHT_TRACE_CONTENT=true: real
statements carry names, addresses and plates.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any

log = logging.getLogger(__name__)

_client: Any = None
_resolved = False


def _langfuse():
    """The Langfuse client, or None when tracing is off. Resolved once."""
    global _client, _resolved
    if _resolved:
        return _client
    _resolved = True
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import get_client
    except ImportError:
        log.warning("LANGFUSE_* is set but langfuse is not installed; tracing is off.")
        return None
    _client = get_client()
    return _client


def set_backend(client: Any) -> None:
    """Swap the client (tests pass a recorder; None turns tracing off)."""
    global _client, _resolved
    _client, _resolved = client, True


def enabled() -> bool:
    return _langfuse() is not None


def content(value: Any) -> Any:
    """Claim text goes to Langfuse only when CLAIMSIGHT_TRACE_CONTENT is on."""
    return value if os.getenv("CLAIMSIGHT_TRACE_CONTENT", "").lower() == "true" else None


@contextmanager
def claim_trace(claim_id: str, loss_type: str, mode: str, tags: list[str] | None = None):
    """Root observation for one claim. Everything inside nests under it."""
    lf = _langfuse()
    if lf is None:
        yield None
        return
    propagate_attributes = getattr(lf, "propagate_attributes", None)
    if propagate_attributes is None:  # the real client: it's a module-level function
        from langfuse import propagate_attributes

    with propagate_attributes(
        trace_name="claim",
        session_id=claim_id,  # a claim's submit and review land in one session
        tags=[f"loss:{loss_type}", f"mode:{mode}", *(tags or [])],
        metadata={"claim_id": claim_id},
    ):
        with lf.start_as_current_observation(name="claim", as_type="agent") as root:
            yield root


def span(name: str):
    """Decorator: run the function inside a span named `name` (a no-op when off)."""

    def wrap(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            lf = _langfuse()
            ctx = lf.start_as_current_observation(name=name) if lf else nullcontext()
            with ctx:
                return fn(*args, **kwargs)

        return inner

    return wrap


def generation(name: str, usage: dict | None, cost: float, output: Any = None, **metadata):
    """Record one model call that already happened (live or replayed from a cassette)."""
    lf = _langfuse()
    # Only inside a claim's trace: eval sets scored outside the graph would otherwise each
    # start a stray trace of their own.
    if lf is None or not usage or lf.get_current_trace_id() is None:
        return
    details = {
        "input": usage.get("input", 0),
        "output": usage.get("output", 0),
        "cache_read_input_tokens": usage.get("cache_read", 0),
        "cache_creation_input_tokens": usage.get("cache_write", 0),
    }
    with lf.start_as_current_observation(
        name=name,
        as_type="generation",
        model=usage.get("model"),
        usage_details=details,
        cost_details={"total": round(cost, 6)},
        output=output,
        metadata=metadata or None,
    ):
        pass


def score(name: str, value: float | str | bool, data_type: str | None = None) -> None:
    """Score the current trace. bool -> BOOLEAN, str -> CATEGORICAL, number -> NUMERIC."""
    lf = _langfuse()
    if lf is None:
        return
    if data_type is None:
        data_type = (
            "BOOLEAN"
            if isinstance(value, bool)
            else "CATEGORICAL"
            if isinstance(value, str)
            else "NUMERIC"
        )
    if isinstance(value, bool):
        value = float(value)
    lf.score_current_trace(name=name, value=value, data_type=data_type)


def score_claim(packet, expected: str | None = None) -> None:
    """The per-claim scores the dashboards group and filter on."""
    if not enabled():
        return
    decision = packet.decision.value
    score("decision", decision)
    score("escalated", decision == "escalate")
    score("confidence", packet.confidence)
    score("fraud_flags", float(len(packet.fraud_flags)))
    review = packet.coverage.review
    if review is not None:
        score("review_verdict", review.verdict)
        score("review_escalated", review.applied)
        score("review_route", review.route or "single")
        score("review_cost_usd", review.cost_usd)
    if expected is not None:
        score("decision_correct", decision == expected)


def flush() -> None:
    lf = _langfuse()
    if lf is not None:
        lf.flush()
