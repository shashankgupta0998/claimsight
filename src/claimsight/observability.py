"""Langfuse tracing, enabled only when keys are configured and the `llm` extra is installed."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def get_callbacks() -> list:
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return []
    try:
        from langfuse.langchain import CallbackHandler  # langfuse >= 3
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler  # langfuse 2.x
        except ImportError:
            log.warning("LANGFUSE_* is set but langfuse is not installed; tracing is off.")
            return []
    return [CallbackHandler()]
