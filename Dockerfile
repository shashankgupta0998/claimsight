# ClaimSight API and queue worker, replay mode (no model weights, no API keys needed).
#
#   docker build -t claimsight .
#   docker run -p 8000:8000 claimsight                                   # API, inline
#   docker run claimsight python -m claimsight.queue                     # worker
#
# On Fly.io the same image runs both process groups (fly.toml). Live mode needs the `ml`
# and `llm` extras and model weights; that image is a separate step (README, "Deploy").

FROM python:3.12-slim-bookworm AS build
# uv pinned to the version that wrote uv.lock (`uv --version`).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra pg --extra queue
COPY src ./src
COPY evals ./evals
COPY infra ./infra
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra pg --extra queue

FROM python:3.12-slim-bookworm
WORKDIR /app
RUN useradd --create-home --uid 1000 app
COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CLAIMSIGHT_MODE=replay
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)"
# One process unless WEB_CONCURRENCY says otherwise (uvicorn reads it). More than one only
# with CLAIMSIGHT_CHECKPOINTER=postgres: in-memory state isn't shared between processes.
CMD ["uvicorn", "claimsight.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
