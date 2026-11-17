"""Redis job queue (arq): the API enqueues a claim, a worker runs the graph.

    CLAIMSIGHT_QUEUE=redis CLAIMSIGHT_CHECKPOINTER=postgres make worker
    CLAIMSIGHT_QUEUE=redis CLAIMSIGHT_CHECKPOINTER=postgres make api

With `CLAIMSIGHT_QUEUE=redis`, POST /claims returns 202 as soon as the claim is on the
queue; GET /claims/{id} shows queued -> processing -> decided or awaiting_review. The worker
and the API are different processes, so the graph's state has to live in Postgres (the
checkpointer): queued mode refuses to start with the in-memory one.

* The job id is the claim id. arq won't enqueue a second job with an id it already holds,
  so a double submit can't run a claim twice.
* arq doesn't retry a job that raises; `process_claim` raises `Retry` itself, up to
  `queue_max_tries`. A retry continues the graph from its last checkpoint
  (ClaimService.run), so nodes that finished don't run again.
* Results are kept for `queue_keep_result_s`: after that a failed claim whose graph never
  started reads as not found.
"""

from __future__ import annotations

import asyncio
import logging

from claimsight.config import Settings, get_settings
from claimsight.schemas import ClaimInput

log = logging.getLogger(__name__)

JOB = "process_claim"


def job_id(claim_id: str) -> str:
    return f"claim:{claim_id}"


def redis_settings(s: Settings):
    from arq.connections import RedisSettings

    return RedisSettings.from_dsn(s.redis_url)


async def enqueue(pool, claim: ClaimInput) -> bool:
    """False when a job for this claim already exists (queued, running or kept)."""
    job = await pool.enqueue_job(JOB, claim.model_dump(mode="json"), _job_id=job_id(claim.claim_id))
    return job is not None


async def job_state(pool, claim_id: str) -> tuple[str, str | None]:
    """(queued | processing | failed | done | not_found, error message if it failed)."""
    from arq.jobs import Job, JobStatus

    job = Job(job_id(claim_id), pool)
    status = await job.status()
    if status in (JobStatus.queued, JobStatus.deferred):
        return "queued", None
    if status == JobStatus.in_progress:
        return "processing", None
    if status == JobStatus.complete:
        info = await job.result_info()
        if info is not None and not info.success:
            return "failed", f"{type(info.result).__name__}: {info.result}"
        return "done", None
    return "not_found", None


async def process_claim(ctx: dict, claim: dict) -> str:
    """The worker's job: run one claim through the graph (in a thread: the graph is sync)."""
    from arq import Retry

    service, s = ctx["service"], ctx["settings"]
    parsed = ClaimInput.model_validate(claim)
    try:
        result = await asyncio.to_thread(service.run, parsed)
    except Exception as exc:
        if ctx["job_try"] < s.queue_max_tries:
            log.warning("claim %s failed (try %s): %s", parsed.claim_id, ctx["job_try"], exc)
            raise Retry(defer=s.queue_retry_delay_s * ctx["job_try"]) from exc
        raise
    return result.status


def build_service(s: Settings):
    """The graph a worker or API process runs claims on."""
    from claimsight.checkpoint import make_checkpointer
    from claimsight.deps import build_deps
    from claimsight.graph import build_graph
    from claimsight.service import ClaimService

    if s.queue == "redis" and s.checkpointer != "postgres":
        raise ValueError(
            "CLAIMSIGHT_QUEUE=redis needs CLAIMSIGHT_CHECKPOINTER=postgres: the API and the "
            "worker are separate processes and must share the graph's state."
        )
    deps = build_deps(s)
    graph = build_graph(deps, checkpointer=make_checkpointer(s), hitl=True)
    return ClaimService(graph, s.mode)


async def startup(ctx: dict) -> None:
    s = ctx.get("settings") or get_settings()
    ctx["settings"] = s
    ctx.setdefault("service", build_service(s))


def worker(s: Settings | None = None, service=None, **kwargs):
    """An arq Worker for `s` (tests pass a ready service and burst=True)."""
    from arq import Worker, func

    s = s or get_settings()
    ctx = {"settings": s}
    if service is not None:
        ctx["service"] = service
    return Worker(
        functions=[func(process_claim, name=JOB, max_tries=s.queue_max_tries)],
        redis_settings=redis_settings(s),
        on_startup=startup,
        ctx=ctx,
        max_jobs=s.queue_concurrency,
        poll_delay=s.queue_poll_s,
        job_timeout=s.queue_job_timeout_s,
        keep_result=s.queue_keep_result_s,
        **kwargs,
    )


def main() -> None:
    worker().run()


if __name__ == "__main__":
    main()
