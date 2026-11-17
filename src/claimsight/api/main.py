"""HTTP API: submit a claim, read its status, and record an adjuster's review.

Two ways to run a claim (`CLAIMSIGHT_QUEUE`):

* `off` (default): the graph runs inside the request, which returns the decision.
* `redis`: the request puts the claim on a Redis queue and returns 202; a worker
  (`make worker`, claimsight/queue.py) runs the graph. Poll GET /claims/{id}:
  queued -> processing -> decided | awaiting_review, or failed after the last retry.

Either way the graph's state lives in the checkpointer (claimsight/checkpoint.py), keyed by
claim id: with `CLAIMSIGHT_CHECKPOINTER=postgres` any API process can read a claim's status
or record the adjuster's review, and claims waiting for review survive restarts.
"""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Response

from claimsight.checkpoint import make_checkpointer
from claimsight.deps import Deps, build_deps
from claimsight.graph import build_graph
from claimsight.schemas import ClaimInput
from claimsight.service import ClaimService, ClaimStatus, ReviewInput

__all__ = ["ClaimStatus", "ReviewInput", "create_app"]


def create_app(deps: Deps | None = None, checkpointer=None, queue_pool=None) -> FastAPI:
    deps = deps or build_deps()
    s = deps.settings
    queued = s.queue == "redis"
    if queued and checkpointer is None and s.checkpointer != "postgres":
        raise ValueError(
            "CLAIMSIGHT_QUEUE=redis needs CLAIMSIGHT_CHECKPOINTER=postgres: the API and the "
            "worker are separate processes and must share the graph's state."
        )
    checkpointer = checkpointer or make_checkpointer(s)
    service = ClaimService(build_graph(deps, checkpointer=checkpointer, hitl=True), s.mode)
    pool = {"redis": queue_pool}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if queued and pool["redis"] is None:
            from arq import create_pool

            from claimsight.queue import redis_settings

            pool["redis"] = await create_pool(redis_settings(s))
        yield
        if queued and queue_pool is None and pool["redis"] is not None:
            await pool["redis"].aclose()

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if s.api_key and not hmac.compare_digest(x_api_key or "", s.api_key):
            raise HTTPException(401, "Missing or wrong X-API-Key.")

    app = FastAPI(title="ClaimSight", version="0.1.0", lifespan=lifespan)
    guarded = [Depends(require_key)]
    filed = {"claims": 0}  # demo mode's cap: in-memory state grows with every claim
    if s.demo:
        from claimsight.api.demo import router as demo_router

        app.include_router(demo_router(s.golden_dir))

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "mode": s.mode,
            "queue": s.queue,
            "models": deps.model_versions(),
        }

    @app.post("/claims", response_model=ClaimStatus, dependencies=guarded)
    async def submit(claim: ClaimInput, response: Response) -> ClaimStatus:
        if service.status(claim.claim_id) is not None:
            raise HTTPException(409, f"Claim {claim.claim_id} was already submitted.")
        if s.demo:
            if filed["claims"] >= s.demo_max_claims:
                raise HTTPException(503, "The demo is full; it empties when the Space restarts.")
            filed["claims"] += 1
        if not queued:
            return await asyncio.to_thread(service.run, claim)
        from claimsight.queue import enqueue

        if not await enqueue(pool["redis"], claim):
            raise HTTPException(409, f"Claim {claim.claim_id} was already submitted.")
        response.status_code = 202
        return ClaimStatus(claim_id=claim.claim_id, status="queued")

    @app.get("/claims/{claim_id}", response_model=ClaimStatus, dependencies=guarded)
    async def get_claim(claim_id: str) -> ClaimStatus:
        if queued:
            from claimsight.queue import job_state

            # The job first, then the graph: the worker writes the graph's state before the
            # job completes, so a finished job always has state to read. The other order
            # raced (load test): graph read empty, worker finished, job read "done" -> 404.
            job, error = await job_state(pool["redis"], claim_id)
            if job == "failed":  # out of retries; the graph may have stopped partway
                return ClaimStatus(claim_id=claim_id, status="failed", error=error)
            graph_state = service.status(claim_id)
            if graph_state is None and job in ("queued", "processing"):
                return ClaimStatus(claim_id=claim_id, status=job)
        else:
            graph_state = service.status(claim_id)
        if graph_state is None:
            raise HTTPException(404, f"Claim {claim_id} not found.")
        return graph_state

    @app.post("/claims/{claim_id}/review", response_model=ClaimStatus, dependencies=guarded)
    async def review(claim_id: str, body: ReviewInput) -> ClaimStatus:
        current = service.status(claim_id)
        if current is None:
            raise HTTPException(404, f"Claim {claim_id} not found.")
        if current.status != "awaiting_review":
            raise HTTPException(409, f"Claim {claim_id} is not waiting for review.")
        return await asyncio.to_thread(service.resume, claim_id, body)

    return app


app = create_app()
