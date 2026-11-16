"""HTTP API: submit a claim, read its status, and record an adjuster's review.

The graph runs inside the request. Its state lives in the checkpointer from Settings
(claimsight/checkpoint.py): with `CLAIMSIGHT_CHECKPOINTER=postgres` a claim waiting for an
adjuster survives a restart and can be reviewed through any API process.
Weeks 11-12: enqueue to Redis (arq) and write packets to the `decision_packets` table.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from claimsight import tracing
from claimsight.checkpoint import make_checkpointer
from claimsight.deps import Deps, build_deps
from claimsight.graph import build_graph
from claimsight.schemas import ClaimInput, Decision, DecisionPacket


class ClaimStatus(BaseModel):
    claim_id: str
    status: Literal["decided", "awaiting_review"]
    packet: DecisionPacket | None = None
    review_request: dict | None = None


class ReviewInput(BaseModel):
    decision: Decision
    reviewer: str
    note: str | None = None


def create_app(deps: Deps | None = None, checkpointer=None) -> FastAPI:
    deps = deps or build_deps()
    checkpointer = checkpointer or make_checkpointer(deps.settings)
    graph = build_graph(deps, checkpointer=checkpointer, hitl=True)
    app = FastAPI(title="ClaimSight", version="0.1.0")

    def config(claim_id: str) -> dict:
        return {"configurable": {"thread_id": claim_id}}

    def status(claim_id: str) -> ClaimStatus:
        snap = graph.get_state(config(claim_id))
        if not snap.values:
            raise HTTPException(404, f"Claim {claim_id} not found.")
        if snap.next:
            pending = [i.value for t in snap.tasks for i in t.interrupts]
            return ClaimStatus(
                claim_id=claim_id,
                status="awaiting_review",
                review_request=pending[0] if pending else None,
            )
        return ClaimStatus(claim_id=claim_id, status="decided", packet=snap.values.get("packet"))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "mode": deps.settings.mode, "models": deps.model_versions()}

    @app.post("/claims", response_model=ClaimStatus)
    def submit(claim: ClaimInput) -> ClaimStatus:
        if graph.get_state(config(claim.claim_id)).values:
            raise HTTPException(409, f"Claim {claim.claim_id} was already submitted.")
        mode = deps.settings.mode
        with tracing.claim_trace(claim.claim_id, claim.loss_type.value, mode, ["api"]):
            graph.invoke({"claim": claim}, config(claim.claim_id))
            result = status(claim.claim_id)
            if result.packet is not None:
                tracing.score_claim(result.packet)
            else:
                tracing.score("decision", "awaiting_review")
        tracing.flush()
        return result

    @app.get("/claims/{claim_id}", response_model=ClaimStatus)
    def get_claim(claim_id: str) -> ClaimStatus:
        return status(claim_id)

    @app.post("/claims/{claim_id}/review", response_model=ClaimStatus)
    def review(claim_id: str, body: ReviewInput) -> ClaimStatus:
        if status(claim_id).status != "awaiting_review":
            raise HTTPException(409, f"Claim {claim_id} is not waiting for review.")
        claim = graph.get_state(config(claim_id)).values["claim"]
        mode = deps.settings.mode
        with tracing.claim_trace(claim_id, claim.loss_type.value, mode, ["api", "review"]):
            graph.invoke(Command(resume=body.model_dump(mode="json")), config(claim_id))
            result = status(claim_id)
            tracing.score_claim(result.packet)
            tracing.score("reviewed_by_adjuster", True)
        tracing.flush()
        return result

    return app


app = create_app()
