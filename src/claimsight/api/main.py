"""HTTP API: submit a claim, read its status, and record an adjuster's review.

v0 runs the graph inside the request and keeps state in an in-memory checkpointer.
Week 9-10: enqueue to Redis (arq), persist checkpoints with langgraph's Postgres saver,
and write packets to the `decision_packets` table.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from claimsight.deps import Deps, build_deps
from claimsight.graph import build_graph
from claimsight.observability import get_callbacks
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


def create_app(deps: Deps | None = None) -> FastAPI:
    deps = deps or build_deps()
    graph = build_graph(deps, checkpointer=MemorySaver(), hitl=True)
    app = FastAPI(title="ClaimSight", version="0.1.0")

    def config(claim_id: str) -> dict:
        return {"configurable": {"thread_id": claim_id}, "callbacks": get_callbacks()}

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
        graph.invoke({"claim": claim}, config(claim.claim_id))
        return status(claim.claim_id)

    @app.get("/claims/{claim_id}", response_model=ClaimStatus)
    def get_claim(claim_id: str) -> ClaimStatus:
        return status(claim_id)

    @app.post("/claims/{claim_id}/review", response_model=ClaimStatus)
    def review(claim_id: str, body: ReviewInput) -> ClaimStatus:
        if status(claim_id).status != "awaiting_review":
            raise HTTPException(409, f"Claim {claim_id} is not waiting for review.")
        graph.invoke(Command(resume=body.model_dump(mode="json")), config(claim_id))
        return status(claim_id)

    return app


app = create_app()
