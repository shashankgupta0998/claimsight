"""Running a claim through the graph: what the API (inline) and the queue worker share.

The graph's state is keyed by claim id in the checkpointer, so whoever runs a claim, any
process with the same checkpointer can read its status or record the adjuster's review.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import Command
from pydantic import BaseModel

from claimsight import tracing
from claimsight.schemas import ClaimInput, Decision, DecisionPacket


class ClaimStatus(BaseModel):
    claim_id: str
    status: Literal["queued", "processing", "failed", "awaiting_review", "decided"]
    packet: DecisionPacket | None = None
    review_request: dict | None = None
    error: str | None = None


class ReviewInput(BaseModel):
    decision: Decision
    reviewer: str
    note: str | None = None


class ClaimService:
    def __init__(self, graph, mode: str):
        self.graph, self.mode = graph, mode

    @staticmethod
    def config(claim_id: str) -> dict:
        return {"configurable": {"thread_id": claim_id}}

    def status(self, claim_id: str) -> ClaimStatus | None:
        """Where the graph got to, or None if it never started on this claim."""
        snap = self.graph.get_state(self.config(claim_id))
        if not snap.values:
            return None
        if snap.next == ("human_review",):
            pending = [i.value for t in snap.tasks for i in t.interrupts]
            return ClaimStatus(
                claim_id=claim_id,
                status="awaiting_review",
                review_request=pending[0] if pending else None,
            )
        if snap.next:  # a run that stopped partway (a node raised)
            return ClaimStatus(claim_id=claim_id, status="processing")
        return ClaimStatus(claim_id=claim_id, status="decided", packet=snap.values.get("packet"))

    def run(self, claim: ClaimInput) -> ClaimStatus:
        """Run a claim to a decision or to the adjuster. Safe to call again after a failure:
        a run that stopped partway continues from its last checkpoint instead of restarting.
        """
        config = self.config(claim.claim_id)
        snap = self.graph.get_state(config)
        if snap.values and (not snap.next or snap.next == ("human_review",)):
            return self.status(claim.claim_id)  # already done: a retry after the run finished
        with tracing.claim_trace(claim.claim_id, claim.loss_type.value, self.mode, ["api"]):
            self.graph.invoke(None if snap.values else {"claim": claim}, config)
            result = self.status(claim.claim_id)
            if result.packet is not None:
                tracing.score_claim(result.packet)
            else:
                tracing.score("decision", "awaiting_review")
        tracing.flush()
        return result

    def resume(self, claim_id: str, body: ReviewInput) -> ClaimStatus:
        claim = self.graph.get_state(self.config(claim_id)).values["claim"]
        with tracing.claim_trace(claim_id, claim.loss_type.value, self.mode, ["api", "review"]):
            self.graph.invoke(Command(resume=body.model_dump(mode="json")), self.config(claim_id))
            result = self.status(claim_id)
            tracing.score_claim(result.packet)
            tracing.score("reviewed_by_adjuster", True)
        tracing.flush()
        return result
