"""The claims graph.

    intake ─┬─ vision ── estimator ─┐
            ├─ coverage ────────────┼─ critic ─┬─ finalize
            └─ verification ────────┘          └─ human_review ─ finalize

Vision, coverage and verification run in parallel. The critic waits for all three.
With `hitl=True` and a checkpointer, escalated claims pause at `human_review` until an
adjuster resumes the thread with `Command(resume={"decision": ..., "reviewer": ...})`.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from claimsight.agents import CoverageAgent, Critic, Estimator, VerificationAgent, VisionAgent
from claimsight.deps import Deps
from claimsight.graph.state import ClaimState
from claimsight.schemas import Decision, DecisionPacket


def build_graph(deps: Deps, *, checkpointer=None, hitl: bool = False):
    s = deps.settings
    vision = VisionAgent(deps.detector, min_confidence=s.min_detection_confidence)
    coverage = CoverageAgent(
        deps.retriever, deps.policies, reviewer=deps.coverage_reviewer, screening=deps.screening
    )
    verification = VerificationAgent(
        deps.vin_decoder,
        deps.weather,
        deps.photo_index,
        hail_radius_km=s.hail_radius_km,
        flood_min_precip_mm=s.flood_min_precip_mm,
        max_photo_distance_km=s.max_photo_distance_km,
        duplicate_max_hamming=s.duplicate_max_hamming,
        photo_embedder=deps.photo_embedder,
        embedding_index=deps.embedding_index,
        duplicate_min_cosine=s.duplicate_min_cosine,
    )
    estimator = Estimator()
    critic = Critic(
        s.escalation_confidence,
        s.auto_approve_limit_usd,
        impact=deps.impact_classifier,
        photo_checker=deps.photo_checker,
    )
    versions = {**deps.model_versions(), "estimator": estimator.name}

    def intake_node(state: ClaimState) -> dict:
        claim = state["claim"]
        # The voice note is the primary evidence when we can hear it. Without a
        # transcriber (replay mode) the recorded transcript on the claim stands in for it.
        if claim.voice_note_uri and deps.transcriber:
            return {
                "statement": deps.transcriber.transcribe(claim.voice_note_uri),
                "transcribed": True,
            }
        return {"statement": claim.statement, "transcribed": False}

    def vision_node(state: ClaimState) -> dict:
        return {"damage": vision.run(state["claim"])}

    def coverage_node(state: ClaimState) -> dict:
        finding, retrieved = coverage.run(state["claim"], state["statement"])
        return {"coverage": finding, "retrieved_clause_ids": retrieved}

    def verification_node(state: ClaimState) -> dict:
        return {"signals": verification.run(state["claim"])}

    def estimator_node(state: ClaimState) -> dict:
        return {"estimate": estimator.run(state["damage"])}

    def critic_node(state: ClaimState) -> dict:
        result = critic.run(
            state["claim"],
            state["statement"],
            state["damage"],
            state["coverage"],
            state.get("signals", []),
            state["estimate"],
        )
        return {
            "signals": result.new_signals,
            "confidence": result.confidence,
            "decision": result.decision,
            "reasons": result.reasons,
            "decided_by": "system",
        }

    def route(state: ClaimState) -> str:
        return "human_review" if Decision(state["decision"]) == Decision.ESCALATE else "finalize"

    def human_review_node(state: ClaimState) -> dict:
        if not hitl:
            return {}  # offline/eval mode: leave the claim escalated
        answer = interrupt(
            {
                "claim_id": state["claim"].claim_id,
                "proposed_decision": Decision(state["decision"]).value,
                "confidence": state["confidence"],
                "reasons": state["reasons"],
            }
        )
        decision = Decision(answer["decision"])
        reviewer = answer.get("reviewer", "adjuster")
        note = answer.get("note")
        reasons = [*state["reasons"], f"Adjuster {reviewer} decided: {decision.value}"]
        if note:
            reasons.append(f"Adjuster note: {note}")
        return {"decision": decision, "decided_by": reviewer, "reasons": reasons}

    def finalize_node(state: ClaimState) -> dict:
        packet = DecisionPacket(
            claim_id=state["claim"].claim_id,
            decision=Decision(state["decision"]),
            decided_by=state.get("decided_by", "system"),
            confidence=state["confidence"],
            damage=state["damage"],
            coverage=state["coverage"],
            signals=state.get("signals", []),
            estimate=state["estimate"],
            reasons=state["reasons"],
            model_versions=versions,
        )
        return {"packet": packet}

    g = StateGraph(ClaimState)
    g.add_node("intake", intake_node)
    g.add_node("vision", vision_node)
    g.add_node("coverage", coverage_node)
    g.add_node("verification", verification_node)
    g.add_node("estimator", estimator_node)
    g.add_node("critic", critic_node)
    g.add_node("human_review", human_review_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "intake")
    for node in ("vision", "coverage", "verification"):
        g.add_edge("intake", node)
    g.add_edge("vision", "estimator")
    g.add_edge(["estimator", "coverage", "verification"], "critic")
    g.add_conditional_edges(
        "critic", route, {"human_review": "human_review", "finalize": "finalize"}
    )
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
