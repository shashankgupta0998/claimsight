import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from claimsight.agents.critic import statement_zone
from claimsight.graph import build_graph
from claimsight.schemas import Decision

GOLDEN_IDS = [f"CLM-{i:04d}" for i in range(1, 11)]


@pytest.fixture(scope="module")
def graph(deps):
    return build_graph(deps)


@pytest.mark.parametrize("claim_id", GOLDEN_IDS)
def test_golden_decisions_and_flags(graph, claim, golden_cases, claim_id):
    labels = golden_cases[claim_id]["labels"]
    packet = graph.invoke({"claim": claim(claim_id)})["packet"]
    assert packet.decision.value == labels["decision"], packet.reasons
    assert sorted(packet.fraud_flags) == sorted(labels["fraud_flags"])


def test_low_confidence_detections_are_dropped(graph, claim):
    packet = graph.invoke({"claim": claim("CLM-0001")})["packet"]
    assert ("door", "scratch") not in {(d.part, d.damage_type) for d in packet.damage}


def test_every_coverage_decision_is_cited(graph, claim):
    packet = graph.invoke({"claim": claim("CLM-0005")})["packet"]
    assert packet.decision == Decision.DENY
    assert {c.clause_id for c in packet.coverage.citations} == {"PP-D-COLL", "PP-DEC-DED"}


def test_animal_strike_is_comprehensive_not_collision(graph, claim):
    packet = graph.invoke({"claim": claim("CLM-0010")})["packet"]
    assert packet.coverage.coverage_part == "comprehensive"
    assert any("auto-approval limit" in r for r in packet.reasons)


def test_hitl_pauses_then_resumes_with_adjuster_decision(deps, claim):
    graph = build_graph(deps, checkpointer=MemorySaver(), hitl=True)
    config = {"configurable": {"thread_id": "CLM-0004"}}
    graph.invoke({"claim": claim("CLM-0004")}, config)

    snap = graph.get_state(config)
    assert snap.next == ("human_review",)
    request = snap.tasks[0].interrupts[0].value
    assert request["proposed_decision"] == "escalate"

    graph.invoke(Command(resume={"decision": "deny", "reviewer": "adj-17"}), config)
    packet = graph.get_state(config).values["packet"]
    assert packet.decision == Decision.DENY and packet.decided_by == "adj-17"


@pytest.mark.parametrize(
    ("text", "zone"),
    [
        ("Someone rear-ended me at a light", "rear"),
        ("I rear-ended the car ahead", "front"),
        ("I hit a deer", "front"),
        ("Hail fell on my car", None),
    ],
)
def test_statement_zone(text, zone):
    assert statement_zone(text) == zone
