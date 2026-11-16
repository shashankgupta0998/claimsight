import json
from dataclasses import replace

import pytest

from claimsight.agents.coverage import CoverageAgent
from claimsight.agents.coverage_review import (
    PROMPT_VERSION,
    CassetteReviewer,
    MissingReview,
    ReviewRequest,
    RoutedReviewer,
    answer_cost,
    check_quotes,
    cost_usd,
)
from claimsight.retrieval.base import RetrievedChunk

BUSINESS = "PP-D-EXCL-BUSINESS"
UBER = "I was driving for Uber with a paying passenger when a truck hit me."


class FakeReviewer:
    name = "fake-llm"

    def __init__(self, answer: dict | None):
        self.answer, self.requests = answer, []

    def review(self, request: ReviewRequest) -> dict | None:
        self.requests.append(request)
        return self.answer


def _agent(deps, answer):
    reviewer = FakeReviewer(answer)
    agent = CoverageAgent(
        deps.retriever, deps.policies, reviewer=reviewer, screening=deps.screening
    )
    return agent, reviewer


def _with(claim, policy_id="P-100", statement=UBER):
    return claim.model_copy(update={"policy_id": policy_id, "statement": statement})


def _chunk(cid, text):
    return RetrievedChunk(cid, "doc", 1, text, 0.0)


def test_quote_check_accepts_verbatim_and_rejects_the_rest():
    shown = {"X": _chunk("X", "We do not pay for loss to your covered auto while it is used.")}
    got = check_quotes(
        [
            {"clause_id": "X", "quote": "we do not  pay for LOSS to your covered auto"},
            {"clause_id": "X", "quote": "We never pay for loss to any auto at all"},
            {"clause_id": "Y", "quote": "We do not pay for loss to your covered auto"},
            {"clause_id": "X", "quote": "We do not"},
        ],
        shown,
    )
    assert [q.valid for q in got] == [True, False, False, False]
    assert [q.problem for q in got][1:] == [
        "quote is not in the clause text",
        "clause was not shown to the reviewer",
        "quote too short to check",
    ]


def test_reviewer_sees_statement_and_every_part_d_exclusion(deps, claim):
    agent, reviewer = _agent(deps, {"verdict": "covered", "quotes": [], "explanation": ""})
    agent.run(_with(claim("CLM-0003")), UBER)
    request = reviewer.requests[0]
    shown = {c.clause_id for c in request.shown}
    assert {c.clause_id for c in request.fixed} >= {BUSINESS, "PP-D-EXCL", "PP-F-TERRITORY"}
    # Default layout: statement first, one block. Cached layout: screening clauses first,
    # as the block every request shares, statement last.
    assert request.parts()[0] == "" and request.render().index(
        "<statement>"
    ) < request.render().index("<clause")
    cached = replace(request, layout="cached_prefix")
    static, dynamic = cached.parts()
    assert static.count("<clause") == len(request.fixed) and "<statement>" in dynamic
    assert cached.render() != request.render()  # different cassette keys
    assert {BUSINESS, "PP-D-EXCL", "PP-D-EXCL-GOV", "PP-F-TERRITORY", "PP-D-COLL"} <= shown
    assert "PP-A-EXCL" not in shown  # liability exclusions don't bear on damage to your auto
    assert f"<statement>{UBER}</statement>" in request.render()


def test_checked_exclusion_quote_escalates_never_denies(deps, claim):
    quote = "while logged into a transportation network company app"
    agent, _ = _agent(
        deps,
        {
            "verdict": "excluded",
            "quotes": [{"clause_id": BUSINESS, "quote": quote}],
            "explanation": "Carrying a paying passenger for Uber.",
        },
    )
    finding, _ = agent.run(_with(claim("CLM-0003")), UBER)
    assert finding.applies is None  # the critic escalates; it never denies on the LLM's word
    assert finding.coverage_part == "collision" and finding.deductible == 1000
    assert finding.citations[-1].clause_id == BUSINESS
    assert finding.citations[-1].snippet == quote
    assert finding.review.applied and quote in finding.rationale


def test_invented_quote_leaves_the_rules_in_charge(deps, claim):
    agent, _ = _agent(
        deps,
        {
            "verdict": "excluded",
            "quotes": [{"clause_id": BUSINESS, "quote": "rideshare drivers are never covered"}],
            "explanation": "",
        },
    )
    finding, _ = agent.run(_with(claim("CLM-0003")), UBER)
    assert finding.applies is True and not finding.review.applied
    assert finding.review.quotes[0].problem == "quote is not in the clause text"


def test_exclusion_must_be_quoted_from_an_exclusion(deps, claim):
    # A real quote, but from the granting clause: it can't back an exclusion.
    agent, _ = _agent(
        deps,
        {
            "verdict": "excluded",
            "quotes": [
                {"clause_id": "PP-D-COLL", "quote": "impact with another vehicle or object"}
            ],
            "explanation": "",
        },
    )
    finding, _ = agent.run(_with(claim("CLM-0003")), UBER)
    assert finding.applies is True and not finding.review.applied


def test_checked_granting_quote_replaces_the_truncated_snippet(deps, claim):
    quote = "the upset of your covered auto or its impact with another vehicle or object"
    agent, _ = _agent(
        deps,
        {
            "verdict": "covered",
            "quotes": [{"clause_id": "PP-D-COLL", "quote": quote}],
            "explanation": "",
        },
    )
    finding, _ = agent.run(_with(claim("CLM-0003"), statement="Rear-ended at a light."), "x")
    assert finding.applies is True
    assert next(c for c in finding.citations if c.clause_id == "PP-D-COLL").snippet == quote


def test_no_review_when_the_rules_already_say_no(deps, claim):
    agent, reviewer = _agent(deps, None)
    finding, _ = agent.run(claim("CLM-0005"), "I backed into a pole.")  # P-200: no collision
    assert finding.applies is False and reviewer.requests == []


def test_refusal_or_missing_review_keeps_the_rules(deps, claim, tmp_path):
    agent, _ = _agent(deps, None)
    finding, _ = agent.run(_with(claim("CLM-0003")), UBER)
    assert finding.applies is True and finding.review is None

    offline = CassetteReviewer(tmp_path / "coverage.json")
    agent = CoverageAgent(deps.retriever, deps.policies, reviewer=offline, screening=deps.screening)
    finding, _ = agent.run(_with(claim("CLM-0003")), UBER)
    assert finding.applies is True and finding.review is None


def test_cassette_records_once_then_replays(tmp_path):
    path = tmp_path / "coverage.json"
    answer = {"verdict": "covered", "quotes": [], "explanation": "ok"}
    fake = FakeReviewer(answer)
    req = ReviewRequest("collision", "collision", "hit a pole", [_chunk("X", "text")])
    live = CassetteReviewer(path, fake)
    assert live.review(req) == answer and live.review(req) == answer
    assert len(fake.requests) == 1
    assert json.loads(path.read_text())["prompt"] == PROMPT_VERSION

    assert CassetteReviewer(path).review(req) == answer
    with pytest.raises(MissingReview):
        CassetteReviewer(path).review(ReviewRequest("hail", "x", "y", []))


def test_cassette_from_an_old_prompt_is_ignored_offline_and_refused_online(tmp_path):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"model": "fake-llm", "prompt": "old", "reviews": {"k": {}}}))
    with pytest.raises(MissingReview):
        CassetteReviewer(path).review(ReviewRequest("hail", "x", "y", []))
    with pytest.raises(ValueError, match="re-record"):
        CassetteReviewer(path, FakeReviewer(None))


class Scripted:
    def __init__(self, name, answer):
        self.name, self.answer, self.calls = name, answer, 0

    def review(self, request):
        self.calls += 1
        return self.answer


def _req():
    return ReviewRequest("collision", "collision", "hit a pole", [_chunk("X", "text")])


def test_router_keeps_a_cheap_covered_and_escalates_everything_else():
    usage = {"model": "claude-haiku-5-5", "input": 1000, "output": 100}
    cheap = Scripted("haiku", {"verdict": "covered", "quotes": [], "usage": usage})
    strong = Scripted("sonnet", {"verdict": "excluded", "quotes": []})
    got = RoutedReviewer(cheap, strong).review(_req())
    assert got["routed"] == "cheap" and strong.calls == 0

    for first in ({"verdict": "unclear"}, {"verdict": "excluded"}, {"refused": True}, None):
        strong.calls = 0
        got = RoutedReviewer(Scripted("haiku", first), strong).review(_req())
        assert got["routed"] == "strong" and got["verdict"] == "excluded" and strong.calls == 1


def test_cost_counts_cache_reads_cheaply_and_both_calls_when_routed_up():
    sonnet = {"model": "claude-sonnet-5-5", "input": 500, "output": 100, "cache_read": 1500}
    assert cost_usd(sonnet) == pytest.approx((500 * 2 + 100 * 10 + 1500 * 0.10) / 1e6)
    assert cost_usd(sonnet, cached=False) == pytest.approx((2000 * 2 + 100 * 10) / 1e6)
    haiku = {"model": "claude-haiku-5-5", "input": 2000, "output": 100}
    both = {"verdict": "excluded", "routed": "strong", "calls": [haiku, sonnet]}
    assert answer_cost(both) == pytest.approx(cost_usd(haiku) + cost_usd(sonnet))
    assert cost_usd({"model": "unknown", "input": 1, "output": 1}) == 0.0
