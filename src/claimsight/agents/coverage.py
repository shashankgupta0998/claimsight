"""Coverage agent: retrieve policy language, decide which coverage part applies, cite it.

Retrieval runs two targeted queries instead of one search over the claimant's statement:

* the peril, in the policy's own vocabulary, taken from the structured loss type that intake
  already captured ("vandalism" -> "malicious mischief or vandalism");
* the declarations page, which holds the deductibles.

Searching with the whole statement let incidental details win: "the windshield is smashed"
pulled the glass clauses ahead of the vandalism one, and "someone keyed both doors" matched
nothing useful. The statement still matters for exclusions: that is the optional LLM review
(`coverage_review.py`), which reads the statement against the retrieved clauses and every
Part D exclusion, and must back what it says with quotes that are checked against the text.

The base decision is deterministic: clause metadata says which coverage part a clause
grants, and the policy says whether the insured carries it. The review can only turn an
approval into an escalation, never into a denial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from claimsight.agents.coverage_review import (
    CoverageReviewer,
    MissingReview,
    ReviewRequest,
    check_quotes,
)
from claimsight.policies import JsonPolicyStore
from claimsight.retrieval.base import RetrievedChunk, Retriever
from claimsight.schemas import Citation, ClaimInput, CoverageFinding, CoverageReview

log = logging.getLogger(__name__)


def _cite(chunk: RetrievedChunk, max_chars: int = 180) -> Citation:
    snippet = chunk.text if len(chunk.text) <= max_chars else chunk.text[: max_chars - 1] + "…"
    return Citation(
        doc_id=chunk.doc_id, clause_id=chunk.clause_id, page=chunk.page, snippet=snippet
    )


# Each loss type in the policy's own words. Paraphrased, not copied from any clause text.
PERIL_QUERIES = {
    "collision": "collision: impact with another vehicle or object, or upset of the auto",
    "hail": "hail damage to the auto",
    "flood": "water or flood damage to the auto",
    "animal": "contact with a bird or animal",
    "glass": "breakage of glass such as a windshield",
    "theft": "theft or larceny of the auto",
    "vandalism": "malicious mischief or vandalism",
}
DECLARATIONS_QUERY = "declarations page: deductible for each physical damage coverage"


def screening_clauses(chunks: list[dict]) -> list[RetrievedChunk]:
    """Clauses the reviewer always sees: every Part D exclusion, plus the territory condition.

    There are few of them and missing one is the costly error, so they are listed in full
    instead of left to retrieval, which searches by peril and would rarely surface them.
    """
    keep = [
        c
        for c in chunks
        if (c["metadata"].get("kind") == "exclusion" and c["clause_id"].startswith("PP-D"))
        or c["clause_id"] == "PP-F-TERRITORY"
    ]
    return [
        RetrievedChunk(c["clause_id"], c["doc_id"], c.get("page"), c["text"], 0.0, c["metadata"])
        for c in keep
    ]


def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    return [c for c in chunks if not (c.clause_id in seen or seen.add(c.clause_id))]


@dataclass
class CoverageAgent:
    retriever: Retriever
    policies: JsonPolicyStore
    k: int = 5
    reviewer: CoverageReviewer | None = None
    screening: list[RetrievedChunk] = field(default_factory=list)
    review_layout: str = "statement_first"

    def retrieve(self, claim: ClaimInput) -> list[RetrievedChunk]:
        peril = PERIL_QUERIES.get(claim.loss_type.value, claim.loss_type.value)
        peril_hits = self.retriever.search(peril, k=self.k)
        declarations = self.retriever.search(DECLARATIONS_QUERY, k=1)
        # The decision needs the best peril clause and the declarations (deductibles): rank
        # those first, then the rest of the peril hits as context.
        return _dedupe(peril_hits[:1] + declarations + peril_hits[1:])[: self.k]

    def run(self, claim: ClaimInput, statement: str) -> tuple[CoverageFinding, list[str]]:
        chunks = self.retrieve(claim)
        finding = self.decide(claim, chunks)
        if self.reviewer is not None and finding.applies and statement.strip():
            finding = self.review(claim, statement, chunks, finding)
        return finding, [c.clause_id for c in chunks]

    def review(
        self,
        claim: ClaimInput,
        statement: str,
        chunks: list[RetrievedChunk],
        finding: CoverageFinding,
    ) -> CoverageFinding:
        """Run the LLM review over an approval and apply what its checked quotes support."""
        fixed_ids = {c.clause_id for c in self.screening}
        retrieved = [c for c in _dedupe(chunks) if c.clause_id not in fixed_ids]
        request = ReviewRequest(
            claim.loss_type.value,
            finding.coverage_part or "",
            statement,
            clauses=retrieved,
            fixed=self.screening,
            layout=self.review_layout,
        )
        shown = request.shown
        try:
            raw = self.reviewer.review(request)
        except MissingReview:
            log.warning("no recorded coverage review for %s; keeping the rules", claim.claim_id)
            return finding
        if not raw or raw.get("refused"):  # refusal or empty answer: the rules stand
            return finding

        by_id = {c.clause_id: c for c in shown}
        quotes = check_quotes(raw.get("quotes", []), by_id)
        review = CoverageReview(
            reviewer=self.reviewer.name,
            verdict=raw["verdict"],
            explanation=raw.get("explanation", ""),
            quotes=quotes,
        )
        screening_ids = {c.clause_id for c in self.screening}
        backing = [q for q in quotes if q.valid and q.clause_id in screening_ids]

        if review.verdict in ("excluded", "unclear") and backing:
            review.applied = True
            q = backing[0]
            label = "Possible exclusion" if review.verdict == "excluded" else "Exclusion to check"
            exclusion = Citation(
                doc_id=by_id[q.clause_id].doc_id,
                clause_id=q.clause_id,
                page=by_id[q.clause_id].page,
                snippet=q.quote,
            )
            return finding.model_copy(
                update={
                    "applies": None,
                    "citations": finding.citations + [exclusion],
                    "rationale": f'{label} ({q.clause_id}: "{q.quote}"). {review.explanation}',
                    "review": review,
                }
            )

        # Covered, or an exclusion the reviewer couldn't back with a quote: the rules stand.
        # A checked quote from the granting clause is a better snippet than a truncated one.
        granting = {q.clause_id: q.quote for q in quotes if q.valid}
        citations = [
            c.model_copy(update={"snippet": granting[c.clause_id]})
            if c.clause_id in granting
            else c
            for c in finding.citations
        ]
        return finding.model_copy(update={"citations": citations, "review": review})

    def decide(self, claim: ClaimInput, chunks: list[RetrievedChunk]) -> CoverageFinding:
        """The rules: loss type -> coverage part -> does the policy carry it."""

        policy = self.policies.get(claim.policy_id)
        if policy is None:
            return CoverageFinding(applies=None, rationale=f"Policy {claim.policy_id} not found.")

        covering = [c for c in chunks if claim.loss_type.value in c.metadata.get("covers", [])]
        if not covering:
            return CoverageFinding(
                applies=None,
                rationale=f"No retrieved clause addresses a {claim.loss_type.value} loss.",
            )

        clause = covering[0]
        part = clause.metadata["coverage_part"]
        applies = part in policy.coverages
        citations = [_cite(clause)]
        declarations = next((c for c in chunks if c.metadata.get("kind") == "declarations"), None)
        if declarations:
            citations.append(_cite(declarations))

        if applies:
            deductible = policy.deductibles.get(part)
            rationale = (
                f"A {claim.loss_type.value} loss falls under {part} coverage, which policy "
                f"{policy.policy_id} carries with a ${deductible:,.0f} deductible."
            )
        else:
            deductible = None
            rationale = (
                f"A {claim.loss_type.value} loss falls under {part} coverage, which policy "
                f"{policy.policy_id} does not carry."
            )
        return CoverageFinding(
            applies=applies,
            coverage_part=part,
            deductible=deductible,
            citations=citations,
            rationale=rationale,
        )
