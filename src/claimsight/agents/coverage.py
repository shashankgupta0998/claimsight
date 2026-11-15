"""Coverage agent: retrieve policy language, decide which coverage part applies, cite it.

Retrieval runs two targeted queries instead of one search over the claimant's statement:

* the peril, in the policy's own vocabulary, taken from the structured loss type that intake
  already captured ("vandalism" -> "malicious mischief or vandalism");
* the declarations page, which holds the deductibles.

Searching with the whole statement let incidental details win: "the windshield is smashed"
pulled the glass clauses ahead of the vandalism one, and "someone keyed both doors" matched
nothing useful. The statement still matters for exclusions; that is the job of the LLM
coverage step on the roadmap, which reads the retrieved clauses alongside the statement.

The decision is still deterministic: clause metadata says which coverage part a clause
grants, and the policy says whether the insured carries it.
"""

from __future__ import annotations

from dataclasses import dataclass

from claimsight.policies import JsonPolicyStore
from claimsight.retrieval.base import RetrievedChunk, Retriever
from claimsight.schemas import Citation, ClaimInput, CoverageFinding


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


def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    return [c for c in chunks if not (c.clause_id in seen or seen.add(c.clause_id))]


@dataclass
class CoverageAgent:
    retriever: Retriever
    policies: JsonPolicyStore
    k: int = 5

    def retrieve(self, claim: ClaimInput) -> list[RetrievedChunk]:
        peril = PERIL_QUERIES.get(claim.loss_type.value, claim.loss_type.value)
        peril_hits = self.retriever.search(peril, k=self.k)
        declarations = self.retriever.search(DECLARATIONS_QUERY, k=1)
        # The decision needs the best peril clause and the declarations (deductibles): rank
        # those first, then the rest of the peril hits as context.
        return _dedupe(peril_hits[:1] + declarations + peril_hits[1:])[: self.k]

    def run(self, claim: ClaimInput, statement: str) -> tuple[CoverageFinding, list[str]]:
        chunks = self.retrieve(claim)
        retrieved_ids = [c.clause_id for c in chunks]

        policy = self.policies.get(claim.policy_id)
        if policy is None:
            return (
                CoverageFinding(applies=None, rationale=f"Policy {claim.policy_id} not found."),
                retrieved_ids,
            )

        covering = [c for c in chunks if claim.loss_type.value in c.metadata.get("covers", [])]
        if not covering:
            return (
                CoverageFinding(
                    applies=None,
                    rationale=f"No retrieved clause addresses a {claim.loss_type.value} loss.",
                ),
                retrieved_ids,
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
        return (
            CoverageFinding(
                applies=applies,
                coverage_part=part,
                deductible=deductible,
                citations=citations,
                rationale=rationale,
            ),
            retrieved_ids,
        )
