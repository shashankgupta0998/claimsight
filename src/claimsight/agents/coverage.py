"""Coverage agent: retrieve policy language, decide which coverage part applies, cite it.

v0 is deterministic (clause metadata + declarations). Week 3-4 upgrade: an LLM reads the
retrieved clauses, must quote the clause it relies on, and a validator rejects any
citation whose clause_id was not in the retrieved set.
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


@dataclass
class CoverageAgent:
    retriever: Retriever
    policies: JsonPolicyStore
    k: int = 5

    def run(self, claim: ClaimInput, statement: str) -> tuple[CoverageFinding, list[str]]:
        query = f"{claim.loss_type.value} {statement} deductible"
        chunks = self.retriever.search(query, k=self.k)
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
