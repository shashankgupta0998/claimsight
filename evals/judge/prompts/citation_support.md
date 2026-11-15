You are auditing an insurance claims system. Decide whether the cited policy clause
supports the system's coverage statement.

<coverage_statement>
{statement}
</coverage_statement>

<cited_clause id="{clause_id}" page="{page}">
{clause_text}
</cited_clause>

Rules:
- "supported": the clause, read on its own, establishes the statement's coverage claim.
- "partial": the clause is relevant but the statement relies on facts the clause does not contain.
- "unsupported": the clause does not establish the claim, or contradicts it.
- Judge only the text above. Do not use outside knowledge of insurance policies.

Answer with one word on the last line: supported, partial or unsupported.
