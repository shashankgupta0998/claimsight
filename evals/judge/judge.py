"""LLM judge for citation support. Needs the `llm` extra and ANTHROPIC_API_KEY.

Do not trust its scores until `calibrate.py` shows acceptable agreement with your own labels.
"""

from __future__ import annotations

from pathlib import Path

LABELS = ("supported", "partial", "unsupported")
PROMPT = (Path(__file__).parent / "prompts" / "citation_support.md").read_text()


def parse_label(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        word = line.strip().strip(".").lower()
        if word in LABELS:
            return word
    return "unsupported"  # unparseable answers count against the system, never for it


def judge_citation(
    statement: str,
    clause_id: str,
    clause_text: str,
    page: int | None,
    model: str = "claude-sonnet-5-5",
) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    statement=statement, clause_id=clause_id, clause_text=clause_text, page=page
                ),
            }
        ],
    )
    text = "".join(block.text for block in msg.content if block.type == "text")
    return parse_label(text)
