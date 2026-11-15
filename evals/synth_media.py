"""Optional realism passes on generated claims: LLM paraphrase and TTS voice notes.

Both are cached or written to disk so the golden set stays reproducible without
re-calling a model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PARAPHRASE_SYSTEM = """You rewrite insurance claim statements so they sound like a real claimant
talking on the phone to their insurer. Rules:
- Keep every fact: who hit whom, which part of the car, the place, the cause.
- Do not add facts, injuries, amounts or names.
- Vary length and tone: some people are brief, some ramble, some are upset.
- Output only the rewritten statement, with no preamble."""


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def paraphrase_statements(
    cases: list[dict], cache_path: Path, model: str = "claude-haiku-5-5"
) -> int:
    """Rewrite each templated statement once; reuse cached rewrites on later runs.

    The template text stays in labels["template_statement"] so you can measure how much
    the critic's accuracy drops on natural language versus templates.
    Returns the number of new model calls.
    """
    import anthropic

    cache: dict[str, str] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    client = anthropic.Anthropic()
    calls = 0
    for case in cases:
        template = case["claim"]["statement"]
        key = _key(template)
        if key not in cache:
            msg = client.messages.create(
                model=model,
                max_tokens=300,
                system=PARAPHRASE_SYSTEM,
                messages=[{"role": "user", "content": template}],
            )
            cache[key] = "".join(b.text for b in msg.content if b.type == "text").strip()
            calls += 1
        case["labels"]["template_statement"] = template
        case["claim"]["statement"] = cache[key]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return calls


def synthesize_voice_notes(
    cases: list[dict], out_dir: Path, model: str = "facebook/mms-tts-eng", limit: int | None = None
) -> int:
    """Speak each statement with a TTS model so the ASR stage has audio to transcribe.

    Sets claim.voice_note_uri and labels.reference_transcript (for word error rate).
    The written statement stays on the claim: replay mode reads it, live mode transcribes
    the audio instead. Needs the `ml` extra.
    """
    import wave

    import numpy as np
    from transformers import pipeline

    tts = pipeline("text-to-speech", model=model)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for case in cases[:limit]:
        text = case["claim"]["statement"]
        out = tts(text)
        audio = np.clip(np.asarray(out["audio"]).squeeze(), -1, 1)
        path = out_dir / f"{case['claim']['claim_id']}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(out["sampling_rate"]))
            w.writeframes((audio * 32767).astype("<i2").tobytes())
        case["claim"]["voice_note_uri"] = str(path)
        case["labels"]["reference_transcript"] = text
        done += 1
    return done
