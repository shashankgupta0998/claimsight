from __future__ import annotations


class WhisperTranscriber:
    """Claimant voice notes -> text. Needs the `ml` extra (and ffmpeg for most audio formats)."""

    def __init__(self, model: str, device: int | str | None = None):
        from transformers import pipeline

        self._pipe = pipeline("automatic-speech-recognition", model=model, device=device)
        self.name = model

    def transcribe(self, uri: str) -> str:
        out = self._pipe(uri, return_timestamps=True)  # timestamps allow clips longer than 30 s
        return out["text"].strip()
