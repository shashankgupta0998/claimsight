"""Seams between agents and the outside world.

Every model and external API sits behind one of these protocols, with a live and a
replay implementation. Agents never know which one they got, so the same graph runs in
production, in CI (replay cassettes) and in evals.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from claimsight.schemas import DamageItem, Photo


class Detector(Protocol):
    name: str

    def detect(self, photo: Photo) -> list[DamageItem]: ...


class Transcriber(Protocol):
    name: str

    def transcribe(self, uri: str) -> str: ...


class VinDecoder(Protocol):
    name: str

    def decode(self, vin: str) -> dict[str, str] | None: ...


class WeatherEvidence(Protocol):
    name: str

    def hail_reports(self, lat: float, lon: float, day: date, radius_km: float) -> list[dict]: ...

    def daily(self, lat: float, lon: float, day: date) -> dict | None: ...


class PhotoIndex(Protocol):
    name: str

    def nearest(self, phash: str, exclude_claim_id: str) -> tuple[dict, int] | None: ...
