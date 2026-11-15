"""NHTSA vPIC VIN decoding (free, no key): https://vpic.nhtsa.dot.gov/api/"""

from __future__ import annotations

import httpx

VPIC_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}"


class NHTSAVinDecoder:
    name = "nhtsa-vpic"

    def __init__(self, client: httpx.Client | None = None, timeout: float = 10.0):
        self._client = client or httpx.Client(timeout=timeout)

    def decode(self, vin: str) -> dict[str, str] | None:
        resp = self._client.get(VPIC_URL.format(vin=vin), params={"format": "json"})
        resp.raise_for_status()
        results = resp.json().get("Results") or []
        if not results:
            return None
        row = results[0]
        make = (row.get("Make") or "").strip()
        if not make:
            return None
        return {
            "make": make.title(),
            "model": (row.get("Model") or "").strip(),
            "year": (row.get("ModelYear") or "").strip(),
            "error_code": (row.get("ErrorCode") or "").strip(),  # "0" = clean decode
        }


class ReplayVinDecoder:
    name = "replay-vpic"

    def __init__(self, cassette: dict[str, dict]):
        self._cassette = cassette

    def decode(self, vin: str) -> dict[str, str] | None:
        return self._cassette.get(vin)
