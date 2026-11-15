"""Verification agent: check the claim against outside evidence.

Each check returns PASS, FLAG or UNKNOWN. A failing API never fails the claim: it yields
UNKNOWN, which lowers confidence instead of blocking the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from claimsight.interfaces import PhotoIndex, VinDecoder, WeatherEvidence
from claimsight.schemas import ClaimInput, LossType, SignalStatus, VerificationSignal
from claimsight.tools.geo import haversine_km
from claimsight.tools.weather import HAIL_WMO_CODES

PASS, FLAG, UNKNOWN = SignalStatus.PASS, SignalStatus.FLAG, SignalStatus.UNKNOWN


@dataclass
class VerificationAgent:
    vin_decoder: VinDecoder
    weather: WeatherEvidence
    photo_index: PhotoIndex
    hail_radius_km: float = 25.0
    flood_min_precip_mm: float = 25.0
    max_photo_distance_km: float = 50.0
    duplicate_max_hamming: int = 6

    def run(self, claim: ClaimInput) -> list[VerificationSignal]:
        checks: list[tuple[str, Callable[[ClaimInput], VerificationSignal | None]]] = [
            ("vin_decode", self.check_vin),
            ("weather_corroboration", self.check_weather),
            ("exif_consistency", self.check_exif),
            ("duplicate_photos", self.check_duplicates),
        ]
        signals = []
        for name, check in checks:
            try:
                signal = check(claim)
            except Exception as exc:  # external APIs fail; record it and move on
                signal = VerificationSignal(
                    check=name, status=UNKNOWN, detail=f"Check failed: {exc}"
                )
            if signal is not None:
                signals.append(signal)
        return signals

    def check_vin(self, claim: ClaimInput) -> VerificationSignal | None:
        if not claim.vin:
            return VerificationSignal(check="vin_decode", status=UNKNOWN, detail="No VIN supplied.")
        decoded = self.vin_decoder.decode(claim.vin)
        if not decoded:
            return VerificationSignal(
                check="vin_decode", status=FLAG, detail=f"VIN {claim.vin} did not decode."
            )
        reported = set((claim.reported_vehicle or "").lower().split())
        model_head = decoded["model"].lower().split()[0] if decoded["model"] else ""
        mismatches = [
            field
            for field, value in (
                ("make", decoded["make"].lower()),
                ("model", model_head),
                ("year", decoded["year"]),
            )
            if value and value not in reported
        ]
        vehicle = f"{decoded['year']} {decoded['make']} {decoded['model']}"
        if mismatches:
            return VerificationSignal(
                check="vin_decode",
                status=FLAG,
                detail=f"VIN decodes to {vehicle}; claimant reported "
                f"'{claim.reported_vehicle}' ({', '.join(mismatches)} differ).",
                evidence=decoded,
            )
        return VerificationSignal(
            check="vin_decode", status=PASS, detail=f"VIN decodes to {vehicle}.", evidence=decoded
        )

    def check_weather(self, claim: ClaimInput) -> VerificationSignal | None:
        loc, day = claim.location, claim.loss_date
        if claim.loss_type == LossType.HAIL:
            reports = self.weather.hail_reports(loc.lat, loc.lon, day, self.hail_radius_km)
            daily = self.weather.daily(loc.lat, loc.lon, day)
            if reports:
                top = reports[0]
                return VerificationSignal(
                    check="weather_corroboration",
                    status=PASS,
                    detail=f"NOAA SPC: {top['size_in']:.2f} in hail {top['distance_km']} km away.",
                    evidence={"hail_reports": reports[:3], "daily": daily},
                )
            if daily and daily.get("weather_code") in HAIL_WMO_CODES:
                return VerificationSignal(
                    check="weather_corroboration",
                    status=PASS,
                    detail=f"Open-Meteo: weather code {daily['weather_code']} (hail) that day.",
                    evidence={"daily": daily},
                )
            if daily is None:
                return VerificationSignal(
                    check="weather_corroboration", status=UNKNOWN, detail="No weather data found."
                )
            return VerificationSignal(
                check="weather_corroboration",
                status=FLAG,
                detail=f"No hail reported within {self.hail_radius_km:.0f} km on {day}.",
                evidence={"daily": daily},
            )
        if claim.loss_type == LossType.FLOOD:
            daily = self.weather.daily(loc.lat, loc.lon, day)
            if daily is None:
                return VerificationSignal(
                    check="weather_corroboration", status=UNKNOWN, detail="No weather data found."
                )
            precip = daily.get("precipitation_sum_mm") or 0
            status = PASS if precip >= self.flood_min_precip_mm else FLAG
            return VerificationSignal(
                check="weather_corroboration",
                status=status,
                detail=f"{precip} mm of rain recorded on {day}.",
                evidence={"daily": daily},
            )
        return None  # weather is not evidence for this loss type

    def check_exif(self, claim: ClaimInput) -> VerificationSignal | None:
        dated = [p for p in claim.photos if p.exif_taken_at]
        if not dated:
            return VerificationSignal(
                check="exif_consistency", status=UNKNOWN, detail="No photo has EXIF metadata."
            )
        problems = []
        earliest_ok = claim.loss_date - timedelta(days=1)
        for p in dated:
            taken = p.exif_taken_at.date()
            if taken < earliest_ok:
                problems.append(
                    f"{p.photo_id} taken {(claim.loss_date - taken).days} days before the loss"
                )
            if p.exif_lat is not None and p.exif_lon is not None:
                km = haversine_km(claim.location.lat, claim.location.lon, p.exif_lat, p.exif_lon)
                if km > self.max_photo_distance_km:
                    problems.append(f"{p.photo_id} taken {km:.0f} km from the loss location")
        if problems:
            return VerificationSignal(
                check="exif_consistency", status=FLAG, detail="; ".join(problems) + "."
            )
        return VerificationSignal(
            check="exif_consistency",
            status=PASS,
            detail="Photo timestamps and locations match the reported loss.",
        )

    def check_duplicates(self, claim: ClaimInput) -> VerificationSignal | None:
        hashed = [p for p in claim.photos if p.phash]
        if not hashed:
            return None
        matches = []
        for p in hashed:
            hit = self.photo_index.nearest(p.phash, exclude_claim_id=claim.claim_id)
            if hit and hit[1] <= self.duplicate_max_hamming:
                entry, distance = hit
                matches.append(
                    {"photo_id": p.photo_id, "matches": entry["claim_id"], "hamming": distance}
                )
        if matches:
            claims = sorted({m["matches"] for m in matches})
            return VerificationSignal(
                check="duplicate_photos",
                status=FLAG,
                detail=f"{len(matches)} photo(s) match earlier claim(s) {', '.join(claims)}.",
                evidence={"matches": matches},
            )
        return VerificationSignal(
            check="duplicate_photos", status=PASS, detail="No photo matches an earlier claim."
        )
