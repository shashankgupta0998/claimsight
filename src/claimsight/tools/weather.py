"""Weather evidence for hail and flood claims.

* NOAA SPC daily storm reports (hail size + coordinates): https://www.spc.noaa.gov/climo/reports/
* Open-Meteo historical archive (daily WMO weather code, precipitation, gusts)
"""

from __future__ import annotations

import csv
import io
from datetime import date, timedelta

import httpx

from claimsight.tools.geo import haversine_km

SPC_HAIL_URL = "https://www.spc.noaa.gov/climo/reports/{yymmdd}_rpts_hail.csv"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HAIL_WMO_CODES = {96, 99}  # thunderstorm with slight / heavy hail


def parse_spc_hail_csv(text: str) -> list[dict]:
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            lat, lon = float(row["Lat"]), float(row["Lon"])
            size_in = int(row["Size"]) / 100  # SPC reports hail size in hundredths of an inch
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(
            {
                "time_utc": row.get("Time"),
                "lat": lat,
                "lon": lon,
                "size_in": size_in,
                "location": row.get("Location"),
                "state": row.get("State"),
            }
        )
    return rows


def cassette_key(lat: float, lon: float, day: date) -> str:
    return f"{lat:.3f},{lon:.3f},{day.isoformat()}"


def _within(reports: list[dict], lat: float, lon: float, radius_km: float) -> list[dict]:
    out = []
    for r in reports:
        d = haversine_km(lat, lon, r["lat"], r["lon"])
        if d <= radius_km:
            out.append({**r, "distance_km": round(d, 1)})
    return sorted(out, key=lambda r: r["distance_km"])


class LiveWeather:
    name = "spc+open-meteo"

    def __init__(self, client: httpx.Client | None = None, timeout: float = 10.0):
        self._client = client or httpx.Client(timeout=timeout)

    def _spc_day(self, day: date) -> list[dict]:
        resp = self._client.get(SPC_HAIL_URL.format(yymmdd=day.strftime("%y%m%d")))
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return parse_spc_hail_csv(resp.text)

    def hail_reports(self, lat: float, lon: float, day: date, radius_km: float) -> list[dict]:
        # An SPC "day" runs 12Z to 12Z, so a storm the claimant dates as `day` can sit in the
        # previous file (e.g. 2 a.m. local). Check both.
        reports = self._spc_day(day - timedelta(days=1)) + self._spc_day(day)
        return _within(reports, lat, lon, radius_km)

    def daily(self, lat: float, lon: float, day: date) -> dict | None:
        resp = self._client.get(
            OPEN_METEO_ARCHIVE,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": day.isoformat(),
                "end_date": day.isoformat(),
                "daily": "weather_code,precipitation_sum,wind_gusts_10m_max",
                "timezone": "auto",
            },
        )
        resp.raise_for_status()
        d = resp.json().get("daily") or {}
        if not d.get("time"):
            return None
        return {
            "weather_code": d["weather_code"][0],
            "precipitation_sum_mm": d["precipitation_sum"][0],
            "wind_gusts_max_kmh": d["wind_gusts_10m_max"][0],
        }


class ReplayWeather:
    """Reads recorded responses keyed by (lat, lon, day). Record new keys with LiveWeather."""

    name = "replay-weather"

    def __init__(self, cassette: dict[str, dict]):
        self._cassette = cassette

    def hail_reports(self, lat: float, lon: float, day: date, radius_km: float) -> list[dict]:
        entry = self._cassette.get(cassette_key(lat, lon, day), {})
        return _within(entry.get("hail_reports", []), lat, lon, radius_km)

    def daily(self, lat: float, lon: float, day: date) -> dict | None:
        return self._cassette.get(cassette_key(lat, lon, day), {}).get("daily")
