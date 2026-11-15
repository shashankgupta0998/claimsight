from datetime import date

import httpx
import pytest

from claimsight.models.detector import overlap_fraction
from claimsight.tools.geo import haversine_km
from claimsight.tools.nhtsa import NHTSAVinDecoder
from claimsight.tools.photos import InMemoryPhotoIndex, hamming
from claimsight.tools.weather import LiveWeather, parse_spc_hail_csv

SPC_CSV = """Time,Size,Location,County,State,Lat,Lon,Comments
2342,175,2 N PLANO,COLLIN,TX,33.05,-96.72,(FWD)
2351,125,ALLEN,COLLIN,TX,33.12,-96.64,
0105,bad,SOMEWHERE,X,TX,,,
"""


def test_haversine_dallas_to_austin():
    assert 280 < haversine_km(32.777, -96.797, 30.267, -97.743) < 310
    assert haversine_km(33.0, -96.0, 33.0, -96.0) == 0


def test_parse_spc_hail_csv_skips_bad_rows():
    rows = parse_spc_hail_csv(SPC_CSV)
    assert len(rows) == 2
    assert rows[0]["size_in"] == 1.75 and rows[0]["lat"] == 33.05


def test_live_weather_checks_previous_spc_day_and_filters_radius():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("260514_rpts_hail.csv"):
            return httpx.Response(200, text=SPC_CSV)
        return httpx.Response(404)

    weather = LiveWeather(client=httpx.Client(transport=httpx.MockTransport(handler)))
    hits = weather.hail_reports(33.02, -96.699, date(2026, 5, 15), radius_km=10)
    assert any(p.endswith("260514_rpts_hail.csv") for p in seen)
    assert [h["location"] for h in hits] == ["2 N PLANO"]  # Allen is ~15 km away


def test_vpic_decode_parses_results():
    payload = {
        "Results": [{"Make": "TOYOTA", "Model": "Camry", "ModelYear": "2021", "ErrorCode": "0"}]
    }
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    )
    assert NHTSAVinDecoder(client=client).decode("4T1G11AK5MU000417") == {
        "make": "Toyota",
        "model": "Camry",
        "year": "2021",
        "error_code": "0",
    }


def test_vpic_decode_returns_none_without_make():
    payload = {"Results": [{"Make": "", "ErrorCode": "11"}]}
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    )
    assert NHTSAVinDecoder(client=client).decode("BADVIN") is None


def test_photo_index_ignores_same_claim():
    index = InMemoryPhotoIndex([{"claim_id": "A", "photo_id": "1", "phash": "ffff000000000000"}])
    assert hamming("ffff000000000000", "ffff000000000001") == 1
    assert index.nearest("ffff000000000001", exclude_claim_id="A") is None
    entry, dist = index.nearest("ffff000000000001", exclude_claim_id="B")
    assert entry["claim_id"] == "A" and dist == 1


@pytest.mark.parametrize(
    ("inner", "outer", "expected"),
    [
        ((10, 10, 20, 20), (0, 0, 100, 100), 1.0),
        ((90, 90, 110, 110), (0, 0, 100, 100), 0.25),
        ((200, 200, 210, 210), (0, 0, 100, 100), 0.0),
    ],
)
def test_overlap_fraction(inner, outer, expected):
    assert overlap_fraction(inner, outer) == pytest.approx(expected)
