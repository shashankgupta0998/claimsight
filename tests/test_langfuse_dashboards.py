"""The dashboard spec and its build into Langfuse, against a fake Langfuse (no network)."""

import importlib.util
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dash():
    path = ROOT / "scripts/langfuse_dashboards.py"
    spec = importlib.util.spec_from_file_location("langfuse_dashboards", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def spec(dash):
    return json.loads(dash.SPEC.read_text())


def test_every_widget_has_one_metric_and_a_known_chart(dash, spec):
    """Langfuse charts draw only the first metric; a second one would silently vanish."""
    titles = [w["title"] for _, w in dash.widgets(spec)]
    assert len(titles) == len(set(titles)) == 15
    for _, w in dash.widgets(spec):
        assert len(w["query"]["metrics"]) == 1, w["title"]
        assert w["chart"] in dash.CHART_TYPES, w["title"]


def test_widget_body_speaks_the_dashboard_api(dash, spec):
    _, w = next(dash.widgets(spec))
    body = dash.widget_body(w)
    assert body["metrics"] == [{"measure": "count", "agg": "count"}]
    assert body["chartType"] == "BAR_TIME_SERIES"
    assert body["description"].endswith(dash.NOTE)
    assert "timeDimension" not in body


def test_tiles_go_two_to_a_row(dash):
    tiles = dash.placements(["a", "b", "c"])
    assert [(t["x"], t["y"]) for t in tiles] == [(0, 0), (6, 0), (0, 6)]
    assert {t["width"] for t in tiles} == {6} and len({t["id"] for t in tiles}) == 3


def test_check_counts_rows_so_a_real_zero_rate_is_data(dash, spec):
    line = next(w for _, w in dash.widgets(spec) if w["chart"] == "line")
    t = datetime(2026, 11, 17, tzinfo=UTC)
    assert dash.COUNT in dash.query_for(line, t, t)["metrics"]
    pie = next(w for _, w in dash.widgets(spec) if w["chart"] == "pie")
    assert dash.query_for(pie, t, t)["metrics"] == pie["query"]["metrics"]
    assert dash.has_data({"avg_value": 0, "count_count": 230})  # no escalations: real
    assert not dash.has_data({"avg_value": 0, "count_count": 0})  # padded bucket
    assert dash.has_data({"p95_value": 0.0005})


class FakeLangfuse:
    """Just enough of the unstable dashboard API, plus one 429 to exercise the wait."""

    def __init__(self):
        self.widgets, self.boards, self.ids = {}, {}, itertools.count()
        self.rate_limited_once = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path.removeprefix("/api/public"), request.method
        body = json.loads(request.content) if request.content else {}
        if path == "/projects":
            return httpx.Response(200, json={"data": [{"id": "proj"}]})
        if path in ("/unstable/dashboard-widgets", "/unstable/dashboards") and method == "GET":
            store = self.widgets if "widgets" in path else self.boards
            return httpx.Response(
                200, json={"data": list(store.values()), "meta": {"totalPages": 1}}
            )
        if path == "/unstable/dashboard-widgets" and method == "POST":
            if not self.rate_limited_once:
                self.rate_limited_once = True
                return httpx.Response(
                    429, json={"details": {"retryAfterSeconds": 0}}, headers={"retry-after": "0"}
                )
            wid = f"w{next(self.ids)}"
            self.widgets[wid] = {"id": wid, **body}
            return httpx.Response(200, json=self.widgets[wid])
        if path.startswith("/unstable/dashboard-widgets/"):
            wid = path.rsplit("/", 1)[1]
            if method == "DELETE":
                del self.widgets[wid]
                return httpx.Response(200, json={"message": "deleted"})
            self.widgets[wid].update(body)
            return httpx.Response(200, json=self.widgets[wid])
        if path == "/unstable/dashboards" and method == "POST":
            did = f"d{next(self.ids)}"
            self.boards[did] = {"id": did, "definition": {"widgets": []}, **body}
            return httpx.Response(200, json=self.boards[did])
        if path.startswith("/unstable/dashboards/") and method == "PATCH":
            did = path.rsplit("/", 1)[1]
            self.boards[did].update(body)
            return httpx.Response(200, json=self.boards[did])
        return httpx.Response(404, json={"path": path, "method": method})


@pytest.fixture
def fake():
    lf = FakeLangfuse()
    lf.rate_limited_once = True  # build tests are about the bookkeeping, not the wait

    def client():
        return httpx.Client(base_url="https://lf.test", transport=httpx.MockTransport(lf.handler))

    return lf, client


def test_build_creates_then_updates_in_place_and_prunes_only_its_own(dash, spec, fake):
    lf, client = fake
    lf.widgets["mine"] = {"id": "mine", "name": "Escalation rate", "description": "my own chart"}
    with client() as c:
        dash.build(c, spec, apply=False)
    assert set(lf.widgets) == {"mine"} and not lf.boards  # dry run changes nothing

    with client() as c:
        dash.build(c, spec, apply=True)
    ours = [w for w in lf.widgets.values() if w["description"].endswith(dash.NOTE)]
    assert len(ours) == 15 and len(lf.boards) == 2
    tiles = {b["name"]: b["definition"]["widgets"] for b in lf.boards.values()}
    assert len(tiles["ClaimSight: decisions"]) == 7 and len(tiles["ClaimSight: LLM cost"]) == 8
    placed = {t["widgetId"] for ts in tiles.values() for t in ts}
    assert placed == {w["id"] for w in ours} and "mine" not in placed

    stale = {"id": "old", "name": "Tokens by model", "description": f"x {dash.NOTE}"}
    lf.widgets["old"] = stale
    with client() as c:
        dash.build(c, spec, apply=True)
    assert "old" not in lf.widgets  # ours, no longer in the spec: deleted
    assert "mine" in lf.widgets  # same title as one of ours, but not ours: untouched
    assert len(lf.widgets) == 16 and len(lf.boards) == 2  # rerun made no duplicates


def test_client_waits_out_a_429_and_retries(dash, monkeypatch):
    lf = FakeLangfuse()
    waits = []
    monkeypatch.setattr(dash.time, "sleep", waits.append)
    monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: lf.handler(req))
    with dash.api_client() as c:
        r = c.post("/api/public/unstable/dashboard-widgets", json={"name": "x"})
    assert r.status_code == 200 and waits == [1.0]  # retry-after 0, plus a second's margin
