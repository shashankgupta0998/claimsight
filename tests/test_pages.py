import json

import pytest

from claimsight.config import Settings
from claimsight.retrieval.factory import build_retrieval, load_chunks
from claimsight.retrieval.pages import (
    CassettePageScorer,
    MissingPageScore,
    PageLaneRetriever,
    fingerprint,
    page_paths,
    render_pages,
    section_of,
    split_heading,
)
from evals.retrieval_eval import load_queries

GOLDEN = Settings().golden_dir


class FakeScorer:
    name = "fake-colqwen"

    def __init__(self, scores):
        self.scores, self.calls = scores, 0

    def score(self, query):
        self.calls += 1
        return self.scores


def test_committed_pages_match_the_corpus():
    # A clause edit without re-rendering would leave the page lane searching old text.
    chunks = load_chunks(GOLDEN)
    rendered = render_pages(chunks)
    committed = page_paths(GOLDEN / "pages")
    assert set(rendered) == set(committed) == {c["page"] for c in chunks}


def test_headings_and_sections():
    assert split_heading("Exclusions. We do not pay.") == ("Exclusions", "We do not pay.")
    text = "Collision means the upset of your covered auto. More."
    assert split_heading(text) == (None, text)
    assert section_of("PP-D-EXCL-BUSINESS").startswith("PART D")
    assert section_of("END-GAP") == "ENDORSEMENTS"


def test_lane_returns_clauses_of_the_best_page_first():
    chunks = load_chunks(GOLDEN)
    lane = PageLaneRetriever(chunks, FakeScorer({10: 9.0, 9: 5.0, 1: 1.0, 99: 50.0}))
    got = lane.search("uber", k=10)
    assert {c.page for c in got[:8]} == {10} and got[8].page == 9
    assert lane.rank_pages("uber")[0] == (10, 9.0)  # page 99 has no clauses: ignored


def test_cassette_records_once_replays_and_refuses_other_pixels(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "page-01.png").write_bytes(b"one")
    path = tmp_path / "pages.json"
    fake = FakeScorer({1: 3.14159})
    live = CassettePageScorer(path, pages, fake)
    assert live.score("q") == live.score("q") == {1: 3.142}
    assert fake.calls == 1
    assert json.loads(path.read_text())["pages"] == fingerprint(pages)

    assert CassettePageScorer(path, pages).score("q") == {1: 3.142}
    with pytest.raises(MissingPageScore):
        CassettePageScorer(path, pages).score("unrecorded")

    (pages / "page-01.png").write_bytes(b"re-rendered")
    with pytest.raises(ValueError, match="other page images"):
        CassettePageScorer(path, pages)


def test_page_lane_replays_in_the_stack():
    stack = build_retrieval(Settings(page_lane="colqwen2"))
    assert "pages:" in stack.retriever.name and stack.pages is not None
    got = stack.retriever.search(load_queries()[0]["query"], k=5)  # recorded by make record
    assert len(got) == 5
