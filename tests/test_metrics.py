import pytest

from evals.gate import compare
from evals.judge.judge import parse_label
from evals.metrics import cohen_kappa, ndcg_at_k, percentile, precision_recall, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=2) == 0.5
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0


def test_ndcg_perfect_and_reversed():
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], k=3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a", "b"], ["a", "b"], k=3) < 1.0


def test_precision_recall_counts():
    assert precision_recall({1, 2, 3}, {2, 3, 4, 5}) == (2, 3, 4)


def test_cohen_kappa_known_value():
    # observed 0.75, chance 0.5 -> kappa 0.5
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_cohen_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 0])


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 0.5) == 25


def test_gate_flags_only_drops_beyond_tolerance():
    baseline = {"decision_accuracy": 0.90, "fraud_recall": 0.80}
    assert compare({"decision_accuracy": 0.89, "fraud_recall": 0.80}, baseline, 0.02) == []
    failures = compare({"decision_accuracy": 0.85, "fraud_recall": 0.80}, baseline, 0.02)
    assert len(failures) == 1 and failures[0].startswith("decision_accuracy")


def test_judge_label_parsing_defaults_to_unsupported():
    assert parse_label("The clause covers hail.\nsupported") == "supported"
    assert parse_label("I'm not sure.") == "unsupported"
