from __future__ import annotations

import numpy as np
import pytest

from pm_pipeline.config import CostConfig
from pm_pipeline.evaluate import (
    bootstrap_ci,
    confusion_at_threshold,
    cost_per_1000,
    evaluate_scores,
    precision_at_alert_rate,
    recall_at_alert_rate,
    threshold_for_alert_rate,
)

COSTS = CostConfig(
    false_negative=1000.0, true_positive=250.0, false_positive=100.0, true_negative=0.0
)


def test_recall_at_alert_rate_counts_failures_in_top_k() -> None:
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    # budget 20 % of 10 rows = 2 alerts -> top two scores contain one of three failures
    assert recall_at_alert_rate(y, s, 0.2) == pytest.approx(1 / 3)
    assert precision_at_alert_rate(y, s, 0.2) == pytest.approx(0.5)
    assert recall_at_alert_rate(y, s, 0.3) == pytest.approx(2 / 3)


def test_threshold_for_alert_rate_flags_the_budget() -> None:
    scores = np.linspace(0, 1, 1000)
    thr = threshold_for_alert_rate(scores, 0.05)
    assert np.mean(scores >= thr) == pytest.approx(0.05)


def test_cost_per_1000_uses_cost_model() -> None:
    counts = {"tp": 2, "fp": 3, "fn": 1, "tn": 94}
    expected = 1000 * (2 * 250 + 3 * 100 + 1 * 1000) / 100
    assert cost_per_1000(counts, COSTS) == pytest.approx(expected)


def test_perfect_model_beats_no_model() -> None:
    y = np.array([0] * 95 + [1] * 5)
    s = y.astype(float)
    metrics = evaluate_scores(y, s, threshold=0.5, alert_budget=0.05, costs=COSTS)
    assert metrics["recall_at_budget"] == 1.0
    assert metrics["fn"] == 0 and metrics["fp"] == 0
    # no model: 5 failures x 1000 per 100 cycles; perfect: 5 planned stops x 250
    assert metrics["savings_per_1000"] == pytest.approx(1000 * (5000 - 1250) / 100)


def test_downtime_reduction_matches_hand_calculation() -> None:
    hours = CostConfig(
        false_negative=1.8, true_positive=0.4, false_positive=0.25, true_negative=0.0
    )
    y = np.array([1, 1, 0, 0] + [0] * 96)
    s = np.array([0.9, 0.1, 0.8, 0.0] + [0.0] * 96)
    m = evaluate_scores(y, s, threshold=0.5, alert_budget=0.02, costs=COSTS, downtime=hours)
    # flagged: one failure (0.4 h) + one false alarm (0.25 h); missed: one failure (1.8 h)
    assert m["downtime_h_per_1000"] == pytest.approx(1000 * (0.4 + 0.25 + 1.8) / 100)
    assert m["downtime_reduction"] == pytest.approx(1 - 2.45 / 3.6)


def test_confusion_at_threshold_totals() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    counts = confusion_at_threshold(y, s, 0.5)
    assert sum(counts.values()) == 200


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    s = y * 0.5 + rng.random(2000)
    lo, hi = bootstrap_ci(y, s, 0.05, rounds=200, seed=0)
    point = recall_at_alert_rate(y, s, 0.05)
    assert lo <= point <= hi
