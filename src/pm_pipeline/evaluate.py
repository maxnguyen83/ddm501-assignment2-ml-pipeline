"""Stage 6 - evaluation: business-aligned metrics for a rare-event alerting model.

Accuracy is useless here (always predicting "no failure" scores 96.6 %). The maintenance team
can inspect only a fixed share of cycles (the *alert budget*), so the primary metric is the
share of failures caught when the highest-risk ``budget`` fraction of cycles is flagged.
A cost model translates the confusion matrix into USD per 1,000 cycles.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from pm_pipeline.config import CostConfig


def _as_arrays(y_true: object, scores: object) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same shape")
    return y, s


def n_alerts(n_rows: int, rate: float) -> int:
    """Return how many cycles fit in the alert budget (at least one)."""
    return max(1, math.ceil(rate * n_rows))


def recall_at_alert_rate(y_true: object, scores: object, rate: float) -> float:
    """Share of failures among the top ``rate`` fraction of cycles ranked by score."""
    y, s = _as_arrays(y_true, scores)
    positives = y.sum()
    if positives == 0:
        return float("nan")
    top = np.argsort(-s, kind="stable")[: n_alerts(len(y), rate)]
    return float(y[top].sum() / positives)


def precision_at_alert_rate(y_true: object, scores: object, rate: float) -> float:
    """Share of flagged cycles that really fail, when the top ``rate`` fraction is flagged."""
    y, s = _as_arrays(y_true, scores)
    k = n_alerts(len(y), rate)
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / k)


def threshold_for_alert_rate(scores: object, rate: float) -> float:
    """Score threshold that flags roughly ``rate`` of the given (validation) scores."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    return float(s[n_alerts(len(s), rate) - 1])


def confusion_at_threshold(y_true: object, scores: object, threshold: float) -> dict[str, int]:
    """Confusion-matrix counts when cycles with ``score >= threshold`` are flagged."""
    y, s = _as_arrays(y_true, scores)
    flagged = s >= threshold
    return {
        "tp": int(np.sum(flagged & (y == 1))),
        "fp": int(np.sum(flagged & (y == 0))),
        "fn": int(np.sum(~flagged & (y == 1))),
        "tn": int(np.sum(~flagged & (y == 0))),
    }


def cost_per_1000(counts: Mapping[str, int], costs: CostConfig) -> float:
    """Return the expected cost in USD per 1,000 cycles for a confusion matrix."""
    n = sum(counts.values())
    total = (
        counts["fn"] * costs.false_negative
        + counts["tp"] * costs.true_positive
        + counts["fp"] * costs.false_positive
        + counts["tn"] * costs.true_negative
    )
    return 1000.0 * total / n if n else float("nan")


def bootstrap_ci(
    y_true: object,
    scores: object,
    rate: float,
    rounds: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for :func:`recall_at_alert_rate` (resampling cycles)."""
    y, s = _as_arrays(y_true, scores)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        values.append(recall_at_alert_rate(y[idx], s[idx], rate))
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def evaluate_scores(
    y_true: object,
    scores: object,
    threshold: float,
    alert_budget: float,
    costs: CostConfig,
    modes: pd.DataFrame | None = None,
    downtime: CostConfig | None = None,
) -> dict[str, float]:
    """Compute the full metric set for one set of scores.

    Args:
        y_true: 0/1 failure labels.
        scores: predicted failure risk (higher = riskier).
        threshold: operating threshold, fixed on development data for the alert budget.
        alert_budget: share of cycles the maintenance team can inspect.
        costs: cost model.
        modes: optional failure-mode flags for per-mode recall (slice analysis).
        downtime: optional machine-downtime hours per outcome; adds the business KPI
            ``downtime_h_per_1000`` and the reduction versus running without a model.

    Returns:
        Flat dict of metrics, ready for ``mlflow.log_metrics``.

    """
    y, s = _as_arrays(y_true, scores)
    counts = confusion_at_threshold(y, s, threshold)
    n_pos = int(y.sum())
    no_model_cost = cost_per_1000({"tp": 0, "fp": 0, "fn": n_pos, "tn": len(y) - n_pos}, costs)
    model_cost = cost_per_1000(counts, costs)
    flagged = counts["tp"] + counts["fp"]
    recall = counts["tp"] / n_pos if n_pos else float("nan")
    precision = counts["tp"] / flagged if flagged else 0.0
    beta2 = 4.0  # F2 weights recall twice as much as precision
    f2 = (
        (1 + beta2) * precision * recall / (beta2 * precision + recall)
        if precision + recall > 0
        else 0.0
    )
    metrics: dict[str, float] = {
        "recall_at_budget": recall_at_alert_rate(y, s, alert_budget),
        "precision_at_budget": precision_at_alert_rate(y, s, alert_budget),
        "pr_auc": float(average_precision_score(y, s)) if n_pos else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < n_pos < len(y) else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(s, 0, 1))),
        "thr_alert_rate": flagged / len(y),
        "thr_recall": recall,
        "thr_precision": precision,
        "thr_f2": f2,
        "cost_per_1000": model_cost,
        "savings_per_1000": no_model_cost - model_cost,
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "tn": counts["tn"],
        "n_failures": n_pos,
    }
    if downtime is not None:
        no_model_hours = cost_per_1000(
            {"tp": 0, "fp": 0, "fn": n_pos, "tn": len(y) - n_pos}, downtime
        )
        model_hours = cost_per_1000(counts, downtime)
        metrics["downtime_h_per_1000"] = model_hours
        metrics["downtime_reduction"] = (
            1.0 - model_hours / no_model_hours if no_model_hours else float("nan")
        )
    if modes is not None:
        flagged_mask = s >= threshold
        for mode in modes.columns:
            # Only real failures count: RNF flags also appear on rows without a failure.
            mask = (modes[mode].to_numpy() == 1) & (y == 1)
            if mask.sum():
                metrics[f"recall_{mode}"] = float(flagged_mask[mask].mean())
    return metrics
