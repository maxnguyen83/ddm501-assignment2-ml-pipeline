from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from pm_pipeline.config import ExperimentSpec, PipelineConfig
from pm_pipeline.predict import load_champion, score_batch
from pm_pipeline.preprocess import clean, split_xy, temporal_split
from pm_pipeline.register import (
    Selection,
    candidate_runs,
    check_gates,
    promote,
    register_best,
    select_best,
)
from pm_pipeline.results import results_table, to_markdown
from pm_pipeline.train import build_model, cross_val_scores, fit_model, resample

from .conftest import make_raw


def runs_by_name(cfg: PipelineConfig) -> dict[str, str]:
    runs = candidate_runs(cfg)
    return dict(zip(runs["tags.mlflow.runName"], runs["run_id"], strict=True))


def _xy(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    x, y, _ = split_xy(clean(make_raw(n, seed, failure_rate=0.08)))
    return x, y


def test_resampling_only_changes_class_balance() -> None:
    x, y = _xy()
    _, yu = resample(x, y, "undersample", target_ratio=0.5, seed=0)
    assert yu.sum() == y.sum()  # all failures kept
    assert (yu == 0).sum() == round(y.sum() / 0.5)  # majority reduced to the ratio
    _, yo = resample(x, y, "oversample", target_ratio=0.5, seed=0)
    assert (yo == 0).sum() == (y == 0).sum()  # majority untouched
    assert yo.sum() == round((y == 0).sum() * 0.5)
    assert resample(x, y, "none", 0.5, 0)[0] is x


@pytest.mark.parametrize("model", ["rule", "logreg", "random_forest", "hist_gb", "lightgbm"])
def test_every_model_family_builds_and_scores(model: str) -> None:
    x, y = _xy()
    spec = ExperimentSpec(
        name=f"t_{model}",
        model=model,
        feature_set="physics",
        imbalance="none",
        params={"n_estimators": 20} if model in ("random_forest", "lightgbm") else {},
    )
    if model == "rule":
        spec = ExperimentSpec(name="t_rule", model="rule", feature_set="raw", imbalance="none")
    scores = fit_model(spec, x, y, seed=0).predict_proba(x)[:, 1]
    assert scores.shape == (len(x),)
    assert np.all((scores >= 0) & (scores <= 1))


def test_training_is_reproducible_with_fixed_seed() -> None:
    x, y = _xy()
    spec = ExperimentSpec(
        name="rf",
        model="random_forest",
        feature_set="physics",
        imbalance="undersample",
        params={"n_estimators": 30},
    )
    first, _ = cross_val_scores(spec, x, y, folds=3, seed=42)
    second, _ = cross_val_scores(spec, x, y, folds=3, seed=42)
    np.testing.assert_array_equal(first, second)


def test_physics_pipeline_accepts_raw_inputs_only() -> None:
    spec = ExperimentSpec(name="p", model="logreg", feature_set="physics", imbalance="none")
    model = build_model(spec, seed=0)
    assert [name for name, _ in model.steps] == ["physics", "encode", "model"]


def test_end_to_end_train_register_and_score(cfg: PipelineConfig) -> None:
    """Small but complete run: two experiments -> MLflow -> registry alias -> scoring."""
    from pm_pipeline.pipeline import train_and_evaluate

    split = temporal_split(clean(make_raw(900, seed=3, failure_rate=0.08)), 0.2)
    specs = [
        ExperimentSpec(
            name=cfg.registry.baseline_run_name, model="rule", feature_set="raw", imbalance="none"
        ),
        ExperimentSpec(
            name="lgbm_small",
            model="lightgbm",
            feature_set="physics",
            imbalance="none",
            params={"n_estimators": 30},
        ),
    ]
    for spec in specs:
        train_and_evaluate(cfg, spec, split.dev, split.test, data_sha256="test")

    table = results_table(cfg)
    assert set(table["run"]) == {s.name for s in specs}
    assert to_markdown(table).startswith("| run |")

    selection = select_best(cfg)
    # Synthetic labels are random, so force the gates open to exercise the registry path.
    selection.gate_failures = []
    result = register_best(cfg, selection)
    assert result["registered"] and result["alias"] == "champion"
    again = register_best(cfg, selection)  # idempotent on retry
    assert again["version"] == result["version"]
    client = MlflowClient()
    champion = client.get_model_version_by_alias(cfg.registry.model_name, "champion")
    assert str(champion.version) == result["version"]

    model, threshold = load_champion(cfg)
    scored = score_batch(model, threshold, split.test)
    assert {"risk", "alert"} <= set(scored.columns)

    # A second, different candidate only becomes challenger (auto_promote is false) ...
    other = runs_by_name(cfg)["E01_rule_toolwear"]
    second = register_best(cfg, Selection(other, "E01_rule_toolwear", 0.99, 0.0))
    assert second["alias"] == "challenger"
    # ... until a person promotes it; the old champion is kept as `previous` for rollback.
    promote(cfg, second["version"])
    assert (
        str(client.get_model_version_by_alias(cfg.registry.model_name, "champion").version)
        == second["version"]
    )
    assert (
        str(client.get_model_version_by_alias(cfg.registry.model_name, "previous").version)
        == result["version"]
    )


def test_check_gates_requires_lift_over_baseline(cfg: PipelineConfig) -> None:
    assert check_gates(0.90, 0.22, cfg) == []
    assert any("lift" in f for f in check_gates(0.75, 0.70, cfg))
    assert any("below minimum" in f for f in check_gates(0.50, 0.10, cfg))
