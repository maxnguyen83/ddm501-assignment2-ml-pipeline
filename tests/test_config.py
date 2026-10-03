from __future__ import annotations

from pathlib import Path

import pytest

from pm_pipeline.config import ExperimentSpec, apply_env_overrides, load_config, load_experiments

ROOT = Path(__file__).resolve().parents[1]


def test_env_override_is_typed_and_nested() -> None:
    raw = {"seed": 1, "evaluation": {"alert_budget": 0.05, "costs": {"false_negative": 1.0}}}
    env = {
        "PM_EVALUATION__ALERT_BUDGET": "0.03",
        "PM_SEED": "7",
        "PM_EVALUATION__COSTS__FALSE_NEGATIVE": "2500",
        "HOME": "/x",
    }
    out = apply_env_overrides(raw, env)
    assert out["evaluation"]["alert_budget"] == 0.03
    assert out["seed"] == 7
    assert out["evaluation"]["costs"]["false_negative"] == 2500
    assert raw["seed"] == 1  # input not mutated


def test_unknown_override_key_fails_fast() -> None:
    with pytest.raises(KeyError):
        apply_env_overrides({"seed": 1}, {"PM_SEDE": "3"})


def test_reserved_pm_variables_are_not_treated_as_overrides() -> None:
    # Regression: the Airflow image sets PM_PROJECT_ROOT, which used to fail the ingest task.
    env = {"PM_PROJECT_ROOT": "/opt/airflow/project", "PM_ALERT_WEBHOOK_URL": "https://x"}
    assert apply_env_overrides({"seed": 1}, env) == {"seed": 1}


def test_load_config_resolves_paths_against_project_root() -> None:
    cfg = load_config(ROOT / "configs" / "pipeline.yaml", environ={})
    assert cfg.data.raw_path == ROOT / "data" / "raw" / "ai4i2020.csv"
    assert cfg.tracking.tracking_uri == f"sqlite:///{ROOT / 'mlflow.db'}"
    assert cfg.evaluation.costs.false_negative > cfg.evaluation.costs.true_positive


def test_experiment_matrix_is_valid_and_large_enough() -> None:
    specs = load_experiments(ROOT / "configs" / "experiments.yaml")
    assert len(specs) >= 10
    assert {s.model for s in specs} >= {"rule", "logreg", "random_forest", "lightgbm"}
    assert {s.imbalance for s in specs} >= {"none", "class_weight", "undersample", "oversample"}


def test_experiment_spec_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        ExperimentSpec(name="x", model="svm", feature_set="raw", imbalance="none")
