"""Pipeline stages with file-based inputs/outputs, used by both the CLI and the Airflow DAG.

Each ``stage_*`` function reads its inputs from paths, writes its outputs to a run folder and
returns a small JSON-serialisable dict. Airflow passes only these dicts between tasks (XCom),
never DataFrames, so any task can be retried on its own from the files of its parent.
"""

from __future__ import annotations

import pickle
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: the pipeline runs in containers and Airflow workers

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

from pm_pipeline import __version__
from pm_pipeline.config import ExperimentSpec, PipelineConfig
from pm_pipeline.evaluate import (
    bootstrap_ci,
    evaluate_scores,
    threshold_for_alert_rate,
)
from pm_pipeline.features import add_physics_features, check_features, feature_profile
from pm_pipeline.ingest import ingest, load_raw
from pm_pipeline.preprocess import clean, split_xy, temporal_split
from pm_pipeline.tracking import setup_mlflow
from pm_pipeline.train import cross_val_scores, fit_model
from pm_pipeline.utils import get_logger, git_commit, set_global_seed, write_json
from pm_pipeline.validate import assert_valid, validate_raw

log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Data stages
# --------------------------------------------------------------------------------------
def stage_ingest(cfg: PipelineConfig) -> dict[str, Any]:
    """Stage 1: fetch/verify the raw file. Returns its path and sha256."""
    result = ingest(cfg.data)
    return {"raw_path": str(result.path), "sha256": result.sha256, "n_bytes": result.n_bytes}


def stage_validate(cfg: PipelineConfig, raw_path: str, run_dir: str | Path) -> dict[str, Any]:
    """Stage 2: validate the raw data; raises ``DataValidationError`` on blocking errors."""
    report = validate_raw(load_raw(raw_path), cfg.validation)
    out = write_json(report.to_dict(), Path(run_dir) / "validation_report.json")
    for warning in report.warnings:
        log.warning("validation: %s", warning)
    assert_valid(report)
    return {"report_path": str(out), "n_rows": report.n_rows, **report.stats}


def stage_preprocess(cfg: PipelineConfig, raw_path: str, run_dir: str | Path) -> dict[str, Any]:
    """Stage 3: clean, drop leakage/identifier columns and split in production order."""
    split = temporal_split(clean(load_raw(raw_path)), cfg.split.test_fraction)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    dev_path, test_path = run_path / "dev.parquet", run_path / "test.parquet"
    split.dev.to_parquet(dev_path, index=False)
    split.test.to_parquet(test_path, index=False)
    return {
        "dev_path": str(dev_path),
        "test_path": str(test_path),
        "n_dev": len(split.dev),
        "n_test": len(split.test),
        "dev_failure_rate": round(float(split.dev["machine_failure"].mean()), 5),
        "test_failure_rate": round(float(split.test["machine_failure"].mean()), 5),
    }


def stage_features(dev_path: str, run_dir: str | Path) -> dict[str, Any]:
    """Stage 4: materialise physics features, run the feature gate, save the drift profile."""
    features = add_physics_features(pd.read_parquet(dev_path))
    problems = check_features(features)
    if problems:
        raise ValueError(f"feature quality gate failed: {problems}")
    out = write_json(feature_profile(features), Path(run_dir) / "feature_profile.json")
    return {"profile_path": str(out), "n_features": len(feature_profile(features))}


# --------------------------------------------------------------------------------------
# Training + evaluation (one MLflow run per experiment)
# --------------------------------------------------------------------------------------
def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in metrics.items() if np.isfinite(v)}


def _pr_curve_png(y: pd.Series, scores: np.ndarray, title: str, path: Path) -> Path:
    precision, recall, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(recall, precision, color="#1f5a96")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _model_requirements() -> list[str]:
    packages = ("scikit-learn", "lightgbm", "pandas", "numpy", "pyyaml")
    return [f"{p}=={version(p)}" for p in packages] + [f"mlflow=={mlflow.__version__}"]


def _latency_ms_per_1000(model: Any, x: pd.DataFrame, repeats: int = 5) -> float:
    batch = pd.concat([x] * (1000 // len(x) + 1)).iloc[:1000]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(batch)
        timings.append(time.perf_counter() - start)
    return 1000.0 * float(np.median(timings))


def train_and_evaluate(
    cfg: PipelineConfig,
    spec: ExperimentSpec,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    data_sha256: str,
    extra_artifacts: list[Path] | None = None,
) -> dict[str, Any]:
    """Run one experiment: CV on dev, refit, score the time-ordered holdout, log to MLflow.

    The operating threshold is fixed from the out-of-fold scores (top ``alert_budget`` of
    dev cycles) and then applied unchanged to the holdout, exactly as it would be in
    production. Returns the run id and the headline metrics.
    """
    set_global_seed(cfg.seed)
    setup_mlflow(cfg.tracking)
    x_dev, y_dev, modes_dev = split_xy(dev)
    x_test, y_test, modes_test = split_xy(test)
    ev = cfg.evaluation

    with mlflow.start_run(run_name=spec.name) as run:
        mlflow.set_tags(
            {
                "pipeline_stage": "candidate",
                "pipeline_version": __version__,
                "git_commit": git_commit(),
                "data_sha256": data_sha256,
                "model_family": spec.model,
            }
        )
        mlflow.log_params(
            {
                "model": spec.model,
                "feature_set": spec.feature_set,
                "imbalance": spec.imbalance,
                "seed": cfg.seed,
                "cv_folds": cfg.split.cv_folds,
                "alert_budget": ev.alert_budget,
                "n_dev": len(dev),
                "n_test": len(test),
                "data_sha256": data_sha256[:16],
                **{f"hp_{k}": v for k, v in spec.params.items()},
            }
        )

        start = time.perf_counter()
        oof, _ = cross_val_scores(spec, x_dev, y_dev, cfg.split.cv_folds, cfg.seed)
        cv_seconds = time.perf_counter() - start
        threshold = threshold_for_alert_rate(oof, ev.alert_budget)
        cv_metrics = evaluate_scores(
            y_dev, oof, threshold, ev.alert_budget, ev.costs, modes_dev, ev.downtime_hours
        )
        ci_lo, ci_hi = bootstrap_ci(y_dev, oof, ev.alert_budget, ev.bootstrap_rounds, cfg.seed)

        start = time.perf_counter()
        model = fit_model(spec, x_dev, y_dev, cfg.seed)
        fit_seconds = time.perf_counter() - start
        test_scores = model.predict_proba(x_test)[:, 1]
        test_metrics = evaluate_scores(
            y_test, test_scores, threshold, ev.alert_budget, ev.costs, modes_test, ev.downtime_hours
        )
        test_lo, test_hi = bootstrap_ci(
            y_test, test_scores, ev.alert_budget, ev.bootstrap_rounds, cfg.seed
        )

        mlflow.log_metrics(
            {
                **_prefixed("cv", cv_metrics),
                **_prefixed("test", test_metrics),
                "cv_recall_at_budget_ci_low": ci_lo,
                "cv_recall_at_budget_ci_high": ci_hi,
                "test_recall_at_budget_ci_low": test_lo,
                "test_recall_at_budget_ci_high": test_hi,
                "threshold": threshold,
                "cv_seconds": cv_seconds,
                "fit_seconds": fit_seconds,
                "latency_ms_per_1000": _latency_ms_per_1000(model, x_test.head(200)),
                "model_size_kb": len(pickle.dumps(model)) / 1024.0,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_json(
                {
                    "threshold": threshold,
                    "alert_budget": ev.alert_budget,
                    "chosen_on": "out-of-fold dev scores",
                },
                tmp_path / "threshold.json",
            )
            _pr_curve_png(
                y_dev, oof, f"{spec.name} - CV precision/recall", tmp_path / "pr_curve_cv.png"
            )
            importance = permutation_importance(
                model,
                x_test,
                y_test,
                scoring="average_precision",
                n_repeats=5,
                random_state=cfg.seed,
                n_jobs=1,
            )
            pd.DataFrame(
                {
                    "feature": x_test.columns,
                    "importance_mean": importance.importances_mean,
                    "importance_std": importance.importances_std,
                }
            ).sort_values("importance_mean", ascending=False).to_csv(
                tmp_path / "permutation_importance.csv", index=False
            )
            pd.DataFrame({"udi": dev["udi"], "y": y_dev, "oof_score": oof}).to_csv(
                tmp_path / "oof_scores.csv", index=False
            )
            mlflow.log_artifacts(str(tmp_path), artifact_path="evaluation")
        for path in extra_artifacts or []:
            if Path(path).exists():
                mlflow.log_artifact(str(path), artifact_path="data_checks")

        # The pickled pipeline references pm_pipeline classes (PhysicsFeatures, ToolWearRule),
        # so the package source and pinned requirements are stored with the model.
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(x_test, test_scores),
            input_example=x_test.head(3),
            code_paths=[str(Path(__file__).resolve().parent)],
            pip_requirements=_model_requirements(),
        )
        log.info(
            "%s: cv recall@budget=%.3f test recall@budget=%.3f",
            spec.name,
            cv_metrics["recall_at_budget"],
            test_metrics["recall_at_budget"],
        )
        return {
            "run_id": run.info.run_id,
            "name": spec.name,
            "cv_recall_at_budget": cv_metrics["recall_at_budget"],
            "cv_pr_auc": cv_metrics["pr_auc"],
            "test_recall_at_budget": test_metrics["recall_at_budget"],
        }


def stage_train_candidate(
    cfg: PipelineConfig,
    spec: ExperimentSpec,
    dev_path: str,
    test_path: str,
    data_sha256: str,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Stage 5+6 for one experiment, reading the split produced by ``stage_preprocess``."""
    extras = [Path(run_dir) / "validation_report.json", Path(run_dir) / "feature_profile.json"]
    return train_and_evaluate(
        cfg,
        spec,
        pd.read_parquet(dev_path),
        pd.read_parquet(test_path),
        data_sha256,
        extra_artifacts=extras,
    )


def run_all(
    cfg: PipelineConfig, specs: list[ExperimentSpec], run_dir: str | Path
) -> list[dict[str, Any]]:
    """Run every stage up to evaluation for all experiments (CLI equivalent of the DAG)."""
    ingested = stage_ingest(cfg)
    stage_validate(cfg, ingested["raw_path"], run_dir)
    split = stage_preprocess(cfg, ingested["raw_path"], run_dir)
    stage_features(split["dev_path"], run_dir)
    return [
        stage_train_candidate(
            cfg, spec, split["dev_path"], split["test_path"], ingested["sha256"], run_dir
        )
        for spec in specs
    ]
