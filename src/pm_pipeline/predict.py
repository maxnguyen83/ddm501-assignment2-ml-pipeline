"""Deployment-side helpers: load the champion by alias and score a batch of cycles."""

from __future__ import annotations

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from pm_pipeline.config import PipelineConfig
from pm_pipeline.preprocess import INPUT_COLUMNS
from pm_pipeline.tracking import setup_mlflow


def load_champion(cfg: PipelineConfig, alias: str = "champion") -> tuple[object, float]:
    """Return (sklearn pipeline, operating threshold) for the model behind ``alias``."""
    setup_mlflow(cfg.tracking)
    name = cfg.registry.model_name
    version = MlflowClient().get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    return model, float(version.tags["threshold"])


def score_batch(model: object, threshold: float, cycles: pd.DataFrame) -> pd.DataFrame:
    """Add ``risk`` and ``alert`` columns to a frame of raw cycle inputs."""
    missing = [c for c in INPUT_COLUMNS if c not in cycles.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    out = cycles.copy()
    out["risk"] = model.predict_proba(cycles[list(INPUT_COLUMNS)])[:, 1]  # type: ignore[attr-defined]
    out["alert"] = out["risk"] >= threshold
    return out
