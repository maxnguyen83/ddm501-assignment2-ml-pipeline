"""MLflow setup shared by the training, registry and reporting code."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from pm_pipeline.config import TrackingConfig


def setup_mlflow(cfg: TrackingConfig) -> str:
    """Point MLflow at the configured backend and return the experiment id.

    The experiment is created with an explicit artifact location so that artifacts end up
    in the project's ``mlartifacts/`` folder (or an object store URI in production) rather
    than wherever the process happens to run.
    """
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_registry_uri(cfg.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            cfg.experiment_name,
            artifact_location=cfg.artifact_location,
            tags={"project": "ddm501-predictive-maintenance", "dataset": "uci-ai4i-2020"},
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id
