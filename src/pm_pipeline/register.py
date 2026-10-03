"""Stage 7 - model selection, quality gates and registration in the MLflow Model Registry.

Naming convention: one registered model per prediction task (``ai4i-failure-risk``), integer
versions created by MLflow, and *aliases* for the deployment role:

* ``champion``   - the version the scoring service loads (``models:/ai4i-failure-risk@champion``)
* ``challenger`` - a newer version that passed the gates and waits for promotion (shadow week)
* ``previous``   - the last champion, kept so rollback is a one-line alias move

The first model that passes the gates becomes ``champion`` directly; after that, promotion is
a human decision (``promote``) unless ``registry.auto_promote`` is true.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from pm_pipeline.config import PipelineConfig
from pm_pipeline.tracking import setup_mlflow
from pm_pipeline.utils import get_logger

log = get_logger(__name__)


@dataclass
class Selection:
    """Best candidate, baseline value and any failed quality gates."""

    run_id: str
    run_name: str
    value: float
    baseline_value: float
    gate_failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every quality gate passed."""
        return not self.gate_failures


def candidate_runs(cfg: PipelineConfig) -> pd.DataFrame:
    """Return the latest finished candidate run per run name, as an MLflow runs frame."""
    experiment_id = setup_mlflow(cfg.tracking)
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.pipeline_stage = 'candidate' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        return runs
    return runs.drop_duplicates(subset="tags.mlflow.runName", keep="first").reset_index(drop=True)


def check_gates(value: float, baseline_value: float, cfg: PipelineConfig) -> list[str]:
    """Return the list of failed gates (empty list = candidate may be registered)."""
    reg = cfg.registry
    failures = []
    if value < reg.min_primary_metric:
        failures.append(f"{reg.primary_metric}={value:.3f} below minimum {reg.min_primary_metric}")
    if value - baseline_value < reg.min_lift_over_baseline:
        failures.append(
            f"lift over baseline {value - baseline_value:.3f} below {reg.min_lift_over_baseline}"
        )
    return failures


def select_best(cfg: PipelineConfig, runs: pd.DataFrame | None = None) -> Selection:
    """Pick the winning run.

    1. keep candidates whose primary metric is within ``selection_tolerance`` of the best
       (differences smaller than that are noise at ~300 CV failures);
    2. among them, highest CV PR-AUC (ranking quality over all alert budgets);
    3. then lowest inference latency.
    """
    runs = candidate_runs(cfg) if runs is None else runs
    if runs.empty:
        raise RuntimeError("no finished candidate runs to select from")
    metric = f"metrics.{cfg.registry.primary_metric}"
    baseline = runs[runs["tags.mlflow.runName"] == cfg.registry.baseline_run_name]
    baseline_value = float(baseline[metric].iloc[0]) if not baseline.empty else 0.0
    top = runs[runs[metric] >= runs[metric].max() - cfg.registry.selection_tolerance]
    ranked = top.sort_values(
        ["metrics.cv_pr_auc", "metrics.latency_ms_per_1000"], ascending=[False, True]
    )
    best = ranked.iloc[0]
    value = float(best[metric])
    return Selection(
        run_id=str(best["run_id"]),
        run_name=str(best["tags.mlflow.runName"]),
        value=value,
        baseline_value=baseline_value,
        gate_failures=check_gates(value, baseline_value, cfg),
    )


def _alias_version(client: MlflowClient, name: str, alias: str) -> Any | None:
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def promote(cfg: PipelineConfig, version: str) -> dict[str, str]:
    """Make ``version`` the champion; the old champion keeps the ``previous`` alias.

    Rollback is the same call with the previous version number.
    """
    setup_mlflow(cfg.tracking)
    client = MlflowClient()
    name = cfg.registry.model_name
    champion = _alias_version(client, name, "champion")
    if champion is not None and str(champion.version) != str(version):
        client.set_registered_model_alias(name, "previous", champion.version)
    client.set_registered_model_alias(name, "champion", version)
    challenger = _alias_version(client, name, "challenger")
    if challenger is not None and str(challenger.version) == str(version):
        client.delete_registered_model_alias(name, "challenger")
    log.info("promoted %s version %s to champion", name, version)
    return {
        "model_name": name,
        "champion": str(version),
        "previous": str(champion.version) if champion else "",
    }


def register_best(cfg: PipelineConfig, selection: Selection | None = None) -> dict[str, Any]:
    """Register the selected run's model and move aliases.

    Idempotent: if the run was already registered (e.g. an Airflow retry), the existing
    version is reused instead of creating a duplicate.
    """
    selection = selection or select_best(cfg)
    result: dict[str, Any] = {"selection": asdict(selection), "registered": False}
    if not selection.passed:
        log.warning("quality gates failed, not registering: %s", selection.gate_failures)
        return result

    setup_mlflow(cfg.tracking)
    client = MlflowClient()
    name = cfg.registry.model_name
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(
            name, description="Failure risk for the next CNC milling cycle (AI4I 2020)."
        )

    existing = [
        mv
        for mv in client.search_model_versions(f"name = '{name}'")
        if mv.run_id == selection.run_id
    ]
    if existing:
        version = existing[0]
    else:
        version = mlflow.register_model(f"runs:/{selection.run_id}/model", name)
    run = client.get_run(selection.run_id)
    tags = {
        "run_name": selection.run_name,
        cfg.registry.primary_metric: f"{selection.value:.4f}",
        "data_sha256": run.data.tags.get("data_sha256", ""),
        "git_commit": run.data.tags.get("git_commit", ""),
        "feature_set": run.data.params.get("feature_set", ""),
        "threshold": f"{run.data.metrics.get('threshold', float('nan')):.6f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, value)

    champion = _alias_version(client, name, "champion")
    champion_value = (
        float(champion.tags.get(cfg.registry.primary_metric, "-inf")) if champion else float("-inf")
    )
    if champion is not None and champion.version == version.version:
        role = "champion"
    elif champion is None or (cfg.registry.auto_promote and selection.value > champion_value):
        promote(cfg, str(version.version))
        role = "champion"
    else:
        client.set_registered_model_alias(name, "challenger", version.version)
        role = "challenger"
    log.info("registered %s version %s as %s", name, version.version, role)
    result.update(
        {"registered": True, "model_name": name, "version": str(version.version), "alias": role}
    )
    return result
