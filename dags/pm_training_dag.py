"""Weekly retraining DAG for the CNC failure-risk model (AI4I 2020 proxy data).

Schedule: every Monday 02:00 Asia/Ho_Chi_Minh. The plant closes the week's maintenance work
orders (the labels) on Sunday evening, and Monday 02:00 is inside the night shift when the
on-prem server is least loaded. The DAG can also be triggered manually, e.g. by the drift
monitor, through the Airflow REST API.

Flow: ingest -> validate -> preprocess -> build_features -> train_candidate (one mapped task
per experiment) -> results_table + select_and_register -> smoke_test_champion.

Only small JSON dicts (paths, hashes, metrics) travel through XCom; data stays in files under
``data/interim/airflow_<ts>/``, so every task can be retried on its own. Package imports are
inside the tasks so the scheduler can parse this file without the ML dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import timedelta
from pathlib import Path
from typing import Any

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.operators.python import get_current_context

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PM_PROJECT_ROOT", "/opt/airflow/project"))
CONFIG_PATH = os.environ.get("PM_CONFIG", str(PROJECT_ROOT / "configs" / "pipeline.yaml"))
EXPERIMENTS_PATH = os.environ.get(
    "PM_EXPERIMENTS", str(PROJECT_ROOT / "configs" / "experiments.yaml")
)


def _alert(text: str) -> None:
    """Log an alert and, if configured, post it to a chat webhook.

    The webhook URL comes from the environment (injected from the Airflow secrets backend),
    never from this file, so the repository stays free of credentials.
    """
    log.error(text)
    url = os.environ.get("PM_ALERT_WEBHOOK_URL")
    if not url:
        return
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except OSError as exc:  # alerting must never mask the original failure
        log.warning("alert webhook failed: %s", exc)


def notify_failure(context: dict[str, Any]) -> None:
    """``on_failure_callback``: called once a task has exhausted its retries."""
    ti = context["task_instance"]
    _alert(
        f"[pm-training] task {ti.task_id} failed (run {context['run_id']}, "
        f"try {ti.try_number}): {context.get('exception')}"
    )


def notify_sla_miss(
    dag: Any, task_list: str, blocking_task_list: str, slas: list[Any], blocking_tis: list[Any]
) -> None:
    """``sla_miss_callback``: training is late, the Monday champion may not be ready."""
    _alert(f"[pm-training] SLA missed in {dag.dag_id}: {task_list}")


def _cfg() -> Any:
    from pm_pipeline.config import load_config

    return load_config(CONFIG_PATH)


def _run_dir(cfg: Any) -> Path:
    """Run-scoped working folder shared by all tasks of one DAG run."""
    ts = get_current_context()["ts_nodash"]
    return cfg.data.work_dir / f"airflow_{ts}"


default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(minutes=30),
    "on_failure_callback": notify_failure,
}


@dag(
    dag_id="pm_failure_risk_training",
    description="Weekly retraining of the CNC failure-risk model",
    schedule="0 2 * * 1",
    start_date=pendulum.datetime(2026, 1, 5, tz="Asia/Ho_Chi_Minh"),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=3),
    default_args=default_args,
    sla_miss_callback=notify_sla_miss,
    tags=["ddm501", "predictive-maintenance", "training"],
    doc_md=__doc__,
)
def pm_failure_risk_training() -> None:
    """Build the task graph (see module docstring)."""

    @task
    def ingest() -> dict[str, Any]:
        """Fetch/verify the raw extract; returns path + sha256."""
        from pm_pipeline.pipeline import stage_ingest

        return stage_ingest(_cfg())

    @task
    def validate(ingested: dict[str, Any]) -> dict[str, Any]:
        """Run the data quality gate; bad data fails at once because a retry cannot fix it."""
        from pm_pipeline.pipeline import stage_validate
        from pm_pipeline.validate import DataValidationError

        cfg = _cfg()
        try:
            return stage_validate(cfg, ingested["raw_path"], _run_dir(cfg))
        except DataValidationError as exc:
            raise AirflowFailException(f"data validation failed: {exc}") from exc

    @task
    def preprocess(ingested: dict[str, Any], validated: dict[str, Any]) -> dict[str, Any]:
        """Clean, drop leakage columns, time-ordered split -> parquet files."""
        from pm_pipeline.pipeline import stage_preprocess

        cfg = _cfg()
        return stage_preprocess(cfg, ingested["raw_path"], _run_dir(cfg))

    @task
    def build_features(split: dict[str, Any]) -> dict[str, Any]:
        """Feature gate + drift reference profile."""
        from pm_pipeline.pipeline import stage_features

        return stage_features(split["dev_path"], _run_dir(_cfg()))

    @task
    def list_experiments() -> list[str]:
        """Experiment names from the matrix file; drives the dynamic task mapping."""
        from pm_pipeline.config import load_experiments

        return [spec.name for spec in load_experiments(EXPERIMENTS_PATH)]

    # Airflow does not support SLAs on mapped tasks, so the SLA sits on the registration
    # step that waits for all of them (see select_and_register).
    @task(max_active_tis_per_dag=4)
    def train_candidate(
        name: str, split: dict[str, Any], ingested: dict[str, Any], features: dict[str, Any]
    ) -> dict[str, Any]:
        """Train + evaluate one configuration and log it as an MLflow run."""
        from pm_pipeline.config import load_experiments
        from pm_pipeline.pipeline import stage_train_candidate

        cfg = _cfg()
        spec = next(s for s in load_experiments(EXPERIMENTS_PATH) if s.name == name)
        return stage_train_candidate(
            cfg, spec, split["dev_path"], split["test_path"], ingested["sha256"], _run_dir(cfg)
        )

    @task
    def results_table(candidates: list[dict[str, Any]]) -> str:
        """Regenerate the results table from MLflow for the weekly report."""
        from pm_pipeline.results import write_results

        cfg = _cfg()
        _, md_path = write_results(cfg, _run_dir(cfg))
        return str(md_path)

    @task(sla=timedelta(hours=2))
    def select_and_register(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply the quality gates; register the winner and move the aliases."""
        from pm_pipeline.register import register_best

        result = register_best(_cfg())
        if not result["registered"]:
            _alert(f"[pm-training] no model registered: {result['selection']['gate_failures']}")
            raise AirflowSkipException("quality gates not met; champion unchanged")
        return result

    @task
    def smoke_test_champion(registration: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
        """Load the new version by its alias, as the scoring service would, and score the holdout.

        The alias is ``champion`` for the first model and ``challenger`` afterwards: promotion
        to champion is a human decision after the shadow week (``pm-pipeline promote``).
        """
        import pandas as pd

        from pm_pipeline.predict import load_champion, score_batch

        cfg = _cfg()
        model, threshold = load_champion(cfg, alias=registration["alias"])
        scored = score_batch(model, threshold, pd.read_parquet(split["test_path"]))
        alert_rate = float(scored["alert"].mean())
        if not 0.0 < alert_rate <= 3 * cfg.evaluation.alert_budget:
            raise AirflowFailException(f"alert rate {alert_rate:.3f} is implausible")
        return {
            "version": registration["version"],
            "alias": registration["alias"],
            "alert_rate": alert_rate,
        }

    ingested = ingest()
    validated = validate(ingested)
    split = preprocess(ingested, validated)
    features = build_features(split)
    candidates = train_candidate.partial(split=split, ingested=ingested, features=features).expand(
        name=list_experiments()
    )
    results_table(candidates)
    registration = select_and_register(candidates)
    smoke_test_champion(registration, split)


pm_failure_risk_training()
