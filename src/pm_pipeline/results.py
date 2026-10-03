"""Build the experiment results table straight from MLflow (no hand-copied numbers)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pm_pipeline.config import PipelineConfig
from pm_pipeline.register import candidate_runs

COLUMNS: dict[str, str] = {
    "tags.mlflow.runName": "run",
    "params.model": "model",
    "params.feature_set": "features",
    "params.imbalance": "imbalance",
    "metrics.cv_recall_at_budget": "cv_recall@5%",
    "metrics.cv_recall_at_budget_ci_low": "ci_low",
    "metrics.cv_recall_at_budget_ci_high": "ci_high",
    "metrics.cv_pr_auc": "cv_pr_auc",
    "metrics.cv_roc_auc": "cv_roc_auc",
    "metrics.cv_cost_per_1000": "cv_cost/1k",
    "metrics.cv_downtime_reduction": "cv_downtime_red",
    "metrics.test_recall_at_budget": "test_recall@5%",
    "metrics.test_pr_auc": "test_pr_auc",
    "metrics.test_thr_alert_rate": "test_alert_rate",
    "metrics.test_thr_recall": "test_thr_recall",
    "metrics.test_cost_per_1000": "test_cost/1k",
    "metrics.test_downtime_reduction": "test_downtime_red",
    "metrics.latency_ms_per_1000": "ms/1k",
    "metrics.fit_seconds": "fit_s",
}


def results_table(cfg: PipelineConfig) -> pd.DataFrame:
    """One row per experiment, sorted by the primary metric then CV PR-AUC."""
    runs = candidate_runs(cfg)
    if runs.empty:
        return pd.DataFrame(columns=list(COLUMNS.values()))
    table = runs[[c for c in COLUMNS if c in runs.columns]].rename(columns=COLUMNS)
    table = table.sort_values(["cv_recall@5%", "cv_pr_auc"], ascending=False)
    return table.reset_index(drop=True)


def to_markdown(table: pd.DataFrame, digits: int = 3) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table without extra dependencies."""

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, sep, *rows]) + "\n"


def write_results(cfg: PipelineConfig, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``experiment_results.csv`` and ``experiment_results.md`` to ``out_dir``."""
    table = results_table(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out / "experiment_results.csv", out / "experiment_results.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(table), encoding="utf-8")
    return csv_path, md_path
