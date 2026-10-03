"""Build the report figures and slice tables from the MLflow runs (no hand-typed numbers).

Usage: python scripts/make_figures.py [--out report/figures]
Reads the out-of-fold scores logged by each run, so the curves use exactly the data the
results table was computed from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from pm_pipeline.config import load_config
from pm_pipeline.evaluate import confusion_at_threshold, cost_per_1000, recall_at_alert_rate
from pm_pipeline.register import candidate_runs, select_best

# Categorical slots 1-4 of the reference palette (validated adjacent-pair CVD separation).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _oof(run_id: str) -> pd.DataFrame:
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="evaluation/oof_scores.csv"
    )
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="report/figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    runs = candidate_runs(cfg)
    best = select_best(cfg, runs)
    by_name = runs.set_index("tags.mlflow.runName")
    names = ["E01_rule_toolwear", "E02_logreg_raw", "E05_rf_raw", best.run_name]
    labels = [
        "Rule: tool wear (current practice)",
        "Logistic regression (raw)",
        "Random forest (raw)",
        f"Selected: {best.run_name}",
    ]

    budgets = np.linspace(0.01, 0.10, 46)
    fig_r, ax_r = plt.subplots(figsize=(6.4, 4.0))
    fig_c, ax_c = plt.subplots(figsize=(6.4, 4.0))
    curves = {}
    for name, label, color in zip(names, labels, COLORS, strict=True):
        oof = _oof(by_name.loc[name, "run_id"])
        y, s = oof["y"].to_numpy(), oof["oof_score"].to_numpy()
        recall = [recall_at_alert_rate(y, s, b) for b in budgets]
        cost = []
        for b in budgets:
            thr = np.sort(s)[::-1][max(1, int(np.ceil(b * len(s)))) - 1]
            cost.append(cost_per_1000(confusion_at_threshold(y, s, thr), cfg.evaluation.costs))
        curves[name] = {"recall": recall, "cost": cost}
        ax_r.plot(budgets * 100, recall, color=color, linewidth=2, label=label)
        ax_c.plot(budgets * 100, cost, color=color, linewidth=2, label=label)
    no_model = 1000 * cfg.evaluation.costs.false_negative * float(y.mean())
    ax_c.axhline(no_model, color=MUTED, linewidth=1, linestyle="--")
    ax_c.text(10, no_model, "  no model", color=MUTED, fontsize=8, va="bottom", ha="right")
    for ax, fig, ylabel, fname in (
        (ax_r, fig_r, "Share of failures caught (recall)", "recall_vs_budget.png"),
        (ax_c, fig_c, "Expected cost per 1,000 cycles (USD)", "cost_vs_budget.png"),
    ):
        _style(ax)
        ax.axvline(cfg.evaluation.alert_budget * 100, color=MUTED, linewidth=1, linestyle=":")
        ax.set_xlabel("Alert budget (% of cycles flagged for inspection)", fontsize=9, color=INK)
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
        ax.legend(
            fontsize=7.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2)
        )
        fig.tight_layout()
        fig.savefig(out / fname, dpi=160)
        plt.close(fig)

    # Permutation importance of the selected model (holdout, average-precision drop).
    imp_path = mlflow.artifacts.download_artifacts(
        run_id=best.run_id, artifact_path="evaluation/permutation_importance.csv"
    )
    imp = pd.read_csv(imp_path).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.barh(
        imp["feature"],
        imp["importance_mean"],
        xerr=imp["importance_std"],
        color=COLORS[0],
        height=0.55,
        error_kw={"ecolor": MUTED, "lw": 1},
    )
    _style(ax)
    ax.set_xlabel("Drop in holdout PR-AUC when the input is shuffled", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / "permutation_importance.png", dpi=160)
    plt.close(fig)

    # Per-failure-mode recall at the operating threshold (slice analysis).
    # RNF is left out: only one RNF row is also a labelled failure, so its recall is noise.
    modes = ["twf", "hdf", "pwf", "osf"]
    rows = []
    for name in names:
        row = {"run": name}
        for split in ("cv", "test"):
            for m in modes:
                key = f"metrics.{split}_recall_{m}"
                row[f"{split}_{m}"] = by_name.loc[name, key] if key in by_name else np.nan
        rows.append(row)
    slices = pd.DataFrame(rows)
    slices.to_csv(out / "recall_by_failure_mode.csv", index=False)

    # Compact results table for the report (same numbers as reports/experiment_results.md).
    def ci(prefix: str, row: pd.Series) -> str:
        return (
            f"{row[f'metrics.{prefix}_recall_at_budget']:.3f} "
            f"[{row[f'metrics.{prefix}_recall_at_budget_ci_low']:.2f}, "
            f"{row[f'metrics.{prefix}_recall_at_budget_ci_high']:.2f}]"
        )

    ordered = runs.sort_values(
        ["metrics.cv_recall_at_budget", "metrics.cv_pr_auc"], ascending=False
    )
    lines = [
        "| # | Run | CV recall@5% [95% CI] | CV PR-AUC | CV cost/1k cycles (USD) "
        "| CV downtime reduction | Holdout recall@5% [95% CI] | Holdout recall at fixed "
        "threshold | Holdout alert rate | Holdout downtime reduction |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['tags.mlflow.runName']} | {ci('cv', row)} "
            f"| {row['metrics.cv_pr_auc']:.3f} | {row['metrics.cv_cost_per_1000']:,.0f} "
            f"| {row['metrics.cv_downtime_reduction']:.1%} | {ci('test', row)} "
            f"| {row['metrics.test_thr_recall']:.3f} | {row['metrics.test_thr_alert_rate']:.1%} "
            f"| {row['metrics.test_downtime_reduction']:.1%} |"
        )
    (out / "results_compact.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "selected": best.run_name,
        "selected_run_id": best.run_id,
        "budget_curve_points": {
            n: {
                "budget_3pct_recall": c["recall"][10],
                "budget_5pct_recall": c["recall"][20],
                "budget_8pct_recall": c["recall"][35],
            }
            for n, c in curves.items()
        },
        "no_model_cost_per_1000_dev": no_model,
    }
    (out / "figure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(slices.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
