"""Command-line entry point: ``pm-pipeline <command>`` (see ``--help``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pm_pipeline.config import load_config, load_experiments
from pm_pipeline.pipeline import run_all, stage_ingest, stage_validate
from pm_pipeline.predict import load_champion, score_batch
from pm_pipeline.register import promote, register_best
from pm_pipeline.results import write_results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pm-pipeline", description=__doc__)
    parser.add_argument("--config", help="pipeline YAML (default: $PM_CONFIG or configs/)")
    parser.add_argument("--experiments", help="experiment matrix YAML")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="ingest + validate the raw data")
    run = sub.add_parser("run", help="run the experiment matrix (all stages up to evaluation)")
    run.add_argument("--only", nargs="*", help="run only these experiment names")
    sub.add_parser("results", help="write reports/experiment_results.{csv,md} from MLflow")
    sub.add_parser("register", help="select the best run, apply gates, register + alias")
    promote_cmd = sub.add_parser("promote", help="move @champion to a registered version")
    promote_cmd.add_argument("--version", required=True, help="model version to promote")
    score = sub.add_parser("score", help="score a CSV of cycles with the champion model")
    score.add_argument("csv", help="CSV with the six raw input columns")
    sub.add_parser("all", help="run + results + register")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the pipeline stages."""
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    run_dir = cfg.data.work_dir / "cli"

    if args.command == "validate":
        ingested = stage_ingest(cfg)
        print(json.dumps(stage_validate(cfg, ingested["raw_path"], run_dir), indent=2))
    elif args.command in ("run", "all"):
        specs = load_experiments(args.experiments)
        if getattr(args, "only", None):
            specs = [s for s in specs if s.name in set(args.only)]
        for row in run_all(cfg, specs, run_dir):
            print(json.dumps(row))
    if args.command in ("results", "all"):
        _, md_path = write_results(cfg, cfg.root / "reports")
        print(md_path.read_text(encoding="utf-8"))
    if args.command in ("register", "all"):
        print(json.dumps(register_best(cfg), indent=2, default=str))
    if args.command == "promote":
        print(json.dumps(promote(cfg, args.version), indent=2))
    if args.command == "score":
        model, threshold = load_champion(cfg)
        scored = score_batch(model, threshold, pd.read_csv(Path(args.csv)))
        print(scored.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
