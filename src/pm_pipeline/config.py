"""Configuration loading: one YAML file plus environment-variable overrides.

The same configuration object is used by the CLI, the tests and the Airflow DAG, so a run
in any of those places is described by exactly the same keys. Precedence (highest first):

1. Environment variables ``PM_<SECTION>__<KEY>`` (e.g. ``PM_EVALUATION__ALERT_BUDGET=0.03``).
2. The YAML file given by ``--config`` or ``PM_CONFIG``.
3. Nothing else: there are no hidden defaults, a missing key is an error.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "PM_"
CONFIG_PATH_ENV = "PM_CONFIG"
EXPERIMENTS_PATH_ENV = "PM_EXPERIMENTS"
# PM_* variables that are not config keys (file locations, deployment settings, secrets).
RESERVED_ENV = frozenset(
    {CONFIG_PATH_ENV, EXPERIMENTS_PATH_ENV, "PM_PROJECT_ROOT", "PM_ALERT_WEBHOOK_URL"}
)
DEFAULT_CONFIG_PATH = Path("configs/pipeline.yaml")
DEFAULT_EXPERIMENTS_PATH = Path("configs/experiments.yaml")

ALLOWED_MODELS = ("rule", "logreg", "random_forest", "hist_gb", "lightgbm")
ALLOWED_FEATURE_SETS = ("raw", "physics")
ALLOWED_IMBALANCE = ("none", "class_weight", "undersample", "oversample")


@dataclass(frozen=True)
class DataConfig:
    """Where the raw data lives and how to check it is the expected file."""

    raw_path: Path
    source_url: str
    expected_sha256: str
    work_dir: Path


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds for the data-validation quality gate."""

    min_rows: int
    ranges: dict[str, tuple[float, float]]
    allowed_types: tuple[str, ...]
    failure_rate_band: tuple[float, float]


@dataclass(frozen=True)
class SplitConfig:
    """Time-ordered holdout size and number of cross-validation folds."""

    test_fraction: float
    cv_folds: int


@dataclass(frozen=True)
class CostConfig:
    """Weight of each confusion-matrix outcome for one cycle (USD cost or downtime hours)."""

    false_negative: float
    true_positive: float
    false_positive: float
    true_negative: float


@dataclass(frozen=True)
class EvaluationConfig:
    """Alert budget, bootstrap settings, the cost model and the downtime model."""

    alert_budget: float
    bootstrap_rounds: int
    costs: CostConfig
    downtime_hours: CostConfig


@dataclass(frozen=True)
class TrackingConfig:
    """MLflow tracking location."""

    tracking_uri: str
    experiment_name: str
    artifact_location: str


@dataclass(frozen=True)
class RegistryConfig:
    """Model-registry name and the quality gates for registration."""

    model_name: str
    primary_metric: str
    selection_tolerance: float
    min_primary_metric: float
    min_lift_over_baseline: float
    baseline_run_name: str
    auto_promote: bool


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level, immutable pipeline configuration."""

    seed: int
    data: DataConfig
    validation: ValidationConfig
    split: SplitConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig
    registry: RegistryConfig
    root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], root: Path) -> PipelineConfig:
        """Build a config from a plain dict; relative paths are resolved against ``root``."""
        d, v, s = raw["data"], raw["validation"], raw["split"]
        e, t, r = raw["evaluation"], raw["tracking"], raw["registry"]
        return cls(
            seed=int(raw["seed"]),
            data=DataConfig(
                raw_path=_resolve(root, d["raw_path"]),
                source_url=str(d["source_url"]),
                expected_sha256=str(d["expected_sha256"]),
                work_dir=_resolve(root, d["work_dir"]),
            ),
            validation=ValidationConfig(
                min_rows=int(v["min_rows"]),
                ranges={k: (float(lo), float(hi)) for k, (lo, hi) in v["ranges"].items()},
                allowed_types=tuple(v["allowed_types"]),
                failure_rate_band=(
                    float(v["failure_rate_band"][0]),
                    float(v["failure_rate_band"][1]),
                ),
            ),
            split=SplitConfig(test_fraction=float(s["test_fraction"]), cv_folds=int(s["cv_folds"])),
            evaluation=EvaluationConfig(
                alert_budget=float(e["alert_budget"]),
                bootstrap_rounds=int(e["bootstrap_rounds"]),
                costs=CostConfig(**{k: float(val) for k, val in e["costs"].items()}),
                downtime_hours=CostConfig(
                    **{k: float(val) for k, val in e["downtime_hours"].items()}
                ),
            ),
            tracking=TrackingConfig(
                tracking_uri=_resolve_sqlite_uri(root, str(t["tracking_uri"])),
                experiment_name=str(t["experiment_name"]),
                artifact_location=_resolve(root, t["artifact_location"]).as_uri()
                if "://" not in str(t["artifact_location"])
                else str(t["artifact_location"]),
            ),
            registry=RegistryConfig(
                model_name=str(r["model_name"]),
                primary_metric=str(r["primary_metric"]),
                selection_tolerance=float(r["selection_tolerance"]),
                min_primary_metric=float(r["min_primary_metric"]),
                min_lift_over_baseline=float(r["min_lift_over_baseline"]),
                baseline_run_name=str(r["baseline_run_name"]),
                auto_promote=bool(r["auto_promote"]),
            ),
            root=root,
        )


@dataclass(frozen=True)
class ExperimentSpec:
    """One row of the experiment matrix (one MLflow run)."""

    name: str
    model: str
    feature_set: str
    imbalance: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject unknown algorithms/options early, before any compute is spent."""
        if self.model not in ALLOWED_MODELS:
            raise ValueError(f"{self.name}: unknown model '{self.model}'")
        if self.feature_set not in ALLOWED_FEATURE_SETS:
            raise ValueError(f"{self.name}: unknown feature_set '{self.feature_set}'")
        if self.imbalance not in ALLOWED_IMBALANCE:
            raise ValueError(f"{self.name}: unknown imbalance '{self.imbalance}'")


def apply_env_overrides(
    raw: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
    prefix: str = ENV_PREFIX,
) -> dict[str, Any]:
    """Return a copy of ``raw`` with ``PM_A__B=value`` applied to ``raw['a']['b']``.

    Values are parsed with ``yaml.safe_load`` so ``0.03`` becomes a float and ``true`` a
    bool. Unknown keys raise ``KeyError``: a typo in an override should fail the run, not be
    silently ignored.
    """
    env = os.environ if environ is None else environ
    result: dict[str, Any] = copy.deepcopy(dict(raw))
    for key, value in sorted(env.items()):
        if not key.startswith(prefix) or key in RESERVED_ENV:
            continue
        path = key[len(prefix) :].lower().split("__")
        node: dict[str, Any] = result
        for part in path[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                raise KeyError(f"{key}: '{part}' is not a config section")
            node = child
        if path[-1] not in node:
            raise KeyError(f"{key}: unknown config key '{path[-1]}'")
        node[path[-1]] = yaml.safe_load(value)
    return result


def load_config(
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> PipelineConfig:
    """Load the YAML config, apply env overrides and return a typed ``PipelineConfig``.

    The project root (used for relative paths) is the parent of the ``configs/`` folder, so
    the CLI behaves the same whatever the current working directory is.
    """
    env = os.environ if environ is None else environ
    config_path = Path(path or env.get(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH).resolve()
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    merged = apply_env_overrides(raw, env)
    return PipelineConfig.from_dict(merged, root=config_path.parent.parent)


def load_experiments(
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[ExperimentSpec]:
    """Load the experiment matrix; names must be unique because they become run names."""
    env = os.environ if environ is None else environ
    exp_path = Path(path or env.get(EXPERIMENTS_PATH_ENV) or DEFAULT_EXPERIMENTS_PATH)
    with exp_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    specs = [ExperimentSpec(**item) for item in raw["experiments"]]
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("experiment names must be unique")
    return specs


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _resolve_sqlite_uri(root: Path, uri: str) -> str:
    """Make ``sqlite:///relative.db`` absolute so every process points at the same file."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri.startswith(prefix + "/"):
        return prefix + str(_resolve(root, uri[len(prefix) :]))
    return uri
