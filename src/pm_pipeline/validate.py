"""Stage 2 - data validation: the quality gate between raw data and training.

Errors stop the pipeline (schema broken, values physically impossible, label rate outside the
sanity band). Warnings are logged and stored with the run but do not block (for example the
known AI4I label inconsistencies, which are real properties of the source data).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from pm_pipeline.config import ValidationConfig

REQUIRED_COLUMNS: dict[str, str] = {
    "udi": "i",
    "product_id": "O",
    "product_type": "O",
    "air_temp_k": "f",
    "process_temp_k": "f",
    "rotational_speed_rpm": "i",
    "torque_nm": "f",
    "tool_wear_min": "i",
    "machine_failure": "i",
    "twf": "i",
    "hdf": "i",
    "pwf": "i",
    "osf": "i",
    "rnf": "i",
}
BINARY_COLUMNS = ("machine_failure", "twf", "hdf", "pwf", "osf", "rnf")
MODE_COLUMNS = ("twf", "hdf", "pwf", "osf", "rnf")


class DataValidationError(ValueError):
    """Raised when a validation report contains blocking errors."""


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_raw`; serialised to JSON and logged to MLflow."""

    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when there are no blocking errors."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON/MLflow logging."""
        return {**asdict(self), "passed": self.passed}


def _kind_ok(series: pd.Series, kind: str) -> bool:
    if kind == "f":
        return pd.api.types.is_numeric_dtype(series)
    if kind == "i":
        return pd.api.types.is_integer_dtype(series)
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def validate_raw(df: pd.DataFrame, cfg: ValidationConfig) -> ValidationReport:
    """Check schema, completeness, ranges, label sanity and label consistency.

    Args:
        df: raw frame after :func:`pm_pipeline.ingest.load_raw`.
        cfg: validation thresholds from the config.

    Returns:
        A report; call :func:`assert_valid` to turn errors into an exception.

    """
    report = ValidationReport(n_rows=len(df))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.errors.append(f"missing columns: {missing}")
        return report  # nothing else is meaningful without the schema

    for col, kind in REQUIRED_COLUMNS.items():
        if not _kind_ok(df[col], kind):
            report.errors.append(f"column {col} has dtype {df[col].dtype}, expected kind {kind}")

    if len(df) < cfg.min_rows:
        report.errors.append(f"only {len(df)} rows, minimum is {cfg.min_rows}")

    n_null = int(df[list(REQUIRED_COLUMNS)].isna().sum().sum())
    if n_null:
        report.errors.append(f"{n_null} null values in required columns")

    if df["udi"].duplicated().any():
        report.errors.append(f"{int(df['udi'].duplicated().sum())} duplicate udi values")

    bad_types = sorted(set(df["product_type"].dropna()) - set(cfg.allowed_types))
    if bad_types:
        report.errors.append(f"unknown product_type values: {bad_types}")

    for col, (lo, hi) in cfg.ranges.items():
        out_of_range = int(((df[col] < lo) | (df[col] > hi)).sum())
        if out_of_range:
            report.errors.append(f"{out_of_range} rows of {col} outside [{lo}, {hi}]")

    for col in BINARY_COLUMNS:
        if not df[col].isin([0, 1]).all():
            report.errors.append(f"{col} is not binary")

    rate = float(df["machine_failure"].mean()) if len(df) else 0.0
    lo, hi = cfg.failure_rate_band
    if not lo <= rate <= hi:
        report.errors.append(f"failure rate {rate:.4f} outside sanity band [{lo}, {hi}]")

    # Label consistency: known quirks of AI4I, kept as warnings so they stay visible.
    any_mode = df[list(MODE_COLUMNS)].sum(axis=1) > 0
    fail_no_mode = int(((df["machine_failure"] == 1) & ~any_mode).sum())
    mode_no_fail = int(((df["machine_failure"] == 0) & any_mode).sum())
    if fail_no_mode:
        report.warnings.append(f"{fail_no_mode} failures have no failure-mode flag")
    if mode_no_fail:
        report.warnings.append(f"{mode_no_fail} rows have a mode flag but machine_failure=0")

    report.stats = {
        "failure_rate": round(rate, 5),
        "n_failures": int(df["machine_failure"].sum()),
        "mode_counts": {m: int(df[m].sum()) for m in MODE_COLUMNS},
        "type_share": {
            k: round(float(v), 4)
            for k, v in df["product_type"].value_counts(normalize=True).items()
        },
        "fail_without_mode": fail_no_mode,
        "mode_without_fail": mode_no_fail,
    }
    return report


def assert_valid(report: ValidationReport) -> None:
    """Raise :class:`DataValidationError` if the report has blocking errors."""
    if not report.passed:
        raise DataValidationError("; ".join(report.errors))
