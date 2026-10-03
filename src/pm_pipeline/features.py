"""Stage 4 - feature engineering.

Physics features come from how a milling machine fails: heat dissipation depends on the gap
between process and air temperature, power failures on mechanical power (torque x angular
speed), and overstrain on tool wear x torque. The transformer below is used twice: by the
feature stage (to materialise and profile features) and *inside* the model pipeline, so the
serving code sends raw sensor values and can never compute a feature differently from
training (no training/serving skew).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_NUMERIC = (
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)
CATEGORICAL = ("product_type",)
PHYSICS = ("temp_diff_k", "power_w", "strain_nm_min")
PRODUCT_TYPES = ("L", "M", "H")


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the three physics features appended.

    * ``temp_diff_k``   = process temperature - air temperature [K]
    * ``power_w``       = torque [Nm] x rotational speed [rpm] x 2*pi/60  [W]
    * ``strain_nm_min`` = tool wear [min] x torque [Nm]
    """
    out = df.copy()
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * (2.0 * math.pi / 60.0)
    out["strain_nm_min"] = out["tool_wear_min"] * out["torque_nm"]
    return out


class PhysicsFeatures(TransformerMixin, BaseEstimator):
    """Stateless sklearn transformer wrapping :func:`add_physics_features`."""

    def fit(self, x: pd.DataFrame, y: Any = None) -> PhysicsFeatures:
        """No fitting needed; present for the sklearn API."""
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Append the physics columns."""
        return add_physics_features(x)


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) column lists for a feature set name."""
    numeric = list(RAW_NUMERIC)
    if feature_set == "physics":
        numeric += list(PHYSICS)
    elif feature_set != "raw":
        raise ValueError(f"unknown feature_set {feature_set!r}")
    return numeric, list(CATEGORICAL)


def build_column_transformer(feature_set: str, scale: bool) -> ColumnTransformer:
    """One-hot encode product type; standardise numerics only for linear models."""
    numeric, categorical = feature_columns(feature_set)
    num_step: Any = StandardScaler() if scale else "passthrough"
    transformer = ColumnTransformer(
        transformers=[
            ("num", num_step, numeric),
            (
                "cat",
                OneHotEncoder(
                    categories=[list(PRODUCT_TYPES)], handle_unknown="ignore", sparse_output=False
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    # Keep column names downstream so feature importances stay readable.
    return transformer.set_output(transform="pandas")


def feature_profile(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summary statistics per numeric feature; stored as the drift-monitoring reference."""
    numeric = [c for c in (*RAW_NUMERIC, *PHYSICS) if c in df.columns]
    profile: dict[str, dict[str, float]] = {}
    for col in numeric:
        values = df[col].to_numpy(dtype=float)
        profile[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p01": float(np.quantile(values, 0.01)),
            "p50": float(np.quantile(values, 0.50)),
            "p99": float(np.quantile(values, 0.99)),
        }
    return profile


def check_features(df: pd.DataFrame) -> list[str]:
    """Feature quality gate: derived features must be finite and physically plausible."""
    problems: list[str] = []
    derived = df[[c for c in PHYSICS if c in df.columns]]
    if not np.isfinite(derived.to_numpy(dtype=float)).all():
        problems.append("non-finite values in physics features")
    if "power_w" in df and (df["power_w"] <= 0).any():
        problems.append("non-positive mechanical power")
    if "temp_diff_k" in df and (df["temp_diff_k"] <= 0).any():
        problems.append("process temperature not above air temperature")
    return problems
