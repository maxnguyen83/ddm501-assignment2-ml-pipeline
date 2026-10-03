"""Stage 3 - preprocessing: clean types, remove leakage, split in time order.

Leakage is the main risk in this dataset: the five failure-mode flags (TWF, HDF, PWF, OSF,
RNF) are recorded *after* the failure and together almost equal the target. They are kept
only as evaluation slices and never reach the model. ``product_id`` is an identifier that
also encodes the product type, so it is dropped as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pm_pipeline.validate import MODE_COLUMNS

TARGET = "machine_failure"
ORDER_KEY = "udi"
ID_COLUMNS = ("product_id",)
INPUT_COLUMNS = (
    "product_type",
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)


@dataclass(frozen=True)
class SplitData:
    """Development (train + CV) and time-ordered holdout frames."""

    dev: pd.DataFrame
    test: pd.DataFrame


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by production order, normalise dtypes and drop exact duplicate rows."""
    out = df.drop_duplicates().sort_values(ORDER_KEY).reset_index(drop=True)
    out["product_type"] = out["product_type"].astype(str).str.strip().str.upper()
    float_cols = ["air_temp_k", "process_temp_k", "torque_nm"]
    int_cols = ["rotational_speed_rpm", "tool_wear_min", TARGET, *MODE_COLUMNS]
    out[float_cols] = out[float_cols].astype("float64")
    out[int_cols] = out[int_cols].astype("int64")
    return out.drop(columns=list(ID_COLUMNS), errors="ignore")


def temporal_split(df: pd.DataFrame, test_fraction: float) -> SplitData:
    """Hold out the *last* ``test_fraction`` of rows in production order.

    A random split would let the model see cycles from the same period it is tested on;
    the time-ordered holdout is closer to deployment, where the model always scores the
    future (Huyen 2022, ch. 5 on leakage from random splits of time-correlated data).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    ordered = df.sort_values(ORDER_KEY).reset_index(drop=True)
    cut = round(len(ordered) * (1.0 - test_fraction))
    return SplitData(
        dev=ordered.iloc[:cut].reset_index(drop=True),
        test=ordered.iloc[cut:].reset_index(drop=True),
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return model inputs X, target y and the failure-mode slice frame (never a feature)."""
    x = df[list(INPUT_COLUMNS)].copy()
    # Numeric inputs are float in the model signature: a sensor gap at inference time then
    # arrives as NaN instead of breaking an integer schema.
    numeric = [c for c in INPUT_COLUMNS if c != "product_type"]
    x[numeric] = x[numeric].astype("float64")
    y = df[TARGET].astype(int)
    modes = df[list(MODE_COLUMNS)].copy()
    return x, y, modes
