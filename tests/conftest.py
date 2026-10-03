from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pm_pipeline.config import PipelineConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def make_raw(n: int = 400, seed: int = 0, failure_rate: float = 0.05) -> pd.DataFrame:
    """Synthetic frame with the AI4I schema (after load_raw) and physically valid values."""
    rng = np.random.default_rng(seed)
    air = rng.normal(300.0, 2.0, n).round(1)
    fail = (rng.random(n) < failure_rate).astype(int)
    df = pd.DataFrame(
        {
            "udi": np.arange(1, n + 1),
            "product_id": [f"L{50000 + i}" for i in range(n)],
            "product_type": rng.choice(["L", "M", "H"], n, p=[0.6, 0.3, 0.1]),
            "air_temp_k": air,
            "process_temp_k": (air + rng.normal(10.0, 1.0, n)).round(1),
            "rotational_speed_rpm": rng.integers(1200, 2800, n),
            "torque_nm": rng.normal(40.0, 10.0, n).clip(5, 75).round(1),
            "tool_wear_min": rng.integers(0, 250, n),
            "machine_failure": fail,
        }
    )
    for mode in ("twf", "hdf", "pwf", "osf", "rnf"):
        df[mode] = 0
    df.loc[df["machine_failure"] == 1, "osf"] = 1
    return df


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture()
def cfg(tmp_path: Path) -> PipelineConfig:
    """Real config, but MLflow and work files redirected to a temp folder."""
    env = {
        "PM_TRACKING__TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "PM_TRACKING__ARTIFACT_LOCATION": str(tmp_path / "artifacts"),
        "PM_DATA__WORK_DIR": str(tmp_path / "interim"),
        "PM_EVALUATION__BOOTSTRAP_ROUNDS": "20",
        "PM_SPLIT__CV_FOLDS": "3",
    }
    return load_config(ROOT / "configs" / "pipeline.yaml", environ=env)
