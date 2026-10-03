from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from pm_pipeline.config import PipelineConfig
from pm_pipeline.features import add_physics_features, check_features
from pm_pipeline.ingest import IngestError, ingest, load_raw
from pm_pipeline.preprocess import INPUT_COLUMNS, clean, split_xy, temporal_split
from pm_pipeline.validate import DataValidationError, assert_valid, validate_raw


def test_ingest_verifies_pinned_hash(cfg: PipelineConfig) -> None:
    result = ingest(cfg.data, allow_download=False)
    assert result.sha256 == cfg.data.expected_sha256


def test_ingest_rejects_modified_file(cfg: PipelineConfig, tmp_path) -> None:
    fake = tmp_path / "ai4i2020.csv"
    fake.write_text("UDI\n1\n")
    data_cfg = replace(cfg.data, raw_path=fake)
    with pytest.raises(IngestError):
        ingest(data_cfg, allow_download=False)


def test_real_data_passes_validation_with_known_warnings(cfg: PipelineConfig) -> None:
    report = validate_raw(load_raw(cfg.data.raw_path), cfg.validation)
    assert report.passed, report.errors
    assert report.stats["n_failures"] == 339
    assert report.stats["fail_without_mode"] == 9
    assert len(report.warnings) == 2


def test_validation_catches_missing_column(cfg: PipelineConfig, raw_df: pd.DataFrame) -> None:
    report = validate_raw(raw_df.drop(columns=["torque_nm"]), cfg.validation)
    assert not report.passed
    with pytest.raises(DataValidationError):
        assert_valid(report)


def test_validation_catches_out_of_range_and_bad_type(
    cfg: PipelineConfig, raw_df: pd.DataFrame
) -> None:
    bad = raw_df.copy()
    bad.loc[0, "air_temp_k"] = 3000.0  # Celsius/Kelvin mix-up
    bad.loc[1, "product_type"] = "X"
    report = validate_raw(bad, cfg.validation)
    joined = " ".join(report.errors)
    assert "air_temp_k" in joined and "product_type" in joined


def test_preprocess_drops_leakage_and_ids(raw_df: pd.DataFrame) -> None:
    x, y, modes = split_xy(clean(raw_df))
    assert list(x.columns) == list(INPUT_COLUMNS)
    assert not set(x.columns) & {"twf", "hdf", "pwf", "osf", "rnf", "product_id", "udi"}
    assert set(modes.columns) == {"twf", "hdf", "pwf", "osf", "rnf"}
    assert y.isin([0, 1]).all()


def test_temporal_split_keeps_order_and_has_no_overlap(raw_df: pd.DataFrame) -> None:
    shuffled = raw_df.sample(frac=1.0, random_state=1)
    split = temporal_split(clean(shuffled), test_fraction=0.25)
    assert len(split.test) == 100
    assert split.dev["udi"].max() < split.test["udi"].min()
    assert set(split.dev["udi"]).isdisjoint(split.test["udi"])


def test_physics_features_match_hand_calculation() -> None:
    row = pd.DataFrame(
        {
            "air_temp_k": [300.0],
            "process_temp_k": [309.5],
            "rotational_speed_rpm": [1500.0],
            "torque_nm": [40.0],
            "tool_wear_min": [100.0],
        }
    )
    out = add_physics_features(row).iloc[0]
    assert out["temp_diff_k"] == pytest.approx(9.5)
    assert out["power_w"] == pytest.approx(40.0 * 1500.0 * 2 * math.pi / 60)  # ~6283 W
    assert out["strain_nm_min"] == pytest.approx(4000.0)
    assert check_features(add_physics_features(row)) == []
