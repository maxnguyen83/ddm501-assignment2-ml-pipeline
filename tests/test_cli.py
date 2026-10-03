from __future__ import annotations

import json
from pathlib import Path

import pytest

from pm_pipeline.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PM_TRACKING__TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("PM_TRACKING__ARTIFACT_LOCATION", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PM_DATA__WORK_DIR", str(tmp_path / "interim"))
    return tmp_path


def test_cli_validate_writes_report(isolated_env: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["--config", str(ROOT / "configs" / "pipeline.yaml"), "validate"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_failures"] == 339
    assert (isolated_env / "interim" / "cli" / "validation_report.json").exists()


def test_cli_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit):
        main(["deploy-to-prod"])
