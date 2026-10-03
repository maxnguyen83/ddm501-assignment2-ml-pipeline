"""Stage 1 - ingestion: make sure the raw CSV is present and is the expected version.

In the plant this stage would pull one week of cycle records from the historian/MES export.
For the public AI4I 2020 data it downloads the UCI archive once and pins it by sha256, so a
silently changed file can never reach training.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pm_pipeline.config import DataConfig
from pm_pipeline.utils import get_logger, sha256_file

log = get_logger(__name__)

CSV_NAME_IN_ZIP = "ai4i2020.csv"

# Source column -> internal snake_case name. Units stay in the name to avoid unit bugs.
COLUMN_MAP: dict[str, str] = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "product_type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


class IngestError(RuntimeError):
    """Raised when the raw file cannot be obtained or does not match the pinned hash."""


@dataclass(frozen=True)
class IngestResult:
    """Location and fingerprint of the ingested raw file."""

    path: Path
    sha256: str
    n_bytes: int


def download_dataset(url: str, dest: Path, timeout: int = 60) -> Path:
    """Download the UCI zip archive and extract the CSV to ``dest``."""
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        data = archive.read(CSV_NAME_IN_ZIP)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def ingest(cfg: DataConfig, allow_download: bool = True) -> IngestResult:
    """Ensure the raw CSV exists locally and matches ``cfg.expected_sha256``.

    Args:
        cfg: data section of the pipeline config.
        allow_download: fetch from ``cfg.source_url`` when the file is missing.

    Returns:
        IngestResult with path, sha256 and size - the sha256 is logged to every MLflow run
        so each model can be traced back to the exact bytes it was trained on.

    Raises:
        IngestError: file missing (and download disabled) or hash mismatch.

    """
    path = cfg.raw_path
    if not path.exists():
        if not allow_download:
            raise IngestError(f"raw file not found: {path}")
        download_dataset(cfg.source_url, path)
    digest = sha256_file(path)
    if digest != cfg.expected_sha256:
        raise IngestError(
            f"sha256 mismatch for {path}: got {digest}, expected {cfg.expected_sha256}. "
            "If the source was intentionally updated, review it and update the config."
        )
    log.info("ingested %s (sha256 %s...)", path.name, digest[:12])
    return IngestResult(path=path, sha256=digest, n_bytes=path.stat().st_size)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV (UTF-8 with BOM) and rename columns to snake_case."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    unknown = set(df.columns) - set(COLUMN_MAP)
    if unknown:
        log.warning("unexpected columns ignored: %s", sorted(unknown))
    renamed = df.rename(columns=COLUMN_MAP)
    return renamed[[c for c in COLUMN_MAP.values() if c in renamed.columns]]
