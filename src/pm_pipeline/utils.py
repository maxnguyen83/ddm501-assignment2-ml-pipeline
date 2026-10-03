"""Small helpers shared by all stages: seeding, hashing, provenance and JSON output."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "pm_pipeline"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a module logger with a single, timestamped stream handler."""
    logger = logging.getLogger(name)
    if not logging.getLogger(LOGGER_NAME).handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger(LOGGER_NAME)
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False  # MLflow/Airflow configure the root logger; avoid duplicates
    return logger


def set_global_seed(seed: int) -> None:
    """Seed every random source the pipeline touches.

    Estimators also receive ``random_state=seed`` explicitly; this function covers the
    global generators (resampling, bootstrap) and hash randomisation for subprocesses.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks so large files are fine."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(default: str = "unknown") -> str:
    """Return the current git commit SHA (or ``GIT_COMMIT`` env var inside containers)."""
    env_sha = os.environ.get("GIT_COMMIT")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return default
    return out.stdout.strip() or default


def write_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as pretty JSON, creating parent folders; returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out
