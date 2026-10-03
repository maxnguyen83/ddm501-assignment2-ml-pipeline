"""Stage 5 - training: model factory, imbalance handling and cross-validated scoring.

Every candidate is a single sklearn ``Pipeline`` (features -> encoding -> estimator) so the
artifact logged to MLflow is self-contained: it takes the six raw sensor/product inputs and
returns a failure probability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from pm_pipeline.config import ExperimentSpec
from pm_pipeline.features import PhysicsFeatures, build_column_transformer

RESAMPLING_KEYS = ("target_ratio",)


class ToolWearRule(ClassifierMixin, BaseEstimator):
    """Rule-based baseline that mirrors current practice: the more worn the tool, the riskier.

    The score is the raw feature value scaled to [0, 1] by the training maximum. With a fixed
    alert budget this is "inspect the machines with the most worn tools first".
    """

    def __init__(self, feature: str = "tool_wear_min") -> None:
        self.feature = feature

    def fit(self, x: pd.DataFrame, y: pd.Series) -> ToolWearRule:
        """Remember the training maximum used to scale scores."""
        self.max_: float = float(max(x[self.feature].max(), 1.0))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return [1 - score, score] where score = feature / training max, clipped to [0, 1]."""
        score = np.clip(x[self.feature].to_numpy(dtype=float) / self.max_, 0.0, 1.0)
        return np.column_stack([1.0 - score, score])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Hard prediction at score >= 0.5 (only used by sklearn utilities)."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def build_estimator(spec: ExperimentSpec, seed: int) -> Any:
    """Instantiate the estimator for ``spec`` with its hyperparameters and seed."""
    params = {k: v for k, v in spec.params.items() if k not in RESAMPLING_KEYS}
    balanced = spec.imbalance == "class_weight"
    if spec.model == "rule":
        return ToolWearRule(**params)
    if spec.model == "logreg":
        return LogisticRegression(
            max_iter=2000,
            random_state=seed,
            class_weight="balanced" if balanced else None,
            **params,
        )
    if spec.model == "random_forest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample" if balanced else None,
            **params,
        )
    if spec.model == "hist_gb":
        return HistGradientBoostingClassifier(
            random_state=seed, class_weight="balanced" if balanced else None, **params
        )
    if spec.model == "lightgbm":
        return LGBMClassifier(
            random_state=seed,
            verbose=-1,
            n_jobs=1,
            class_weight="balanced" if balanced else None,
            **params,
        )
    raise ValueError(f"unknown model {spec.model!r}")


def build_model(spec: ExperimentSpec, seed: int) -> Pipeline:
    """Assemble features -> column transformer -> estimator into one sklearn Pipeline."""
    steps: list[tuple[str, Any]] = []
    if spec.feature_set == "physics":
        steps.append(("physics", PhysicsFeatures()))
    if spec.model != "rule":  # the rule reads the raw column directly
        steps.append(
            ("encode", build_column_transformer(spec.feature_set, scale=spec.model == "logreg"))
        )
    steps.append(("model", build_estimator(spec, seed)))
    return Pipeline(steps)


def resample(
    x: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    target_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Random under- or over-sampling to reach ``minority / majority = target_ratio``.

    Applied to training folds only - never to validation or test data, otherwise the
    measured precision and alert rate would not reflect the real failure rate.
    """
    if strategy not in ("undersample", "oversample"):
        return x, y
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y.to_numpy() == 1)
    neg_idx = np.flatnonzero(y.to_numpy() == 0)
    if strategy == "undersample":
        n_neg = min(len(neg_idx), round(len(pos_idx) / target_ratio))
        neg_idx = rng.choice(neg_idx, size=n_neg, replace=False)
    else:
        n_pos = max(len(pos_idx), round(len(neg_idx) * target_ratio))
        pos_idx = rng.choice(pos_idx, size=n_pos, replace=True)
    idx = rng.permutation(np.concatenate([pos_idx, neg_idx]))
    return x.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def fit_model(spec: ExperimentSpec, x: pd.DataFrame, y: pd.Series, seed: int) -> Pipeline:
    """Resample (if configured) and fit a fresh pipeline for ``spec``."""
    ratio = float(spec.params.get("target_ratio", 0.33))
    x_fit, y_fit = resample(x, y, spec.imbalance, ratio, seed)
    model = build_model(spec, seed)
    model.fit(x_fit, y_fit)
    return model


def cross_val_scores(
    spec: ExperimentSpec,
    x: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold failure probabilities from stratified K-fold CV.

    Returns:
        (oof_scores, fold_id) - one score per development row and the fold it was scored in.

    """
    oof = np.zeros(len(y), dtype=float)
    fold_id = np.full(len(y), -1, dtype=int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        model = fit_model(spec, x.iloc[train_idx], y.iloc[train_idx], seed + k)
        oof[val_idx] = model.predict_proba(x.iloc[val_idx])[:, 1]
        fold_id[val_idx] = k
    return oof, fold_id
