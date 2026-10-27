"""Train-only fitted feature baselines."""

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ecg_workbench.data.labels import CLASS_ORDER


def train_feature_models(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    if set(y) - set(CLASS_ORDER):
        raise ValueError("labels must be N/S/V")
    if X.ndim != 2 or len(X) != len(y) or not np.all(np.isfinite(X)):
        raise ValueError("training features must be finite and aligned")
    if set(y) != set(CLASS_ORDER):
        raise ValueError("training requires all N/S/V classes")
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
        ),
        "forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        ),
    }
    for model in models.values():
        model.fit(X, y)
    return models


def record_capped_class_weights(
    y: np.ndarray, record_ids: np.ndarray, cap: int = 100,
) -> np.ndarray:
    """Keep S weight balanced while limiting one source record's share."""
    if len(y) != len(record_ids) or cap < 1 or not np.any(y == "S"):
        raise ValueError("record weights require aligned S labels and a positive cap")
    weights = compute_sample_weight("balanced", y)
    capped = weights.copy()
    for record_id in np.unique(record_ids[y == "S"]):
        in_record = (record_ids == record_id) & (y == "S")
        capped[in_record] *= min(1.0, cap / int(in_record.sum()))
    capped[y == "S"] *= weights[y == "S"].sum() / capped[y == "S"].sum()
    return capped


def train_feature_models_v2(
    X: np.ndarray, y: np.ndarray, record_ids: np.ndarray, seed: int,
) -> dict:
    """Bounded comparison with aligned timing and weighted tree alternatives."""
    if (set(y) != set(CLASS_ORDER) or X.ndim != 2 or
            len(X) != len(y) or len(record_ids) != len(y)):
        raise ValueError("training requires aligned N/S/V features")
    if not np.all(np.isfinite(X)):
        raise ValueError("training features must be finite")
    models = {
        "forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        "histogram_boosting": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, learning_rate=0.08,
            random_state=seed,
        ),
        "histogram_boosting_recordcap": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, learning_rate=0.08,
            random_state=seed,
        ),
    }
    weights = compute_sample_weight("balanced", y)
    capped_weights = record_capped_class_weights(y, record_ids)
    for name, model in models.items():
        sample_weight = (
            capped_weights if name == "histogram_boosting_recordcap" else
            weights if name == "histogram_boosting" else None
        )
        model.fit(X, y, sample_weight=sample_weight)
    return models


def train_feature_models_v3(
    X: np.ndarray, y: np.ndarray, record_ids: np.ndarray, seed: int,
) -> dict:
    """Compare S record caps with a bounded tree control on offline context."""
    if (set(y) != set(CLASS_ORDER) or X.ndim != 2 or
            len(X) != len(y) or len(record_ids) != len(y) or
            not np.all(np.isfinite(X))):
        raise ValueError("training requires finite aligned N/S/V features")
    models = {
        f"histogram_cap{cap}": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, learning_rate=0.08,
            random_state=seed,
        )
        for cap in (100, 25, 10)
    }
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=200, min_samples_leaf=5,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    for name, model in models.items():
        cap = int(name.removeprefix("histogram_cap")) if name.startswith("histogram_cap") else None
        weights = record_capped_class_weights(y, record_ids, cap) if cap is not None else None
        model.fit(X, y, sample_weight=weights)
    return models
