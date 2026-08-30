from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler

from .features import CaseSequence, aggregate_features


METHOD_NAME = "Isolation Forest baseline"
FIXED_MODEL_CONTRACT: dict[str, object] = {
    "n_estimators": 100,
    "contamination": 0.03,
    "max_samples": "auto",
    "max_features": 1.0,
    "bootstrap": False,
    "n_jobs": 1,
}


def validate_model_contract(settings: dict[str, object]) -> None:
    """Reject parameter drift from the pre-specified experiment."""

    for key, expected in FIXED_MODEL_CONTRACT.items():
        observed = settings.get(key)
        if observed != expected:
            raise ValueError(
                f"Isolation Forest {key} is fixed at {expected!r}; found {observed!r}"
            )


def build_case_feature_dict(
    sequence: CaseSequence,
    *,
    elapsed_boundaries: tuple[float, ...],
) -> dict[str, float | int]:
    """Build the registered case features without outcome metadata."""

    features: dict[str, float | int] = dict(
        aggregate_features(sequence, elapsed_boundaries=elapsed_boundaries)
    )
    features[f"task:{sequence.task_id}"] = 1
    return features


def fit_predict_fold(
    train_sequences: Sequence[CaseSequence],
    test_sequences: Sequence[CaseSequence],
    *,
    settings: dict[str, object],
    elapsed_boundaries: tuple[float, ...],
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit on outer-fold normal cases and predict the held-out fold."""

    validate_model_contract(settings)
    if not train_sequences or not test_sequences:
        raise ValueError("An Isolation Forest fold requires non-empty training and test cases")
    if any(sequence.data_type != "normal" for sequence in train_sequences):
        raise ValueError("Isolation Forest training accepts normal cases only")

    train_features = [
        build_case_feature_dict(sequence, elapsed_boundaries=elapsed_boundaries)
        for sequence in train_sequences
    ]
    test_features = [
        build_case_feature_dict(sequence, elapsed_boundaries=elapsed_boundaries)
        for sequence in test_sequences
    ]
    vectorizer = DictVectorizer(sparse=False, sort=True)
    train_matrix = vectorizer.fit_transform(train_features)
    test_matrix = vectorizer.transform(test_features)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    test_scaled = scaler.transform(test_matrix)

    model = IsolationForest(
        n_estimators=int(settings["n_estimators"]),
        contamination=float(settings["contamination"]),
        max_samples=settings["max_samples"],
        max_features=float(settings["max_features"]),
        bootstrap=bool(settings["bootstrap"]),
        n_jobs=int(settings["n_jobs"]),
        random_state=int(random_state),
    )
    model.fit(train_scaled)
    raw_predictions = model.predict(test_scaled)
    score_samples = model.score_samples(test_scaled)
    decision_function = model.decision_function(test_scaled)
    vocabulary = set(vectorizer.vocabulary_)

    rows: list[dict[str, object]] = []
    for sequence, feature_dict, raw, score, decision in zip(
        test_sequences,
        test_features,
        raw_predictions,
        score_samples,
        decision_function,
        strict=True,
    ):
        novel = sorted(set(feature_dict).difference(vocabulary))
        rows.append(
            {
                "Case ID": sequence.case_id,
                "Task ID": sequence.task_id,
                "Prediction": "error" if int(raw) == -1 else "normal",
                "Isolation Forest Prediction": int(raw),
                "Score Samples": float(score),
                "Decision Function": float(decision),
                "Novel Feature Count": len(novel),
                "Novel Features": "; ".join(novel),
            }
        )

    train_raw = model.predict(train_scaled)
    diagnostics: dict[str, object] = {
        "Training Cases": len(train_sequences),
        "Training Error Cases": 0,
        "Test Cases": len(test_sequences),
        "Feature Count": int(train_scaled.shape[1]),
        "Training Predicted Anomalies": int(np.sum(train_raw == -1)),
        "N Estimators": int(settings["n_estimators"]),
        "Contamination": float(settings["contamination"]),
        "Max Samples": settings["max_samples"],
        "Max Features": float(settings["max_features"]),
        "Bootstrap": bool(settings["bootstrap"]),
        "N Jobs": int(settings["n_jobs"]),
        "Random State": int(random_state),
        "Scikit-learn Version": sklearn.__version__,
    }
    return pd.DataFrame(rows), diagnostics


def runtime_versions() -> dict[str, str]:
    import scipy

    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
