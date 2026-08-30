from __future__ import annotations

import hashlib
import math
import os
import statistics
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sensor2disco.config import load_config, resolve_config_path
from sensor2disco.ingestion import read_sensor_folder

from .features import CaseSequence, build_case_sequences
from .protocol import build_fold_assignments, participant_from_case_id


METHOD_NAME = "Cook-based skipped-step detector"


def _sequence_features(sequence: CaseSequence) -> dict[str, int]:
    features = Counter(f"token:{token}" for token in sequence.tokens)
    previous = "<START>"
    for token in sequence.tokens:
        features[f"transition:{previous}->{token}"] += 1
        previous = token
    features[f"transition:{previous}-><END>"] += 1
    return dict(features)


@dataclass(frozen=True)
class Decision:
    is_error: bool
    reasons: tuple[str, ...]


class FrequencyProfile:
    def __init__(self, *, standard_deviations: float, minimum_prevalence: float) -> None:
        self.standard_deviations = float(standard_deviations)
        self.minimum_prevalence = float(minimum_prevalence)
        self.bounds: dict[str, tuple[float, float]] = {}
        self.known_features: set[str] = set()

    def fit(self, sequences: list[CaseSequence]) -> "FrequencyProfile":
        if not sequences:
            raise ValueError("At least one normal training sequence is required")
        rows = [_sequence_features(sequence) for sequence in sequences]
        self.known_features = {feature for row in rows for feature in row}
        bounds = {}
        for feature in sorted(self.known_features):
            values = [float(row.get(feature, 0)) for row in rows]
            prevalence = sum(value > 0 for value in values) / len(values)
            if prevalence < self.minimum_prevalence:
                continue
            mean = statistics.fmean(values)
            sigma = statistics.pstdev(values) if len(values) >= 2 else 0.0
            bounds[feature] = (
                max(0.0, mean - self.standard_deviations * sigma),
                mean + self.standard_deviations * sigma,
            )
        self.bounds = bounds
        return self

    def decide(self, sequence: CaseSequence) -> Decision:
        observed = _sequence_features(sequence)
        reasons = []
        for feature, (lower, upper) in sorted(self.bounds.items()):
            value = float(observed.get(feature, 0))
            if value + 1e-9 < lower:
                reasons.append(f"{feature}={value:g} below {lower:.3f}")
            elif value - 1e-9 > upper:
                reasons.append(f"{feature}={value:g} above {upper:.3f}")
        for feature in sorted(set(observed).difference(self.known_features)):
            reasons.append(f"novel {feature}")
        return Decision(bool(reasons), tuple(reasons[:8]))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["Ground Truth"].astype(str)
    prediction = frame["Prediction"].astype(str)
    tp = int((truth.eq("error") & prediction.eq("error")).sum())
    fp = int((truth.eq("normal") & prediction.eq("error")).sum())
    tn = int((truth.eq("normal") & prediction.eq("normal")).sum())
    fn = int((truth.eq("error") & prediction.eq("normal")).sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "Cases": len(frame),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "Evidence Coverage": 1.0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_manifest(output_dir: Path) -> Path:
    path = output_dir / "verification" / "output_manifest.csv"
    rows = []
    for file in sorted(output_dir.rglob("*")):
        if file.is_file() and file != path:
            rows.append(
                {
                    "Path": file.relative_to(output_dir).as_posix(),
                    "Bytes": file.stat().st_size,
                    "SHA256": _sha256(file),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _build(config: dict[str, object], output_dir: Path) -> dict[str, Path]:
    config_path = Path(str(config["_config_path"]))
    dataset_config = load_config((config_path.parent / str(config["dataset_config"])).resolve())
    normal_dir = resolve_config_path(dataset_config, str(dataset_config["normal_input_dir"]))
    error_dir = resolve_config_path(dataset_config, str(dataset_config["error_input_dir"]))
    source_root = Path(os.path.commonpath([str(normal_dir.parent), str(error_dir.parent)]))
    normal = read_sensor_folder(normal_dir, "normal", dataset_config, source_root)
    error = read_sensor_folder(error_dir, "error_trace", dataset_config, source_root)
    events = pd.concat([normal, error], ignore_index=True).sort_values(
        ["Case ID", "Timestamp", "_Original Row"], kind="mergesort"
    )
    feature_config = config["feature_contract"]  # type: ignore[assignment]
    sequences = build_case_sequences(
        events,
        dataset_config,
        analog_delta_threshold=float(feature_config["analog_delta_threshold"]),
    )
    if len(sequences) != 220:
        raise ValueError("Expected 220 task sequences")

    expected = dataset_config["expected_case_counts"]  # type: ignore[assignment]
    folds = int(config["outer_protocol"]["folds"])  # type: ignore[index]
    assignments = build_fold_assignments(
        expected["normal"]["participants"],  # type: ignore[index]
        expected["scripted_error"]["participants"],  # type: ignore[index]
        folds,
    )
    settings = config["detector"]  # type: ignore[assignment]
    rows = []
    for fold in range(folds):
        test_participants = set(assignments[assignments["Fold"].eq(fold)]["Participant ID"].astype(str))
        training_normal = [
            sequence
            for sequence in sequences.values()
            if sequence.data_type == "normal"
            and sequence.participant_id not in test_participants
        ]
        profiles = {
            task_id: FrequencyProfile(
                standard_deviations=float(settings["standard_deviations"]),
                minimum_prevalence=float(settings["minimum_prevalence"]),
            ).fit([sequence for sequence in training_normal if sequence.task_id == task_id])
            for task_id in ("t1", "t2", "t3", "t4", "t5")
        }
        for sequence in sequences.values():
            if sequence.participant_id not in test_participants:
                continue
            decision = profiles[sequence.task_id].decide(sequence)
            condition = "normal" if sequence.data_type == "normal" else "scripted_error"
            rows.append(
                {
                    "Case ID": sequence.case_id,
                    "Participant ID": sequence.participant_id,
                    "Task ID": sequence.task_id,
                    "Condition": condition,
                    "Ground Truth": "normal" if condition == "normal" else "error",
                    "Fold": fold,
                    "Source File": sequence.source_file,
                    "Method": METHOD_NAME,
                    "Prediction": "error" if decision.is_error else "normal",
                    "Evidence": "; ".join(decision.reasons) or "within learned frequency bounds",
                    "Training Data": "normal_outer_training_participants_only",
                }
            )

    predictions = pd.DataFrame(rows).sort_values("Case ID", kind="mergesort")
    if len(predictions) != 220 or predictions["Case ID"].nunique() != 220:
        raise ValueError("Cook-based evaluation must produce 220 unique predictions")
    metric_rows = []
    for task_id, group in list(predictions.groupby("Task ID", sort=True)) + [
        ("overall", predictions)
    ]:
        metric_rows.append({"Method": METHOD_NAME, "Task ID": task_id, **_metrics(group)})
    metrics = pd.DataFrame(metric_rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    assignments.to_csv(output_dir / "fold_assignments.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / "case_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    manifest = _write_manifest(output_dir)
    return {
        "output_dir": output_dir,
        "predictions": output_dir / "case_predictions.csv",
        "metrics": output_dir / "metrics_summary.csv",
        "fold_assignments": output_dir / "fold_assignments.csv",
        "manifest": manifest,
    }


def run_cook_evaluation(config: dict[str, object]) -> dict[str, Path]:
    output_dir = resolve_config_path(config, str(config["output_dir"]))
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.staging-{uuid.uuid4().hex}")
    result = _build(config, staging)
    staging.rename(output_dir)
    return {
        key: output_dir / value.relative_to(staging) if value != staging else output_dir
        for key, value in result.items()
    }
