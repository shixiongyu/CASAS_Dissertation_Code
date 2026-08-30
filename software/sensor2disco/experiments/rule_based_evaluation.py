from __future__ import annotations

import hashlib
import math
import os
import uuid
from pathlib import Path

import pandas as pd

from sensor2disco.config import load_config, resolve_config_path
from sensor2disco.ingestion import read_sensor_folder

from .protocol import build_fold_assignments, participant_from_case_id
from .rule_based_detector import calibrate_rule_profile, detect_rule_based_cases


METHOD_NAME = "Proposed rule-based method"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def _binary_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["Ground Truth"].astype(str)
    prediction = frame["Prediction"].astype(str)
    is_error = truth.eq("error")
    is_normal = truth.eq("normal")
    predicted_error = prediction.eq("error")
    predicted_normal = prediction.eq("normal")
    unknown = prediction.eq("unknown")
    tp = int((is_error & predicted_error).sum())
    fp = int((is_normal & predicted_error).sum())
    tn = int((is_normal & predicted_normal).sum())
    fn = int((is_error & ~predicted_error).sum())
    unknown_error = int((is_error & unknown).sum())
    unknown_normal = int((is_normal & unknown).sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp + unknown_normal)
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
        "Insufficient Error": unknown_error,
        "Insufficient Normal": unknown_normal,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "Evidence Coverage": _ratio(int((~unknown).sum()), len(frame)),
    }


def _metadata(events: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    frame = (
        events.groupby("Case ID", sort=True)
        .agg({"Data Type": "first", "_Task ID": "first", "Source File": "first"})
        .reset_index()
    )
    frame["Participant ID"] = frame["Case ID"].map(participant_from_case_id)
    frame["Condition"] = frame["Data Type"].map(
        {"normal": "normal", "error_trace": "scripted_error"}
    )
    frame["Ground Truth"] = frame["Data Type"].map(
        {"normal": "normal", "error_trace": "error"}
    )
    frame = frame.merge(
        assignments,
        on=["Condition", "Participant ID"],
        how="left",
        validate="many_to_one",
    )
    if frame["Fold"].isna().any():
        raise ValueError("Some cases have no participant-grouped fold")
    frame["Fold"] = frame["Fold"].astype(int)
    return frame.rename(columns={"_Task ID": "Task ID"})


def _predict_fold(
    events: pd.DataFrame,
    truth: pd.DataFrame,
    dataset_config: dict[str, object],
    profile: dict[str, float],
    fold: int,
) -> pd.DataFrame:
    masked = events.copy()
    masked["Data Type"] = "masked_before_prediction"
    masked["Source File"] = "masked_before_prediction"
    detected = detect_rule_based_cases(
        masked,
        dataset_config,
        profile,
        profile_label="primary",
        rule_version="rule-based-error-detection",
    )
    detected = detected.merge(
        truth[
            [
                "Case ID",
                "Task ID",
                "Participant ID",
                "Condition",
                "Ground Truth",
                "Source File",
            ]
        ],
        on="Case ID",
        how="left",
        suffixes=("", " Truth"),
        validate="one_to_one",
    )
    if not detected["Task ID"].astype(str).eq(
        detected["Task ID Truth"].astype(str)
    ).all():
        raise ValueError("Task identity changed during prediction")
    detected["Fold"] = fold
    detected["Method"] = METHOD_NAME
    detected["Prediction"] = detected["Rule Outcome"].map(
        {"error": "error", "no_error": "normal", "insufficient_evidence": "unknown"}
    )
    detected["Data Type"] = detected["Condition"].map(
        {"normal": "normal", "scripted_error": "error_trace"}
    )
    detected["Source File"] = detected["Source File Truth"]
    return detected.drop(columns=["Task ID Truth", "Source File Truth"])


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_id, group in list(predictions.groupby("Task ID", sort=True)) + [
        ("overall", predictions)
    ]:
        rows.append({"Method": METHOD_NAME, "Task ID": task_id, **_binary_metrics(group)})
    return pd.DataFrame(rows)


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

    expected = dataset_config["expected_case_counts"]  # type: ignore[assignment]
    folds = int(config["outer_protocol"]["folds"])  # type: ignore[index]
    assignments = build_fold_assignments(
        expected["normal"]["participants"],  # type: ignore[index]
        expected["scripted_error"]["participants"],  # type: ignore[index]
        folds,
    )
    metadata = _metadata(events, assignments)
    prediction_frames = []
    threshold_frames = []
    minimum_cases = int(config["calibration"]["minimum_normal_evidence_cases"])  # type: ignore[index]
    for fold in range(folds):
        normal_train_participants = set(
            assignments[
                assignments["Condition"].eq("normal")
                & assignments["Fold"].ne(fold)
            ]["Participant ID"].astype(str)
        )
        event_participants = events["Case ID"].map(participant_from_case_id)
        train = events[
            events["Data Type"].eq("normal")
            & event_participants.isin(normal_train_participants)
        ].copy()
        profile, thresholds, _, _ = calibrate_rule_profile(
            train,
            dataset_config,
            fold=fold,
            minimum_normal_evidence_cases=minimum_cases,
        )
        threshold_frames.append(thresholds)
        test_cases = set(metadata[metadata["Fold"].eq(fold)]["Case ID"].astype(str))
        test_events = events[events["Case ID"].astype(str).isin(test_cases)].copy()
        prediction_frames.append(
            _predict_fold(
                test_events,
                metadata[metadata["Fold"].eq(fold)],
                dataset_config,
                profile,
                fold,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        "Case ID", kind="mergesort"
    )
    if len(predictions) != 220 or predictions["Case ID"].nunique() != 220:
        raise ValueError("Rule-based evaluation must produce 220 unique predictions")
    metrics = _metrics(predictions)

    output_dir.mkdir(parents=True, exist_ok=False)
    assignments.to_csv(output_dir / "fold_assignments.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / "case_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(threshold_frames, ignore_index=True).to_csv(
        output_dir / "fold_thresholds.csv", index=False, encoding="utf-8-sig"
    )
    manifest = _write_manifest(output_dir)
    return {
        "output_dir": output_dir,
        "predictions": output_dir / "case_predictions.csv",
        "metrics": output_dir / "metrics_summary.csv",
        "fold_assignments": output_dir / "fold_assignments.csv",
        "manifest": manifest,
    }


def run_rule_based_evaluation(config: dict[str, object]) -> dict[str, Path]:
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
