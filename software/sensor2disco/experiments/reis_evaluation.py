from __future__ import annotations

import hashlib
import math
import os
import uuid
from pathlib import Path

import pandas as pd

from sensor2disco.config import load_config, resolve_config_path
from sensor2disco.ingestion import read_sensor_folder

from .features import build_case_sequences
from .protocol import build_fold_assignments
from .reis_isolation_forest import fit_predict_fold


METHOD_NAME = "Isolation Forest baseline"


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
    expected = dataset_config["expected_case_counts"]  # type: ignore[assignment]
    folds = int(config["outer_protocol"]["folds"])  # type: ignore[index]
    assignments = build_fold_assignments(
        expected["normal"]["participants"],  # type: ignore[index]
        expected["scripted_error"]["participants"],  # type: ignore[index]
        folds,
    )
    model_settings = config["isolation_forest"]  # type: ignore[assignment]
    elapsed_boundaries = tuple(float(value) for value in feature_config["elapsed_boundaries_seconds"])
    base_seed = int(config["random_seed"])
    rows = []
    diagnostics = []
    for fold in range(folds):
        test_participants = set(assignments[assignments["Fold"].eq(fold)]["Participant ID"].astype(str))
        train_sequences = [
            sequence
            for sequence in sequences.values()
            if sequence.data_type == "normal"
            and sequence.participant_id not in test_participants
        ]
        test_sequences = [
            sequence
            for sequence in sequences.values()
            if sequence.participant_id in test_participants
        ]
        predicted, diagnostic = fit_predict_fold(
            train_sequences,
            test_sequences,
            settings=model_settings,
            elapsed_boundaries=elapsed_boundaries,
            random_state=base_seed + fold,
        )
        diagnostic["Fold"] = fold
        diagnostics.append(diagnostic)
        by_case = {sequence.case_id: sequence for sequence in test_sequences}
        for _, row in predicted.iterrows():
            sequence = by_case[str(row["Case ID"])]
            condition = "normal" if sequence.data_type == "normal" else "scripted_error"
            record = row.to_dict()
            record.update(
                {
                    "Participant ID": sequence.participant_id,
                    "Condition": condition,
                    "Ground Truth": "normal" if condition == "normal" else "error",
                    "Fold": fold,
                    "Source File": sequence.source_file,
                    "Method": METHOD_NAME,
                    "Detection Label State": "masked_before_prediction",
                    "Ground Truth Attached Stage": "post_prediction",
                }
            )
            rows.append(record)

    predictions = pd.DataFrame(rows).sort_values("Case ID", kind="mergesort")
    if len(predictions) != 220 or predictions["Case ID"].nunique() != 220:
        raise ValueError("Isolation Forest evaluation must produce 220 unique predictions")
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
    pd.DataFrame(diagnostics).to_csv(
        output_dir / "fold_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    manifest = _write_manifest(output_dir)
    return {
        "output_dir": output_dir,
        "predictions": output_dir / "case_predictions.csv",
        "metrics": output_dir / "metrics_summary.csv",
        "fold_assignments": output_dir / "fold_assignments.csv",
        "manifest": manifest,
    }


def run_reis_evaluation(config: dict[str, object]) -> dict[str, Path]:
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
