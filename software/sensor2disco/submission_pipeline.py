from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pandas as pd

from .config import resolve_config_path
from .ingestion import read_sensor_folder
from .mapping import build_mapped_event_log
from .quality import build_quality_summary, check_nearby_motion_file, check_raw_data
from .repository_room_abstraction import build_room_action_log, build_room_level_log


CONDITION_BY_SOURCE = {"normal": "normal", "error_trace": "scripted_error"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _prepare_activity_rows(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    output = frame.copy()
    prefix = "normal_" if condition == "normal" else "error_"
    output["Case ID"] = prefix + output["Case ID"].astype(str)
    output["Task ID"] = output["_Task ID"]
    output["Condition"] = condition
    output["Event Type"] = "activity"
    output["Source Data Type"] = output["Data Type"]
    return output


def _combine(normal: pd.DataFrame, error: pd.DataFrame) -> pd.DataFrame:
    output = pd.concat(
        [
            _prepare_activity_rows(normal, "normal"),
            _prepare_activity_rows(error, "scripted_error"),
        ],
        ignore_index=True,
        sort=False,
    )
    return output.sort_values(
        ["Condition", "Case ID", "Timestamp", "_Original Row"],
        kind="mergesort",
    ).reset_index(drop=True)


def _sensor_log(
    normal_raw: pd.DataFrame,
    error_raw: pd.DataFrame,
    normal_mapped: pd.DataFrame,
    error_mapped: pd.DataFrame,
) -> pd.DataFrame:
    normal = normal_mapped.copy()
    normal["Raw Activity"] = normal_raw["Activity"].to_numpy()
    normal["Mapped Activity"] = normal["Activity"]
    error = error_mapped.copy()
    error["Raw Activity"] = error_raw["Activity"].to_numpy()
    error["Mapped Activity"] = error["Activity"]
    return _combine(normal, error)


def _write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    available = [column for column in columns if column in frame.columns]
    output = frame.loc[:, available].copy()
    if "Timestamp" in output.columns:
        output["Timestamp"] = pd.to_datetime(output["Timestamp"]).dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )
    output.to_csv(path, index=False, encoding="utf-8-sig")


def _write_manifest(output_dir: Path) -> Path:
    path = output_dir / "verification" / "output_manifest.csv"
    rows = []
    for file in sorted(output_dir.rglob("*")):
        if not file.is_file() or file == path:
            continue
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
    normal_dir = resolve_config_path(config, str(config["normal_input_dir"]))
    error_dir = resolve_config_path(config, str(config["error_input_dir"]))
    source_root = Path(os.path.commonpath([str(normal_dir.parent), str(error_dir.parent)]))

    normal_raw = read_sensor_folder(normal_dir, "normal", config, source_root)
    error_raw = read_sensor_folder(error_dir, "error_trace", config, source_root)
    normal_mapped = build_mapped_event_log(normal_raw, config)
    error_mapped = build_mapped_event_log(error_raw, config)
    normal_room_action = build_room_action_log(normal_mapped, config)
    error_room_action = build_room_action_log(error_mapped, config)

    sensor = _sensor_log(normal_raw, error_raw, normal_mapped, error_mapped)
    room_action = _combine(normal_room_action, error_room_action)
    room_path = build_room_level_log(room_action, config)

    expected_rows = config["expected_event_rows"]  # type: ignore[assignment]
    observed = {
        "sensor": len(sensor),
        "room_action": len(room_action),
        "room_path": len(room_path),
    }
    if observed != {key: int(value) for key, value in expected_rows.items()}:
        raise ValueError(f"Event-log row counts differ: {observed}")
    if any(set(frame["Event Type"].astype(str)) != {"activity"} for frame in (sensor, room_action, room_path)):
        raise ValueError("Generated event logs must contain activity events only")
    if any(frame["Case ID"].nunique() != 220 for frame in (sensor, room_action, room_path)):
        raise ValueError("Each event log must contain all 220 cases")

    event_dir = output_dir / "event_logs"
    sensor_path = event_dir / "sensor_event_log.csv"
    room_action_path = event_dir / "room_action_event_log.csv"
    room_path_path = event_dir / "room_path_event_log.csv"
    _write_csv(
        sensor,
        sensor_path,
        ["Case ID", "Task ID", "Condition", "Timestamp", "Activity", "Event Type", "Sensor", "Message", "Raw Activity", "Mapped Activity", "Source File", "Source Data Type"],
    )
    _write_csv(
        room_action,
        room_action_path,
        ["Case ID", "Task ID", "Condition", "Timestamp", "Activity", "Event Type", "Sensor", "Message", "Source File", "Source Data Type"],
    )
    _write_csv(
        room_path,
        room_path_path,
        ["Case ID", "Task ID", "Condition", "Timestamp", "Activity", "Room", "Original Activity", "Event Type", "Sensor", "Message", "Source File", "Source Data Type"],
    )

    raw_check = check_raw_data(normal_dir, error_dir, config, source_root)
    nearby = pd.concat(
        [
            check_nearby_motion_file(normal_mapped, "normal", config),
            check_nearby_motion_file(error_mapped, "error_trace", config),
        ],
        ignore_index=True,
    )
    quality_dir = output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    raw_check.to_csv(quality_dir / "raw_data_check.csv", index=False, encoding="utf-8-sig")
    nearby.to_csv(quality_dir / "room_mapping_check.csv", index=False, encoding="utf-8-sig")
    build_quality_summary(raw_check, nearby).to_csv(
        quality_dir / "quality_summary.csv", index=False, encoding="utf-8-sig"
    )
    manifest = _write_manifest(output_dir)
    return {
        "output_dir": output_dir,
        "sensor_event_log": sensor_path,
        "room_action_event_log": room_action_path,
        "room_path_event_log": room_path_path,
        "manifest": manifest,
    }


def run_submission_pipeline(config: dict[str, object]) -> dict[str, Path]:
    output_dir = resolve_config_path(config, str(config["output_dir"]))
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.staging-{uuid.uuid4().hex}")
    try:
        result = _build(config, staging)
        staging.rename(output_dir)
    except Exception:
        raise
    return {
        key: output_dir / value.relative_to(staging) if value != staging else output_dir
        for key, value in result.items()
    }
