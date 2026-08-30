from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .mapping import MOTION_RE
from .repository_room_abstraction import action_sensor_room, motion_sensor_room


CASE_FILE_RE = re.compile(r"^(p\d+)\.(t\d+)\.csv$", re.IGNORECASE)


def raw_files(normal_dir: Path, error_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    files.extend(("normal", path) for path in sorted(normal_dir.glob("*.csv")))
    files.extend(("error", path) for path in sorted(error_dir.glob("*.csv")))
    return files


def check_raw_data(
    normal_dir: Path, error_dir: Path, config: dict[str, object], root_dir: Path
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required = set(config["required_columns"])
    defined_sensors = {
        "D01",
        "AD1-A",
        "AD1-B",
        "AD1-C",
        "ASTERISK",
        *{f"M{i:02d}" for i in range(1, 27)},
        *{f"I{i:02d}" for i in range(1, 9)},
    }

    for data_type, path in raw_files(normal_dir, error_dir):
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        relative = str(path.relative_to(root_dir)).replace("\\", "/")
        missing = sorted(required.difference(raw.columns))

        rows.append(
            {
                "Data Type": data_type,
                "Source File": relative,
                "Check": "file name pattern",
                "Status": "Pass" if CASE_FILE_RE.match(path.name) else "Warning",
                "Details": "Matches pXX.tX.csv" if CASE_FILE_RE.match(path.name) else "Does not match pXX.tX.csv",
            }
        )
        rows.append(
            {
                "Data Type": data_type,
                "Source File": relative,
                "Check": "required columns",
                "Status": "Pass" if not missing else "Fail",
                "Details": "date/time/sensor/message present" if not missing else "Missing: " + ", ".join(missing),
            }
        )
        rows.append(
            {
                "Data Type": data_type,
                "Source File": relative,
                "Check": "empty file",
                "Status": "Pass" if len(raw) > 0 else "Fail",
                "Details": f"{len(raw)} sensor event rows",
            }
        )

        if missing:
            continue
        for row_number, sensor in enumerate(raw["sensor"], start=2):
            sensor_text = str(sensor).strip().upper()
            if sensor_text not in defined_sensors:
                rows.append(
                    {
                        "Data Type": data_type,
                        "Source File": relative,
                        "Check": "undefined sensor",
                        "Status": "Warning",
                        "Details": f"Line {row_number}: {sensor_text} is not defined by the dataset sensor description",
                    }
                )

    return pd.DataFrame(rows)


def expected_room_for_sensor(
    sensor: object,
    config: dict[str, object],
) -> str | None:
    return action_sensor_room(sensor, config)


def nearest_motion(case_df: pd.DataFrame, index: int, direction: int, config: dict[str, object]) -> pd.Series | None:
    indices = range(index - 1, -1, -1) if direction < 0 else range(index + 1, len(case_df))
    for candidate in indices:
        event = case_df.iloc[candidate]
        sensor_text = str(event["Sensor"]).strip().upper()
        message_text = str(event["Message"]).strip().upper()
        if MOTION_RE.match(sensor_text) and message_text == "ON":
            return event
    return None


def check_nearby_motion_file(
    mapped_log: pd.DataFrame, data_type: str, config: dict[str, object]
) -> pd.DataFrame:
    mapped = mapped_log.copy()
    mapped["TimestampParsed"] = pd.to_datetime(mapped["Timestamp"], errors="coerce")
    mapped = mapped.sort_values(["Case ID", "TimestampParsed"], kind="mergesort")
    rows: list[dict[str, object]] = []

    for case_id, case_df in mapped.groupby("Case ID", sort=True):
        case_df = case_df.reset_index(drop=True)
        for index, event in case_df.iterrows():
            activity = str(event["Activity"]).strip()
            expected = expected_room_for_sensor(event["Sensor"], config)
            if expected is None:
                continue

            before = nearest_motion(case_df, index, -1, config)
            after = nearest_motion(case_df, index, 1, config)
            before_sensor = "" if before is None else str(before["Sensor"]).strip().upper()
            after_sensor = "" if after is None else str(after["Sensor"]).strip().upper()
            before_room = "" if before is None else motion_sensor_room(before_sensor, config)
            after_room = "" if after is None else motion_sensor_room(after_sensor, config)

            if before_room == expected or after_room == expected:
                status = "Pass"
                reason = "Before or after motion evidence supports expected room"
            elif before is None and after is None:
                status = "Unknown"
                reason = "No nearby Mxx ON event found in this case"
            else:
                status = "Warning"
                reason = "Neither nearest before nor nearest after Mxx ON supports expected room"

            rows.append(
                {
                    "Data Type": data_type,
                    "Case ID": case_id,
                    "Timestamp": event["Timestamp"],
                    "Source File": event["Source File"],
                    "Activity": activity,
                    "Expected Room": expected,
                    "Before Motion Sensor": before_sensor,
                    "Before Motion Room": before_room,
                    "Before Motion Timestamp": "" if before is None else before["Timestamp"],
                    "After Motion Sensor": after_sensor,
                    "After Motion Room": after_room,
                    "After Motion Timestamp": "" if after is None else after["Timestamp"],
                    "Status": status,
                    "Reason": reason,
                }
            )

    return pd.DataFrame(rows)


def build_quality_summary(raw_report: pd.DataFrame, nearby_report: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    raw_counts = raw_report["Status"].value_counts().to_dict()
    nearby_counts = nearby_report["Status"].value_counts().to_dict()

    rows.append(
        {
            "Check": "Raw data quality check",
            "Pass": int(raw_counts.get("Pass", 0)),
            "Warning": int(raw_counts.get("Warning", 0)),
            "Fail": int(raw_counts.get("Fail", 0)),
            "Unknown": 0,
            "Details": "Checks file names, required columns, empty files, and undefined sensors",
        }
    )
    rows.append(
        {
            "Check": "Nearby motion check",
            "Pass": int(nearby_counts.get("Pass", 0)),
            "Warning": int(nearby_counts.get("Warning", 0)),
            "Fail": 0,
            "Unknown": int(nearby_counts.get("Unknown", 0)),
            "Details": "Checks whether nearby Mxx ON evidence supports kitchen and phone actions",
        }
    )

    for data_type, group in nearby_report.groupby("Data Type", sort=True):
        counts = group["Status"].value_counts().to_dict()
        rows.append(
            {
                "Check": f"Nearby motion check ({data_type})",
                "Pass": int(counts.get("Pass", 0)),
                "Warning": int(counts.get("Warning", 0)),
                "Fail": 0,
                "Unknown": int(counts.get("Unknown", 0)),
                "Details": f"{len(group)} checked kitchen/phone actions",
            }
        )

    return pd.DataFrame(rows)
