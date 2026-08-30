from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


CASE_RE = re.compile(r"^(p\d+)\.(t\d+)\.csv$", re.IGNORECASE)


def parse_case_file_name(path: Path) -> tuple[str, str, str]:
    match = CASE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected sensor log file name: {path.name}")
    participant_id, task_id = match.groups()
    return participant_id.lower(), task_id.lower(), f"{participant_id}_{task_id}".lower()


def infer_hour_for_short_times(times: Iterable[str]) -> int:
    for value in times:
        text = str(value).strip()
        if re.match(r"^\d{1,2}:\d{2}:\d{2}(?:\.\d+)?$", text):
            return int(text.split(":", 1)[0])
    return 0


def parse_timestamp(date_value: object, time_value: object, inferred_hour: int) -> pd.Timestamp:
    date_text = str(date_value).strip().replace("/", "-")
    time_text = str(time_value).strip()

    if re.match(r"^\d{1,2}:\d{2}:\d{2}(?:\.\d+)?$", time_text):
        return pd.to_datetime(f"{date_text} {time_text}")

    short_match = re.match(r"^(\d+):(\d{2})(?:\.(\d+))?$", time_text)
    if short_match:
        minutes = int(short_match.group(1))
        seconds = int(short_match.group(2))
        fraction = short_match.group(3) or "0"
        microseconds = int((fraction + "000000")[:6])
        base = pd.to_datetime(f"{date_text} {inferred_hour:02d}:00:00")
        return base + pd.Timedelta(
            minutes=minutes, seconds=seconds, microseconds=microseconds
        )

    return pd.to_datetime(f"{date_text} {time_text}")


def make_raw_activity(sensor: object, message: object) -> str:
    sensor_text = str(sensor).strip()
    message_text = str(message).strip()
    if sensor_text.startswith("AD1-"):
        return f"{sensor_text} value {message_text}"
    if sensor_text.lower() == "asterisk":
        return f"Phone sensor {message_text}"
    return f"{sensor_text} {message_text}"


def read_sensor_folder(
    folder: Path, data_type: str, config: dict[str, object], root_dir: Path
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required = set(config["required_columns"])
    excluded = {str(value).upper() for value in config.get("excluded_sensors", [])}

    for csv_path in sorted(folder.glob("*.csv")):
        _, task_id, case_id = parse_case_file_name(csv_path)
        raw = pd.read_csv(csv_path)
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")

        inferred_hour = infer_hour_for_short_times(raw["time"].tolist())
        source_file = str(csv_path.relative_to(root_dir)).replace("\\", "/")

        for row_index, row in raw.iterrows():
            sensor_text = str(row["sensor"]).strip()
            message_text = str(row["message"]).strip()
            if sensor_text.upper() in excluded:
                continue

            rows.append(
                {
                    "Case ID": case_id,
                    "Timestamp": parse_timestamp(row["date"], row["time"], inferred_hour),
                    "Sensor": sensor_text,
                    "Message": message_text,
                    "Activity": make_raw_activity(sensor_text, message_text),
                    "Source File": source_file,
                    "Data Type": data_type,
                    "_Task ID": task_id,
                    "_Original Row": row_index,
                }
            )

    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError(f"No events found in {folder}")
    return output.sort_values(
        ["Case ID", "Timestamp", "_Original Row"], kind="mergesort"
    ).reset_index(drop=True)
