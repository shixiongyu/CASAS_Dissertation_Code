from __future__ import annotations

import re

import pandas as pd

from .ingestion import make_raw_activity


MOTION_RE = re.compile(r"^M\d{2,3}$", re.IGNORECASE)


def make_mapped_activity(sensor: object, message: object, config: dict[str, object]) -> str:
    sensor_text = str(sensor).strip().upper()
    message_text = str(message).strip().upper()
    item_sensors: dict[str, str] = config["item_sensors"]  # type: ignore[assignment]
    analog_sensors: dict[str, object] = config["analog_sensors"]  # type: ignore[assignment]
    analog_activities: dict[str, str] = analog_sensors["activities"]  # type: ignore[assignment]

    if MOTION_RE.match(sensor_text):
        if message_text == "ON":
            return f"Motion detected at {sensor_text}"
        if message_text == "OFF":
            return f"Motion stopped at {sensor_text}"
        return f"Motion sensor {sensor_text} {message_text.lower()}"

    if sensor_text in item_sensors:
        item_name = item_sensors[sensor_text]
        if message_text == "ABSENT":
            return f"Take {item_name}"
        if message_text == "PRESENT":
            return f"Return {item_name}"
        return f"{item_name.capitalize()} sensor {message_text.lower()}"

    if sensor_text == "D01":
        if message_text == "OPEN":
            return "Open kitchen cabinet"
        if message_text in {"CLOSE", "CLOSED"}:
            return "Close kitchen cabinet"
        return f"Kitchen cabinet door {message_text.lower()}"

    if sensor_text in analog_activities:
        return analog_activities[sensor_text]

    if sensor_text == "ASTERISK":
        if message_text == "START":
            return "Start phone use"
        if message_text == "END":
            return "End phone use"
        return f"Phone use {message_text.lower()}"

    if message_text in {"START_INSTRUCT", "STOP_INSTRUCT"}:
        return f"Instruction marker {message_text.lower()}"

    return make_raw_activity(sensor, message)


def build_mapped_event_log(sensor_log: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    mapped = sensor_log.copy()
    mapped["Activity"] = mapped.apply(
        lambda row: make_mapped_activity(row["Sensor"], row["Message"], config), axis=1
    )
    return mapped.sort_values(
        ["Case ID", "Timestamp", "_Original Row"], kind="mergesort"
    ).reset_index(drop=True)
