from __future__ import annotations

import re

import pandas as pd

from .mapping import MOTION_RE


def motion_sensor_room(sensor: object, config: dict[str, object]) -> str:
    rooms: dict[str, str] = config["motion_rooms"]  # type: ignore[assignment]
    return rooms.get(str(sensor).strip().upper(), "unmapped motion area")


def action_sensor_room(sensor: object, config: dict[str, object]) -> str | None:
    rooms: dict[str, str] = config["action_rooms"]  # type: ignore[assignment]
    return rooms.get(str(sensor).strip().upper())


def room_action_activity(
    sensor: object,
    mapped_activity: object,
    config: dict[str, object],
) -> str:
    sensor_text = str(sensor).strip().upper()
    analog_sensors: dict[str, object] = config.get("analog_sensors", {})  # type: ignore[assignment]
    analog_actions: dict[str, str] = analog_sensors.get(  # type: ignore[assignment]
        "room_action_activities", {}
    )
    return analog_actions.get(sensor_text, str(mapped_activity))


def make_room_movement(previous_room: str | None, current_room: str) -> str:
    if previous_room is None:
        return f"Enter {current_room}"
    if previous_room == current_room:
        return f"Move within {current_room}"
    return f"Move from {previous_room} to {current_room}"


def build_room_action_log(
    mapped_log: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bursty_actions = set(config.get("bursty_actions", []))
    cooldown = int(config.get("bursty_action_cooldown_seconds", 30))

    for _, case_events in mapped_log.groupby("Case ID", sort=True):
        case_events = case_events.sort_values(
            ["Timestamp", "_Original Row"], kind="mergesort"
        )
        current_room: str | None = None
        last_activity: str | None = None
        last_bursty_time: dict[str, pd.Timestamp] = {}

        for _, event in case_events.iterrows():
            sensor_text = str(event["Sensor"]).strip().upper()
            message_text = str(event["Message"]).strip().upper()
            timestamp = pd.Timestamp(event["Timestamp"])

            if MOTION_RE.match(sensor_text):
                if message_text != "ON":
                    continue
                detected_room = motion_sensor_room(sensor_text, config)
                activity = make_room_movement(current_room, detected_room)
                current_room = detected_room
            else:
                if message_text in {"START_INSTRUCT", "STOP_INSTRUCT"}:
                    continue
                activity = room_action_activity(sensor_text, event["Activity"], config)
                implied_room = action_sensor_room(sensor_text, config)
                if implied_room is not None:
                    current_room = implied_room

            if activity in bursty_actions:
                previous_time = last_bursty_time.get(activity)
                if (
                    previous_time is not None
                    and (timestamp - previous_time).total_seconds() <= cooldown
                ):
                    continue
                last_bursty_time[activity] = timestamp

            if activity == last_activity:
                continue
            row = event.to_dict()
            row["Activity"] = activity
            rows.append(row)
            last_activity = activity

    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("No room/action events generated")
    return output.sort_values(
        ["Case ID", "Timestamp", "_Original Row"], kind="mergesort"
    ).reset_index(drop=True)


def infer_room_level_activity(
    activity: object,
    sensor: object,
    config: dict[str, object],
) -> tuple[str, str]:
    activity_text = str(activity).strip()
    move_from_match = re.match(r"^Move from (.+) to (.+)$", activity_text)
    if move_from_match:
        room = move_from_match.group(2).strip()
        return room, room

    for prefix in ("Enter ", "Move within "):
        if activity_text.startswith(prefix):
            room = activity_text[len(prefix) :].strip()
            return room, room

    room = action_sensor_room(sensor, config)
    if room:
        return room, room
    return activity_text, "unknown"


def build_room_level_log(
    room_action_log: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, case_events in room_action_log.groupby("Case ID", sort=True):
        sort_columns = ["Timestamp"]
        if "_Original Row" in case_events.columns:
            sort_columns.append("_Original Row")
        case_events = case_events.sort_values(sort_columns, kind="mergesort")
        last_activity: str | None = None
        for _, event in case_events.iterrows():
            activity, room = infer_room_level_activity(
                event["Activity"], event.get("Sensor", ""), config
            )
            if activity == last_activity:
                continue
            row = event.to_dict()
            row["Original Activity"] = row["Activity"]
            row["Activity"] = activity
            row["Room"] = room
            rows.append(row)
            last_activity = activity

    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("No room-level events generated")
    sort_columns = ["Case ID", "Timestamp"]
    if "_Original Row" in output.columns:
        sort_columns.append("_Original Row")
    return output.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
