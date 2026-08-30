from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


RULE_IDS = {
    "t1": "R1",
    "t2": "R2",
    "t3": "R3",
    "t4": "R4",
    "t5": "R5",
}

EVIDENCE_STRENGTH = {
    "t1": "direct completed-episode sequence",
    "t2": "derived terminal analog state",
    "t3": "derived terminal analog state",
    "t4": "direct event-order evidence",
    "t5": "whole-case analog process evidence",
}

_FLOAT_TOLERANCE = 1e-9
_PROFILE_KEYS = {
    "t2_min": "hand_washing_min_use_amplitude",
    "t2_terminal": "hand_washing_max_normal_terminal_ratio",
    "t3_min": "cooking_min_use_amplitude",
    "t3_terminal": "cooking_max_normal_terminal_ratio",
    "t5_min": "cleaning_min_use_amplitude",
}


def _normalised_sensor_set(config: dict[str, object], role: str) -> set[str]:
    analog_sensors: dict[str, object] = config["analog_sensors"]  # type: ignore[assignment]
    values: list[str] = analog_sensors[role]  # type: ignore[assignment]
    return {str(value).strip().upper() for value in values}


def _sorted_events(events: pd.DataFrame) -> pd.DataFrame:
    return events.sort_values(["Timestamp", "_Original Row"], kind="mergesort")


def _phone_result(case_events: pd.DataFrame) -> dict[str, object]:
    sensor = case_events["Sensor"].astype(str).str.strip().str.upper()
    message = case_events["Message"].astype(str).str.strip().str.upper()
    phone = _sorted_events(case_events[sensor.eq("ASTERISK")].copy())
    phone_messages = phone["Message"].astype(str).str.strip().str.upper()
    phone = phone[phone_messages.isin(["START", "END"])]

    first_start_position: int | None = None
    for position, value in enumerate(
        phone["Message"].astype(str).str.strip().str.upper().tolist()
    ):
        if value == "START":
            first_start_position = position
            break
    if first_start_position is None:
        return {
            "Rule Outcome": "insufficient_evidence",
            "Decision Timestamp": pd.NaT,
            "Evidence Summary": "No complete first phone START/END episode",
        }

    first_end_position: int | None = None
    ordered_messages = phone["Message"].astype(str).str.strip().str.upper().tolist()
    for position in range(first_start_position + 1, len(phone)):
        value = ordered_messages[position]
        if value == "START":
            return {
                "Rule Outcome": "insufficient_evidence",
                "Decision Timestamp": pd.NaT,
                "Evidence Summary": (
                    "Overlapping phone START occurred before the first episode END"
                ),
            }
        if value == "END":
            first_end_position = position
            break
    if first_end_position is None:
        return {
            "Rule Outcome": "insufficient_evidence",
            "Decision Timestamp": pd.NaT,
            "Evidence Summary": "No complete first phone START/END episode",
        }

    for position in range(first_end_position + 1, len(phone)):
        if ordered_messages[position] == "START":
            timestamp = pd.Timestamp(phone.iloc[position]["Timestamp"])
            return {
                "Rule Outcome": "error",
                "Decision Timestamp": timestamp,
                "Evidence Summary": (
                    "Second phone START detected after a completed first episode"
                ),
            }
    return {
        "Rule Outcome": "no_error",
        "Decision Timestamp": pd.NaT,
        "Evidence Summary": "One completed phone episode and no later START",
    }


def _channel_metrics(
    case_events: pd.DataFrame,
    sensors: set[str],
) -> list[dict[str, object]]:
    selected = case_events[
        case_events["Sensor"].astype(str).str.strip().str.upper().isin(sensors)
    ].copy()
    selected["Evidence Sensor"] = (
        selected["Sensor"].astype(str).str.strip().str.upper()
    )
    selected["Numeric Value"] = pd.to_numeric(selected["Message"], errors="coerce")
    selected = selected[np.isfinite(selected["Numeric Value"].astype(float))]

    metrics: list[dict[str, object]] = []
    for sensor, events in selected.groupby("Evidence Sensor", sort=True):
        events = _sorted_events(events)
        if len(events) < 2 or events["Timestamp"].nunique(dropna=True) < 2:
            continue
        values = events["Numeric Value"].astype(float)
        minimum = float(values.min())
        maximum = float(values.max())
        amplitude = maximum - minimum
        if amplitude <= _FLOAT_TOLERANCE:
            continue
        last = float(values.iloc[-1])
        metrics.append(
            {
                "Evidence Sensor": str(sensor),
                "Use Amplitude": amplitude,
                "Terminal Ratio": (last - minimum) / amplitude,
                "Minimum Value": minimum,
                "Maximum Value": maximum,
                "Last Value": last,
            }
        )
    return metrics


def _terminal_analog_result(
    case_events: pd.DataFrame,
    *,
    sensors: set[str],
    minimum_amplitude: float,
    maximum_normal_terminal_ratio: float,
) -> dict[str, object]:
    case_end = pd.Timestamp(case_events["Timestamp"].max())
    candidates = [
        metric
        for metric in _channel_metrics(case_events, sensors)
        if float(metric["Use Amplitude"]) + _FLOAT_TOLERANCE
        >= minimum_amplitude
    ]
    if not candidates:
        return {
            "Rule Outcome": "insufficient_evidence",
            "Decision Timestamp": pd.NaT,
            "Evidence Summary": (
                "No configured channel had two time-separated numeric readings "
                "with sufficient dynamic amplitude"
            ),
            "Evidence Sensor": pd.NA,
            "Use Amplitude": np.nan,
            "Terminal Ratio": np.nan,
        }

    matching = [
        metric
        for metric in candidates
        if float(metric["Terminal Ratio"]) + _FLOAT_TOLERANCE
        >= maximum_normal_terminal_ratio
    ]
    selected = max(
        matching or candidates,
        key=lambda metric: (
            float(metric["Terminal Ratio"]),
            float(metric["Use Amplitude"]),
            str(metric["Evidence Sensor"]),
        ),
    )
    outcome = "error" if matching else "no_error"
    return {
        "Rule Outcome": outcome,
        "Decision Timestamp": case_end if matching else pd.NaT,
        "Evidence Summary": (
            f"{selected['Evidence Sensor']}: "
            f"amplitude={float(selected['Use Amplitude']):.6f}, "
            f"terminal_ratio={float(selected['Terminal Ratio']):.6f}"
        ),
        "Evidence Sensor": selected["Evidence Sensor"],
        "Use Amplitude": selected["Use Amplitude"],
        "Terminal Ratio": selected["Terminal Ratio"],
    }


def _medicine_result(
    case_events: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    motion_rooms: dict[str, str] = config["motion_rooms"]  # type: ignore[assignment]
    normalised_rooms = {
        str(sensor).strip().upper(): str(room).strip().lower()
        for sensor, room in motion_rooms.items()
    }
    sensor = case_events["Sensor"].astype(str).str.strip().str.upper()
    message = case_events["Message"].astype(str).str.strip().str.upper()
    room = sensor.map(normalised_rooms)
    dining = _sorted_events(case_events[message.eq("ON") & room.eq("dining room")])
    if dining.empty:
        return {
            "Rule Outcome": "insufficient_evidence",
            "Decision Timestamp": pd.NaT,
            "Evidence Summary": "No dining-room motion ON event",
        }

    entry_time = pd.Timestamp(dining.iloc[0]["Timestamp"])
    medicine = case_events[
        sensor.eq("I06")
        & message.eq("ABSENT")
        & (case_events["Timestamp"] < entry_time)
    ]
    if medicine.empty:
        return {
            "Rule Outcome": "error",
            "Decision Timestamp": entry_time,
            "Evidence Summary": "No I06 ABSENT event before first dining-room entry",
        }
    return {
        "Rule Outcome": "no_error",
        "Decision Timestamp": pd.NaT,
        "Evidence Summary": "I06 ABSENT detected before first dining-room entry",
    }


def _cleaning_result(
    case_events: pd.DataFrame,
    *,
    sensors: set[str],
    minimum_amplitude: float,
) -> dict[str, object]:
    case_end = pd.Timestamp(case_events["Timestamp"].max())
    candidates = [
        metric
        for metric in _channel_metrics(case_events, sensors)
        if float(metric["Use Amplitude"]) + _FLOAT_TOLERANCE
        >= minimum_amplitude
    ]
    if candidates:
        selected = max(
            candidates,
            key=lambda metric: (
                float(metric["Use Amplitude"]),
                str(metric["Evidence Sensor"]),
            ),
        )
        return {
            "Rule Outcome": "no_error",
            "Decision Timestamp": pd.NaT,
            "Evidence Summary": (
                f"{selected['Evidence Sensor']}: valid whole-case water process, "
                f"amplitude={float(selected['Use Amplitude']):.6f}"
            ),
            "Evidence Sensor": selected["Evidence Sensor"],
            "Use Amplitude": selected["Use Amplitude"],
            "Terminal Ratio": selected["Terminal Ratio"],
        }
    return {
        "Rule Outcome": "error",
        "Decision Timestamp": case_end,
        "Evidence Summary": (
            "No configured water channel had a valid whole-case dynamic process"
        ),
        "Evidence Sensor": pd.NA,
        "Use Amplitude": np.nan,
        "Terminal Ratio": np.nan,
    }


def detect_rule_based_cases(
    sensor_log: pd.DataFrame,
    config: dict[str, object],
    profile: dict[str, float],
    *,
    profile_label: str = "primary",
    rule_version: str = "rule-based-error-detection",
) -> pd.DataFrame:
    """Apply the five label-blind task-specific rules once per case."""

    error_rules: dict[str, str] = config["error_rules"]  # type: ignore[assignment]
    water_sensors = _normalised_sensor_set(config, "water")
    burner_sensors = _normalised_sensor_set(config, "burner")
    rows: list[dict[str, object]] = []

    for case_id, raw_case_events in sensor_log.groupby("Case ID", sort=True):
        case_events = _sorted_events(raw_case_events)
        task_values = case_events["_Task ID"].astype(str).unique().tolist()
        if len(task_values) != 1:
            raise ValueError(f"Case {case_id!s} does not have exactly one Task ID")
        task_id = task_values[0]
        if task_id == "t1":
            result = _phone_result(case_events)
        elif task_id == "t2":
            minimum = float(profile[_PROFILE_KEYS["t2_min"]])
            terminal = float(profile[_PROFILE_KEYS["t2_terminal"]])
            result = (
                _terminal_analog_result(
                    case_events,
                    sensors=water_sensors,
                    minimum_amplitude=minimum,
                    maximum_normal_terminal_ratio=terminal,
                )
                if math.isfinite(minimum) and math.isfinite(terminal)
                else {
                    "Rule Outcome": "insufficient_evidence",
                    "Decision Timestamp": pd.NaT,
                    "Evidence Summary": "Normal-training calibration threshold unavailable",
                }
            )
        elif task_id == "t3":
            minimum = float(profile[_PROFILE_KEYS["t3_min"]])
            terminal = float(profile[_PROFILE_KEYS["t3_terminal"]])
            result = (
                _terminal_analog_result(
                    case_events,
                    sensors=burner_sensors,
                    minimum_amplitude=minimum,
                    maximum_normal_terminal_ratio=terminal,
                )
                if math.isfinite(minimum) and math.isfinite(terminal)
                else {
                    "Rule Outcome": "insufficient_evidence",
                    "Decision Timestamp": pd.NaT,
                    "Evidence Summary": "Normal-training calibration threshold unavailable",
                }
            )
        elif task_id == "t4":
            result = _medicine_result(case_events, config)
        elif task_id == "t5":
            minimum = float(profile[_PROFILE_KEYS["t5_min"]])
            result = (
                _cleaning_result(
                    case_events,
                    sensors=water_sensors,
                    minimum_amplitude=minimum,
                )
                if math.isfinite(minimum)
                else {
                    "Rule Outcome": "insufficient_evidence",
                    "Decision Timestamp": pd.NaT,
                    "Evidence Summary": "Normal-training calibration threshold unavailable",
                }
            )
        else:
            raise ValueError(f"Rule-based detector not implemented for task: {task_id}")

        row: dict[str, object] = {
            "Case ID": case_id,
            "Task ID": task_id,
            "Case Start": pd.Timestamp(case_events["Timestamp"].min()),
            "Case End": pd.Timestamp(case_events["Timestamp"].max()),
            "Rule ID": RULE_IDS[task_id],
            "Rule Outcome": result["Rule Outcome"],
            "Error Type": error_rules[task_id],
            "Decision Timestamp": result["Decision Timestamp"],
            "Evidence Strength": EVIDENCE_STRENGTH[task_id],
            "Evidence Summary": result["Evidence Summary"],
            "Evidence Sensor": result.get("Evidence Sensor", pd.NA),
            "Use Amplitude": result.get("Use Amplitude", np.nan),
            "Terminal Ratio": result.get("Terminal Ratio", np.nan),
            "Source File": str(case_events["Source File"].iloc[0]),
            "Data Type": str(case_events["Data Type"].iloc[0]),
            "Profile": profile_label,
            "Threshold Profile": profile_label,
            "Mapping Version": str(config.get("mapping_version", "unknown")),
            "Rule Version": rule_version,
        }
        rows.append(row)

    columns = [
        "Case ID",
        "Task ID",
        "Case Start",
        "Case End",
        "Rule ID",
        "Rule Outcome",
        "Error Type",
        "Decision Timestamp",
        "Evidence Strength",
        "Evidence Summary",
        "Evidence Sensor",
        "Use Amplitude",
        "Terminal Ratio",
        "Source File",
        "Data Type",
        "Profile",
        "Threshold Profile",
        "Mapping Version",
        "Rule Version",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["Case ID"], kind="mergesort"
    ).reset_index(drop=True)


def _quantile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, quantile, method="linear"))


def _task_metrics(
    events: pd.DataFrame,
    task_id: str,
    sensors: set[str],
) -> tuple[dict[str, list[dict[str, object]]], int]:
    task_events = events[events["_Task ID"].astype(str).eq(task_id)]
    case_metrics: dict[str, list[dict[str, object]]] = {}
    for case_id, case_events in task_events.groupby("Case ID", sort=True):
        case_metrics[str(case_id)] = _channel_metrics(case_events, sensors)
    return case_metrics, int(task_events["Case ID"].nunique())


def _case_maxima(
    metrics: dict[str, list[dict[str, object]]],
    key: str,
    *,
    minimum_amplitude: float | None = None,
) -> list[float]:
    values: list[float] = []
    for case_values in metrics.values():
        eligible = case_values
        if minimum_amplitude is not None:
            eligible = [
                value
                for value in eligible
                if float(value["Use Amplitude"]) + _FLOAT_TOLERANCE
                >= minimum_amplitude
            ]
        if eligible:
            values.append(max(float(value[key]) for value in eligible))
    return values


def _calibration_record(
    *,
    fold: int,
    task_id: str,
    parameter: str,
    quantile: float,
    distribution: list[float],
    training_cases: int,
    minimum_cases: int,
) -> tuple[float, dict[str, object]]:
    sufficient = len(distribution) >= minimum_cases
    value = _quantile(distribution, quantile) if sufficient else float("nan")
    status = "calibrated" if sufficient else "insufficient_evidence"
    finite = np.asarray(distribution, dtype=float)
    return value, {
        "Fold": fold,
        "Profile": "primary",
        "Task ID": task_id,
        "Rule ID": RULE_IDS[task_id],
        "Parameter": parameter,
        "Quantile": quantile,
        "Value": value,
        "Evidence Cases": len(distribution),
        "Training Normal Cases": training_cases,
        "Training Error Cases": 0,
        "Label Access": "normal_training_only",
        "Calibration Status": status,
        "Distribution Min": float(np.min(finite)) if finite.size else np.nan,
        "Distribution Median": float(np.median(finite)) if finite.size else np.nan,
        "Distribution Max": float(np.max(finite)) if finite.size else np.nan,
    }


def _structural_evidence(
    training_normal_events: pd.DataFrame,
    config: dict[str, object],
    fold: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task_id, parameter in (
        ("t1", "complete_first_episode_prevalence"),
        ("t4", "medicine_before_dining_prevalence"),
    ):
        task_events = training_normal_events[
            training_normal_events["_Task ID"].astype(str).eq(task_id)
        ]
        total = int(task_events["Case ID"].nunique())
        supported = 0
        observable = 0
        for _, case_events in task_events.groupby("Case ID", sort=True):
            if task_id == "t1":
                result = _phone_result(_sorted_events(case_events))
                if result["Rule Outcome"] != "insufficient_evidence":
                    observable += 1
                    supported += 1
            else:
                result = _medicine_result(_sorted_events(case_events), config)
                if result["Rule Outcome"] != "insufficient_evidence":
                    observable += 1
                    if result["Rule Outcome"] == "no_error":
                        supported += 1
        rows.append(
            {
                "Fold": fold,
                "Profile": "primary",
                "Task ID": task_id,
                "Rule ID": RULE_IDS[task_id],
                "Parameter": parameter,
                "Supported Cases": supported,
                "Observable Cases": observable,
                "Training Normal Cases": total,
                "Prevalence": supported / total if total else np.nan,
                "Calibration Status": "structural_predefined",
                "Training Error Cases": 0,
                "Label Access": "normal_training_only",
            }
        )
    return pd.DataFrame(rows)


def calibrate_rule_profile(
    training_normal_events: pd.DataFrame,
    config: dict[str, object],
    *,
    fold: int,
    minimum_normal_evidence_cases: int = 15,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit only the five numerical thresholds from normal training cases."""

    if minimum_normal_evidence_cases < 1:
        raise ValueError("minimum_normal_evidence_cases must be positive")
    if training_normal_events.empty:
        raise ValueError("Rule calibration requires normal training events")
    observed_data_types = set(
        training_normal_events["Data Type"].astype(str).str.strip().str.lower()
    )
    if observed_data_types != {"normal"}:
        raise ValueError("Rule calibration may use normal training cases only")

    water = _normalised_sensor_set(config, "water")
    burner = _normalised_sensor_set(config, "burner")
    t2_metrics, t2_cases = _task_metrics(training_normal_events, "t2", water)
    t3_metrics, t3_cases = _task_metrics(training_normal_events, "t3", burner)
    t5_metrics, t5_cases = _task_metrics(training_normal_events, "t5", water)

    threshold_rows: list[dict[str, object]] = []
    t2_amplitudes = _case_maxima(t2_metrics, "Use Amplitude")
    t2_min, record = _calibration_record(
        fold=fold,
        task_id="t2",
        parameter=_PROFILE_KEYS["t2_min"],
        quantile=0.05,
        distribution=t2_amplitudes,
        training_cases=t2_cases,
        minimum_cases=minimum_normal_evidence_cases,
    )
    threshold_rows.append(record)
    provisional_t2_min = _quantile(t2_amplitudes, 0.05)
    t2_residuals = _case_maxima(
        t2_metrics,
        "Terminal Ratio",
        minimum_amplitude=provisional_t2_min,
    )
    t2_terminal, record = _calibration_record(
        fold=fold,
        task_id="t2",
        parameter=_PROFILE_KEYS["t2_terminal"],
        quantile=0.95,
        distribution=t2_residuals,
        training_cases=t2_cases,
        minimum_cases=minimum_normal_evidence_cases,
    )
    threshold_rows.append(record)

    t3_amplitudes = _case_maxima(t3_metrics, "Use Amplitude")
    t3_min, record = _calibration_record(
        fold=fold,
        task_id="t3",
        parameter=_PROFILE_KEYS["t3_min"],
        quantile=0.05,
        distribution=t3_amplitudes,
        training_cases=t3_cases,
        minimum_cases=minimum_normal_evidence_cases,
    )
    threshold_rows.append(record)
    provisional_t3_min = _quantile(t3_amplitudes, 0.05)
    t3_residuals = _case_maxima(
        t3_metrics,
        "Terminal Ratio",
        minimum_amplitude=provisional_t3_min,
    )
    t3_terminal, record = _calibration_record(
        fold=fold,
        task_id="t3",
        parameter=_PROFILE_KEYS["t3_terminal"],
        quantile=0.95,
        distribution=t3_residuals,
        training_cases=t3_cases,
        minimum_cases=minimum_normal_evidence_cases,
    )
    threshold_rows.append(record)

    t5_amplitudes = _case_maxima(t5_metrics, "Use Amplitude")
    t5_min, record = _calibration_record(
        fold=fold,
        task_id="t5",
        parameter=_PROFILE_KEYS["t5_min"],
        quantile=0.05,
        distribution=t5_amplitudes,
        training_cases=t5_cases,
        minimum_cases=minimum_normal_evidence_cases,
    )
    threshold_rows.append(record)

    profile = {
        _PROFILE_KEYS["t2_min"]: t2_min,
        _PROFILE_KEYS["t2_terminal"]: t2_terminal,
        _PROFILE_KEYS["t3_min"]: t3_min,
        _PROFILE_KEYS["t3_terminal"]: t3_terminal,
        _PROFILE_KEYS["t5_min"]: t5_min,
    }
    thresholds = pd.DataFrame(threshold_rows).sort_values(
        ["Task ID", "Parameter"], kind="mergesort"
    ).reset_index(drop=True)

    status_rows: list[dict[str, object]] = []
    for task_id in ("t1", "t2", "t3", "t4", "t5"):
        task_thresholds = thresholds[thresholds["Task ID"].eq(task_id)]
        if task_id in {"t1", "t4"}:
            status = "structural_predefined"
            evidence_cases = int(
                training_normal_events[
                    training_normal_events["_Task ID"].astype(str).eq(task_id)
                ]["Case ID"].nunique()
            )
        else:
            status = (
                "calibrated"
                if not task_thresholds.empty
                and task_thresholds["Calibration Status"].eq("calibrated").all()
                else "insufficient_evidence"
            )
            evidence_cases = (
                int(task_thresholds["Evidence Cases"].min())
                if not task_thresholds.empty
                else 0
            )
        status_rows.append(
            {
                "Fold": fold,
                "Profile": "primary",
                "Task ID": task_id,
                "Rule ID": RULE_IDS[task_id],
                "Calibration Status": status,
                "Evidence Cases": evidence_cases,
                "Required Evidence Cases": minimum_normal_evidence_cases,
                "Training Normal Cases": int(
                    training_normal_events[
                        training_normal_events["_Task ID"].astype(str).eq(task_id)
                    ]["Case ID"].nunique()
                ),
                "Training Error Cases": 0,
                "Label Access": "normal_training_only",
            }
        )
    status_frame = pd.DataFrame(status_rows)
    structural = _structural_evidence(training_normal_events, config, fold)
    return profile, thresholds, status_frame, structural
