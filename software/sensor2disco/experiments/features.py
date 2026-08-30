from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from .protocol import participant_from_case_id


@dataclass(frozen=True)
class CaseQuality:
    """Label-blind data-quality facts available while a case is encoded."""

    empty_token: bool = False
    non_numeric_analog_count: int = 0
    usable_analog_reading_pairs: int = 0
    rule_required_evidence_status: str = "not_audited"
    rule_required_evidence_detail: str = "not_audited"


@dataclass(frozen=True)
class CaseQualityAssessment:
    """Per-case quality facts relative to a supplied training reference."""

    case_id: str
    empty_token: bool
    non_numeric_analog_count: int
    usable_analog_reading_pairs: int
    novel_state_count: int
    novel_transition_count: int
    novel_states: tuple[str, ...]
    novel_transitions: tuple[str, ...]
    rule_required_evidence_status: str
    rule_required_evidence_detail: str

    def as_record(self) -> dict[str, object]:
        """Return a deterministic, flat record suitable for a CSV audit table."""

        return {
            "Case ID": self.case_id,
            "Empty Token": self.empty_token,
            "Non-numeric Analog Count": self.non_numeric_analog_count,
            "Usable Analog Reading Pairs": self.usable_analog_reading_pairs,
            "Novel State Count": self.novel_state_count,
            "Novel Transition Count": self.novel_transition_count,
            "Novel States": "; ".join(self.novel_states),
            "Novel Transitions": "; ".join(self.novel_transitions),
            "Rule Required Evidence Status": self.rule_required_evidence_status,
            "Rule Required Evidence Detail": self.rule_required_evidence_detail,
        }


@dataclass(frozen=True)
class CaseSequence:
    case_id: str
    participant_id: str
    task_id: str
    data_type: str
    source_file: str
    tokens: tuple[str, ...]
    elapsed_seconds: tuple[float, ...]
    duration_seconds: float
    quality: CaseQuality = field(default_factory=CaseQuality)


def _slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return text.strip("_")


def _elapsed_bin(seconds: float, boundaries: tuple[float, ...]) -> str:
    for boundary in boundaries:
        if seconds <= boundary:
            return f"le_{boundary:g}s"
    return f"gt_{boundaries[-1]:g}s"


def _analog_names(config: dict[str, object]) -> dict[str, str]:
    analog: dict[str, object] = config["analog_sensors"]  # type: ignore[assignment]
    activities: dict[str, str] = analog["activities"]  # type: ignore[assignment]
    return {
        str(sensor).upper(): _slug(activity)
        for sensor, activity in activities.items()
    }


def _normalized_sensor_message(
    case_events: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    return (
        case_events["Sensor"].astype(str).str.strip().str.upper(),
        case_events["Message"].astype(str).str.strip().str.upper(),
    )


def _numeric_analog_values(
    case_events: pd.DataFrame,
    sensors: set[str],
) -> pd.DataFrame:
    sensor_text, _ = _normalized_sensor_message(case_events)
    selected = case_events[sensor_text.isin(sensors)].copy()
    selected["Numeric Value"] = pd.to_numeric(
        selected["Message"], errors="coerce"
    )
    return selected.dropna(subset=["Numeric Value"])


def _rule_required_evidence(
    case_events: pd.DataFrame,
    config: dict[str, object],
    task_id: str,
) -> tuple[str, str]:
    """Summarise only whether the configured rule has its required evidence.

    This check uses the known task ID and sensor data but never reads a
    normal/error outcome label.
    """

    sensor_text, message_text = _normalized_sensor_message(case_events)
    analog: dict[str, object] = config["analog_sensors"]  # type: ignore[assignment]
    motion_rooms: dict[str, str] = config["motion_rooms"]  # type: ignore[assignment]

    if task_id == "t1":
        phone = case_events[sensor_text.eq("ASTERISK")].sort_values(
            ["Timestamp", "_Original Row"], kind="mergesort"
        )
        phone_message = (
            phone["Message"].astype(str).str.strip().str.upper()
        )
        starts = phone[phone_message.eq("START")]
        if len(starts) >= 2:
            return "sufficient", "at least two phone START events"
        if len(starts) == 1:
            later_end = phone[
                phone_message.eq("END")
                & (phone["Timestamp"] >= starts.iloc[0]["Timestamp"])
            ]
            if not later_end.empty:
                return "sufficient", "one complete phone START/END episode"
        return "insufficient_evidence", "no complete or repeated phone episode"

    if task_id == "t2":
        water_sensors = {
            str(value).strip().upper()
            for value in analog["water"]  # type: ignore[index]
        }
        numeric = _numeric_analog_values(case_events, water_sensors)
        counts = numeric.groupby(
            numeric["Sensor"].astype(str).str.strip().str.upper(),
            sort=True,
        ).size()
        if not counts.empty and int(counts.max()) >= 2:
            return "sufficient", "water channel has at least two numeric readings"
        return (
            "insufficient_evidence",
            "no water channel has two numeric readings",
        )

    if task_id == "t3":
        burner_sensors = {
            str(value).strip().upper()
            for value in analog["burner"]  # type: ignore[index]
        }
        if not _numeric_analog_values(case_events, burner_sensors).empty:
            return "sufficient", "numeric burner reading is available"
        return "insufficient_evidence", "no numeric burner reading"

    rooms = sensor_text.map(motion_rooms)
    if task_id == "t4":
        if bool((message_text.eq("ON") & rooms.eq("dining room")).any()):
            return "sufficient", "dining-room motion ON is available"
        return "insufficient_evidence", "no dining-room motion ON event"

    if task_id == "t5":
        if bool((message_text.eq("ON") & rooms.eq("kitchen")).any()):
            return "sufficient", "kitchen motion ON is available"
        return "insufficient_evidence", "no kitchen motion ON event"

    return "not_audited", f"no rule evidence contract for {task_id}"


def build_case_sequence(
    case_events: pd.DataFrame,
    config: dict[str, object],
    *,
    analog_delta_threshold: float,
) -> CaseSequence:
    ordered = case_events.sort_values(
        ["Timestamp", "_Original Row"], kind="mergesort"
    )
    case_id = str(ordered["Case ID"].iloc[0])
    task_id = str(ordered["_Task ID"].iloc[0])
    data_type = str(ordered["Data Type"].iloc[0])
    source_file = str(ordered["Source File"].iloc[0])
    motion_rooms: dict[str, str] = config["motion_rooms"]  # type: ignore[assignment]
    analog_names = _analog_names(config)

    tokens: list[str] = []
    elapsed: list[float] = []
    analog_previous: dict[str, float] = {}
    analog_numeric_counts: Counter[str] = Counter()
    non_numeric_analog_count = 0
    previous_raw_time: pd.Timestamp | None = None

    for _, row in ordered.iterrows():
        timestamp = pd.Timestamp(row["Timestamp"])
        gap = (
            max((timestamp - previous_raw_time).total_seconds(), 0.0)
            if previous_raw_time is not None
            else 0.0
        )
        previous_raw_time = timestamp

        sensor = str(row["Sensor"]).strip().upper()
        message = str(row["Message"]).strip().upper()
        token: str | None = None

        if sensor in motion_rooms and message == "ON":
            token = f"location:{_slug(motion_rooms[sensor])}"
        elif sensor in analog_names:
            value = pd.to_numeric(pd.Series([row["Message"]]), errors="coerce").iloc[0]
            if not pd.isna(value):
                numeric = float(value)
                analog_numeric_counts[sensor] += 1
                if sensor in analog_previous:
                    delta = numeric - analog_previous[sensor]
                    if delta >= analog_delta_threshold:
                        token = f"analog:{analog_names[sensor]}:up"
                    elif delta <= -analog_delta_threshold:
                        token = f"analog:{analog_names[sensor]}:down"
                analog_previous[sensor] = numeric
            else:
                non_numeric_analog_count += 1
        elif sensor == "D01" and message in {"OPEN", "CLOSE", "CLOSED"}:
            normalized = "close" if message in {"CLOSE", "CLOSED"} else "open"
            token = f"cabinet:{normalized}"
        elif re.fullmatch(r"I0[1-8]", sensor) and message in {"ABSENT", "PRESENT"}:
            token = f"item:{sensor.lower()}:{message.lower()}"
        elif sensor == "ASTERISK" and message in {"START", "END"}:
            token = f"phone:{message.lower()}"

        if token is not None:
            tokens.append(token)
            elapsed.append(float(gap))

    empty_token = not tokens
    if empty_token:
        tokens = ["empty"]
        elapsed = [0.0]

    evidence_status, evidence_detail = _rule_required_evidence(
        ordered, config, task_id
    )
    quality = CaseQuality(
        empty_token=empty_token,
        non_numeric_analog_count=non_numeric_analog_count,
        usable_analog_reading_pairs=sum(
            max(count - 1, 0) for count in analog_numeric_counts.values()
        ),
        rule_required_evidence_status=evidence_status,
        rule_required_evidence_detail=evidence_detail,
    )
    start = pd.Timestamp(ordered["Timestamp"].min())
    end = pd.Timestamp(ordered["Timestamp"].max())
    return CaseSequence(
        case_id=case_id,
        participant_id=participant_from_case_id(case_id),
        task_id=task_id,
        data_type=data_type,
        source_file=source_file,
        tokens=tuple(tokens),
        elapsed_seconds=tuple(elapsed),
        duration_seconds=max((end - start).total_seconds(), 0.0),
        quality=quality,
    )


def build_case_sequences(
    sensor_events: pd.DataFrame,
    config: dict[str, object],
    *,
    analog_delta_threshold: float,
) -> dict[str, CaseSequence]:
    sequences: dict[str, CaseSequence] = {}
    for case_id, case_events in sensor_events.groupby("Case ID", sort=True):
        sequence = build_case_sequence(
            case_events,
            config,
            analog_delta_threshold=analog_delta_threshold,
        )
        if sequence.case_id in sequences:
            raise ValueError(f"Duplicate case sequence: {case_id}")
        sequences[sequence.case_id] = sequence
    return sequences


def aggregate_features(
    sequence: CaseSequence,
    *,
    elapsed_boundaries: tuple[float, ...],
) -> dict[str, int]:
    features: Counter[str] = Counter()
    for token, seconds in zip(
        sequence.tokens,
        sequence.elapsed_seconds,
        strict=True,
    ):
        features[f"event_count:{token}"] += 1
        features[
            f"elapsed_count:{_elapsed_bin(seconds, elapsed_boundaries)}"
        ] += 1

    duration_bin = _elapsed_bin(sequence.duration_seconds, elapsed_boundaries)
    features[f"duration:{duration_bin}"] = 1
    features["sequence_length"] = len(sequence.tokens)
    return dict(features)


def frequency_features(sequence: CaseSequence) -> dict[str, int]:
    features: Counter[str] = Counter()
    previous = "<START>"
    for token in sequence.tokens:
        features[f"state:{token}"] += 1
        features[f"transition:{previous}->{token}"] += 1
        previous = token
    features[f"transition:{previous}-><END>"] += 1
    return dict(features)


def _sequence_transitions(sequence: CaseSequence) -> tuple[str, ...]:
    transitions: list[str] = []
    previous = "<START>"
    for token in sequence.tokens:
        transitions.append(f"{previous}->{token}")
        previous = token
    transitions.append(f"{previous}-><END>")
    return tuple(transitions)


def assess_case_quality(
    sequence: CaseSequence,
    reference_sequences: list[CaseSequence],
) -> CaseQualityAssessment:
    """Assess novel states/transitions against an explicit training reference."""

    if not reference_sequences:
        raise ValueError("At least one reference sequence is required")
    known_states = {
        token for reference in reference_sequences for token in reference.tokens
    }
    known_transitions = {
        transition
        for reference in reference_sequences
        for transition in _sequence_transitions(reference)
    }
    novel_state_occurrences = [
        token for token in sequence.tokens if token not in known_states
    ]
    novel_transition_occurrences = [
        transition
        for transition in _sequence_transitions(sequence)
        if transition not in known_transitions
    ]
    return CaseQualityAssessment(
        case_id=sequence.case_id,
        empty_token=sequence.quality.empty_token,
        non_numeric_analog_count=sequence.quality.non_numeric_analog_count,
        usable_analog_reading_pairs=sequence.quality.usable_analog_reading_pairs,
        novel_state_count=len(novel_state_occurrences),
        novel_transition_count=len(novel_transition_occurrences),
        novel_states=tuple(sorted(set(novel_state_occurrences))),
        novel_transitions=tuple(sorted(set(novel_transition_occurrences))),
        rule_required_evidence_status=(
            sequence.quality.rule_required_evidence_status
        ),
        rule_required_evidence_detail=(
            sequence.quality.rule_required_evidence_detail
        ),
    )


def sequence_is_finite(sequence: CaseSequence) -> bool:
    return all(math.isfinite(value) and value >= 0 for value in sequence.elapsed_seconds)
