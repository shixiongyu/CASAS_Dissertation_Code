from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


PARTICIPANT_RE = re.compile(r"^(p\d+)_t\d+$", re.IGNORECASE)


def participant_from_case_id(case_id: str) -> str:
    match = PARTICIPANT_RE.match(str(case_id).strip())
    if not match:
        raise ValueError(f"Unexpected case identifier: {case_id}")
    return match.group(1).lower()


def build_fold_assignments(
    normal_participants: Iterable[str],
    error_participants: Iterable[str],
    n_folds: int,
) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("At least two folds are required")

    rows: list[dict[str, object]] = []
    for condition, participants in (
        ("normal", normal_participants),
        ("scripted_error", error_participants),
    ):
        unique = sorted({str(value).lower() for value in participants})
        if len(unique) < n_folds:
            raise ValueError(
                f"{condition} has fewer participants than folds: "
                f"{len(unique)} < {n_folds}"
            )
        for index, participant_id in enumerate(unique):
            rows.append(
                {
                    "Condition": condition,
                    "Participant ID": participant_id,
                    "Fold": index % n_folds,
                }
            )

    assignments = pd.DataFrame(rows).sort_values(
        ["Condition", "Participant ID"], kind="mergesort"
    )
    validate_fold_assignments(assignments, n_folds)
    return assignments.reset_index(drop=True)


def validate_fold_assignments(
    assignments: pd.DataFrame,
    n_folds: int,
) -> None:
    required = {"Condition", "Participant ID", "Fold"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Fold assignments are missing columns: {sorted(missing)}")
    if assignments.duplicated(["Condition", "Participant ID"]).any():
        raise ValueError("A participant appears more than once in fold assignments")
    if set(assignments["Fold"].astype(int)) != set(range(n_folds)):
        raise ValueError("Fold identifiers are incomplete")

    for condition, group in assignments.groupby("Condition", sort=True):
        counts = group.groupby("Fold").size()
        if counts.max() - counts.min() > 1:
            raise ValueError(f"{condition} participants are not balanced across folds")


def attach_folds(
    cases: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    output = cases.copy()
    output["Participant ID"] = output["Case ID"].map(participant_from_case_id)
    condition = output["Data Type"].map(
        {"normal": "normal", "error_trace": "scripted_error"}
    )
    if condition.isna().any():
        unexpected = sorted(
            output.loc[condition.isna(), "Data Type"].astype(str).unique()
        )
        raise ValueError(f"Unexpected data types: {unexpected}")
    output["Condition"] = condition
    merged = output.merge(
        assignments,
        on=["Condition", "Participant ID"],
        how="left",
        validate="many_to_one",
    )
    if merged["Fold"].isna().any():
        missing = sorted(merged.loc[merged["Fold"].isna(), "Case ID"].unique())
        raise ValueError(f"Cases have no fold assignment: {missing}")
    merged["Fold"] = merged["Fold"].astype(int)
    return merged
