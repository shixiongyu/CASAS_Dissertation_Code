# CASAS dissertation code

## Purpose

This repository supports a dissertation examining whether smart home sensor records can be transformed into structured event logs for descriptive process mining and whether explicit task-specific rules can detect scripted task errors.

The analysis contains 220 participant-task cases: 120 normal cases and 100 scripted error cases across telephone use, hand washing, cooking, medicine-container use and cleaning.

## Repository structure

- `software/`: the four-command Python interface and fixed configurations.
- `evidence/event_logs/`: contains verified activity-only Sensor, Room/Action and Room Path event-log snapshots generated from the CASAS dataset.
- `REPRODUCE.md`: step-by-step reproduction commands.
- `DATA_SOURCE.md`: dataset DOI, download information and placement instructions.

## Reported methods

- Descriptive process mining in Fluxicon Disco using Room/Action and Room Path event logs.
- A proposed rule-based method with one explicit rule for each task.
- A Cook-based skipped-step detector.
- An Isolation Forest baseline adapted from Reis and Serodio (2025).

## Core reported results

- Proposed rule-based method: `65/13/102/35` TP/FP/TN/FN, F1 `0.7303`.
- Cook-based skipped-step detector: `80/83/37/20`, F1 `0.6084`.
- Isolation Forest baseline: `6/5/115/94`, F1 `0.1081`.

All three methods use the same participant-grouped held-out folds. The process-mining analysis is descriptive and is not a third classifier.

The verified event-log snapshots contain Sensor `11,584`, Room/Action `3,166` and Room Path `515` rows. Reproduced outputs are written under `software/outputs/reproduction/`.

## Scope

The software is a reproducible research prototype for one controlled smart-apartment dataset. Formal conformance checking, clinical diagnosis, continuous monitoring and automated intervention are outside the study.
