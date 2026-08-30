# Reproducing the dissertation results

These instructions regenerate the three activity-only event logs and the three held-out comparisons reported in the dissertation.

## 1. Download the data

Download the two archives identified in `DATA_SOURCE.md`.

- Extract exactly 120 no-error CSV files to `adl_noerror/`.
- Extract exactly 100 scripted-error CSV files to `adl_error/`.
- Do not rename or edit the CSV files.
- Do not place ZIP files in the data folders.

## 2. Create the Python environment

Python 3.11 or newer is required. Run the following commands from `software/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Generate the event logs

```powershell
.\.venv\Scripts\python.exe sensor2disco.py run
```

The generated logs are written under `software/outputs/reproduction/event_logs/event_logs/`. Expected row counts are Sensor `11,584`, Room/Action `3,166` and Room Path `515`.

## 4. Run the rule-based method

```powershell
.\.venv\Scripts\python.exe sensor2disco.py evaluate-rules
```

## 5. Run the Cook-based detector

```powershell
.\.venv\Scripts\python.exe sensor2disco.py evaluate-cook
```

## 6. Run the Isolation Forest baseline

```powershell
.\.venv\Scripts\python.exe sensor2disco.py evaluate-reis-if
```

## 7. Inspect the logs in Disco

Import a generated `room_action_event_log.csv` or `room_path_event_log.csv` into Fluxicon Disco. Assign `Case ID` as the case, `Activity` as the activity and `Timestamp` as the timestamp. Use `Condition` and `Task ID` only as filters.

Generated outputs are written under `software/outputs/reproduction/`.
