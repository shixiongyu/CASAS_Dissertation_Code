from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML configuration file.

    The config file uses a JSON syntax subset so the prototype has no external
    YAML dependency. JSON is valid YAML, so the file keeps the .yaml extension.
    """

    config_path = path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_config_path(config: dict[str, Any], value: str) -> Path:
    return (Path(config["_config_dir"]) / value).resolve()
