from __future__ import annotations

import argparse
from pathlib import Path

from sensor2disco.config import load_config
from sensor2disco.experiments import (
    run_cook_evaluation,
    run_reis_evaluation,
    run_rule_based_evaluation,
)
from sensor2disco.submission_pipeline import run_submission_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform smart home sensor logs into Disco-compatible event logs."
    )
    parser.add_argument(
        "command",
        choices=["run", "evaluate-rules", "evaluate-cook", "evaluate-reis-if"],
        help="Generate event logs or run one of the three dissertation evaluations.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a JSON-compatible YAML config file. When omitted, the "
            "submission configuration for the selected command is used."
        ),
    )
    args = parser.parse_args()

    config_name = {
        "run": "casas_submission.yaml",
        "evaluate-rules": "rule_based_method_submission.yaml",
        "evaluate-cook": "cook_skipped_step_submission.yaml",
        "evaluate-reis-if": "reis_isolation_forest_submission.yaml",
    }[args.command]
    config_path = (
        Path(args.config)
        if args.config
        else Path(__file__).parent / "config" / config_name
    )
    config = load_config(config_path)
    if args.command == "evaluate-rules":
        outputs = run_rule_based_evaluation(config)
    elif args.command == "evaluate-cook":
        outputs = run_cook_evaluation(config)
    elif args.command == "evaluate-reis-if":
        outputs = run_reis_evaluation(config)
    else:
        outputs = run_submission_pipeline(config)

    print(f"Sensor2Disco {args.command} completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
