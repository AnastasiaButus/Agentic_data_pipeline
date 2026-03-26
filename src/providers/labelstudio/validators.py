"""Validation helpers for Label Studio task payloads."""

from __future__ import annotations

from typing import Any


def validate_labelstudio_tasks(tasks: list[dict[str, Any]]) -> None:
    """Validate the minimal Label Studio task structure used by this step."""

    if not isinstance(tasks, list):
        raise ValueError("Label Studio tasks must be provided as a list")

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {index} must be a mapping")
        if "data" not in task:
            raise ValueError(f"Task at index {index} is missing data")
        if not isinstance(task["data"], dict):
            raise ValueError(f"Task at index {index} must contain a data mapping")
        if "predictions" not in task:
            raise ValueError(f"Task at index {index} is missing predictions")
        if not isinstance(task["predictions"], list):
            raise ValueError(f"Task at index {index} must contain a predictions list")

        for pred_index, prediction in enumerate(task["predictions"]):
            if not isinstance(prediction, dict):
                raise ValueError(f"Prediction at task {index}, index {pred_index} must be a mapping")
            if "result" not in prediction:
                raise ValueError(f"Prediction at task {index}, index {pred_index} is missing result")
            if not isinstance(prediction["result"], list):
                raise ValueError(f"Prediction at task {index}, index {pred_index} must contain a result list")