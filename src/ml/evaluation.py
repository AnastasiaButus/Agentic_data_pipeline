"""Evaluation helpers for the active learning baseline."""

from __future__ import annotations

from typing import Sequence

try:  # pragma: no cover - exercised when sklearn is installed in the runtime.
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:  # pragma: no cover - fallback keeps local/offline tests runnable.
    accuracy_score = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]


def compute_accuracy(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Compute accuracy while staying safe on empty inputs."""

    if len(y_true) == 0:
        return 0.0
    if accuracy_score is not None:
        return float(accuracy_score(y_true, y_pred))

    matches = sum(1 for left, right in zip(y_true, y_pred) if left == right)
    return matches / len(y_true)


def compute_macro_f1(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Compute macro F1 in a deterministic way for binary and multiclass cases."""

    if len(y_true) == 0:
        return 0.0

    if f1_score is not None:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    labels = sorted({str(label) for label in y_true} | {str(label) for label in y_pred})
    if not labels:
        return 0.0

    y_true_str = [str(label) for label in y_true]
    y_pred_str = [str(label) for label in y_pred]

    f1_scores: list[float] = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true_str, y_pred_str) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true_str, y_pred_str) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true_str, y_pred_str) if truth == label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores)