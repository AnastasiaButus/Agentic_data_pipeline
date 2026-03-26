"""Baseline model builders for the active learning loop."""

from __future__ import annotations

from typing import Iterable

try:  # pragma: no cover - exercised when sklearn is installed in the runtime.
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover - fallback keeps local/offline tests runnable.
    DummyClassifier = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]

if LogisticRegression is None:
    import math
    import random
    from collections import defaultdict


class SimpleLogisticRegression:
    """A tiny multiclass logistic-regression baseline implemented without external deps."""

    def __init__(self, random_state: int | None = None, max_iter: int = 1000) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.classes_: list[str] = []
        self._weights: list[dict[str, float]] = []
        self._bias: list[float] = []
        self._constant_class: str | None = None

    def fit(self, X: Iterable[dict[str, float]], y: Iterable[object]) -> "SimpleLogisticRegression":
        """Fit a softmax regression model on sparse feature dictionaries."""

        features = [dict(row) for row in X]
        targets = [str(label) for label in y]
        self.classes_ = sorted(set(targets))
        if not self.classes_:
            self._constant_class = None
            self._weights = []
            self._bias = []
            return self

        if len(self.classes_) == 1:
            self._constant_class = self.classes_[0]
            self._weights = []
            self._bias = []
            return self

        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        self._weights = [defaultdict(float) for _ in self.classes_]
        self._bias = [0.0 for _ in self.classes_]

        order = list(range(len(features)))
        rng = random.Random(self.random_state)

        for epoch in range(self.max_iter):
            if len(order) > 1:
                rng.shuffle(order)
            learning_rate = 0.5 / math.sqrt(epoch + 1)

            for row_index in order:
                x = features[row_index]
                target_index = class_to_index[targets[row_index]]
                scores = self._scores(x)
                probabilities = self._softmax(scores)

                for class_index, probability in enumerate(probabilities):
                    error = probability - (1.0 if class_index == target_index else 0.0)
                    self._bias[class_index] -= learning_rate * error
                    weights = self._weights[class_index]
                    for feature, value in x.items():
                        weights[feature] -= learning_rate * error * value

        return self

    def predict_proba(self, X: Iterable[dict[str, float]]) -> list[list[float]]:
        """Predict class probabilities for sparse feature dictionaries."""

        if self._constant_class is not None:
            return [[1.0] for _ in X]
        if not self.classes_:
            return []

        return [self._softmax(self._scores(dict(row))) for row in X]

    def predict(self, X: Iterable[dict[str, float]]) -> list[str]:
        """Predict the most likely class for each sparse feature dictionary."""

        if self._constant_class is not None:
            return [self._constant_class for _ in X]

        predictions: list[str] = []
        for row in X:
            probabilities = self._softmax(self._scores(dict(row)))
            best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
            predictions.append(self.classes_[best_index])
        return predictions

    def _scores(self, x: dict[str, float]) -> list[float]:
        """Compute linear scores for every class."""

        scores: list[float] = []
        for class_index, weights in enumerate(self._weights):
            score = self._bias[class_index]
            for feature, value in x.items():
                score += weights.get(feature, 0.0) * value
            scores.append(score)
        return scores

    def _softmax(self, scores: list[float]) -> list[float]:
        """Convert scores into numerically stable probabilities."""

        if not scores:
            return []

        max_score = max(scores)
        exponentials = [math.exp(score - max_score) for score in scores]
        normalizer = sum(exponentials)
        if normalizer <= 0:
            return [0.0 for _ in scores]
        return [value / normalizer for value in exponentials]


def build_logreg_model(
    random_state: int | None = None,
    max_iter: int = 1000,
) -> object:
    """Build a multiclass-safe logistic-regression baseline."""

    if LogisticRegression is not None:
        return LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver="lbfgs",
        )

    return SimpleLogisticRegression(random_state=random_state, max_iter=max_iter)