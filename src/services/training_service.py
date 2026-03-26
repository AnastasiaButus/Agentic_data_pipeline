"""Training helpers for the text-classification baseline used in the pipeline demo."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from src.core.context import PipelineContext
from src.core.exceptions import ValidationError
from src.ml.evaluation import compute_accuracy, compute_macro_f1
from src.ml.models import build_logreg_model
from src.ml.vectorizers import build_tfidf_vectorizer
from src.services.artifact_registry import ArtifactRegistry

try:  # pragma: no cover - exercised when sklearn is installed in the runtime.
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - fallback keeps local/offline tests runnable.
    train_test_split = None  # type: ignore[assignment]


class TrainingService:
    """Train and persist a text baseline over the canonical review contract."""

    def __init__(self, ctx: PipelineContext, registry: ArtifactRegistry | None = None, random_state: int | None = None) -> None:
        """Bind the service to the active context and artifact registry."""

        self.ctx = ctx
        self.registry = registry if registry is not None else ArtifactRegistry(ctx)
        self.random_state = random_state if random_state is not None else getattr(ctx.config.project, "seed", 42)

    def train(self, df: Any) -> tuple[dict[str, str], dict[str, float]]:
        """Train TF-IDF plus LogisticRegression and persist the model artifacts and metrics."""

        records = self._filter_records(self._to_records(df))
        if not records:
            raise ValidationError("Training requires non-empty data with text and effect_label")

        labels = [self._normalize_text(row.get("effect_label")) for row in records]
        if not any(labels):
            raise ValidationError("Training requires effect_label values after filtering")
        if len(set(label for label in labels if label)) < 2:
            raise ValidationError("Training requires at least two effect_label classes")

        train_records, valid_records, test_records = self._split_records(records)

        if not train_records:
            raise ValidationError("Training split is empty")

        vectorizer = build_tfidf_vectorizer()
        model = build_logreg_model(random_state=self.random_state)

        train_texts = [self._normalize_text(row.get("text")) for row in train_records]
        train_targets = [self._normalize_text(row.get("effect_label")) for row in train_records]
        vectorizer_fit = vectorizer.fit(train_texts)
        train_features = vectorizer_fit.transform(train_texts)
        model.fit(train_features, train_targets)

        evaluation_records = test_records or valid_records or train_records
        evaluation_texts = [self._normalize_text(row.get("text")) for row in evaluation_records]
        evaluation_targets = [self._normalize_text(row.get("effect_label")) for row in evaluation_records]
        evaluation_features = vectorizer_fit.transform(evaluation_texts)
        predictions = model.predict(evaluation_features)

        metrics = {
            "accuracy": compute_accuracy(evaluation_targets, predictions),
            "f1": compute_macro_f1(evaluation_targets, predictions),
        }
        artifacts = self._save_artifacts(model, vectorizer_fit, metrics)
        return artifacts, metrics

    def _split_records(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Split records into train/validation/test using a stratified path when possible."""

        if len(records) < 6 or self._min_class_count(records) < 2 or train_test_split is None:
            return self._fallback_split(records)

        texts = [self._normalize_text(row.get("text")) for row in records]
        targets = [self._normalize_text(row.get("effect_label")) for row in records]

        try:
            train_valid_records, test_records = train_test_split(
                records,
                test_size=0.2,
                random_state=self.random_state,
                stratify=targets,
            )
            train_valid_targets = [self._normalize_text(row.get("effect_label")) for row in train_valid_records]
            train_records, valid_records = train_test_split(
                train_valid_records,
                test_size=0.25,
                random_state=self.random_state,
                stratify=train_valid_targets,
            )
            return list(train_records), list(valid_records), list(test_records)
        except Exception:
            return self._fallback_split(records)

    def _fallback_split(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Use a deterministic split for small datasets where stratified splitting is not viable."""

        ordered = sorted(records, key=lambda row: str(row.get("id", "")))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in ordered:
            label = self._normalize_text(row.get("effect_label"))
            grouped.setdefault(label, []).append(row)

        train: list[dict[str, Any]] = []
        used_ids: set[str] = set()

        for label in sorted(grouped):
            if grouped[label]:
                row = grouped[label][0]
                train.append(row)
                used_ids.add(self._record_id(row))

        desired_train_size = max(len(grouped), int(round(len(ordered) * 0.6)))
        for row in ordered:
            if len(train) >= desired_train_size:
                break
            row_id = self._record_id(row)
            if row_id in used_ids:
                continue
            train.append(row)
            used_ids.add(row_id)

        remaining = [row for row in ordered if self._record_id(row) not in used_ids]
        validation_size = 1 if len(remaining) >= 2 else 0
        validation = remaining[:validation_size]
        test = remaining[validation_size:]
        return train, validation, test

    def _save_artifacts(self, model: Any, vectorizer: Any, metrics: dict[str, float]) -> dict[str, str]:
        """Persist the model, vectorizer, and metrics to project-relative artifacts."""

        model_path = self._artifact_path("data/interim/model_artifact.pkl")
        vectorizer_path = self._artifact_path("data/interim/vectorizer_artifact.pkl")
        metrics_path = "data/interim/model_metrics.json"

        model_path.parent.mkdir(parents=True, exist_ok=True)
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        with vectorizer_path.open("wb") as handle:
            pickle.dump(vectorizer, handle)
        self.registry.save_json(metrics_path, metrics)

        return {
            "model_path": str(model_path),
            "vectorizer_path": str(vectorizer_path),
            "metrics_path": str(self._artifact_path(metrics_path)),
        }

    def _artifact_path(self, relative_path: str) -> Path:
        """Resolve a project-relative artifact path without changing registry behavior."""

        if hasattr(self.registry, "_resolve"):
            return Path(self.registry._resolve(relative_path))  # type: ignore[attr-defined]
        root_dir = Path(getattr(self.registry, "root_dir", self.ctx.paths.root_dir))
        return root_dir / Path(relative_path)

    def _to_records(self, df: Any) -> list[dict[str, Any]]:
        """Materialize dataframe-like inputs into row dictionaries."""

        if hasattr(df, "to_dict"):
            try:
                records = df.to_dict(orient="records")
            except TypeError:
                records = df.to_dict()
            if isinstance(records, list):
                return [dict(row) for row in records]
            if isinstance(records, dict):
                columns = list(records.keys())
                row_count = len(records[columns[0]]) if columns else 0
                return [{column: records[column][index] for column in columns} for index in range(row_count)]
            return [dict(row) for row in records]

        if isinstance(df, list):
            return [dict(row) for row in df]

        return []

    def _filter_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only rows with usable text and effect labels."""

        filtered: list[dict[str, Any]] = []
        for row in records:
            text = self._normalize_text(row.get("text"))
            effect_label = self._normalize_text(row.get("effect_label"))
            if text and effect_label:
                filtered.append(dict(row))
        return filtered

    def _normalize_text(self, value: Any) -> str:
        """Normalize text-like values for stable filtering and modeling."""

        if value is None:
            return ""
        return str(value).strip()

    def _record_id(self, row: dict[str, Any]) -> str:
        """Return a stable string id for a record."""

        return self._normalize_text(row.get("id"))

    def _min_class_count(self, records: list[dict[str, Any]]) -> int:
        """Return the size of the smallest label class after filtering."""

        counts: dict[str, int] = {}
        for row in records:
            label = self._normalize_text(row.get("effect_label"))
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
        return min(counts.values()) if counts else 0