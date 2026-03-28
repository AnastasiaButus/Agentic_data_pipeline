"""Active learning agent for an offline text classification baseline."""

from __future__ import annotations

import random
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.constants import DEFAULT_RANDOM_SEED
from src.ml.evaluation import compute_accuracy, compute_macro_f1
from src.ml.models import build_logreg_model
from src.ml.uncertainty import entropy_sampling, margin_sampling, random_sampling
from src.ml.vectorizers import build_tfidf_vectorizer
from src.services.artifact_registry import ArtifactRegistry

try:  # pragma: no cover - exercised when sklearn is installed in the runtime.
    from sklearn.dummy import DummyClassifier
except ImportError:  # pragma: no cover - fallback keeps local/offline tests runnable.
    DummyClassifier = None  # type: ignore[assignment]


class SimpleFrame:
    """Tiny dataframe-like container used for the offline AL simulation."""

    def __init__(self, records: list[dict[str, Any]] | None = None, columns: list[str] | None = None) -> None:
        self._records = [dict(row) for row in (records or [])]
        self._columns = list(columns or [])
        if not self._columns and self._records:
            self._columns = list(self._records[0].keys())

    @property
    def empty(self) -> bool:
        return not self._records

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def records(self) -> list[dict[str, Any]]:
        """Expose a copy of the underlying row dictionaries for internal helpers and tests."""

        return [dict(row) for row in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def copy(self) -> "SimpleFrame":
        """Return a shallow copy of the frame-like container."""

        return SimpleFrame(self._records, self._columns)

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        """Return rows in records orientation, matching the project convention."""

        if orient != "records":
            raise ValueError("Only records orientation is supported")
        return [dict(row) for row in self._records]

    def take(self, indices: list[int]) -> "SimpleFrame":
        """Select rows by positional indices while keeping the existing column order."""

        selected = [self._records[index] for index in indices]
        return SimpleFrame(selected, self._columns)

    def drop_indices(self, indices: list[int]) -> "SimpleFrame":
        """Drop rows by positional indices while keeping the existing column order."""

        index_set = set(indices)
        remaining = [row for index, row in enumerate(self._records) if index not in index_set]
        return SimpleFrame(remaining, self._columns)

    def extend(self, rows: list[dict[str, Any]]) -> "SimpleFrame":
        """Append rows and preserve the existing schema when possible."""

        combined = self.records + [dict(row) for row in rows]
        columns = list(self._columns)
        if not columns and combined:
            columns = list(combined[0].keys())
        return SimpleFrame(combined, columns)


class ActiveLearningAgent(BaseAgent):
    """Train a baseline text classifier and query uncertain rows for human labeling."""

    feature_column = "text"
    target_column = "effect_label"

    def __init__(self, ctx: Any, registry: ArtifactRegistry | None = None, random_state: int | None = None) -> None:
        """Bind the agent to the execution context and a deterministic random seed."""

        super().__init__(ctx, registry if registry is not None else ArtifactRegistry(ctx))
        self.random_state = random_state if random_state is not None else getattr(ctx.config.project, "seed", DEFAULT_RANDOM_SEED)

    def split_seed_and_pool(self, df: Any, seed_size: int = 50) -> tuple[SimpleFrame, SimpleFrame]:
        """Split a labeled frame into a seed set and a pool for AL simulation."""

        frame = self._to_frame(df)
        usable = self._filter_labeled_records(frame.records)
        if not usable:
            return self._empty_like(frame), self._empty_like(frame)

        if seed_size <= 0:
            return self._empty_like(frame), SimpleFrame(usable, frame.columns)

        if len(usable) <= seed_size:
            return SimpleFrame(usable, frame.columns), self._empty_like(frame)

        shuffled = usable[:]
        random.Random(self.random_state).shuffle(shuffled)

        seed_indices: list[int] = []
        used_indices: set[int] = set()
        class_to_first_index: dict[str, int] = {}

        for index, row in enumerate(shuffled):
            label = str(row.get(self.target_column, "")).strip()
            if label and label not in class_to_first_index:
                class_to_first_index[label] = index

        for label in sorted(class_to_first_index):
            if len(seed_indices) >= seed_size:
                break
            index = class_to_first_index[label]
            if index not in used_indices:
                seed_indices.append(index)
                used_indices.add(index)

        for index in range(len(shuffled)):
            if len(seed_indices) >= seed_size:
                break
            if index not in used_indices:
                seed_indices.append(index)
                used_indices.add(index)

        seed_records = [shuffled[index] for index in seed_indices]
        pool_records = [row for index, row in enumerate(shuffled) if index not in used_indices]
        return SimpleFrame(seed_records, frame.columns), SimpleFrame(pool_records, frame.columns)

    def fit(self, labeled_df: Any) -> dict[str, Any]:
        """Fit the TF-IDF plus logistic-regression baseline on text -> effect_label."""

        frame = self._filter_labeled_records(self._to_frame(labeled_df).records)
        bundle: dict[str, Any] = {
            "vectorizer": build_tfidf_vectorizer(),
            "model": None,
            "classes": [],
            "feature_column": self.feature_column,
            "target_column": self.target_column,
            "random_state": self.random_state,
        }

        if not frame:
            return bundle

        texts = [str(row.get(self.feature_column, "")) for row in frame]
        targets = [str(row.get(self.target_column, "")).strip() for row in frame]
        unique_targets = sorted(set(targets))
        if not texts or not unique_targets:
            return bundle

        vectorizer = build_tfidf_vectorizer()
        features = vectorizer.fit_transform(texts)
        if len(unique_targets) < 2 and DummyClassifier is not None:
            model = DummyClassifier(strategy="most_frequent")
            model.fit(features, targets)
        else:
            model = build_logreg_model(random_state=self.random_state)
            model.fit(features, targets)

        bundle.update(
            {
                "vectorizer": vectorizer,
                "model": model,
                "classes": list(getattr(model, "classes_", unique_targets)),
            }
        )
        return bundle

    def query(self, model_bundle: dict[str, Any], pool_df: Any, strategy: str, batch_size: int) -> list[int]:
        """Select the next batch of pool indices using the requested uncertainty strategy."""

        pool = self._filter_pool_records(self._to_frame(pool_df).records)
        if not pool or batch_size <= 0:
            return []

        strategy_name = strategy.lower().strip()
        if strategy_name == "random":
            return random_sampling(len(pool), batch_size, self.random_state)

        if strategy_name not in {"entropy", "margin"}:
            raise ValueError(f"Unknown active learning strategy: {strategy}")

        model = model_bundle.get("model")
        vectorizer = model_bundle.get("vectorizer")
        if model is None or vectorizer is None or not hasattr(vectorizer, "transform"):
            return random_sampling(len(pool), batch_size, self.random_state)

        texts = [str(row.get(self.feature_column, "")) for row in pool]
        try:
            probabilities = model.predict_proba(vectorizer.transform(texts))
        except Exception:
            return random_sampling(len(pool), batch_size, self.random_state)

        scores = entropy_sampling(probabilities) if strategy_name == "entropy" else margin_sampling(probabilities)
        if not scores:
            return random_sampling(len(pool), batch_size, self.random_state)

        ranked_indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return ranked_indices[: min(len(pool), batch_size)]

    def evaluate(self, model_bundle: dict[str, Any], df: Any) -> dict[str, float]:
        """Evaluate the fitted baseline on a labeled dataframe-like object."""

        frame = self._filter_labeled_records(self._to_frame(df).records)
        if not frame:
            return {"accuracy": 0.0, "f1": 0.0}

        model = model_bundle.get("model")
        vectorizer = model_bundle.get("vectorizer")
        if model is None or vectorizer is None or not hasattr(vectorizer, "transform"):
            return {"accuracy": 0.0, "f1": 0.0}

        texts = [str(row.get(self.feature_column, "")) for row in frame]
        targets = [str(row.get(self.target_column, "")).strip() for row in frame]
        try:
            predictions = model.predict(vectorizer.transform(texts))
        except Exception:
            return {"accuracy": 0.0, "f1": 0.0}

        return {
            "accuracy": compute_accuracy(targets, predictions),
            "f1": compute_macro_f1(targets, predictions),
        }

    def run_cycle(
        self,
        df: Any,
        strategy: str = "entropy",
        seed_size: int = 50,
        n_iterations: int = 5,
        batch_size: int = 20,
    ) -> tuple[list[dict[str, Any]], SimpleFrame]:
        """Run an offline AL simulation using effect_label as the target signal."""

        seed_df, pool_df = self.split_seed_and_pool(df, seed_size=seed_size)
        labeled_df = seed_df.copy()
        pool_frame = pool_df.copy()
        history: list[dict[str, Any]] = []

        if labeled_df.empty and pool_frame.empty:
            return history, labeled_df

        for iteration in range(1, n_iterations + 1):
            model_bundle = self.fit(labeled_df)
            evaluation_frame = pool_frame if not pool_frame.empty else labeled_df
            metrics = self.evaluate(model_bundle, evaluation_frame)
            queried_indices = self.query(model_bundle, pool_frame, strategy=strategy, batch_size=batch_size)

            if queried_indices:
                queried_rows = pool_frame.take(queried_indices)
                labeled_df = labeled_df.extend(queried_rows.records)
                pool_frame = pool_frame.drop_indices(queried_indices)

            history.append(
                {
                    "iteration": iteration,
                    "n_labeled": len(labeled_df),
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                }
            )

            if pool_frame.empty or not queried_indices:
                break

        return history, labeled_df

    def compare_strategies(
        self,
        df: Any,
        strategies: tuple[str, ...] = ("entropy", "random"),
        seed_size: int = 50,
        n_iterations: int = 5,
        batch_size: int = 20,
    ) -> list[dict[str, Any]]:
        """Run the offline AL loop for multiple strategies and return flat comparison rows.

        The method keeps the existing ``run_cycle`` behavior intact by delegating to it once per
        strategy and flattening the per-iteration histories into a single table-friendly list.
        """

        comparison_rows: list[dict[str, Any]] = []
        for strategy in strategies:
            history, _ = self.run_cycle(
                df,
                strategy=strategy,
                seed_size=seed_size,
                n_iterations=n_iterations,
                batch_size=batch_size,
            )
            for row in history:
                comparison_rows.append(
                    {
                        "strategy": strategy,
                        "iteration": row.get("iteration"),
                        "n_labeled": row.get("n_labeled"),
                        "accuracy": row.get("accuracy"),
                        "f1": row.get("f1"),
                    }
                )

        return comparison_rows

    def summarize_strategy_comparison(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize the final AL outcomes for entropy and random in a report-friendly shape."""

        strategies: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            strategy = str(row.get("strategy", "")).strip().lower()
            if not strategy:
                continue
            strategies.setdefault(strategy, []).append(dict(row))

        final_by_strategy: dict[str, dict[str, Any]] = {}
        for strategy, strategy_rows in strategies.items():
            if not strategy_rows:
                continue
            ordered = sorted(
                strategy_rows,
                key=lambda item: (
                    int(item.get("iteration", 0) or 0),
                    int(item.get("n_labeled", 0) or 0),
                ),
            )
            final_by_strategy[strategy] = ordered[-1]

        entropy_final = final_by_strategy.get("entropy")
        random_final = final_by_strategy.get("random")
        delta_accuracy = None
        delta_f1 = None
        if entropy_final and random_final:
            delta_accuracy = float(entropy_final.get("accuracy", 0.0)) - float(random_final.get("accuracy", 0.0))
            delta_f1 = float(entropy_final.get("f1", 0.0)) - float(random_final.get("f1", 0.0))

        best_strategy = ""
        if final_by_strategy:
            best_strategy = max(
                final_by_strategy.items(),
                key=lambda item: (
                    float(item[1].get("f1", 0.0)),
                    float(item[1].get("accuracy", 0.0)),
                    -int(item[1].get("n_labeled", 0) or 0),
                ),
            )[0]

        notes: list[str] = []
        if entropy_final and random_final:
            notes.append("Entropy and random are compared on the same offline text baseline with identical seed, iteration, and batch-size settings.")
            if delta_f1 is not None and delta_f1 > 0:
                notes.append("Entropy finishes with a higher macro-F1 than random for this run.")
            elif delta_f1 is not None and delta_f1 < 0:
                notes.append("Random finishes with a higher macro-F1 than entropy for this run.")
            else:
                notes.append("Entropy and random finish with the same macro-F1 for this run.")
        else:
            notes.append("Not enough strategy rows were produced to compare entropy and random.")

        return {
            "comparison_scope": "entropy_vs_random_active_learning",
            "strategies": sorted(strategies.keys()),
            "rows": rows,
            "final_by_strategy": final_by_strategy,
            "delta_accuracy_entropy_minus_random": delta_accuracy,
            "delta_f1_entropy_minus_random": delta_f1,
            "best_strategy": best_strategy,
            "notes": notes,
        }

    def _filter_labeled_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only rows with usable text and effect labels for the AL baseline."""

        filtered: list[dict[str, Any]] = []
        for row in records:
            text = str(row.get(self.feature_column, "")).strip()
            label = str(row.get(self.target_column, "")).strip()
            if text and label:
                filtered.append(dict(row))
        return filtered

    def _filter_pool_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep pool rows usable for acquisition while allowing offline effect labels to remain hidden."""

        filtered: list[dict[str, Any]] = []
        for row in records:
            text = str(row.get(self.feature_column, "")).strip()
            if text:
                filtered.append(dict(row))
        return filtered

    def _to_frame(self, df: Any) -> SimpleFrame:
        """Materialize dataframe-like inputs into the local frame container."""

        if isinstance(df, SimpleFrame):
            return df.copy()
        if hasattr(df, "to_dict"):
            try:
                records = df.to_dict(orient="records")
            except TypeError:
                records = df.to_dict()
            if isinstance(records, dict):
                records = [dict(zip(records.keys(), values)) for values in zip(*records.values())] if records else []
            return SimpleFrame([dict(row) for row in records])
        return SimpleFrame([dict(row) for row in df])

    def _empty_like(self, frame: SimpleFrame) -> SimpleFrame:
        """Return an empty frame with the same schema when possible."""

        return SimpleFrame([], frame.columns)
