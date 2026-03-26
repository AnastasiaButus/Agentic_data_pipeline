"""Hugging Face dataset loader with a small, test-friendly dataframe adapter."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Iterable


try:
    datasets = import_module("datasets")
except ModuleNotFoundError:
    class _DatasetsShim:
        """Fallback shim so tests can monkeypatch the datasets loader without the dependency."""

        def load_dataset(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("datasets is required to load Hugging Face datasets")

    datasets = _DatasetsShim()


@dataclass(slots=True)
class SimpleDataFrame:
    """Minimal dataframe-like object used when pandas is unavailable."""

    _rows: list[dict[str, Any]]
    _columns: list[str]

    @property
    def empty(self) -> bool:
        """Return whether the frame has any rows."""

        return not self._rows

    @property
    def columns(self) -> list[str]:
        """Return the column labels."""

        return list(self._columns)

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        """Return row records using the standard records orientation."""

        if orient != "records":
            raise ValueError("SimpleDataFrame only supports records orientation")
        return [dict(row) for row in self._rows]

    def __len__(self) -> int:
        return len(self._rows)


class HFDatasetLoader:
    """Load Hugging Face datasets and adapt them to tabular structures."""

    def load(self, dataset_name: str, split: str = "train", streaming: bool = False) -> Any:
        """Load a dataset using the local datasets backend."""

        return datasets.load_dataset(dataset_name, split=split, streaming=streaming)

    def to_dataframe(self, dataset: Any, limit: int | None = None) -> Any:
        """Convert list-like or tabular-like dataset inputs into a dataframe-like object."""

        if limit == 0:
            columns = self._columns_for_empty_frame(dataset)
            return self._build_frame([], columns)

        records = self._extract_records(dataset, limit=limit)
        columns = self._extract_columns(dataset, records)
        return self._build_frame(records, columns)

    def _extract_records(self, dataset: Any, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Materialize records from list-like or tabular-like inputs."""

        if hasattr(dataset, "to_dict"):
            try:
                records = list(dataset.to_dict(orient="records"))
            except TypeError:
                records = list(dataset.to_dict())
        elif isinstance(dataset, dict):
            records = [dict(dataset)]
        else:
            records = []
            for index, item in enumerate(dataset if isinstance(dataset, Iterable) else [dataset]):
                if limit is not None and index >= limit:
                    break
                if isinstance(item, dict):
                    records.append(dict(item))
                else:
                    records.append(dict(item))

        if limit is not None:
            return records[:limit]
        return records

    def _extract_columns(self, dataset: Any, records: list[dict[str, Any]] | None = None) -> list[str]:
        """Infer column names from the dataset or from the materialized records."""

        columns = list(getattr(dataset, "columns", []))
        if columns:
            return columns

        if records is None:
            records = self._extract_records(dataset)

        if records:
            return list(records[0].keys())
        return []

    def _columns_for_empty_frame(self, dataset: Any) -> list[str]:
        """Infer columns without forcing dataset materialization."""

        columns = list(getattr(dataset, "columns", []))
        if columns:
            return columns

        column_names = getattr(dataset, "column_names", None)
        if column_names:
            return list(column_names)

        return []

    def _build_frame(self, records: list[dict[str, Any]], columns: list[str]) -> Any:
        """Build a pandas frame when available, otherwise use the local fallback."""

        try:
            import pandas as pd  # type: ignore[import-not-found]

            return pd.DataFrame(records, columns=columns or None)
        except Exception:
            return SimpleDataFrame(records, columns)
