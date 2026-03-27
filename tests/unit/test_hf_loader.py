"""Tests for the Hugging Face dataset loader provider."""

from __future__ import annotations

import pytest

from src.providers.datasets.hf_loader import HFDatasetLoader


def test_hf_loader_to_dataframe_converts_records() -> None:
    """List-like records should be converted into a dataframe-like object."""

    loader = HFDatasetLoader()
    dataset = [
        {"id": 1, "text": "good product"},
        {"id": 2, "text": "too expensive"},
    ]

    frame = loader.to_dataframe(dataset)

    assert not frame.empty
    assert list(frame.columns) == ["id", "text"]
    assert frame.to_dict(orient="records") == dataset


def test_hf_loader_to_dataframe_limit_zero_returns_empty_frame() -> None:
    """A zero limit should return an empty dataframe-like object."""

    loader = HFDatasetLoader()
    dataset = [
        {"id": 1, "text": "good product"},
        {"id": 2, "text": "too expensive"},
    ]

    frame = loader.to_dataframe(dataset, limit=0)

    assert frame.empty
    assert list(frame.columns) == []
    assert frame.to_dict(orient="records") == []


def test_hf_loader_limit_zero_does_not_materialize_dataset() -> None:
    """A zero limit should not iterate or materialize the dataset at all."""

    class ExplodingDataset:
        """Dataset stub that fails if the loader tries to materialize it."""

        @property
        def columns(self) -> list[str]:
            return []

        @property
        def column_names(self) -> list[str]:
            return []

        def __iter__(self):
            raise AssertionError("dataset should not be iterated for limit=0")

        def to_dict(self, orient: str = "records"):
            raise AssertionError("dataset should not be materialized for limit=0")

    loader = HFDatasetLoader()
    frame = loader.to_dataframe(ExplodingDataset(), limit=0)

    assert frame.empty
    assert list(frame.columns) == []


def test_hf_loader_limit_zero_uses_safe_columns_without_materialization() -> None:
    """When available, column metadata should be used without materializing records."""

    class ColumnOnlyDataset:
        """Dataset stub exposing columns metadata only."""

        columns = ["id", "text", "label"]

        def __iter__(self):
            raise AssertionError("dataset should not be iterated for limit=0")

        def to_dict(self, orient: str = "records"):
            raise AssertionError("dataset should not be materialized for limit=0")

    loader = HFDatasetLoader()
    frame = loader.to_dataframe(ColumnOnlyDataset(), limit=0)

    assert frame.empty
    assert list(frame.columns) == ["id", "text", "label"]


def test_hf_loader_preserves_canonical_dataset_id() -> None:
    """Canonical dataset ids should remain unchanged during normalization."""

    loader = HFDatasetLoader()

    assert loader._normalize_dataset_name("owner/name") == "owner/name"


def test_hf_loader_normalizes_standard_dataset_url() -> None:
    """A standard Hugging Face dataset URL should normalize to the canonical dataset id."""

    loader = HFDatasetLoader()

    assert loader._normalize_dataset_name("https://huggingface.co/datasets/owner/name") == "owner/name"


def test_hf_loader_normalizes_tree_and_viewer_urls() -> None:
    """Tree and viewer dataset URLs should also normalize to the canonical dataset id."""

    loader = HFDatasetLoader()

    assert loader._normalize_dataset_name("https://huggingface.co/datasets/owner/name/tree/main") == "owner/name"
    assert loader._normalize_dataset_name("https://huggingface.co/datasets/owner/name/viewer/default/train") == "owner/name"


def test_hf_loader_accepts_www_huggingface_host() -> None:
    """The canonical www host should normalize exactly like huggingface.co."""

    loader = HFDatasetLoader()

    assert loader._normalize_dataset_name("https://www.huggingface.co/datasets/owner/name") == "owner/name"


@pytest.mark.parametrize(
    "url",
    [
        "https://evilhuggingface.co/datasets/owner/name",
        "https://huggingface.co.evil.com/datasets/owner/name",
        "https://huggingface.co@evil.com/datasets/owner/name",
    ],
)
def test_hf_loader_rejects_tricky_non_huggingface_hosts(url: str) -> None:
    """Deceptive hosts must not be treated as Hugging Face dataset URLs."""

    loader = HFDatasetLoader()

    assert loader._normalize_dataset_name(url) == url
