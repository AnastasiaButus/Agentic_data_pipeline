"""Tests for the DataCollectionAgent collect-stage behavior."""

from __future__ import annotations

from pathlib import Path

from src.agents.data_collection_agent import DataCollectionAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.domain import SourceCandidate


class FakeRegistry:
    """Capture raw artifact writes without touching the filesystem."""

    def __init__(self) -> None:
        self.saved: tuple[str, object] | None = None

    def save_dataframe(self, path: str | Path, df: object) -> Path:
        self.saved = (str(path), df)
        return Path(path)


class StubHFLoader:
    """Return deterministic tabular data for hf_dataset collection tests."""

    def __init__(self) -> None:
        self.loaded_dataset_names: list[str] = []

    def load(self, dataset_name: str, split: str = "train", streaming: bool = False) -> list[dict[str, object]]:
        self.loaded_dataset_names.append(dataset_name)
        return [
            {"text": "Great product", "rating": 5, "product_name": "Protein Powder"},
            {"text": "Great product", "rating": 5, "product_name": "Protein Powder"},
        ]

    def to_dataframe(self, dataset: list[dict[str, object]]) -> Any:
        return _Frame(dataset)


class StubNormalizer:
    """Return a canonical frame-like object while capturing the merge input."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def normalize_reviews(self, df: object, source_name: str, source_type: str) -> Any:
        self.calls.append(df)
        if _is_empty_frame(df):
            return _Frame([], columns=["id", "source", "text", "label", "rating", "created_at", "split", "meta_json"])
        return _Frame(
            [
                {"id": "1", "source": source_name, "text": "Great product", "label": None, "rating": 5, "created_at": "now", "split": None, "meta_json": "{}"}
            ],
            columns=["id", "source", "text", "label", "rating", "created_at", "split", "meta_json"],
        )


class _Frame:
    """Tiny dataframe-like helper used by tests to avoid depending on pandas."""

    def __init__(self, rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
        self._rows = [dict(row) for row in rows]
        self._columns = list(columns or (list(rows[0].keys()) if rows else []))

    @property
    def empty(self) -> bool:
        return not self._rows

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
        if orient != "records":
            raise ValueError("Only records orientation is supported")
        if not self._columns:
            return [dict(row) for row in self._rows]
        return [{column: row.get(column) for column in self._columns} for row in self._rows]


def _is_empty_frame(df: object) -> bool:
    """Check emptiness without triggering pandas comparison semantics."""

    if df is None:
        return True
    empty = getattr(df, "empty", None)
    if isinstance(empty, bool):
        return empty
    try:
        return len(df) == 0  # type: ignore[arg-type]
    except Exception:
        return df == []


def _make_context(tmp_path: Path) -> PipelineContext:
    """Build a minimal pipeline context for collection tests."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    return PipelineContext.from_config(config)


def test_data_collection_agent_run_on_hf_dataset_source(tmp_path: Path) -> None:
    """HF dataset sources should flow through collection, merge, and normalization."""

    registry = FakeRegistry()
    agent = DataCollectionAgent(
        _make_context(tmp_path),
        hf_loader=StubHFLoader(),
        normalizer=StubNormalizer(),
        registry=registry,
    )

    result = agent.run([SourceCandidate("hf-1", "hf_dataset", "HF", "https://huggingface.co/datasets/fitness-dataset")])

    assert list(result.columns) == ["id", "source", "text", "label", "rating", "created_at", "split", "meta_json"]
    assert result.to_dict(orient="records")[0]["source"] == "HF"
    assert registry.saved is not None
    assert registry.saved[0] == "data/raw/merged_raw.parquet"
    assert agent.hf_loader.loaded_dataset_names == ["https://huggingface.co/datasets/fitness-dataset"]


def test_hf_collection_failure_is_safe(tmp_path: Path) -> None:
    """HF load failures should fall back to an empty frame instead of crashing collect."""

    class FailingHFLoader(StubHFLoader):
        def load(self, dataset_name: str, split: str = "train", streaming: bool = False) -> list[dict[str, object]]:
            raise RuntimeError("hf unavailable")

    registry = FakeRegistry()
    agent = DataCollectionAgent(
        _make_context(tmp_path),
        hf_loader=FailingHFLoader(),
        normalizer=StubNormalizer(),
        registry=registry,
    )

    result = agent.run([SourceCandidate("hf-1", "hf_dataset", "HF", "https://huggingface.co/datasets/fitness-dataset")])

    assert result.empty
    assert registry.saved is not None
    assert registry.saved[0] == "data/raw/merged_raw.parquet"


def test_merge_skews_multiple_frame_like_objects(tmp_path: Path) -> None:
    """The merge step should combine frame-like inputs and remove safe duplicates."""

    agent = DataCollectionAgent(_make_context(tmp_path), hf_loader=StubHFLoader(), normalizer=StubNormalizer(), registry=FakeRegistry())
    merged = agent.merge([
        _Frame([{"text": "A", "rating": 1}]),
        _Frame([{"text": "A", "rating": 1}]),
        _Frame([{"text": "B", "rating": 2}]),
    ])

    rows = merged.to_dict(orient="records")
    assert len(rows) == 2


def test_scrape_source_converts_to_frame_like_object(tmp_path: Path) -> None:
    """Local HTML scraping should parse review blocks into a frame-like object."""

    from src.providers.web.scraper import parse_review_blocks

    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "sample_scraped_html.html"
    html = fixture_path.read_text(encoding="utf-8")

    frame = parse_review_blocks(html)

    assert not frame.empty
    assert frame.to_dict(orient="records")[0]["text"] == "Great product"


def test_empty_sources_return_empty_normalized_frame(tmp_path: Path) -> None:
    """An empty source list should return an empty canonical frame."""

    registry = FakeRegistry()
    agent = DataCollectionAgent(
        _make_context(tmp_path),
        hf_loader=StubHFLoader(),
        normalizer=StubNormalizer(),
        registry=registry,
    )

    result = agent.run([])

    assert result.empty
    assert list(result.columns) == ["id", "source", "text", "label", "rating", "created_at", "split", "meta_json"]
    assert registry.saved is not None


def test_empty_source_does_not_break_pipeline(tmp_path: Path) -> None:
    """A source that produces an empty frame should be skipped safely."""

    class EmptyHFLoader(StubHFLoader):
        def to_dataframe(self, dataset: list[dict[str, object]]) -> Any:
            return _Frame([])

    agent = DataCollectionAgent(
        _make_context(tmp_path),
        hf_loader=EmptyHFLoader(),
        normalizer=StubNormalizer(),
        registry=FakeRegistry(),
    )

    result = agent.run([SourceCandidate("hf-1", "hf_dataset", "HF", "fitness-dataset")])

    assert list(result.columns) == ["id", "source", "text", "label", "rating", "created_at", "split", "meta_json"]


def test_html_without_review_blocks_returns_empty_frame(tmp_path: Path) -> None:
    """HTML without review blocks should produce an empty frame-like object."""

    from src.providers.web.scraper import parse_review_blocks

    frame = parse_review_blocks("<html><body><p>No reviews here</p></body></html>")

    assert frame.empty


def test_api_source_is_ignored_without_crashing(tmp_path: Path) -> None:
    """API sources should be supported as a skip path for this step."""

    agent = DataCollectionAgent(
        _make_context(tmp_path),
        hf_loader=StubHFLoader(),
        normalizer=StubNormalizer(),
        registry=FakeRegistry(),
    )

    result = agent.run([SourceCandidate("api-1", "api", "API", "https://example.com/api")])

    assert list(result.columns) == ["id", "source", "text", "label", "rating", "created_at", "split", "meta_json"]
    assert result.empty


def test_data_collection_agent_is_compatible_with_base_agent(tmp_path: Path) -> None:
    """The collect agent should still be constructible with the shared base-agent contract."""

    agent = DataCollectionAgent(
        _make_context(tmp_path),
        hf_loader=StubHFLoader(),
        normalizer=StubNormalizer(),
        registry=FakeRegistry(),
    )

    assert agent.name == "DataCollectionAgent"
