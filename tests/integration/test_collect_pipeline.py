"""Integration coverage for the collect stage and raw artifact persistence."""

from __future__ import annotations

from pathlib import Path

from src.agents.data_collection_agent import DataCollectionAgent
from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.domain import SourceCandidate


class StubRegistry:
    """Record saved raw artifacts without performing filesystem assertions in the test body."""

    def __init__(self) -> None:
        self.saved_paths: list[str] = []

    def save_dataframe(self, path: str | Path, df: object) -> Path:
        self.saved_paths.append(str(path))
        return Path(path)


class StubHFLoader:
    """Return deterministic HF records for the collect integration test."""

    def load(self, dataset_name: str, split: str = "train", streaming: bool = False) -> list[dict[str, object]]:
        return [{"text": "Great product", "rating": 5, "product_name": "Protein Powder"}]

    def to_dataframe(self, dataset: list[dict[str, object]]) -> Any:
        return _Frame(dataset)


class _Frame:
    """Tiny dataframe-like helper used by the integration test."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [dict(row) for row in rows]

    @property
    def empty(self) -> bool:
        return not self._rows

    @property
    def columns(self) -> list[str]:
        return list(self._rows[0].keys()) if self._rows else []

    def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
        if orient != "records":
            raise ValueError("Only records orientation is supported")
        return [dict(row) for row in self._rows]


def _make_context(tmp_path: Path) -> PipelineContext:
    """Build a minimal pipeline context for the integration test."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    return PipelineContext.from_config(config)


def test_collect_pipeline_hf_and_scrape_sources_return_canonical_schema(tmp_path: Path) -> None:
    """The collect pipeline should return the canonical schema and persist the merged raw artifact."""

    from src.providers.web.scraper import parse_review_blocks

    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "sample_scraped_html.html"
    html = fixture_path.read_text(encoding="utf-8")

    class HtmlScraper:
        def __call__(self, html_text: str) -> Any:
            return parse_review_blocks(html_text)

    agent = DataCollectionAgent(
        _make_context(tmp_path),
        hf_loader=StubHFLoader(),
        scraper=HtmlScraper(),
        registry=StubRegistry(),
    )

    result = agent.run(
        [
            SourceCandidate("hf-1", "hf_dataset", "HF", "https://huggingface.co/datasets/fitness-dataset"),
            SourceCandidate("scrape-1", "scrape", "Web", str(tmp_path / "sample.html"), metadata={"html": html}),
        ]
    )

    assert list(result.columns) == ["id", "source", "text", "label", "rating", "created_at", "split", "meta_json"]
    rows = result.to_dict(orient="records")
    assert len(rows) == 3
    assert [row["source"] for row in rows] == ["HF", "Web", "Web"]
    assert [row["text"] for row in rows] == ["Great product", "Great product", "Too sweet"]
    assert agent.registry.saved_paths == ["data/raw/merged_raw.parquet"]
