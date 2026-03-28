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


class StubAPIClient:
    """Return deterministic JSON payloads for api collection integration coverage."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def fetch_json(
        self,
        endpoint: str,
        *,
        params: object | None = None,
        headers: object | None = None,
        timeout: object | None = None,
    ) -> object:
        self.calls.append(
            {
                "endpoint": endpoint,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return {
            "items": [
                {"review_text": "Balanced energy support", "score": 4, "product_name": "Focus Blend"},
            ]
        }


class StubWebScraper:
    """Return deterministic rows for selector-based scrape integration tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        url: str,
        selector: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        html: str | None = None,
    ) -> Any:
        self.calls.append(
            {
                "url": url,
                "selector": selector,
                "headers": headers,
                "timeout": timeout,
                "html": html,
            }
        )
        return _Frame([{"text": "Selector scrape row", "rating": 3, "title": "Selector Card"}])


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


def test_collect_pipeline_api_source_returns_canonical_schema(tmp_path: Path) -> None:
    """API sources should participate in collection and produce canonical text rows."""

    api_client = StubAPIClient()
    agent = DataCollectionAgent(
        _make_context(tmp_path),
        hf_loader=StubHFLoader(),
        api_client=api_client,
        registry=StubRegistry(),
    )

    result = agent.run(
        [
            SourceCandidate(
                "api-1",
                "api",
                "Review API",
                "https://example.com/api/reviews",
                metadata={
                    "params": {"topic": "fitness supplements"},
                    "field_map": {"text": "review_text", "rating": "score"},
                },
            )
        ]
    )

    rows = result.to_dict(orient="records")

    assert len(rows) == 1
    assert rows[0]["source"] == "Review API"
    assert rows[0]["text"] == "Balanced energy support"
    assert rows[0]["rating"] == 4
    assert api_client.calls == [
        {
            "endpoint": "https://example.com/api/reviews",
            "params": {"topic": "fitness supplements"},
            "headers": None,
            "timeout": None,
        }
    ]


def test_collect_pipeline_selector_scrape_source_returns_canonical_schema(tmp_path: Path) -> None:
    """Selector-based scrape sources should flow through the collect stage and normalization."""

    web_scraper = StubWebScraper()
    agent = DataCollectionAgent(
        _make_context(tmp_path),
        hf_loader=StubHFLoader(),
        web_scraper=web_scraper,
        registry=StubRegistry(),
    )

    result = agent.run(
        [
            SourceCandidate(
                "scrape-1",
                "scrape",
                "Web Selector",
                "https://example.com/reviews",
                metadata={"selector": ".review-card", "headers": {"User-Agent": "test"}, "timeout": 8},
            )
        ]
    )

    rows = result.to_dict(orient="records")

    assert len(rows) == 1
    assert rows[0]["source"] == "Web Selector"
    assert rows[0]["text"] == "Selector scrape row"
    assert rows[0]["rating"] == 3
    assert web_scraper.calls == [
        {
            "url": "https://example.com/reviews",
            "selector": ".review-card",
            "headers": {"User-Agent": "test"},
            "timeout": 8.0,
            "html": None,
        }
    ]
