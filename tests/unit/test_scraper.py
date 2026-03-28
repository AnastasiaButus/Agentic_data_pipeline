"""Unit tests for selector-based web scraping helpers."""

from __future__ import annotations

from src.providers.web.scraper import parse_selector_blocks, scrape_url


class FakeResponse:
    """Simple response double for remote HTML fetching tests."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.raise_called = False

    def raise_for_status(self) -> None:
        self.raise_called = True


def test_parse_selector_blocks_extracts_text_and_rating() -> None:
    """Selector-based parsing should build row dictionaries from matched HTML elements."""

    html = (
        "<html><body>"
        '<article class="review-card" data-text="Great product" data-rating="5" data-title="Protein Powder">Loved it</article>'
        '<article class="review-card" data-text="Too sweet" data-rating="2" data-category="supplements">Too sweet for me</article>'
        "</body></html>"
    )

    frame = parse_selector_blocks(html, ".review-card")
    rows = frame.to_dict(orient="records")

    assert rows[0]["text"] == "Great product"
    assert rows[0]["rating"] == 5
    assert rows[0]["title"] == "Protein Powder"
    assert rows[1]["text"] == "Too sweet"
    assert rows[1]["category"] == "supplements"


def test_scrape_url_fetches_remote_html_and_applies_selector(monkeypatch) -> None:
    """Remote scraping should fetch HTML first and then parse the selected elements."""

    captured: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse(
            "<html><body>"
            '<div class="review" data-text="Remote review" data-rating="4">Nice</div>'
            "</body></html>"
        )

    monkeypatch.setattr("src.providers.web.scraper.requests.get", fake_get)

    frame = scrape_url(
        "https://example.com/reviews",
        ".review",
        headers={"User-Agent": "pipeline-test"},
        timeout=9,
    )
    rows = frame.to_dict(orient="records")

    assert rows == [{"text": "Remote review", "rating": 4, "content": "Nice"}]
    assert captured == {
        "url": "https://example.com/reviews",
        "kwargs": {
            "headers": {"User-Agent": "pipeline-test"},
            "timeout": 9,
        },
    }
