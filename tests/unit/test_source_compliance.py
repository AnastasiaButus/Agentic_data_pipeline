"""Unit tests for source compliance helpers used by discovery and approval."""

from __future__ import annotations

from src.services.source_compliance import (
    build_candidate_compliance_metadata,
    extract_github_license,
    extract_huggingface_license,
    robots_txt_url_for_uri,
)


def test_extract_huggingface_license_reads_carddata_and_tags() -> None:
    """HF license extraction should accept both card metadata and license tags."""

    assert extract_huggingface_license({"cardData": {"license": "apache-2.0"}}) == "apache-2.0"
    assert extract_huggingface_license({"tags": ["text-classification", "license:mit"]}) == "mit"


def test_extract_github_license_reads_structured_payload() -> None:
    """GitHub license extraction should prefer SPDX identifiers when present."""

    assert extract_github_license({"license": {"spdx_id": "MIT", "name": "MIT License"}}) == "MIT"


def test_build_candidate_compliance_metadata_marks_demo_scrape_as_local_fixture() -> None:
    """Offline demo scrape sources should not pretend to be external web scraping."""

    compliance = build_candidate_compliance_metadata(
        "scrape",
        "demo://fitness-supplements",
        metadata={"demo_mode": True},
    )

    assert compliance["license"] == "offline_demo_fixture"
    assert compliance["license_status"] == "demo_fixture"
    assert compliance["robots_txt_status"] == "not_applicable_local_demo"
    assert compliance["robots_txt_url"] == ""


def test_build_candidate_compliance_metadata_builds_robots_url_for_real_web_scrape() -> None:
    """Real web pages should surface a robots.txt URL for human approval review."""

    compliance = build_candidate_compliance_metadata("scrape", "https://example.com/reviews")

    assert compliance["license"] == "unknown"
    assert compliance["license_status"] == "review_required"
    assert compliance["robots_txt_status"] == "review_required"
    assert compliance["robots_txt_url"] == "https://example.com/robots.txt"


def test_robots_txt_url_for_uri_ignores_non_http_targets() -> None:
    """Local paths and demo URIs should not produce fake robots.txt links."""

    assert robots_txt_url_for_uri("demo://fitness-supplements") == ""
    assert robots_txt_url_for_uri("data/raw/source.html") == ""
