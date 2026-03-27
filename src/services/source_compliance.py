"""Helpers for lightweight source compliance metadata in discovery and approval flows."""

from __future__ import annotations

from urllib.parse import urlparse
from typing import Any


COMPLIANCE_KEYS = {
    "license",
    "license_status",
    "robots_txt_status",
    "robots_txt_url",
    "approval_notes",
}


def normalize_compliance_text(value: Any) -> str:
    """Normalize arbitrary values into compact strings for compliance metadata."""

    if value is None:
        return ""
    return str(value).strip()


def extract_huggingface_license(item: dict[str, Any]) -> str:
    """Extract a dataset license from a Hugging Face search payload when present."""

    direct_candidates = [
        item.get("license"),
        item.get("license_name"),
    ]
    card_data = item.get("cardData")
    if isinstance(card_data, dict):
        direct_candidates.extend(
            [
                card_data.get("license"),
                card_data.get("license_name"),
            ]
        )

    for candidate in direct_candidates:
        normalized = normalize_compliance_text(candidate)
        if normalized and normalized.lower() not in {"unknown", "other", "noassertion"}:
            return normalized

    tags = item.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            normalized_tag = normalize_compliance_text(tag)
            if normalized_tag.lower().startswith("license:"):
                value = normalized_tag.split(":", 1)[1].strip()
                if value:
                    return value

    return ""


def extract_github_license(item: dict[str, Any]) -> str:
    """Extract a repository license from a GitHub search payload when present."""

    license_payload = item.get("license")
    if isinstance(license_payload, dict):
        for key in ("spdx_id", "name", "key"):
            normalized = normalize_compliance_text(license_payload.get(key))
            if normalized and normalized.upper() != "NOASSERTION":
                return normalized

    normalized = normalize_compliance_text(license_payload)
    if normalized and normalized.upper() != "NOASSERTION":
        return normalized

    return ""


def robots_txt_url_for_uri(uri: Any) -> str:
    """Return the origin robots.txt URL for an http(s) URI when applicable."""

    normalized_uri = normalize_compliance_text(uri)
    if not normalized_uri:
        return ""

    parsed = urlparse(normalized_uri)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def build_candidate_compliance_metadata(
    source_type: str,
    uri: Any,
    *,
    metadata: dict[str, Any] | None = None,
    license_label: str = "",
) -> dict[str, str]:
    """Build a lightweight compliance summary for approval-aware discovery."""

    raw_metadata = metadata if isinstance(metadata, dict) else {}
    normalized_license = normalize_compliance_text(license_label) or normalize_compliance_text(raw_metadata.get("license"))
    api_kind = normalize_compliance_text(raw_metadata.get("api_kind")).lower()
    demo_mode = bool(raw_metadata.get("demo_mode")) or normalize_compliance_text(uri).startswith("demo://")

    defaults: dict[str, str]

    if source_type == "hf_dataset":
        defaults = {
            "license": normalized_license or "unknown",
            "license_status": "declared" if normalized_license else "unknown",
            "robots_txt_status": "not_applicable_api",
            "robots_txt_url": "",
            "approval_notes": (
                "dataset license declared in provider metadata"
                if normalized_license
                else "review dataset license before approval"
            ),
        }
    elif source_type == "github_repo":
        defaults = {
            "license": normalized_license or "unknown",
            "license_status": "declared" if normalized_license else "unknown",
            "robots_txt_status": "not_applicable_api",
            "robots_txt_url": "",
            "approval_notes": (
                "repository license declared in GitHub metadata"
                if normalized_license
                else "review repository license and reuse terms before approval"
            ),
        }
    elif source_type == "api":
        if api_kind == "internal":
            defaults = {
                "license": "internal_or_restricted",
                "license_status": "internal_review_required",
                "robots_txt_status": "not_applicable_api",
                "robots_txt_url": "",
                "approval_notes": "internal API candidate requires explicit human approval",
            }
        else:
            defaults = {
                "license": normalized_license or "unknown",
                "license_status": "declared" if normalized_license else "review_required",
                "robots_txt_status": "not_applicable_api",
                "robots_txt_url": "",
                "approval_notes": "verify API terms, quotas and usage policy before approval",
            }
    elif source_type == "scrape":
        if demo_mode:
            defaults = {
                "license": "offline_demo_fixture",
                "license_status": "demo_fixture",
                "robots_txt_status": "not_applicable_local_demo",
                "robots_txt_url": "",
                "approval_notes": "offline demo fixture, no external site access",
            }
        else:
            robots_url = robots_txt_url_for_uri(uri)
            if robots_url:
                defaults = {
                    "license": normalized_license or "unknown",
                    "license_status": "declared" if normalized_license else "review_required",
                    "robots_txt_status": "review_required",
                    "robots_txt_url": robots_url,
                    "approval_notes": "check robots.txt and site terms before enabling scraping",
                }
            else:
                defaults = {
                    "license": normalized_license or "unknown",
                    "license_status": "declared" if normalized_license else "review_required",
                    "robots_txt_status": "not_applicable_local_file",
                    "robots_txt_url": "",
                    "approval_notes": "local HTML/source artifact",
                }
    else:
        defaults = {
            "license": normalized_license or "unknown",
            "license_status": "unknown",
            "robots_txt_status": "unknown",
            "robots_txt_url": "",
            "approval_notes": "review source usage policy before approval",
        }

    for key in COMPLIANCE_KEYS:
        normalized_value = normalize_compliance_text(raw_metadata.get(key))
        if normalized_value:
            defaults[key] = normalized_value

    return defaults
