"""Reporting helpers for the end-to-end demo pipeline."""

from __future__ import annotations

from collections import Counter
import json
import posixpath
import re
from pathlib import Path
from typing import Any

from src.core.context import PipelineContext
from src.services.artifact_registry import ArtifactRegistry
from src.services.source_compliance import COMPLIANCE_KEYS, build_candidate_compliance_metadata

WORD_CLOUD_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
    "в",
    "во",
    "вот",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "есть",
    "еще",
    "же",
    "за",
    "из",
    "или",
    "их",
    "как",
    "ко",
    "к",
    "ли",
    "мы",
    "на",
    "над",
    "не",
    "но",
    "о",
    "об",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "по",
    "под",
    "после",
    "при",
    "про",
    "с",
    "со",
    "так",
    "то",
    "тоже",
    "у",
    "уже",
    "что",
    "это",
    "эти",
    "этот",
    "я",
}


class ReportingService:
    """Render markdown and HTML reports for each pipeline stage and the final summary."""

    def __init__(self, ctx: PipelineContext, registry: ArtifactRegistry | None = None) -> None:
        """Bind the reporting service to the active context and artifact registry."""

        self.ctx = ctx
        self.registry = registry if registry is not None else ArtifactRegistry(ctx)

    def write_source_report(self, sources: list[Any]) -> str:
        """Write a compact Russian shortlist report for manual approval review."""

        approval_candidates = [self._candidate_to_approval_record(candidate) for candidate in sources]
        self.registry.save_json("data/raw/approval_candidates.json", approval_candidates)

        lines = [
            "# Короткий shortlist источников",
            "",
            "Это список найденных источников для ручного просмотра и одобрения перед следующим шагом pipeline.",
            "",
        ]

        if not sources:
            lines.extend([
                "Кандидаты не найдены.",
                "",
                "Чтобы одобрить источники, добавьте их `source_id` в `data/raw/approved_sources.json`.",
                "Формат файла: JSON list of strings.",
            ])
            path = "reports/source_report.md"
            self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
            return path

        lines.extend([
            "Чтобы одобрить источники, добавьте их `source_id` в `data/raw/approved_sources.json`.",
            "Формат файла: JSON list of strings.",
            "",
        ])

        for index, candidate in enumerate(sources, start=1):
            title = self._normalize_text(getattr(candidate, "title", ""))
            source_id = self._normalize_text(getattr(candidate, "source_id", ""))
            source_type = self._normalize_text(getattr(candidate, "source_type", ""))
            uri = self._normalize_text(getattr(candidate, "uri", ""))
            score = self._format_numeric(getattr(candidate, "score", 0.0))
            metadata = getattr(candidate, "metadata", None)
            compliance = self._source_compliance_payload(candidate, metadata)

            lines.append(f"## Источник {index}")
            lines.append(f"- source_id: {source_id}")
            lines.append(f"- source_type: {source_type}")
            lines.append(f"- title: {title}")
            lines.append(f"- uri: {uri}")
            lines.append(f"- score: {score}")
            lines.append(f"- license: {compliance['license']}")
            lines.append(f"- license_status: {compliance['license_status']}")
            lines.append(f"- robots_txt_status: {compliance['robots_txt_status']}")
            if compliance.get("robots_txt_url"):
                lines.append(f"- robots_txt_url: {compliance['robots_txt_url']}")
            if compliance.get("approval_notes"):
                lines.append(f"- approval_notes: {compliance['approval_notes']}")

            metadata_text = self._format_compact_metadata(metadata)
            if metadata_text:
                lines.append(f"- metadata: {metadata_text}")
            lines.append("")

        path = "reports/source_report.md"
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return path

    def write_source_approval_workspace(
        self,
        sources: list[Any],
        *,
        approved_sources_path: str,
        source_report_path: str,
        online_governance_report_path: str,
        dashboard_path: str,
        final_report_path: str,
        approval_status: str = "",
        approval_gate_status: str = "",
        effective_collection_scope: str = "",
        effective_source_count: Any = None,
    ) -> str:
        """Write an interactive HTML workspace for manual source approval."""

        workspace_path = "reports/source_approval_workspace.html"
        rows = [self._candidate_to_approval_record(candidate) for candidate in sources]

        existing_approved_ids: list[str] = []
        if self.registry.exists(approved_sources_path):
            try:
                payload = self.registry.load_json(approved_sources_path)
            except Exception:
                payload = []
            if isinstance(payload, list):
                existing_approved_ids = [
                    self._normalize_text(item)
                    for item in payload
                    if self._normalize_text(item)
                ]

        existing_approved_set = set(existing_approved_ids)
        editor_rows = [
            {
                "source_id": self._normalize_text(row.get("source_id")),
                "source_type": self._normalize_text(row.get("source_type")),
                "title": self._normalize_text(row.get("title")),
                "uri": self._normalize_text(row.get("uri")),
                "score": self._format_numeric(row.get("score")),
                "license": self._normalize_text(row.get("license")),
                "license_status": self._normalize_text(row.get("license_status")),
                "robots_txt_status": self._normalize_text(row.get("robots_txt_status")),
                "robots_txt_url": self._normalize_text(row.get("robots_txt_url")),
                "approval_notes": self._normalize_text(row.get("approval_notes")),
                "approved": self._normalize_text(row.get("source_id")) in existing_approved_set,
            }
            for row in rows
        ]
        editor_rows_json = json.dumps(editor_rows, ensure_ascii=False).replace("</", "<\\/")
        approved_ids_json = json.dumps(existing_approved_ids, ensure_ascii=False).replace("</", "<\\/")
        approved_sources_download_name = Path(approved_sources_path).name or "approved_sources.json"
        source_type_tags = sorted(
            {self._normalize_text(row.get("source_type")) for row in rows if self._normalize_text(row.get("source_type"))}
        )

        current_status = self._normalize_text(approval_status)
        current_gate_status = self._normalize_text(approval_gate_status)
        current_scope = self._normalize_text(effective_collection_scope)
        current_effective_source_count = self._format_numeric(effective_source_count if effective_source_count is not None else len(rows))

        if not rows:
            status_title = "No source candidates found"
            status_body = (
                "This run did not produce discovery candidates, so there is nothing to approve yet. "
                "Check the config, runtime mode, and online governance notes before the next rerun."
            )
        elif current_status == "applied":
            status_title = "Approval gate is currently active"
            status_body = (
                "approved_sources.json was present for this run, so collection used only the approved subset. "
                "Update the selection here if you want the next rerun to use a different approved scope."
            )
        elif current_status == "applied_empty_subset":
            status_title = "Approval gate produced an empty subset"
            status_body = (
                "approved_sources.json was present, but it matched no discovered source_id values. "
                "Update the selection here before the next rerun if you want collection to receive sources again."
            )
        elif existing_approved_ids:
            status_title = "Approval input already exists"
            status_body = (
                "An existing approved_sources.json was detected. You can keep the current selection, "
                "change it here, and download a refreshed file before the next pipeline rerun."
            )
        else:
            status_title = "Approval file is missing"
            status_body = (
                "Select the candidates you want to allow, download approved_sources.json, place it in the expected path, "
                "and rerun the pipeline if you want collection to be constrained to explicit source approval."
            )

        checklist_items = [
            "Review source type, title, and URI before approving a candidate.",
            "Check license, robots status, and approval notes for any compliance warnings.",
            "Select only the sources you want to allow in the next rerun.",
            "Download approved_sources.json and place it in the expected input path.",
            "Rerun the pipeline and confirm the approved subset in the dashboard and final report.",
        ]
        checklist_html = "".join(f"<li>{self._escape_html(item)}</li>" for item in checklist_items)

        quick_links = [
            {
                "label": "Open source shortlist",
                "path": source_report_path,
                "description": "Markdown shortlist with the same discovery candidates and compliance fields.",
                "expected": False,
            },
            {
                "label": "Open online governance report",
                "path": online_governance_report_path,
                "description": "Remote provider limits, auth mode, fallback notes and operator guidance.",
                "expected": False,
            },
            {
                "label": "Open operator dashboard",
                "path": dashboard_path,
                "description": "Return to the full run dashboard after the approval decision is made.",
                "expected": False,
            },
            {
                "label": "Open final report",
                "path": final_report_path,
                "description": "Inspect the compact markdown summary for this run.",
                "expected": False,
            },
        ]
        quick_links_html = "".join(
            self._render_dashboard_link_tile(
                workspace_path,
                item["label"],
                item["path"],
                item["description"],
                expected=bool(item["expected"]),
            )
            for item in quick_links
        )

        file_items = [
            {
                "label": "Approval candidates JSON",
                "path": "data/raw/approval_candidates.json",
                "note": "Machine-readable shortlist with compliance metadata for the current run.",
            },
            {
                "label": "Approved sources input",
                "path": approved_sources_path,
                "note": "Expected reviewer output file that constrains the next rerun to approved sources.",
                "expected": True,
            },
            {
                "label": "Source shortlist markdown",
                "path": source_report_path,
                "note": "Human-facing shortlist summary for discovery and approval review.",
            },
            {
                "label": "Online governance report",
                "path": online_governance_report_path,
                "note": "Operational notes about rate limits, fallback behavior, and auth mode.",
            },
        ]
        file_items_html = "".join(
            self._render_dashboard_artifact_item(workspace_path, item)
            for item in file_items
        )

        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Source Approval Workspace</title>
  <style>
    :root {{
      --bg: #f3ede3;
      --panel: rgba(255, 250, 243, 0.94);
      --ink: #1f2a30;
      --muted: #5d6a72;
      --accent: #1f6f78;
      --accent-soft: #cbe5df;
      --warm: #d97706;
      --warm-soft: #fde7c7;
      --line: rgba(31, 42, 48, 0.12);
      --shadow: 0 22px 48px rgba(31, 42, 48, 0.10);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI Variable Text", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(31, 111, 120, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(217, 119, 6, 0.18), transparent 24%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
      min-height: 100vh;
    }}
    .shell {{ max-width: 1240px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(255, 250, 243, 0.98), rgba(244, 237, 227, 0.92));
      border: 1px solid var(--line);
      border-radius: calc(var(--radius) + 4px);
      box-shadow: var(--shadow);
      padding: 28px;
    }}
    .hero-grid {{ display: grid; grid-template-columns: 1.8fr 1fr; gap: 22px; align-items: start; }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.10em;
      font-size: 12px;
      color: var(--accent);
      margin-bottom: 10px;
      font-weight: 700;
    }}
    h1 {{ font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.03; margin: 0 0 14px; }}
    .lede {{ max-width: 760px; line-height: 1.65; color: var(--muted); margin: 0; }}
    .hero-meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }}
    .meta-pill, .tag {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 9px 14px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.8);
    }}
    .tag-wrap {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-top: 24px; }}
    .metric-card, .panel, .link-tile {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: 0 12px 28px rgba(31, 42, 48, 0.06);
    }}
    .metric-card {{ padding: 18px; }}
    .metric-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .metric-value {{ margin-top: 10px; font-size: 1.6rem; font-weight: 700; }}
    .status-card {{
      padding: 18px;
      border-radius: var(--radius);
      background: linear-gradient(135deg, #fff7eb, #ffe7c3);
      border: 1px solid rgba(217, 119, 6, 0.20);
    }}
    .status-card.ready {{
      background: linear-gradient(135deg, #f4fffc, #ddf4ed);
      border-color: rgba(31, 111, 120, 0.20);
    }}
    .status-card h2 {{ margin: 0 0 10px; font-size: 1.05rem; }}
    .status-card p {{ margin: 0 0 10px; line-height: 1.55; }}
    .layout {{ display: grid; grid-template-columns: 1.15fr 1fr; gap: 20px; margin-top: 22px; }}
    .panel {{ padding: 22px; }}
    .panel h2 {{ margin: 0 0 10px; font-size: 1.15rem; }}
    .panel p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .checklist {{ margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.6; }}
    .quick-links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 16px; }}
    .link-tile {{
      display: block;
      text-decoration: none;
      color: inherit;
      padding: 18px;
      transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }}
    .link-tile:hover {{ transform: translateY(-2px); box-shadow: 0 18px 34px rgba(31, 42, 48, 0.10); border-color: rgba(31, 111, 120, 0.26); }}
    .link-tile .title {{ font-size: 1rem; font-weight: 700; }}
    .link-tile .path {{ margin-top: 8px; font-family: "Cascadia Mono", "Consolas", monospace; font-size: 0.82rem; color: var(--accent); word-break: break-all; }}
    .link-tile .description {{ margin-top: 8px; color: var(--muted); line-height: 1.5; font-size: 0.92rem; }}
    .artifact-list {{ display: grid; gap: 10px; margin-top: 16px; }}
    .artifact-item {{ border-top: 1px solid var(--line); padding-top: 10px; }}
    .artifact-item:first-child {{ border-top: none; padding-top: 0; }}
    .artifact-item a {{ color: var(--ink); text-decoration: none; font-weight: 600; }}
    .artifact-item a:hover {{ color: var(--accent); }}
    .artifact-status {{
      display: inline-flex;
      align-items: center;
      margin-top: 6px;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: #edf7f4;
      color: var(--accent);
    }}
    .artifact-status.expected {{ background: var(--warm-soft); color: #9a5a03; }}
    .artifact-note, .muted {{ color: var(--muted); font-size: 0.92rem; line-height: 1.5; }}
    .artifact-path {{ margin-top: 6px; font-family: "Cascadia Mono", "Consolas", monospace; font-size: 0.8rem; color: var(--accent); word-break: break-all; }}
    .editor-actions {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; }}
    .editor-button {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 10px 16px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--ink);
      font-weight: 700;
      cursor: pointer;
      transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
      box-shadow: 0 10px 22px rgba(31, 42, 48, 0.05);
    }}
    .editor-button:hover {{ transform: translateY(-1px); border-color: rgba(31, 111, 120, 0.26); }}
    .editor-button.primary {{
      background: linear-gradient(135deg, #ddf4ed, #f4fffc);
      border-color: rgba(31, 111, 120, 0.22);
      color: var(--accent);
    }}
    .editor-button.warm {{
      background: linear-gradient(135deg, #fff7eb, #ffe7c3);
      border-color: rgba(217, 119, 6, 0.20);
      color: #9a5a03;
    }}
    .editor-note {{
      margin-top: 12px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.92rem;
    }}
    .editor-status {{
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(203, 229, 223, 0.44);
      border: 1px solid rgba(31, 111, 120, 0.12);
      color: var(--ink);
      line-height: 1.55;
    }}
    .editor-status.warning {{
      background: rgba(253, 231, 199, 0.82);
      border-color: rgba(217, 119, 6, 0.18);
      color: #8c4f00;
    }}
    .editor-shell {{
      margin-top: 16px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.82);
    }}
    .editor-table {{ width: 100%; border-collapse: collapse; min-width: 1160px; }}
    .editor-table th, .editor-table td {{
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 0.94rem;
    }}
    .editor-table th {{
      font-size: 0.78rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      background: rgba(31, 111, 120, 0.06);
      position: sticky;
      top: 0;
    }}
    .editor-table tr:last-child td {{ border-bottom: none; }}
    .editor-table input[type="checkbox"] {{
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }}
    .inline-path {{
      display: block;
      margin-top: 6px;
      font-family: "Cascadia Mono", "Consolas", monospace;
      font-size: 0.8rem;
      color: var(--accent);
      word-break: break-all;
    }}
    @media (max-width: 960px) {{
      .hero-grid, .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Source Approval Entry Point</div>
          <h1>Source Approval Workspace</h1>
          <p class="lede">This page turns discovery output into one human-facing approval step: inspect candidate sources, compare license and robots signals, decide what is allowed, and export `approved_sources.json` for the next pipeline rerun.</p>
          <div class="hero-meta">
            <span class="meta-pill">project: {self._escape_html(getattr(self.ctx.config.project, "name", ""))}</span>
            <span class="meta-pill">topic: {self._escape_html(getattr(self.ctx.config.request, "topic", ""))}</span>
            <span class="meta-pill">approved file: {self._escape_html("present" if existing_approved_ids else "missing")}</span>
          </div>
          <div class="metrics">
            <article class="metric-card">
              <div class="metric-label">Candidates</div>
              <div class="metric-value">{self._escape_html(self._format_numeric(len(rows)))}</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Preapproved</div>
              <div class="metric-value">{self._escape_html(self._format_numeric(len(existing_approved_ids)))}</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Remote types</div>
              <div class="metric-value">{self._escape_html(self._format_numeric(len(source_type_tags)))}</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Effective sources</div>
              <div class="metric-value">{self._escape_html(current_effective_source_count)}</div>
            </article>
          </div>
        </div>
        <div class="status-card{' ready' if existing_approved_ids else ''}">
          <h2>{self._escape_html(status_title)}</h2>
          <p>{self._escape_html(status_body)}</p>
          <p><strong>Current approval status:</strong> {self._escape_html(current_status or "unknown")}</p>
          <p><strong>Gate status:</strong> {self._escape_html(current_gate_status or "unknown")}</p>
          <p><strong>Effective collection scope:</strong> {self._escape_html(current_scope or "unknown")}</p>
          <p><strong>Primary action:</strong> download `approved_sources.json` if you want the next rerun to use only explicit source approvals.</p>
          <p><strong>Next step:</strong> place the file in the expected path and rerun the pipeline.</p>
        </div>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <h2>Approval checklist</h2>
        <p>Keep the approval step explicit and easy to audit. The exported file contains only approved `source_id` values, so the next rerun stays simple and reproducible.</p>
        <div class="tag-wrap">{self._render_dashboard_tag_list(source_type_tags, empty_label="not set")}</div>
        <ol class="checklist">{checklist_html}</ol>
      </div>
      <div class="panel">
        <h2>Open first</h2>
        <p>These links are the fastest way to compare discovery candidates, governance notes, and the current run summary.</p>
        <div class="quick-links">{quick_links_html}</div>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <h2>Approval files and status</h2>
        <p>The expected approval input stays visible even when it is still missing, so the operator always sees what should be created before the next rerun.</p>
        <div class="artifact-list">{file_items_html}</div>
      </div>
      <div class="panel">
        <h2>Approval semantics</h2>
        <p>The exported file is a plain JSON list of approved `source_id` values. Keep it intentionally small and explicit: approve the sources you want, leave the rest out, and rerun.</p>
      </div>
    </section>

    <section class="panel">
      <h2>Interactive source approval editor</h2>
      <p>Toggle the candidates you want to allow, then export `approved_sources.json` for the next pipeline rerun.</p>
      <div class="editor-actions">
        <button class="editor-button primary" type="button" id="download-approved-sources">Download approved_sources.json</button>
        <button class="editor-button" type="button" id="select-all-sources">Select all</button>
        <button class="editor-button" type="button" id="clear-approved-sources">Clear selection</button>
        <button class="editor-button warm" type="button" id="copy-approved-path">Copy expected path</button>
      </div>
      <div class="editor-note">
        Expected input path on rerun: <strong>{self._escape_html(approved_sources_path)}</strong>.
        The downloaded file uses the exact filename <strong>{self._escape_html(approved_sources_download_name)}</strong>.
      </div>
      <div class="editor-status" id="source-approval-status">Select the sources you want to allow, then download approved_sources.json for the next rerun.</div>
      <div class="editor-shell">
        <table class="editor-table">
          <thead>
            <tr>
              <th>approve</th>
              <th>source_id</th>
              <th>type</th>
              <th>title</th>
              <th>score</th>
              <th>license</th>
              <th>license_status</th>
              <th>robots_txt_status</th>
              <th>approval_notes</th>
              <th>uri</th>
            </tr>
          </thead>
          <tbody id="source-approval-body"></tbody>
        </table>
      </div>
    </section>
  </div>
  <script>
    const sourceApprovalRows = {editor_rows_json};
    const initiallyApprovedSourceIds = {approved_ids_json};
    const approvedSourcesPath = {json.dumps(approved_sources_path, ensure_ascii=False)};
    const approvedSourcesFilename = {json.dumps(approved_sources_download_name, ensure_ascii=False)};
    const sourceApprovalBody = document.getElementById("source-approval-body");
    const sourceApprovalStatus = document.getElementById("source-approval-status");

    function escapeHtml(value) {{
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function setApprovalStatus(message, tone = "info") {{
      sourceApprovalStatus.textContent = message;
      sourceApprovalStatus.className = tone === "warning" ? "editor-status warning" : "editor-status";
    }}

    function renderSourceApprovalEditor() {{
      if (!sourceApprovalRows.length) {{
        sourceApprovalBody.innerHTML = '<tr><td colspan="10">No discovery candidates are available for approval in this run.</td></tr>';
        setApprovalStatus("This run did not produce discovery candidates, so there is nothing to export yet.", "warning");
        return;
      }}

      sourceApprovalBody.innerHTML = sourceApprovalRows.map((row, index) => {{
        return `
          <tr>
            <td><input type="checkbox" data-index="${{index}}" data-field="approved"${{row.approved ? " checked" : ""}} /></td>
            <td>${{escapeHtml(row.source_id)}}</td>
            <td>${{escapeHtml(row.source_type)}}</td>
            <td>${{escapeHtml(row.title)}}</td>
            <td>${{escapeHtml(row.score)}}</td>
            <td>${{escapeHtml(row.license)}}</td>
            <td>${{escapeHtml(row.license_status)}}</td>
            <td>${{escapeHtml(row.robots_txt_status)}}</td>
            <td>${{escapeHtml(row.approval_notes || "none")}}</td>
            <td><span class="inline-path">${{escapeHtml(row.uri)}}</span></td>
          </tr>
        `;
      }}).join("");
    }}

    function collectApprovedIds() {{
      return sourceApprovalRows
        .filter((row) => row.approved && String(row.source_id || "").trim())
        .map((row) => String(row.source_id).trim());
    }}

    function downloadApprovedSources() {{
      if (!sourceApprovalRows.length) {{
        setApprovalStatus("There are no discovery candidates to export in this run.", "warning");
        return;
      }}

      const approvedIds = collectApprovedIds();
      const payload = JSON.stringify(approvedIds, null, 2);
      const blob = new Blob([payload], {{ type: "application/json;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = approvedSourcesFilename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      if (approvedIds.length) {{
        setApprovalStatus(`Downloaded ${{approvedSourcesFilename}} with ${{approvedIds.length}} approved source_id values. Place it at ${{approvedSourcesPath}} before the next rerun.`);
      }} else {{
        setApprovalStatus(`Downloaded an empty ${{approvedSourcesFilename}}. This will force an explicit empty approved subset on the next rerun.`, "warning");
      }}
    }}

    function selectAllSources() {{
      sourceApprovalRows.forEach((row) => {{
        row.approved = true;
      }});
      renderSourceApprovalEditor();
      setApprovalStatus("All discovery candidates are currently selected for approval.");
    }}

    function clearApprovedSources() {{
      sourceApprovalRows.forEach((row) => {{
        row.approved = false;
      }});
      renderSourceApprovalEditor();
      setApprovalStatus("All discovery candidates were deselected. Downloading now would create an explicit empty subset.", "warning");
    }}

    async function copyApprovedPath() {{
      try {{
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          await navigator.clipboard.writeText(approvedSourcesPath);
          setApprovalStatus(`Copied expected approved sources path: ${{approvedSourcesPath}}`);
          return;
        }}
      }} catch (error) {{
      }}
      setApprovalStatus(`Clipboard access is unavailable here. Expected path: ${{approvedSourcesPath}}`, "warning");
    }}

    sourceApprovalBody.addEventListener("change", (event) => {{
      const target = event.target;
      const index = Number(target?.dataset?.index);
      const field = target?.dataset?.field;
      if (!Number.isInteger(index) || field !== "approved" || !sourceApprovalRows[index]) {{
        return;
      }}
      sourceApprovalRows[index].approved = Boolean(target.checked);
      const approvedIds = collectApprovedIds();
      setApprovalStatus(`Current approval selection contains ${{approvedIds.length}} source_id value(s). Download approved_sources.json when you are ready.`);
    }});

    document.getElementById("download-approved-sources")?.addEventListener("click", downloadApprovedSources);
    document.getElementById("select-all-sources")?.addEventListener("click", selectAllSources);
    document.getElementById("clear-approved-sources")?.addEventListener("click", clearApprovedSources);
    document.getElementById("copy-approved-path")?.addEventListener("click", copyApprovedPath);

    if (initiallyApprovedSourceIds.length) {{
      setApprovalStatus(`Existing approved_sources.json was detected with ${{initiallyApprovedSourceIds.length}} source_id value(s). Update the selection if needed, then download a refreshed file.`);
    }}
    renderSourceApprovalEditor();
  </script>
</body>
</html>"""

        path = Path(workspace_path)
        self.registry.save_text(path, html)
        return str(path)

    def write_runtime_settings_workspace(self, summary: dict[str, Any]) -> str:
        """Write a run-specific settings workspace covering env keys, providers, and approval semantics."""

        dashboard = summary.get("dashboard", {}) if isinstance(summary.get("dashboard"), dict) else {}
        review = summary.get("review", {}) if isinstance(summary.get("review"), dict) else {}
        settings = summary.get("settings", {}) if isinstance(summary.get("settings"), dict) else {}
        approval = summary.get("approval", {}) if isinstance(summary.get("approval"), dict) else {}
        annotation = summary.get("annotation", {}) if isinstance(summary.get("annotation"), dict) else {}
        runtime = summary.get("runtime", {}) if isinstance(summary.get("runtime"), dict) else {}
        workspace_path = self._normalize_text(settings.get("settings_workspace_path")) or "reports/runtime_settings.html"

        dashboard_path = self._normalize_artifact_reference(settings.get("dashboard_path") or dashboard.get("dashboard_path") or "reports/run_dashboard.html")
        final_report_path = self._normalize_artifact_reference(settings.get("final_report_path") or dashboard.get("final_report_path") or "final_report.md")
        review_workspace_path = self._normalize_artifact_reference(settings.get("review_workspace_path") or review.get("review_workspace_path") or "reports/review_workspace.html")
        source_approval_workspace_path = self._normalize_artifact_reference(settings.get("source_approval_workspace_path") or approval.get("source_approval_workspace_path") or "reports/source_approval_workspace.html")
        launcher_path = self._normalize_artifact_reference(settings.get("launcher_path") or "ui/project_launcher.html")

        quick_links = [
            {
                "label": "Open launcher",
                "path": launcher_path,
                "description": "Static pre-run entry point with configs, tasks, and first-run commands.",
            },
            {
                "label": "Open operator dashboard",
                "path": dashboard_path,
                "description": "Return to the main run dashboard after checking keys and runtime semantics.",
            },
            {
                "label": "Open source approval workspace",
                "path": source_approval_workspace_path,
                "description": "Inspect the current source shortlist and export approved_sources.json for the next rerun.",
            },
            {
                "label": "Open review workspace",
                "path": review_workspace_path,
                "description": "Continue the reviewer-facing HITL flow for low-confidence rows.",
            },
            {
                "label": "Open final report",
                "path": final_report_path,
                "description": "Inspect the compact markdown summary for the current run.",
            },
        ]
        quick_links_html = "".join(
            self._render_dashboard_link_tile(workspace_path, item["label"], item["path"], item["description"], expected=False)
            for item in quick_links
        )

        gemini_api_key_present = bool(settings.get("gemini_api_key_present"))
        gemini_api_key_status = self._normalize_text(settings.get("gemini_api_key_status")) or "unknown"
        gemini_note = self._normalize_text(settings.get("gemini_note"))
        github_token_present = bool(settings.get("github_token_present"))
        github_token_status = self._normalize_text(settings.get("github_token_status")) or "unknown"
        github_note = self._normalize_text(settings.get("github_note"))
        requested_provider = self._normalize_text(settings.get("requested_provider")) or "disabled"
        resolved_provider = self._normalize_text(settings.get("resolved_provider")) or "disabled"
        provider_status = self._normalize_text(settings.get("provider_status")) or self._normalize_text(annotation.get("provider_status")) or "unknown"
        requested_runtime_mode = self._normalize_text(settings.get("requested_runtime_mode")) or self._normalize_text(runtime.get("requested_mode")) or "unknown"
        effective_runtime_mode = self._normalize_text(settings.get("effective_runtime_mode")) or self._normalize_text(runtime.get("effective_mode")) or "unknown"
        github_auth_mode = self._normalize_text(settings.get("github_auth_mode")) or "not_used"

        approval_gate_status = self._normalize_text(settings.get("approval_gate_status") or approval.get("approval_gate_status")) or "unknown"
        effective_collection_scope = self._normalize_text(settings.get("effective_collection_scope") or approval.get("effective_collection_scope")) or "unknown"
        effective_source_count = self._format_numeric(settings.get("effective_source_count") if settings.get("effective_source_count") is not None else approval.get("effective_source_count", 0))
        gate_note = self._normalize_text(settings.get("gate_note") or approval.get("gate_note"))
        approval_status = self._normalize_text(approval.get("approval_status")) or "unknown"

        command_blocks = [
            (
                "Set Gemini key in PowerShell",
                '$env:GEMINI_API_KEY="paste-your-key-here"',
            ),
            (
                "Set GitHub token in PowerShell",
                '$env:GITHUB_TOKEN="paste-your-token-here"',
            ),
            (
                "Run the stable offline demo",
                '.\\.venv\\Scripts\\python.exe run_pipeline.py --config configs\\demo_fitness.yaml',
            ),
            (
                "Open source approval workspace",
                'start .\\reports\\source_approval_workspace.html',
            ),
        ]
        command_cards_html = "".join(
            (
                '<article class="command-card">'
                f'<h3>{self._escape_html(title)}</h3>'
                f'<pre id="cmd-{index}">{self._escape_html(command)}</pre>'
                f'<button class="copy-button" type="button" data-copy-target="cmd-{index}">Copy</button>'
                "</article>"
            )
            for index, (title, command) in enumerate(command_blocks, start=1)
        )

        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Runtime Settings Workspace</title>
  <style>
    :root {{
      --bg: #f2ece1;
      --panel: rgba(255, 250, 243, 0.94);
      --ink: #1f2a30;
      --muted: #5d6a72;
      --accent: #1f6f78;
      --accent-soft: #d8eeea;
      --warm: #d97706;
      --warm-soft: #fde7c7;
      --line: rgba(31, 42, 48, 0.12);
      --shadow: 0 20px 46px rgba(31, 42, 48, 0.09);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI Variable Text", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(31, 111, 120, 0.15), transparent 28%),
        radial-gradient(circle at top right, rgba(217, 119, 6, 0.16), transparent 24%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
      min-height: 100vh;
    }}
    .shell {{ max-width: 1220px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero, .panel, .metric-card, .link-tile, .command-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 28px; }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.10em;
      font-size: 12px;
      color: var(--accent);
      margin-bottom: 10px;
      font-weight: 700;
    }}
    h1 {{ font-size: clamp(2rem, 4vw, 3rem); line-height: 1.03; margin: 0 0 14px; }}
    .lede {{ max-width: 800px; line-height: 1.65; color: var(--muted); margin: 0; }}
    .hero-grid, .layout, .metrics, .quick-links, .command-grid {{
      display: grid;
      gap: 18px;
    }}
    .hero-grid {{ grid-template-columns: 1.6fr 1fr; margin-top: 22px; }}
    .layout {{ grid-template-columns: 1fr 1fr; margin-top: 20px; }}
    .metrics {{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-top: 24px; }}
    .metric-card {{ padding: 18px; }}
    .metric-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .metric-value {{ margin-top: 10px; font-size: 1.55rem; font-weight: 700; }}
    .panel {{ padding: 22px; }}
    .panel h2 {{ margin: 0 0 10px; font-size: 1.15rem; }}
    .panel p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .stack {{ display: grid; gap: 12px; margin-top: 14px; }}
    .callout {{
      margin-top: 14px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(216, 238, 234, 0.45);
      border: 1px solid rgba(31, 111, 120, 0.14);
      color: var(--ink);
      line-height: 1.6;
    }}
    .callout.warning {{
      background: rgba(253, 231, 199, 0.72);
      border-color: rgba(217, 119, 6, 0.18);
      color: #8c4f00;
    }}
    .quick-links {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 16px; }}
    .link-tile {{
      display: block;
      text-decoration: none;
      color: inherit;
      padding: 18px;
      transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }}
    .link-tile:hover {{ transform: translateY(-2px); box-shadow: 0 18px 34px rgba(31, 42, 48, 0.10); border-color: rgba(31, 111, 120, 0.26); }}
    .link-tile .title {{ font-size: 1rem; font-weight: 700; }}
    .link-tile .path {{ margin-top: 8px; font-family: "Cascadia Mono", "Consolas", monospace; font-size: 0.82rem; color: var(--accent); word-break: break-all; }}
    .link-tile .description {{ margin-top: 8px; color: var(--muted); line-height: 1.5; font-size: 0.92rem; }}
    .artifact-status {{
      display: inline-flex;
      align-items: center;
      margin-top: 12px;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: #edf7f4;
      color: var(--accent);
    }}
    .artifact-status.expected {{ background: var(--warm-soft); color: #9a5a03; }}
    .command-grid {{ grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin-top: 16px; }}
    .command-card {{ padding: 18px; }}
    .command-card h3 {{ margin: 0 0 10px; font-size: 1rem; }}
    pre {{
      margin: 0;
      padding: 14px;
      border-radius: 18px;
      overflow-x: auto;
      background: #1f2a30;
      color: #f8f6f1;
      font-family: "Cascadia Mono", "Consolas", monospace;
      font-size: 0.84rem;
      line-height: 1.55;
    }}
    .copy-button {{
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 9px 14px;
      background: rgba(255, 255, 255, 0.92);
      color: var(--ink);
      font-weight: 700;
      cursor: pointer;
    }}
    @media (max-width: 960px) {{
      .hero-grid, .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Runtime Settings Workspace</div>
      <h1>Credentials, provider status, and approval semantics</h1>
      <p class="lede">This page explains how the current run resolved its LLM provider, whether environment-backed keys were available, and how source approval affected collection. It is meant to answer the practical operator questions before the next rerun.</p>
      <p class="lede" style="margin-top: 12px;"><strong>Workspace path:</strong> {self._escape_html(workspace_path)}</p>
      <div class="metrics">
        <article class="metric-card">
          <div class="metric-label">Requested mode</div>
          <div class="metric-value">{self._escape_html(requested_runtime_mode)}</div>
        </article>
        <article class="metric-card">
          <div class="metric-label">Effective mode</div>
          <div class="metric-value">{self._escape_html(effective_runtime_mode)}</div>
        </article>
        <article class="metric-card">
          <div class="metric-label">Resolved provider</div>
          <div class="metric-value">{self._escape_html(resolved_provider)}</div>
        </article>
        <article class="metric-card">
          <div class="metric-label">Approval gate</div>
          <div class="metric-value">{self._escape_html(approval_gate_status)}</div>
        </article>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <h2>LLM and Gemini onboarding</h2>
        <p>LLM assistance is intentionally limited to annotation-sensitive work. Everything else stays deterministic and local, so missing keys do not block the offline baseline.</p>
        <div class="stack">
          <p><strong>Requested provider:</strong> {self._escape_html(requested_provider)}</p>
          <p><strong>Resolved provider:</strong> {self._escape_html(resolved_provider)}</p>
          <p><strong>Provider status:</strong> {self._escape_html(provider_status)}</p>
          <p><strong>GEMINI_API_KEY present:</strong> {self._escape_html("yes" if gemini_api_key_present else "no")}</p>
          <p><strong>GEMINI_API_KEY status:</strong> {self._escape_html(gemini_api_key_status)}</p>
        </div>
        <div class="callout{' warning' if 'missing' in gemini_api_key_status else ''}">{self._escape_html(gemini_note)}</div>
      </div>

      <div class="panel">
        <h2>GitHub and online discovery onboarding</h2>
        <p>GitHub lookup remains optional. Without `GITHUB_TOKEN`, the run can still proceed, but repository search becomes more fragile under rate limits.</p>
        <div class="stack">
          <p><strong>GitHub auth mode:</strong> {self._escape_html(github_auth_mode)}</p>
          <p><strong>GitHub search enabled:</strong> {self._escape_html("yes" if settings.get("github_search_enabled") else "no")}</p>
          <p><strong>GITHUB_TOKEN present:</strong> {self._escape_html("yes" if github_token_present else "no")}</p>
          <p><strong>GITHUB_TOKEN status:</strong> {self._escape_html(github_token_status)}</p>
        </div>
        <div class="callout{' warning' if 'missing' in github_token_status else ''}">{self._escape_html(github_note)}</div>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <h2>Source approval gate semantics</h2>
        <p>This block makes the collection contract explicit: either the run stayed open and used the full shortlist, or an existing `approved_sources.json` constrained the input set.</p>
        <div class="stack">
          <p><strong>Approval status:</strong> {self._escape_html(approval_status)}</p>
          <p><strong>Approval gate status:</strong> {self._escape_html(approval_gate_status)}</p>
          <p><strong>Effective collection scope:</strong> {self._escape_html(effective_collection_scope)}</p>
          <p><strong>Effective source count:</strong> {self._escape_html(effective_source_count)}</p>
        </div>
        <div class="callout{' warning' if approval_gate_status == 'restricted_empty_subset' else ''}">{self._escape_html(gate_note)}</div>
      </div>

      <div class="panel">
        <h2>Open next</h2>
        <p>These links connect the static launcher and the run-generated interfaces so the operator can move from setup to approval, review, and reporting without hunting for files.</p>
        <div class="quick-links">{quick_links_html}</div>
      </div>
    </section>

    <section class="panel" style="margin-top: 20px;">
      <h2>Copyable commands</h2>
      <p>Use these PowerShell snippets to set keys for the current shell or to open the next interface directly.</p>
      <div class="command-grid">{command_cards_html}</div>
    </section>
  </div>
  <script>
    async function copyText(value) {{
      if (!value) {{
        return;
      }}
      try {{
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          await navigator.clipboard.writeText(value);
          return;
        }}
      }} catch (error) {{
      }}
      const area = document.createElement("textarea");
      area.value = value;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      document.body.removeChild(area);
    }}

    document.querySelectorAll("[data-copy-target]").forEach((button) => {{
      button.addEventListener("click", async () => {{
        const targetId = button.getAttribute("data-copy-target");
        const target = targetId ? document.getElementById(targetId) : null;
        if (!target) {{
          return;
        }}
        await copyText(target.textContent || "");
        button.textContent = "Copied";
        window.setTimeout(() => {{
          button.textContent = "Copy";
        }}, 1400);
      }});
    }});
  </script>
</body>
</html>"""

        path = Path(workspace_path)
        self.registry.save_text(path, html)
        return str(path)

    def write_quality_report(self, quality_report: Any) -> str:
        """Write a quality summary that is easy to drop into README-style docs."""

        payload = quality_report.as_dict() if hasattr(quality_report, "as_dict") else dict(quality_report)
        lines = ["# Quality Report", ""]
        lines.append(f"- missing: {payload.get('missing', {})}")
        lines.append(f"- duplicates: {payload.get('duplicates', 0)}")
        lines.append(f"- outliers: {payload.get('outliers', {})}")
        lines.append(f"- imbalance: {payload.get('imbalance', {})}")
        lines.append(f"- warnings: {payload.get('warnings', [])}")
        path = "reports/quality_report.md"
        self.registry.save_markdown(path, "\n".join(lines))
        return path

    def write_eda_report(
        self,
        df_like: Any,
        *,
        raw_df_like: Any | None = None,
        quality_report: Any | None = None,
    ) -> str:
        """Write a compact Russian EDA report for the post-quality dataset."""

        summary = self._build_extended_eda_summary(
            df_like,
            raw_df_like=raw_df_like,
            quality_report=quality_report,
        )
        lines = [
            "# EDA-пакет по данным после quality",
            "",
            "Это расширенный честный EDA-отчет по данным, которые реально идут дальше в pipeline.",
            "Он показывает структуру, сравнение raw/cleaned, пропуски и распределения, не подменяя пустые поля выдуманной статистикой.",
            "",
            f"- n_rows: {summary['n_rows']}",
            f"- column_count: {summary['column_count']}",
            f"- columns: {', '.join(summary['columns']) if summary['columns'] else 'нет'}",
            "",
            "## Raw vs cleaned",
        ]

        raw_vs_cleaned = summary["raw_vs_cleaned"]
        if raw_vs_cleaned["available"]:
            lines.extend([
                f"- raw_rows: {raw_vs_cleaned['raw_rows']}",
                f"- cleaned_rows: {raw_vs_cleaned['cleaned_rows']}",
                f"- dropped_rows: {raw_vs_cleaned['dropped_rows']}",
                f"- kept_fraction: {self._format_numeric(raw_vs_cleaned['kept_fraction'])}",
            ])
        else:
            lines.append(f"- {self._describe_absence(raw_vs_cleaned)}")

        lines.extend([
            "",
            "## Дубликаты",
        ])
        duplicate_summary = summary["duplicate_summary"]
        if duplicate_summary["available"]:
            lines.append(f"- duplicate_rows: {duplicate_summary['duplicate_rows']}")
        else:
            lines.append(f"- {self._describe_absence(duplicate_summary)}")

        lines.extend([
            "",
            "## Распределение source",
        ])

        source_distribution = summary["source_distribution"]
        if source_distribution["available"]:
            lines.append(f"- {self._format_count_map(source_distribution['counts'])}")
        else:
            lines.append(f"- {self._describe_absence(source_distribution)}")

        lines.extend([
            "",
            "## Распределение effect_label",
        ])
        effect_label_distribution = summary["effect_label_distribution"]
        if effect_label_distribution["available"]:
            lines.append(f"- {self._format_count_map(effect_label_distribution['counts'])}")
        else:
            lines.append(f"- {self._describe_absence(effect_label_distribution)}")

        lines.extend([
            "",
            "## Сводка rating",
        ])
        rating_summary = summary["rating_summary"]
        if rating_summary["available"]:
            lines.extend([
                f"- valid_count: {rating_summary['valid_count']}",
                f"- missing_or_invalid_count: {rating_summary['missing_or_invalid_count']}",
                f"- min: {self._format_numeric(rating_summary['min'])}",
                f"- max: {self._format_numeric(rating_summary['max'])}",
                f"- mean: {self._format_numeric(rating_summary['mean'])}",
            ])
        else:
            lines.append(f"- {self._describe_absence(rating_summary)}")

        lines.extend([
            "",
            "## Распределение rating",
        ])
        rating_distribution = summary["rating_distribution"]
        if rating_distribution["available"]:
            lines.append(f"- {self._format_count_map(rating_distribution['counts'])}")
        else:
            lines.append(f"- {self._describe_absence(rating_distribution)}")

        lines.extend([
            "",
            "## Длина text",
        ])
        text_length_summary = summary["text_length_summary"]
        if text_length_summary["available"]:
            lines.extend([
                f"- valid_count: {text_length_summary['valid_count']}",
                f"- missing_or_invalid_count: {text_length_summary['missing_or_invalid_count']}",
                f"- min_chars: {self._format_numeric(text_length_summary['min_chars'])}",
                f"- max_chars: {self._format_numeric(text_length_summary['max_chars'])}",
                f"- mean_chars: {self._format_numeric(text_length_summary['mean_chars'])}",
            ])
        else:
            lines.append(f"- {self._describe_absence(text_length_summary)}")

        lines.extend([
            "",
            "## Бакеты длины text",
        ])
        text_length_buckets = summary["text_length_buckets"]
        if text_length_buckets["available"]:
            lines.append(f"- {self._format_count_map(text_length_buckets['counts'])}")
        else:
            lines.append(f"- {self._describe_absence(text_length_buckets)}")

        lines.extend([
            "",
            "## Пропуски по ключевым колонкам",
        ])
        missing_values_summary = summary["missing_values_summary"]
        if missing_values_summary:
            for column_name, column_summary in missing_values_summary.items():
                if column_summary["available"]:
                    lines.append(
                        f"- {column_name}: missing_count={column_summary['missing_count']} / {summary['n_rows']}"
                    )
                else:
                    lines.append(f"- {column_name}: {self._describe_absence(column_summary)}")
        else:
            lines.append("- Ключевые колонки не найдены.")

        quality_warnings = summary["quality_warnings"]
        if quality_warnings:
            lines.extend(["", "## Quality notes"])
            for warning in quality_warnings:
                lines.append(f"- {warning}")

        if summary["notes"]:
            lines.extend(["", "## Гипотезы и примечания"])
            for note in summary["notes"]:
                lines.append(f"- {note}")

        path = Path("reports/eda_report.md")
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return str(path)

    def write_eda_context(
        self,
        df_like: Any,
        *,
        raw_df_like: Any | None = None,
        quality_report: Any | None = None,
    ) -> str:
        """Write the machine-readable helper artifact for the EDA pack."""

        summary = self._build_extended_eda_summary(
            df_like,
            raw_df_like=raw_df_like,
            quality_report=quality_report,
        )
        path = Path("data/interim/eda_context.json")
        self.registry.save_json(path, summary)
        return str(path)

    def write_eda_html_report(
        self,
        df_like: Any,
        *,
        raw_df_like: Any | None = None,
        quality_report: Any | None = None,
    ) -> str:
        """Write a self-contained offline HTML EDA report."""

        summary = self._build_extended_eda_summary(
            df_like,
            raw_df_like=raw_df_like,
            quality_report=quality_report,
        )
        cards = [
            ("Rows", str(summary["n_rows"])),
            ("Columns", str(summary["column_count"])),
            ("Duplicate rows", str(summary["duplicate_summary"].get("duplicate_rows", 0))),
            ("Warnings", str(len(summary["quality_warnings"]))),
        ]
        cards_html = "".join(
            f'<div class="card"><div class="label">{self._escape_html(label)}</div><div class="value">{self._escape_html(value)}</div></div>'
            for label, value in cards
        )
        sections = [
            self._html_metric_block("Columns", self._escape_html(", ".join(summary["columns"]) or "нет")),
            self._html_metric_block("Raw vs cleaned", self._format_raw_vs_cleaned(summary["raw_vs_cleaned"])),
            self._html_metric_block("Source distribution", self._format_distribution_html(summary["source_distribution"])),
            self._html_metric_block("Effect label distribution", self._format_distribution_html(summary["effect_label_distribution"])),
            self._html_metric_block("Rating summary", self._format_summary_dict_html(summary["rating_summary"])),
            self._html_metric_block("Rating distribution", self._format_distribution_html(summary["rating_distribution"])),
            self._html_metric_block("Text length summary", self._format_summary_dict_html(summary["text_length_summary"])),
            self._html_metric_block("Text length buckets", self._format_distribution_html(summary["text_length_buckets"])),
            self._html_metric_block("Missing values", self._format_missing_values_html(summary["missing_values_summary"])),
        ]
        if summary["quality_warnings"]:
            sections.append(
                self._html_metric_block(
                    "Quality notes",
                    "<ul>" + "".join(f"<li>{self._escape_html(note)}</li>" for note in summary["quality_warnings"]) + "</ul>",
                )
            )
        if summary["notes"]:
            sections.append(
                self._html_metric_block(
                    "Hypotheses",
                    "<ul>" + "".join(f"<li>{self._escape_html(note)}</li>" for note in summary["notes"]) + "</ul>",
                )
            )

        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>EDA Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f4f1ea; color: #1f2933; margin: 0; padding: 32px; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    .intro {{ max-width: 860px; line-height: 1.6; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 24px 0; }}
    .card {{ background: #fffaf5; border: 1px solid #d9c8b4; border-radius: 16px; padding: 18px; box-shadow: 0 10px 24px rgba(31, 41, 51, 0.06); }}
    .label {{ font-size: 13px; color: #7a5c3e; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ font-size: 30px; font-weight: 700; margin-top: 8px; }}
    .section-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .section {{ background: white; border-radius: 18px; padding: 20px; border: 1px solid #eadfce; }}
    .section h3 {{ margin: 0 0 12px 0; }}
    .plot {{ margin-top: 28px; background: white; border-radius: 18px; padding: 20px; border: 1px solid #eadfce; }}
    ul {{ margin: 0; padding-left: 20px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ text-align: left; padding: 6px 0; vertical-align: top; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>EDA Report</h1>
    <p class="intro">Offline-friendly HTML-отчёт по данным после quality stage. Он показывает структуру датасета, сравнение raw/cleaned, пропуски, распределения и краткие аналитические выводы.</p>
    <div class="cards">{cards_html}</div>
    <div class="section-grid">{''.join(sections)}</div>
    <div class="plot">
      <h2>Charts</h2>
      {self._build_eda_plotly_html(summary)}
    </div>
  </div>
</body>
</html>"""
        path = Path("reports/eda_report.html")
        self.registry.save_text(path, html)
        return str(path)

    def write_annotation_report(self, df_labeled: Any, annotation_summary: dict[str, Any] | None = None) -> str:
        """Write a monitoring report that emphasizes effect labels and confidence."""

        rows = self._to_records(df_labeled)
        effect_counts: dict[str, int] = {}
        confidence_values: list[float] = []
        low_confidence = 0
        threshold = 0.7
        if annotation_summary and annotation_summary.get("confidence_threshold") is not None:
            threshold = float(annotation_summary["confidence_threshold"])

        for row in rows:
            effect_label = self._normalize_text(row.get("effect_label")) or "other"
            effect_counts[effect_label] = effect_counts.get(effect_label, 0) + 1
            confidence = self._coerce_float(row.get("confidence"))
            confidence_values.append(confidence)
            if confidence < threshold:
                low_confidence += 1

        confidence_mean = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        lines = ["# Annotation Report", "", f"- n_rows: {len(rows)}", f"- effect_label_distribution: {effect_counts}", f"- confidence_mean: {confidence_mean:.3f}", f"- low_confidence: {low_confidence}"]
        path = "reports/annotation_report.md"
        self.registry.save_markdown(path, "\n".join(lines))
        return path

    def write_annotation_trace_report(self, annotation_trace: dict[str, Any]) -> str:
        """Write a Russian trace report for the annotation prompt and parser contract."""

        prompt_contract = annotation_trace.get("prompt_contract", {}) if isinstance(annotation_trace, dict) else {}
        parser_contract = annotation_trace.get("parser_contract", {}) if isinstance(annotation_trace, dict) else {}
        fallback_rows = annotation_trace.get("fallback_rows", []) if isinstance(annotation_trace, dict) else []

        lines = [
            "# Трассировка annotation contract",
            "",
            "Это компактный отчет о prompt contract, ожидаемом формате ответа и fallback-парсинге.",
            "Он нужен, чтобы будущий real LLM можно было подключить без угадывания контракта.",
            "",
            f"- llm_mode: {annotation_trace.get('llm_mode', 'unknown') if isinstance(annotation_trace, dict) else 'unknown'}",
            f"- n_rows: {annotation_trace.get('n_rows', 0) if isinstance(annotation_trace, dict) else 0}",
            f"- n_fallback_rows: {annotation_trace.get('n_fallback_rows', 0) if isinstance(annotation_trace, dict) else 0}",
            "",
            "## Prompt contract",
            f"- language: {prompt_contract.get('language', 'ru')}",
            f"- input_fields: {', '.join(prompt_contract.get('input_fields', [])) or 'нет'}",
            f"- output_fields: {', '.join(prompt_contract.get('output_fields', [])) or 'нет'}",
            f"- sentiment_labels: {', '.join(prompt_contract.get('sentiment_labels', [])) or 'нет'}",
            f"- effect_labels: {', '.join(prompt_contract.get('effect_labels', [])) or 'нет'}",
            "",
            "## Prompt preview",
            prompt_contract.get('prompt_preview', ''),
            "",
            "## Expected output",
            f"- example: {prompt_contract.get('expected_output_example', {})}",
            "",
            "## Parser contract",
            f"- preferred_format: {parser_contract.get('preferred_format', 'json')}",
            f"- accepted_fallbacks: {', '.join(parser_contract.get('accepted_fallbacks', [])) or 'нет'}",
            f"- parse_status_counts: {parser_contract.get('parse_status_counts', {})}",
            f"- fallback_reason_counts: {parser_contract.get('fallback_reason_counts', {})}",
        ]

        if fallback_rows:
            lines.extend(["", "## Fallback samples"])
            for row in fallback_rows:
                lines.append(
                    "- mode: {mode} | status: {status} | reasons: {reasons} | raw_output: {raw_output}".format(
                        mode=self._normalize_text(row.get("mode")),
                        status=self._normalize_text(row.get("parse_status")),
                        reasons=", ".join(row.get("fallback_reasons", []) or []) or "нет",
                        raw_output=self._normalize_text(row.get("raw_output")),
                    )
                )

        path = Path("reports/annotation_trace_report.md")
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return str(path)

    def write_annotation_trace_context(self, annotation_trace: dict[str, Any]) -> str:
        """Write a machine-readable trace artifact for the annotation prompt contract."""

        path = Path("data/interim/annotation_trace.json")
        self.registry.save_json(path, annotation_trace)
        return str(path)

    def write_al_report(self, history: list[dict[str, Any]]) -> str:
        """Write an active-learning report with the per-iteration history."""

        lines = ["# Active Learning Report", ""]
        for row in history:
            lines.append(
                f"- iteration {row.get('iteration')}: n_labeled={row.get('n_labeled')} accuracy={row.get('accuracy'):.3f} f1={row.get('f1'):.3f}"
            )
        path = "reports/al_report.md"
        self.registry.save_markdown(path, "\n".join(lines))
        return path

    def write_al_comparison_report(self, payload: Any) -> str:
        """Write a compact markdown summary for AL strategy comparison rows and final deltas."""

        if isinstance(payload, dict):
            rows = payload.get("rows", []) if isinstance(payload.get("rows"), list) else []
            final_by_strategy = payload.get("final_by_strategy", {}) if isinstance(payload.get("final_by_strategy"), dict) else {}
            strategies = payload.get("strategies", []) if isinstance(payload.get("strategies"), list) else []
            notes = payload.get("notes", []) if isinstance(payload.get("notes"), list) else []
            best_strategy = self._normalize_text(payload.get("best_strategy"))
            delta_accuracy = payload.get("delta_accuracy_entropy_minus_random")
            delta_f1 = payload.get("delta_f1_entropy_minus_random")
        else:
            rows = payload if isinstance(payload, list) else []
            final_by_strategy = {}
            strategies = sorted({self._normalize_text(row.get("strategy")) for row in rows if self._normalize_text(row.get("strategy"))})
            notes = []
            best_strategy = ""
            delta_accuracy = None
            delta_f1 = None

        lines = ["# Active Learning Comparison Report", ""]
        lines.append(f"- comparison_scope: entropy_vs_random_active_learning")
        lines.append(f"- strategies: {', '.join(strategies) if strategies else 'n/a'}")
        lines.append(f"- best_strategy: {best_strategy or 'n/a'}")
        lines.append(
            f"- delta_accuracy_entropy_minus_random: {self._format_numeric(delta_accuracy) if delta_accuracy is not None else 'n/a'}"
        )
        lines.append(
            f"- delta_f1_entropy_minus_random: {self._format_numeric(delta_f1) if delta_f1 is not None else 'n/a'}"
        )

        if final_by_strategy:
            lines.extend(["", "## Final strategy snapshot"])
            for strategy in sorted(final_by_strategy):
                row = final_by_strategy[strategy]
                lines.append(
                    "- {strategy}: iteration={iteration} n_labeled={n_labeled} accuracy={accuracy} f1={f1}".format(
                        strategy=strategy,
                        iteration=row.get("iteration", ""),
                        n_labeled=row.get("n_labeled", ""),
                        accuracy=self._format_numeric(row.get("accuracy")),
                        f1=self._format_numeric(row.get("f1")),
                    )
                )

        lines.extend(["", "## Iteration table"])
        lines.append("| strategy | iteration | n_labeled | accuracy | f1 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                "| {strategy} | {iteration} | {n_labeled} | {accuracy:.3f} | {f1:.3f} |".format(
                    strategy=row.get("strategy", ""),
                    iteration=row.get("iteration", ""),
                    n_labeled=row.get("n_labeled", ""),
                    accuracy=self._coerce_float(row.get("accuracy")),
                    f1=self._coerce_float(row.get("f1")),
                )
            )

        if notes:
            lines.extend(["", "## Notes"])
            for note in notes:
                lines.append(f"- {self._normalize_text(note)}")

        path = "reports/al_comparison_report.md"
        self.registry.save_markdown(path, "\n".join(lines))
        return path

    def write_al_comparison_context(self, payload: dict[str, Any]) -> str:
        """Write the machine-readable AL strategy comparison payload."""

        path = Path("data/interim/al_comparison.json")
        self.registry.save_json(path, payload)
        return str(path)

    def write_review_queue_report(
        self,
        review_queue: Any,
        confidence_threshold: float,
        label_options: list[str],
    ) -> str:
        """Write a Russian review-pack report for human annotation review."""

        rows = self._to_records(review_queue)
        input_queue_path = "data/interim/review_queue.csv"
        corrected_queue_path = "data/interim/review_queue_corrected.csv"

        lines = [
            "# Очередь ручной проверки",
            "",
            "Это очередь примеров для ручной проверки после авторазметки.",
            "",
            "## Текущий этап",
            "",
            "- Этап pipeline: human review / HITL",
            "- Цель: проверить low-confidence примеры до retrain и финального обучения",
            "",
            "## Reviewer guide",
            "",
            f"- Входной файл очереди: {input_queue_path}",
            f"- Порог confidence: {self._format_numeric(confidence_threshold)}",
            f"- Строк в очереди: {len(rows)}",
            f"- Исправленный файл положите сюда: {corrected_queue_path}",
            "- Проверьте поля: id, source, text, label, effect_label, confidence, reviewed_effect_label, review_comment, human_verified",
            f"- Допустимые effect labels: {', '.join(label_options) if label_options else 'не заданы'}",
            "",
        ]

        if not rows:
            lines.extend(
                [
                    "Очередь пуста, ручная проверка не требуется.",
                    "",
                    "## Next step",
                    "",
                    "- Следующий шаг: active learning / training могут использовать текущий reviewed dataset без ручных правок.",
                ]
            )
        else:
            lines.extend(
                [
                    "## To-do reviewer",
                    "",
                    "1. Откройте `data/interim/review_queue.csv`.",
                    "2. Для спорных строк заполните `reviewed_effect_label`.",
                    "3. При необходимости добавьте `review_comment` и выставьте `human_verified=true`.",
                    "4. Сохраните исправленный файл как `data/interim/review_queue_corrected.csv`.",
                    "5. Перезапустите pipeline, чтобы merge применил ручные правки.",
                    "",
                ]
            )
            lines.append("## Примеры для проверки")
            lines.append("")
            for row in rows:
                lines.append(
                    "- id: {id} | source: {source} | effect_label: {effect_label} | confidence: {confidence} | text: {text}".format(
                        id=self._normalize_text(row.get("id")),
                        source=self._normalize_text(row.get("source")),
                        effect_label=self._normalize_text(row.get("effect_label")),
                        confidence=self._format_numeric(row.get("confidence")),
                        text=self._normalize_text(row.get("text")),
                    )
                )
            lines.extend(
                [
                    "",
                    "## Next step",
                    "",
                    "- Следующий шаг: загрузить corrected queue и повторно запустить pipeline для merge -> retrain.",
                ]
            )

        path = Path("reports/review_queue_report.md")
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return str(path)

    def write_review_queue_context(
        self,
        review_queue: Any,
        confidence_threshold: float,
        label_options: list[str],
    ) -> str:
        """Write a machine-readable helper artifact for review tooling."""

        rows = self._to_records(review_queue)
        payload = {
            "confidence_threshold": confidence_threshold,
            "n_rows": len(rows),
            "label_options": list(label_options),
            "input_queue_path": "data/interim/review_queue.csv",
            "expected_corrected_queue_path": "data/interim/review_queue_corrected.csv",
            "review_required": bool(rows),
            "current_stage": "human_review",
            "next_step": "fill_corrected_queue_and_rerun" if rows else "continue_to_active_learning_and_training",
            "review_columns": [
                "id",
                "source",
                "text",
                "label",
                "effect_label",
                "confidence",
                "reviewed_effect_label",
                "review_comment",
                "human_verified",
            ],
        }
        path = Path("data/interim/review_queue_context.json")
        self.registry.save_json(path, payload)
        return str(path)

    def write_review_workspace(
        self,
        review_queue: Any,
        confidence_threshold: float,
        label_options: list[str],
        *,
        review_required: bool,
        corrected_queue_found: bool,
        corrected_queue_path: str,
        review_status: str,
        next_step: str,
        dashboard_path: str,
        final_report_path: str,
        review_queue_report_path: str,
        review_queue_context_path: str,
        review_merge_report_path: str,
        review_merge_context_path: str,
    ) -> str:
        """Write an HTML workspace for the HITL reviewer flow."""

        rows = self._to_records(review_queue)
        workspace_path = "reports/review_workspace.html"
        preview_limit = 12
        review_attention = review_required and not corrected_queue_found
        visible_rows = rows[:preview_limit]

        status_title = (
            "Waiting for reviewer action"
            if review_attention
            else ("Corrected queue detected" if corrected_queue_found else "Review queue is clear")
        )
        status_body = (
            "Low-confidence rows are waiting for human correction before the next merge and retrain."
            if review_attention
            else (
                "This run found `review_queue_corrected.csv`, so reviewer edits can already be merged into the dataset."
                if corrected_queue_found
                else "The current run does not require manual correction, but the reviewer workspace remains available for inspection."
            )
        )
        primary_action = (
            "Fill `reviewed_effect_label`, optionally add `review_comment`, save `review_queue_corrected.csv`, then rerun the pipeline."
            if review_attention
            else (
                "Inspect `review_merge_report.md` and confirm the reviewed labels were applied as expected."
                if corrected_queue_found
                else "Continue with the dashboard, final report, and training artifacts."
            )
        )

        quick_links = [
            {
                "label": "Open operator dashboard",
                "path": dashboard_path,
                "description": "Return to the top-level run dashboard with the full pipeline state.",
                "expected": False,
            },
            {
                "label": "Open review guide",
                "path": review_queue_report_path,
                "description": "Markdown reviewer guide with step-by-step HITL instructions.",
                "expected": False,
            },
            {
                "label": "Open review queue CSV",
                "path": "data/interim/review_queue.csv",
                "description": "Editable queue exported from the low-confidence annotation rows.",
                "expected": False,
            },
            {
                "label": "Open corrected queue CSV",
                "path": corrected_queue_path,
                "description": "Expected reviewer output file that can be fed back into the pipeline.",
                "expected": True,
            },
        ]
        quick_links_html = "".join(
            self._render_dashboard_link_tile(
                workspace_path,
                item["label"],
                item["path"],
                item["description"],
                expected=bool(item["expected"]),
            )
            for item in quick_links
        )

        file_items = [
            {"label": "Operator dashboard", "path": dashboard_path, "note": "Global entry point for the current run."},
            {"label": "Final report", "path": final_report_path, "note": "Compact markdown summary for the full pipeline run."},
            {"label": "Review guide", "path": review_queue_report_path, "note": "Step-by-step markdown guidance for the reviewer."},
            {"label": "Review queue context", "path": review_queue_context_path, "note": "Machine-readable review metadata: threshold, labels, and expected columns."},
            {"label": "Review queue CSV", "path": "data/interim/review_queue.csv", "note": "Input queue that should be inspected and corrected by the reviewer."},
            {"label": "Corrected queue CSV", "path": corrected_queue_path, "note": "Human-edited CSV that the pipeline will merge on the next run.", "expected": True},
            {"label": "Review merge report", "path": review_merge_report_path, "note": "What happened after the corrected queue was applied."},
            {"label": "Review merge context", "path": review_merge_context_path, "note": "Machine-readable merge status for audit and debugging."},
        ]
        file_items_html = "".join(
            self._render_dashboard_artifact_item(workspace_path, item)
            for item in file_items
            if self._normalize_artifact_reference(item.get("path"))
        )

        if review_attention:
            checklist_items = [
                "Open `data/interim/review_queue.csv` or inspect the table preview below.",
                "Set `reviewed_effect_label` only to one of the allowed effect labels.",
                "Optionally add `review_comment` and mark `human_verified=true` for confirmed rows.",
                "Save the corrected file as `data/interim/review_queue_corrected.csv`.",
                "Rerun the pipeline and confirm the result in `review_merge_report.md` and the dashboard.",
            ]
        elif corrected_queue_found:
            checklist_items = [
                "Inspect `review_merge_report.md` to confirm how reviewer edits affected the dataset.",
                "If additional changes are still needed, update `review_queue_corrected.csv` and rerun the pipeline.",
                "Use the dashboard and final report to confirm the post-review training artifacts.",
            ]
        else:
            checklist_items = [
                "Manual review is not required for this run because the low-confidence queue is empty.",
                "Use the dashboard and final report to inspect the completed training and reporting artifacts.",
            ]
        checklist_html = "".join(f"<li>{self._escape_html(item)}</li>" for item in checklist_items)

        if visible_rows:
            table_headers = [
                "id",
                "source",
                "effect_label",
                "confidence",
                "reviewed_effect_label",
                "review_comment",
                "human_verified",
                "text",
            ]
            header_html = "".join(f"<th>{self._escape_html(column)}</th>" for column in table_headers)
            body_html = "".join(
                "<tr>"
                f"<td>{self._escape_html(self._normalize_text(row.get('id')))}</td>"
                f"<td>{self._escape_html(self._normalize_text(row.get('source')))}</td>"
                f"<td>{self._escape_html(self._normalize_text(row.get('effect_label')))}</td>"
                f"<td>{self._escape_html(self._format_numeric(row.get('confidence')))}</td>"
                f"<td>{self._escape_html(self._normalize_text(row.get('reviewed_effect_label')))}</td>"
                f"<td>{self._escape_html(self._truncate_text(row.get('review_comment'), limit=80))}</td>"
                f"<td>{self._escape_html(self._normalize_text(row.get('human_verified')))}</td>"
                f"<td>{self._escape_html(self._truncate_text(row.get('text'), limit=180))}</td>"
                "</tr>"
                for row in visible_rows
            )
            preview_note = (
                f"Showing the first {preview_limit} rows from the review queue preview."
                if len(rows) > preview_limit
                else "The preview shows the current review queue rows for this run."
            )
            queue_preview_html = (
                '<div class="table-shell">'
                '<table class="queue-table">'
                f"<thead><tr>{header_html}</tr></thead>"
                f"<tbody>{body_html}</tbody>"
                "</table>"
                "</div>"
                f'<p class="table-note">{self._escape_html(preview_note)}</p>'
            )
        else:
            queue_preview_html = (
                '<div class="empty-state">'
                "<strong>No rows currently require manual review.</strong>"
                "<p>The low-confidence queue is empty for this run, so the reviewer can use this page mainly as an audit and navigation hub.</p>"
                "</div>"
            )

        def _review_bool(value: Any) -> bool:
            normalized = self._normalize_text(value).lower()
            return normalized in {"true", "1", "yes", "y"} if normalized else bool(value is True)

        editor_rows = [
            {
                "id": self._normalize_text(row.get("id")),
                "source": self._normalize_text(row.get("source")),
                "text": self._normalize_text(row.get("text")),
                "label": self._normalize_text(row.get("label")),
                "effect_label": self._normalize_text(row.get("effect_label")),
                "confidence": self._format_numeric(row.get("confidence")),
                "reviewed_effect_label": self._normalize_text(row.get("reviewed_effect_label")) or self._normalize_text(row.get("effect_label")),
                "review_comment": self._normalize_text(row.get("review_comment")),
                "human_verified": _review_bool(row.get("human_verified")),
            }
            for row in rows
        ]
        editor_rows_json = json.dumps(editor_rows, ensure_ascii=False).replace("</", "<\\/")
        label_options_json = json.dumps([self._normalize_text(label) for label in label_options], ensure_ascii=False).replace("</", "<\\/")
        corrected_queue_download_name = Path(corrected_queue_path).name or "review_queue_corrected.csv"

        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HITL Review Workspace</title>
  <style>
    :root {{
      --bg: #f3ede3;
      --panel: rgba(255, 250, 243, 0.95);
      --ink: #1f2a30;
      --muted: #5d6a72;
      --accent: #1f6f78;
      --warm-soft: #fde7c7;
      --line: rgba(31, 42, 48, 0.12);
      --shadow: 0 20px 44px rgba(31, 42, 48, 0.10);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI Variable Text", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(31, 111, 120, 0.15), transparent 26%),
        radial-gradient(circle at top right, rgba(217, 119, 6, 0.14), transparent 24%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
      min-height: 100vh;
    }}
    .shell {{ max-width: 1260px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero, .panel, .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 28px; position: relative; overflow: hidden; }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -50px -70px auto;
      width: 210px;
      height: 210px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(31, 111, 120, 0.15) 0%, transparent 70%);
      pointer-events: none;
    }}
    .hero-grid {{ display: grid; grid-template-columns: 1.7fr 1fr; gap: 20px; align-items: start; }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.10em;
      font-size: 12px;
      color: var(--accent);
      margin-bottom: 10px;
      font-weight: 700;
    }}
    h1 {{ margin: 0 0 14px; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.03; }}
    .lede {{ margin: 0; color: var(--muted); line-height: 1.65; max-width: 760px; }}
    .hero-meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }}
    .meta-pill, .tag {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 9px 14px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.82);
    }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-top: 24px; }}
    .metric-card {{ padding: 16px; }}
    .metric-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .metric-value {{ margin-top: 10px; font-size: 1.55rem; font-weight: 700; }}
    .status-card {{
      padding: 18px;
      border-radius: var(--radius);
      border: 1px solid transparent;
      background: linear-gradient(135deg, #fff7eb, #ffe7c3);
    }}
    .status-card.ready {{
      background: linear-gradient(135deg, #f4fffc, #ddf4ed);
      border-color: rgba(31, 111, 120, 0.20);
    }}
    .status-card h2 {{ margin: 0 0 10px; font-size: 1.1rem; }}
    .status-card p {{ margin: 0 0 10px; line-height: 1.55; }}
    .layout {{ display: grid; grid-template-columns: 1.15fr 0.95fr; gap: 20px; margin-top: 20px; }}
    .panel {{ padding: 22px; }}
    .panel h2 {{ margin: 0 0 10px; font-size: 1.15rem; }}
    .panel p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .tag-wrap {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .checklist {{ margin: 16px 0 0; padding-left: 20px; color: var(--ink); }}
    .checklist li {{ margin-top: 10px; line-height: 1.55; }}
    .quick-links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-top: 16px; }}
    .link-tile {{
      display: block;
      text-decoration: none;
      color: inherit;
      padding: 18px;
      border-radius: var(--radius);
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 12px 28px rgba(31, 42, 48, 0.06);
      transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }}
    .link-tile:hover {{ transform: translateY(-2px); box-shadow: 0 18px 34px rgba(31, 42, 48, 0.10); border-color: rgba(31, 111, 120, 0.24); }}
    .link-tile .title {{ font-size: 1rem; font-weight: 700; }}
    .link-tile .path {{ margin-top: 8px; font-family: "Cascadia Mono", "Consolas", monospace; font-size: 0.82rem; color: var(--accent); word-break: break-all; }}
    .link-tile .description {{ margin-top: 8px; color: var(--muted); line-height: 1.5; font-size: 0.92rem; }}
    .artifact-list {{ display: grid; gap: 10px; margin-top: 16px; }}
    .artifact-item {{ border-top: 1px solid var(--line); padding-top: 10px; }}
    .artifact-item:first-child {{ border-top: none; padding-top: 0; }}
    .artifact-item a {{ color: var(--ink); text-decoration: none; font-weight: 600; }}
    .artifact-item a:hover {{ color: var(--accent); }}
    .artifact-status {{
      display: inline-flex;
      align-items: center;
      margin-top: 6px;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: #edf7f4;
      color: var(--accent);
    }}
    .artifact-status.expected {{ background: var(--warm-soft); color: #9a5a03; }}
    .artifact-note {{ color: var(--muted); font-size: 0.92rem; line-height: 1.5; }}
    .artifact-path {{ margin-top: 6px; font-family: "Cascadia Mono", "Consolas", monospace; font-size: 0.8rem; color: var(--accent); word-break: break-all; }}
    .queue-panel {{ margin-top: 20px; }}
    .table-shell {{ overflow-x: auto; margin-top: 16px; }}
    .queue-table {{ width: 100%; border-collapse: collapse; min-width: 980px; }}
    .queue-table th, .queue-table td {{
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 0.94rem;
    }}
    .queue-table th {{
      font-size: 0.78rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      background: rgba(31, 111, 120, 0.06);
      position: sticky;
      top: 0;
    }}
    .table-note, .empty-state p {{ margin: 12px 0 0; color: var(--muted); line-height: 1.6; }}
    .empty-state {{
      margin-top: 16px;
      border-radius: 18px;
      padding: 18px;
      background: rgba(244, 255, 252, 0.92);
      border: 1px solid rgba(31, 111, 120, 0.16);
    }}
    .editor-actions {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; }}
    .editor-button {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 10px 16px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--ink);
      font-weight: 700;
      cursor: pointer;
      transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
      box-shadow: 0 10px 22px rgba(31, 42, 48, 0.05);
    }}
    .editor-button:hover {{ transform: translateY(-1px); border-color: rgba(31, 111, 120, 0.26); }}
    .editor-button.primary {{
      background: linear-gradient(135deg, #ddf4ed, #f4fffc);
      border-color: rgba(31, 111, 120, 0.22);
      color: var(--accent);
    }}
    .editor-button.warm {{
      background: linear-gradient(135deg, #fff7eb, #ffe7c3);
      border-color: rgba(217, 119, 6, 0.20);
      color: #9a5a03;
    }}
    .editor-note {{
      margin-top: 12px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.92rem;
    }}
    .editor-shell {{
      margin-top: 16px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.82);
    }}
    .editor-table {{ width: 100%; border-collapse: collapse; min-width: 1100px; }}
    .editor-table th, .editor-table td {{
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 0.94rem;
    }}
    .editor-table th {{
      font-size: 0.78rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      background: rgba(31, 111, 120, 0.06);
      position: sticky;
      top: 0;
    }}
    .editor-table tr:last-child td {{ border-bottom: none; }}
    .editor-table select,
    .editor-table textarea {{
      width: 100%;
      border: 1px solid rgba(31, 42, 48, 0.16);
      border-radius: 12px;
      padding: 8px 10px;
      font: inherit;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.96);
    }}
    .editor-table textarea {{
      min-height: 72px;
      resize: vertical;
    }}
    .editor-table input[type="checkbox"] {{
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }}
    .editor-status {{
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(203, 229, 223, 0.44);
      border: 1px solid rgba(31, 111, 120, 0.12);
      color: var(--ink);
      line-height: 1.55;
    }}
    .editor-status.warning {{
      background: rgba(253, 231, 199, 0.82);
      border-color: rgba(217, 119, 6, 0.18);
      color: #8c4f00;
    }}
    @media (max-width: 960px) {{
      .hero-grid, .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">HITL Reviewer Entry Point</div>
          <h1>HITL Review Workspace</h1>
          <p class="lede">This page turns the review queue into a single human-facing workspace: what to check, which files to edit, which labels are allowed, and what the next pipeline action should be after reviewer edits.</p>
          <div class="hero-meta">
            <span class="meta-pill">project: {self._escape_html(getattr(self.ctx.config.project, "name", ""))}</span>
            <span class="meta-pill">topic: {self._escape_html(getattr(self.ctx.config.request, "topic", ""))}</span>
            <span class="meta-pill">review_status: {self._escape_html(review_status)}</span>
          </div>
          <div class="metrics">
            <article class="metric-card">
              <div class="metric-label">Queue rows</div>
              <div class="metric-value">{self._escape_html(self._format_numeric(len(rows)))}</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Threshold</div>
              <div class="metric-value">{self._escape_html(self._format_numeric(confidence_threshold))}</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Allowed labels</div>
              <div class="metric-value">{self._escape_html(self._format_numeric(len(label_options)))}</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Corrected queue</div>
              <div class="metric-value">{self._escape_html("yes" if corrected_queue_found else "no")}</div>
            </article>
          </div>
        </div>
        <div class="status-card{' ready' if not review_attention else ''}">
          <h2>{self._escape_html(status_title)}</h2>
          <p>{self._escape_html(status_body)}</p>
          <p><strong>Primary action:</strong> {self._escape_html(primary_action)}</p>
          <p><strong>Next step:</strong> {self._escape_html(next_step)}</p>
        </div>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <h2>Reviewer checklist</h2>
        <p>Use only the approved effect-label vocabulary and keep reviewer edits explicit, traceable, and easy to merge back into the pipeline.</p>
        <div class="tag-wrap">{self._render_dashboard_tag_list(label_options, empty_label="not set")}</div>
        <ol class="checklist">{checklist_html}</ol>
      </div>
      <div class="panel">
        <h2>Open first</h2>
        <p>These links are the fastest path through the HITL workflow for the current run.</p>
        <div class="quick-links">{quick_links_html}</div>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <h2>Review files and status</h2>
        <p>Expected inputs stay visible even when they are still missing, so the reviewer always sees where the next manual file should appear.</p>
        <div class="artifact-list">{file_items_html}</div>
      </div>
      <div class="panel">
        <h2>Editable columns</h2>
        <p>The reviewer mainly works with `reviewed_effect_label`, `review_comment`, and `human_verified`, while keeping the original annotation fields available for context.</p>
        <div class="tag-wrap">
          <span class="tag">reviewed_effect_label</span>
          <span class="tag">review_comment</span>
          <span class="tag">human_verified</span>
        </div>
      </div>
    </section>

    <section class="panel queue-panel">
      <h2>Queue preview</h2>
      <p>This preview mirrors the current low-confidence queue so the reviewer can orient quickly before opening the CSV.</p>
      {queue_preview_html}
    </section>

    <section class="panel queue-panel">
      <h2>Interactive review editor</h2>
      <p>Use this editor to set `reviewed_effect_label`, add optional comments, and export a ready-to-merge `review_queue_corrected.csv` without editing the raw queue by hand.</p>
      <div class="editor-actions">
        <button class="editor-button primary" type="button" id="download-corrected-queue">Download corrected queue CSV</button>
        <button class="editor-button" type="button" id="reset-review-editor">Reset editor</button>
        <button class="editor-button warm" type="button" id="copy-corrected-path">Copy expected path</button>
      </div>
      <div class="editor-note">
        Expected corrected queue path on rerun: <strong>{self._escape_html(corrected_queue_path)}</strong>.
        The downloaded file uses the exact filename <strong>{self._escape_html(corrected_queue_download_name)}</strong>.
      </div>
      <div class="editor-status" id="review-editor-status">Editor is ready. Update the table below, then download the corrected queue CSV for the next pipeline run.</div>
      <div class="editor-shell">
        <table class="editor-table">
          <thead>
            <tr>
              <th>id</th>
              <th>source</th>
              <th>current effect</th>
              <th>confidence</th>
              <th>reviewed_effect_label</th>
              <th>review_comment</th>
              <th>human_verified</th>
              <th>text</th>
            </tr>
          </thead>
          <tbody id="review-editor-body"></tbody>
        </table>
      </div>
    </section>
  </div>
  <script>
    const reviewEditorRows = {editor_rows_json};
    const reviewLabelOptions = {label_options_json};
    const correctedQueuePath = {json.dumps(corrected_queue_path, ensure_ascii=False)};
    const correctedQueueFilename = {json.dumps(corrected_queue_download_name, ensure_ascii=False)};
    const reviewEditorBody = document.getElementById("review-editor-body");
    const reviewEditorStatus = document.getElementById("review-editor-status");
    const initialReviewEditorRows = reviewEditorRows.map((row) => ({{ ...row }}));

    function escapeHtml(value) {{
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function toCsvValue(value) {{
      const normalized = String(value ?? "");
      return `"${{normalized.replace(/"/g, '""')}}"`;
    }}

    function setEditorStatus(message, tone = "info") {{
      reviewEditorStatus.textContent = message;
      reviewEditorStatus.className = tone === "warning" ? "editor-status warning" : "editor-status";
    }}

    function renderReviewEditor() {{
      if (!reviewEditorRows.length) {{
        reviewEditorBody.innerHTML = '<tr><td colspan="8">No review rows are available for interactive editing in this run.</td></tr>';
        setEditorStatus("The low-confidence review queue is empty, so the interactive editor is currently in read-only idle mode.");
        return;
      }}

      reviewEditorBody.innerHTML = reviewEditorRows.map((row, index) => {{
        const optionHtml = reviewLabelOptions.map((label) => {{
          const selected = label === row.reviewed_effect_label ? " selected" : "";
          return `<option value="${{escapeHtml(label)}}"${{selected}}>${{escapeHtml(label)}}</option>`;
        }}).join("");
        const truncatedText = escapeHtml(row.text.length > 220 ? `${{row.text.slice(0, 217)}}...` : row.text);
        return `
          <tr>
            <td>${{escapeHtml(row.id)}}</td>
            <td>${{escapeHtml(row.source)}}</td>
            <td>${{escapeHtml(row.effect_label)}}</td>
            <td>${{escapeHtml(row.confidence)}}</td>
            <td>
              <select data-index="${{index}}" data-field="reviewed_effect_label">
                ${{optionHtml}}
              </select>
            </td>
            <td>
              <textarea data-index="${{index}}" data-field="review_comment" placeholder="Optional reviewer note">${{escapeHtml(row.review_comment)}}</textarea>
            </td>
            <td>
              <input type="checkbox" data-index="${{index}}" data-field="human_verified"${{row.human_verified ? " checked" : ""}} />
            </td>
            <td>${{truncatedText}}</td>
          </tr>
        `;
      }}).join("");
    }}

    function buildCorrectedQueueCsv() {{
      const headers = [
        "id",
        "source",
        "text",
        "label",
        "effect_label",
        "confidence",
        "reviewed_effect_label",
        "review_comment",
        "human_verified"
      ];
      const rows = reviewEditorRows.map((row) => [
        row.id,
        row.source,
        row.text,
        row.label,
        row.effect_label,
        row.confidence,
        row.reviewed_effect_label,
        row.review_comment,
        row.human_verified ? "true" : "false"
      ]);
      return [
        headers.join(","),
        ...rows.map((row) => row.map(toCsvValue).join(","))
      ].join("\\r\\n");
    }}

    function downloadCorrectedQueue() {{
      if (!reviewEditorRows.length) {{
        setEditorStatus("There are no review rows to export in this run.", "warning");
        return;
      }}

      const csvText = buildCorrectedQueueCsv();
      const blob = new Blob([csvText], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = correctedQueueFilename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      setEditorStatus(`Downloaded ${{correctedQueueFilename}}. The next pipeline run expects it at ${{correctedQueuePath}}.`);
    }}

    function resetReviewEditor() {{
      reviewEditorRows.splice(0, reviewEditorRows.length, ...initialReviewEditorRows.map((row) => ({{ ...row }})));
      renderReviewEditor();
      setEditorStatus("The interactive review editor has been reset to the current queue snapshot.");
    }}

    async function copyCorrectedPath() {{
      try {{
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          await navigator.clipboard.writeText(correctedQueuePath);
          setEditorStatus(`Copied expected corrected queue path: ${{correctedQueuePath}}`);
          return;
        }}
      }} catch (error) {{
      }}
      setEditorStatus(`Clipboard access is unavailable here. Expected path: ${{correctedQueuePath}}`, "warning");
    }}

    reviewEditorBody.addEventListener("input", (event) => {{
      const target = event.target;
      const index = Number(target?.dataset?.index);
      const field = target?.dataset?.field;
      if (!Number.isInteger(index) || !field || !reviewEditorRows[index]) {{
        return;
      }}
      if (field === "human_verified") {{
        reviewEditorRows[index][field] = Boolean(target.checked);
      }} else {{
        reviewEditorRows[index][field] = target.value;
      }}
      setEditorStatus("Editor changes are ready. Download the corrected queue CSV when you finish the review.");
    }});

    reviewEditorBody.addEventListener("change", (event) => {{
      const target = event.target;
      const index = Number(target?.dataset?.index);
      const field = target?.dataset?.field;
      if (!Number.isInteger(index) || !field || !reviewEditorRows[index]) {{
        return;
      }}
      if (field === "human_verified") {{
        reviewEditorRows[index][field] = Boolean(target.checked);
      }} else {{
        reviewEditorRows[index][field] = target.value;
      }}
      setEditorStatus("Editor changes are ready. Download the corrected queue CSV when you finish the review.");
    }});

    document.getElementById("download-corrected-queue")?.addEventListener("click", downloadCorrectedQueue);
    document.getElementById("reset-review-editor")?.addEventListener("click", resetReviewEditor);
    document.getElementById("copy-corrected-path")?.addEventListener("click", copyCorrectedPath);

    renderReviewEditor();
  </script>
</body>
</html>"""

        path = Path(workspace_path)
        self.registry.save_text(path, html)
        return str(path)

    def write_review_merge_report(
        self,
        corrected_queue_found: bool,
        corrected_queue_path: str,
        n_corrected_rows: int,
        n_rows_with_reviewed_effect_label: int,
        n_effect_label_changes: int,
        reviewed_effect_labels: list[str],
        review_status: str,
    ) -> str:
        """Write a Russian markdown report for corrected-queue merge results."""

        lines = [
            "# Результат ручного merge",
            "",
            "Это краткий отчет о том, был ли найден corrected queue и что реально изменилось после ручной правки.",
            "",
            f"- corrected_queue_found: {'да' if corrected_queue_found else 'нет'}",
            f"- corrected_queue_path: {corrected_queue_path}",
            f"- review_status: {review_status}",
        ]

        if not corrected_queue_found:
            lines.extend([
                "",
                "Merge не выполнен, потому что corrected queue отсутствует.",
                "",
                "## Next step",
                "",
                "- Если нужна ручная валидация, заполните corrected queue и перезапустите pipeline.",
            ])
        else:
            lines.extend([
                f"- n_corrected_rows: {n_corrected_rows}",
                f"- n_rows_with_reviewed_effect_label: {n_rows_with_reviewed_effect_label}",
                f"- n_effect_label_changes: {n_effect_label_changes}",
                f"- reviewed_effect_labels: {', '.join(reviewed_effect_labels) if reviewed_effect_labels else 'нет'}",
            ])
            if review_status == "merged":
                lines.extend([
                    "",
                    "## Next step",
                    "",
                    "- Ручные правки применены. Следующий шаг: retrain / active learning на reviewed dataset.",
                ])
            else:
                lines.extend([
                    "",
                    "## Next step",
                    "",
                    "- Corrected queue обработан, но effect labels не изменились. Можно продолжать training на текущем датасете.",
                ])

        path = Path("reports/review_merge_report.md")
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return str(path)

    def write_review_merge_context(
        self,
        corrected_queue_found: bool,
        corrected_queue_path: str,
        n_corrected_rows: int,
        n_rows_with_reviewed_effect_label: int,
        n_effect_label_changes: int,
        reviewed_effect_labels: list[str],
        review_status: str,
    ) -> str:
        """Write a machine-readable helper artifact for review merge tooling."""

        payload = {
            "corrected_queue_found": corrected_queue_found,
            "corrected_queue_path": corrected_queue_path,
            "n_corrected_rows": n_corrected_rows,
            "n_rows_with_reviewed_effect_label": n_rows_with_reviewed_effect_label,
            "n_effect_label_changes": n_effect_label_changes,
            "reviewed_effect_labels": list(reviewed_effect_labels),
            "review_status": review_status,
        }
        path = Path("data/interim/review_merge_context.json")
        self.registry.save_json(path, payload)
        return str(path)

    def write_review_agreement_report(self, summary: dict[str, Any]) -> str:
        """Write a human-facing agreement report for the reviewed subset."""

        lines = [
            "# Review agreement report",
            "",
            "This report measures auto-vs-human agreement on the reviewed subset. It is not inter-reviewer agreement between two independent human annotators.",
            "",
            f"- comparison_scope: {self._normalize_text(summary.get('comparison_scope'))}",
            f"- corrected_queue_found: {summary.get('corrected_queue_found')}",
            f"- n_corrected_rows: {summary.get('n_corrected_rows')}",
            f"- n_reviewed_rows: {summary.get('n_reviewed_rows')}",
            f"- compared_rows: {summary.get('compared_rows')}",
            f"- matched_rows: {summary.get('matched_rows')}",
            f"- disagreement_rows: {summary.get('disagreement_rows')}",
            f"- agreement: {self._format_numeric(summary.get('agreement')) if summary.get('agreement') is not None else 'n/a'}",
            f"- kappa: {self._format_numeric(summary.get('kappa')) if summary.get('kappa') is not None else 'n/a'}",
            f"- kappa_status: {self._normalize_text(summary.get('kappa_status'))}",
        ]

        notes = summary.get("notes", []) if isinstance(summary.get("notes"), list) else []
        if notes:
            lines.extend(["", "## Notes"])
            for note in notes:
                lines.append(f"- {self._normalize_text(note)}")

        auto_distribution = summary.get("auto_label_distribution", {})
        human_distribution = summary.get("human_label_distribution", {})
        if isinstance(auto_distribution, dict) or isinstance(human_distribution, dict):
            lines.extend(
                [
                    "",
                    "## Label distribution",
                    f"- auto_label_distribution: {auto_distribution if isinstance(auto_distribution, dict) else {}}",
                    f"- human_label_distribution: {human_distribution if isinstance(human_distribution, dict) else {}}",
                ]
            )

        disagreement_examples = summary.get("disagreement_examples", []) if isinstance(summary.get("disagreement_examples"), list) else []
        if disagreement_examples:
            lines.extend(["", "## Disagreement examples"])
            for row in disagreement_examples:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "- id: {id} | source: {source} | auto_effect_label: {auto} | reviewed_effect_label: {human} | confidence: {confidence} | text_preview: {text}".format(
                        id=self._normalize_text(row.get("id")),
                        source=self._normalize_text(row.get("source")),
                        auto=self._normalize_text(row.get("auto_effect_label")),
                        human=self._normalize_text(row.get("reviewed_effect_label")),
                        confidence=self._format_numeric(row.get("confidence")) if row.get("confidence") is not None else "n/a",
                        text=self._normalize_text(row.get("text_preview")),
                    )
                )

        path = "reports/review_agreement_report.md"
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return path

    def write_review_agreement_context(self, summary: dict[str, Any]) -> str:
        """Write the machine-readable agreement summary for the reviewed subset."""

        path = Path("data/interim/review_agreement_context.json")
        self.registry.save_json(path, summary)
        return str(path)

    def write_training_comparison_report(self, summary: dict[str, Any]) -> str:
        """Write a human-facing comparison between baseline auto-label training and reviewed retrain."""

        lines = [
            "# Training comparison report",
            "",
            f"- comparison_scope: {self._normalize_text(summary.get('comparison_scope'))}",
            f"- baseline_status: {self._normalize_text(summary.get('baseline_status'))}",
            f"- reviewed_status: {self._normalize_text(summary.get('reviewed_status'))}",
            f"- review_status: {self._normalize_text(summary.get('review_status'))}",
            f"- corrected_queue_found: {summary.get('corrected_queue_found')}",
            f"- n_effect_label_changes: {self._format_numeric(summary.get('n_effect_label_changes'))}",
            f"- datasets_identical: {summary.get('datasets_identical')}",
            f"- delta_accuracy: {self._format_numeric(summary.get('delta_accuracy')) if summary.get('delta_accuracy') is not None else 'n/a'}",
            f"- delta_f1: {self._format_numeric(summary.get('delta_f1')) if summary.get('delta_f1') is not None else 'n/a'}",
            "",
            "## Baseline metrics",
        ]

        baseline_metrics = summary.get("baseline_metrics", {}) if isinstance(summary.get("baseline_metrics"), dict) else {}
        reviewed_metrics = summary.get("reviewed_metrics", {}) if isinstance(summary.get("reviewed_metrics"), dict) else {}
        if baseline_metrics:
            for key, value in baseline_metrics.items():
                lines.append(f"- {key}: {self._format_numeric(value) if isinstance(value, (int, float)) else value}")
        else:
            lines.append("- baseline metrics unavailable")

        lines.extend(["", "## Reviewed retrain metrics"])
        if reviewed_metrics:
            for key, value in reviewed_metrics.items():
                lines.append(f"- {key}: {self._format_numeric(value) if isinstance(value, (int, float)) else value}")
        else:
            lines.append("- reviewed retrain metrics unavailable")

        notes = summary.get("notes", []) if isinstance(summary.get("notes"), list) else []
        if notes:
            lines.extend(["", "## Notes"])
            for note in notes:
                lines.append(f"- {self._normalize_text(note)}")

        path = "reports/training_comparison_report.md"
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return path

    def write_training_comparison_context(self, summary: dict[str, Any]) -> str:
        """Write the machine-readable training comparison summary."""

        path = Path("data/interim/training_comparison.json")
        self.registry.save_json(path, summary)
        return str(path)

    def write_final_report(self, summary: dict[str, Any]) -> str:
        """Write the final end-to-end markdown report for the demo pipeline."""

        lines = ["# Final Report", ""]
        section_titles = {
            "runtime": "Runtime",
            "dashboard": "Dashboard",
            "sources": "Sources",
            "online_governance": "Online Governance",
            "quality": "Quality",
            "eda": "EDA",
            "annotation": "Annotation",
            "review": "Review",
            "agreement": "Agreement",
            "settings": "Settings",
            "approval": "Approval",
            "active_learning": "Active Learning",
            "training_comparison": "Training Comparison",
            "training": "Training",
            "artifacts": "Artifacts",
        }
        for section_name in ["runtime", "dashboard", "sources", "online_governance", "quality", "eda", "annotation", "review", "agreement", "settings", "approval", "active_learning", "training_comparison", "training", "artifacts"]:
            section = summary.get(section_name)
            lines.append(f"## {section_titles[section_name]}")
            lines.append("")
            if isinstance(section, dict):
                for key, value in section.items():
                    lines.append(f"- {key}: {value}")
            elif isinstance(section, list):
                for item in section:
                    lines.append(f"- {item}")
            else:
                lines.append(f"- {section}")
            lines.append("")

        path = "final_report.md"
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return path

    def write_online_governance_report(self, summary: dict[str, Any]) -> str:
        """Write a human-facing report about remote-provider limits and fallback behavior."""

        lines = [
            "# Online governance and fallback",
            "",
            "Этот отчёт показывает, какие remote providers были активны, где есть риск лимитов и как pipeline ведёт себя, если online lookup ничего не вернул.",
            "",
            f"- remote_sources_enabled: {summary.get('remote_sources_enabled')}",
            f"- active_provider_count: {summary.get('active_provider_count')}",
            f"- github_auth_mode: {self._normalize_text(summary.get('github_auth_mode')) or 'not_used'}",
            f"- fallback_strategy: {self._normalize_text(summary.get('fallback_strategy'))}",
        ]

        notes = summary.get("notes", []) if isinstance(summary.get("notes"), list) else []
        if notes:
            lines.extend(["", "## Operator notes"])
            for note in notes:
                lines.append(f"- {self._normalize_text(note)}")

        providers = summary.get("providers", []) if isinstance(summary.get("providers"), list) else []
        if providers:
            for index, provider in enumerate(providers, start=1):
                if not isinstance(provider, dict):
                    continue
                lines.extend(
                    [
                        "",
                        f"## Provider {index}",
                        f"- provider_id: {self._normalize_text(provider.get('provider_id'))}",
                        f"- label: {self._normalize_text(provider.get('label'))}",
                        f"- enabled_in_config: {provider.get('enabled_in_config')}",
                        f"- active_in_runtime: {provider.get('active_in_runtime')}",
                        f"- observed_status: {self._normalize_text(provider.get('observed_status'))}",
                        f"- discovered_candidates: {provider.get('discovered_candidates')}",
                        f"- auth_mode: {self._normalize_text(provider.get('auth_mode'))}",
                        f"- implementation_status: {self._normalize_text(provider.get('implementation_status'))}",
                        f"- rate_limit_guidance: {self._normalize_text(provider.get('rate_limit_guidance'))}",
                        f"- fallback_behavior: {self._normalize_text(provider.get('fallback_behavior'))}",
                    ]
                )
                operator_action = self._normalize_text(provider.get("operator_action"))
                if operator_action:
                    lines.append(f"- operator_action: {operator_action}")
        else:
            lines.extend(["", "## Provider summary", "- Remote providers are not configured for this run."])

        path = "reports/online_governance_report.md"
        self.registry.save_markdown(path, "\n".join(lines).strip() + "\n")
        return path

    def write_online_governance_context(self, summary: dict[str, Any]) -> str:
        """Write the machine-readable online governance summary."""

        path = Path("data/raw/online_governance_summary.json")
        self.registry.save_json(path, summary)
        return str(path)

    def write_run_dashboard(self, summary: dict[str, Any]) -> str:
        """Write a single-page operator dashboard that links the user to the main artifacts."""

        runtime = summary.get("runtime", {}) if isinstance(summary.get("runtime"), dict) else {}
        dashboard = summary.get("dashboard", {}) if isinstance(summary.get("dashboard"), dict) else {}
        sources = summary.get("sources", {}) if isinstance(summary.get("sources"), dict) else {}
        online_governance = summary.get("online_governance", {}) if isinstance(summary.get("online_governance"), dict) else {}
        quality = summary.get("quality", {}) if isinstance(summary.get("quality"), dict) else {}
        eda = summary.get("eda", {}) if isinstance(summary.get("eda"), dict) else {}
        annotation = summary.get("annotation", {}) if isinstance(summary.get("annotation"), dict) else {}
        review = summary.get("review", {}) if isinstance(summary.get("review"), dict) else {}
        agreement = summary.get("agreement", {}) if isinstance(summary.get("agreement"), dict) else {}
        settings = summary.get("settings", {}) if isinstance(summary.get("settings"), dict) else {}
        approval = summary.get("approval", {}) if isinstance(summary.get("approval"), dict) else {}
        active_learning = summary.get("active_learning", {}) if isinstance(summary.get("active_learning"), dict) else {}
        training_comparison = summary.get("training_comparison", {}) if isinstance(summary.get("training_comparison"), dict) else {}
        training = summary.get("training", {}) if isinstance(summary.get("training"), dict) else {}
        eda_context_payload = self._load_json_artifact(eda.get("eda_context_path"))
        annotation_trace_payload = self._load_json_artifact(annotation.get("annotation_trace_context_path"))
        review_queue_context_payload = self._load_json_artifact(review.get("review_queue_context_path"))
        review_queue_preview_rows = self._load_dataframe_artifact_records(
            review_queue_context_payload.get("input_queue_path") or "data/interim/review_queue.csv"
        )

        dashboard_path = self._normalize_artifact_reference(dashboard.get("dashboard_path") or "reports/run_dashboard.html")
        final_report_path = self._normalize_artifact_reference(dashboard.get("final_report_path") or "final_report.md")
        pipeline_status = self._normalize_text(dashboard.get("pipeline_status")) or "completed"
        attention_required = pipeline_status == "attention_required"
        review_required = bool(review.get("review_required", False))
        configured_remote = runtime.get("configured_remote_source_types", []) if isinstance(runtime.get("configured_remote_source_types"), list) else []
        active_remote = runtime.get("active_remote_source_types", []) if isinstance(runtime.get("active_remote_source_types"), list) else []
        governance_attention = online_governance.get("providers_requiring_attention", []) if isinstance(online_governance.get("providers_requiring_attention"), list) else []
        governance_fallback = self._normalize_text(online_governance.get("fallback_strategy")) or "empty remote shortlist keeps the run stable"
        github_auth_mode = self._normalize_text(online_governance.get("github_auth_mode")) or "not_used"

        status_cards = [
            ("Runtime mode", self._normalize_text(runtime.get("effective_mode")) or "unknown"),
            ("Source candidates", self._format_numeric(sources.get("n_candidates", 0))),
            ("Review queue", self._format_numeric(review.get("review_queue_rows", 0))),
            ("Approval", self._normalize_text(approval.get("approval_status")) or "unknown"),
            ("Remote ops", self._format_numeric(online_governance.get("active_provider_count", 0))),
            ("Accuracy", self._format_numeric(training.get("accuracy", "n/a"))),
            ("F1", self._format_numeric(training.get("f1", "n/a"))),
        ]
        status_cards_html = "".join(
            (
                '<article class="metric-card">'
                f'<div class="metric-label">{self._escape_html(label)}</div>'
                f'<div class="metric-value">{self._escape_html(value)}</div>'
                "</article>"
            )
            for label, value in status_cards
        )

        step_cards_html = "".join(
            self._render_pipeline_step_card(step)
            for step in self._build_dashboard_pipeline_steps(summary)
        )

        primary_links = [
            {
                "label": "Open final report",
                "path": final_report_path,
                "description": "Сводный markdown-отчёт по всему запуску.",
            },
            {
                "label": "Open EDA HTML",
                "path": eda.get("eda_html_report_path"),
                "description": "Наглядный HTML-отчёт для демонстрации данных.",
            },
            {
                "label": "Open review workspace",
                "path": review.get("review_workspace_path"),
                "description": "Reviewer-facing HTML workspace for HITL queue, files, and next actions.",
            },
            {
                "label": "Open review guide",
                "path": review.get("review_queue_report_path"),
                "description": "Инструкция и очередь ручной проверки для HITL.",
            },
            {
                "label": "Open review merge report",
                "path": review.get("review_merge_report_path"),
                "description": "What the pipeline observed after corrected labels were merged back in.",
            },
            {
                "label": "Open agreement report",
                "path": agreement.get("agreement_report_path"),
                "description": "Auto-vs-human agreement and Cohen's kappa on the reviewed subset.",
            },
            {
                "label": "Open training comparison",
                "path": training_comparison.get("comparison_report_path"),
                "description": "Baseline auto-label metrics versus the reviewed retrain on the same local TF-IDF + LogReg stack.",
            },
            {
                "label": "Open AL comparison",
                "path": active_learning.get("al_comparison_report_path"),
                "description": "Entropy versus random strategy comparison for the offline active-learning loop.",
            },
            {
                "label": "Open source shortlist",
                "path": sources.get("source_report_path"),
                "description": "Shortlist найденных источников и approval guidance.",
            },
            {
                "label": "Open source approval workspace",
                "path": approval.get("source_approval_workspace_path"),
                "description": "Interactive source approval page that exports approved_sources.json for the next rerun.",
            },
            {
                "label": "Open runtime settings",
                "path": settings.get("settings_workspace_path"),
                "description": "Environment key status, provider resolution, and current approval-gate semantics for this run.",
            },
            {
                "label": "Open online governance",
                "path": online_governance.get("governance_report_path"),
                "description": "Rate limits, auth mode, provider status and fallback behavior for remote discovery.",
            },
        ]
        primary_links_html = "".join(
            self._render_dashboard_link_tile(
                dashboard_path,
                item["label"],
                item["path"],
                item["description"],
                expected=False,
            )
            for item in primary_links
        )

        artifact_groups_html = "".join(
            self._render_dashboard_artifact_group(dashboard_path, group)
            for group in self._build_dashboard_artifact_groups(summary)
        )

        warnings = quality.get("warnings", []) if isinstance(quality.get("warnings"), list) else []
        warnings_html = (
            "<ul>" + "".join(f"<li>{self._escape_html(item)}</li>" for item in warnings) + "</ul>"
            if warnings
            else '<p class="muted">Quality stage не вернул предупреждений.</p>'
        )
        word_cloud_html = self._render_dashboard_word_cloud(
            eda_context_payload.get("cleaned_word_cloud", {})
            if isinstance(eda_context_payload, dict)
            else {}
        )
        hitl_panel_html = self._render_dashboard_hitl_panel(
            dashboard_path,
            review,
            review_queue_context_payload,
            review_queue_preview_rows,
            agreement,
            training_comparison,
        )
        llm_panel_html = self._render_dashboard_llm_panel(
            dashboard_path,
            annotation,
            annotation_trace_payload,
        )
        settings_panel_html = self._render_dashboard_settings_panel(
            dashboard_path,
            settings,
            approval,
        )

        next_step = self._normalize_text(dashboard.get("next_step")) or self._normalize_text(review.get("next_step")) or "inspect artifacts"
        primary_action = self._normalize_text(dashboard.get("primary_action")) or "inspect dashboard artifacts"
        action_title = "Требуется действие человека" if attention_required else "Запуск завершён"
        action_class = "action-card attention" if attention_required else "action-card success"
        action_text = (
            "Pipeline дошёл до точки HITL и ждёт ручной проверки before retrain."
            if attention_required
            else "Все основные шаги текущего запуска завершены, артефакты готовы к просмотру."
        )

        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pipeline Operator Dashboard</title>
  <style>
    :root {{
      --bg: #f3ede3;
      --panel: rgba(255, 250, 243, 0.94);
      --ink: #1f2a30;
      --muted: #5d6a72;
      --accent: #1f6f78;
      --accent-soft: #cbe5df;
      --warm: #d97706;
      --warm-soft: #fde7c7;
      --line: rgba(31, 42, 48, 0.12);
      --shadow: 0 22px 48px rgba(31, 42, 48, 0.10);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI Variable Text", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(31, 111, 120, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(217, 119, 6, 0.18), transparent 24%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
      min-height: 100vh;
    }}
    .shell {{ max-width: 1240px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(255, 250, 243, 0.98), rgba(244, 237, 227, 0.92));
      border: 1px solid var(--line);
      border-radius: calc(var(--radius) + 4px);
      box-shadow: var(--shadow);
      overflow: hidden;
      position: relative;
      padding: 28px;
      animation: rise 420ms ease-out;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -60px -70px auto;
      width: 220px;
      height: 220px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(31, 111, 120, 0.16) 0%, transparent 68%);
      pointer-events: none;
    }}
    .hero-grid {{ display: grid; grid-template-columns: 1.8fr 1fr; gap: 22px; align-items: start; }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.10em;
      font-size: 12px;
      color: var(--accent);
      margin-bottom: 10px;
      font-weight: 700;
    }}
    h1 {{ font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.03; margin: 0 0 14px; }}
    .lede {{ max-width: 760px; line-height: 1.65; color: var(--muted); margin: 0; }}
    .hero-meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }}
    .meta-pill, .tag {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 9px 14px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.8);
    }}
    .tag-wrap {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .status-grid, .quick-links, .artifact-grid, .step-grid {{ display: grid; gap: 16px; }}
    .status-grid {{ grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); margin-top: 24px; }}
    .metric-card, .panel, .step-card, .artifact-group, .link-tile {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: 0 12px 28px rgba(31, 42, 48, 0.06);
    }}
    .metric-card {{ padding: 18px; animation: rise 420ms ease-out; }}
    .metric-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .metric-value {{ margin-top: 10px; font-size: 1.65rem; font-weight: 700; }}
    .hero-side {{ display: flex; flex-direction: column; gap: 14px; }}
    .action-card {{ padding: 18px; border-radius: var(--radius); border: 1px solid transparent; }}
    .action-card.attention {{ background: linear-gradient(135deg, #fff7eb, #ffe7c3); border-color: rgba(217, 119, 6, 0.24); }}
    .action-card.success {{ background: linear-gradient(135deg, #f4fffc, #ddf4ed); border-color: rgba(31, 111, 120, 0.20); }}
    .action-card h2 {{ margin: 0 0 10px; font-size: 1.05rem; }}
    .action-card p {{ margin: 0 0 10px; line-height: 1.55; }}
    .layout {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 20px; margin-top: 22px; }}
    .panel {{ padding: 22px; animation: rise 460ms ease-out; }}
    .panel h2 {{ margin: 0 0 10px; font-size: 1.15rem; }}
    .panel p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .step-grid {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 16px; }}
    .step-card {{ padding: 18px; position: relative; overflow: hidden; }}
    .step-card::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 5px; background: var(--accent-soft); }}
    .step-card.attention::before {{ background: var(--warm); }}
    .step-card.complete::before {{ background: var(--accent); }}
    .step-badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      background: #eef7f5;
      color: var(--accent);
    }}
    .step-card.attention .step-badge {{ background: var(--warm-soft); color: #9a5a03; }}
    .step-title {{ margin-top: 12px; font-size: 1rem; font-weight: 700; }}
    .step-detail {{ margin-top: 8px; color: var(--muted); line-height: 1.55; font-size: 0.95rem; }}
    .quick-links {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 16px; }}
    .link-tile {{
      display: block;
      text-decoration: none;
      color: inherit;
      padding: 18px;
      transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }}
    .link-tile:hover {{ transform: translateY(-2px); box-shadow: 0 18px 34px rgba(31, 42, 48, 0.10); border-color: rgba(31, 111, 120, 0.26); }}
    .link-tile .title {{ font-size: 1rem; font-weight: 700; }}
    .link-tile .path {{ margin-top: 8px; font-family: "Cascadia Mono", "Consolas", monospace; font-size: 0.82rem; color: var(--accent); word-break: break-all; }}
    .link-tile .description {{ margin-top: 8px; color: var(--muted); line-height: 1.5; font-size: 0.92rem; }}
    .artifact-grid {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); margin-top: 16px; }}
    .artifact-group {{ padding: 20px; }}
    .artifact-group h3 {{ margin: 0 0 12px; font-size: 1rem; }}
    .artifact-list {{ display: grid; gap: 10px; }}
    .artifact-item {{ border-top: 1px solid var(--line); padding-top: 10px; }}
    .artifact-item:first-child {{ border-top: none; padding-top: 0; }}
    .artifact-item a {{ color: var(--ink); text-decoration: none; font-weight: 600; }}
    .artifact-item a:hover {{ color: var(--accent); }}
    .artifact-status {{
      display: inline-flex;
      align-items: center;
      margin-top: 6px;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: #edf7f4;
      color: var(--accent);
    }}
    .artifact-status.expected {{ background: var(--warm-soft); color: #9a5a03; }}
    .artifact-note, .muted {{ color: var(--muted); font-size: 0.92rem; line-height: 1.5; }}
    .artifact-path {{ margin-top: 6px; font-family: "Cascadia Mono", "Consolas", monospace; font-size: 0.8rem; color: var(--accent); word-break: break-all; }}
    .sub-metric-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); margin-top: 16px; }}
    .sub-metric {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.72);
    }}
    .sub-metric .metric-label {{ font-size: 11px; }}
    .sub-metric .metric-value {{ margin-top: 8px; font-size: 1.1rem; }}
    .stack {{ display: grid; gap: 12px; margin-top: 16px; }}
    .checklist {{ margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.6; }}
    .mini-table-wrap {{
      margin-top: 16px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.78);
    }}
    .mini-table {{ width: 100%; border-collapse: collapse; min-width: 560px; }}
    .mini-table th, .mini-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 0.9rem;
    }}
    .mini-table th {{
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      background: rgba(31, 111, 120, 0.06);
    }}
    .mini-table tr:last-child td {{ border-bottom: none; }}
    .callout {{
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(203, 229, 223, 0.52);
      border: 1px solid rgba(31, 111, 120, 0.14);
      color: var(--ink);
      line-height: 1.55;
    }}
    .callout.warning {{
      background: rgba(253, 231, 199, 0.78);
      border-color: rgba(217, 119, 6, 0.18);
    }}
    .word-cloud-shell {{ display: grid; gap: 14px; }}
    .word-cloud-meta {{ color: var(--muted); font-size: 0.92rem; line-height: 1.5; }}
    .word-cloud {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 14px;
      align-items: flex-end;
      padding: 6px 0 2px;
    }}
    .word-chip {{
      display: inline-flex;
      align-items: baseline;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(203, 229, 223, 0.60);
      border: 1px solid rgba(31, 111, 120, 0.16);
      color: var(--ink);
      line-height: 1;
    }}
    .word-chip:nth-child(3n) {{ background: rgba(253, 231, 199, 0.72); border-color: rgba(217, 119, 6, 0.18); }}
    .word-chip small {{ color: var(--muted); font-size: 0.72em; }}
    @keyframes rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    @media (max-width: 900px) {{ .hero-grid, .layout {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Operator Entry Point</div>
          <h1>Pipeline Run Dashboard</h1>
          <p class="lede">Одна страница для запуска, защиты и ручной работы с пайплайном. Здесь видно режим выполнения, текущий статус цикла агентов, HITL-сигналы и быстрые переходы к ключевым артефактам без поиска по папкам.</p>
          <div class="hero-meta">
            <span class="meta-pill">project: {self._escape_html(getattr(self.ctx.config.project, "name", ""))}</span>
            <span class="meta-pill">topic: {self._escape_html(getattr(self.ctx.config.request, "topic", ""))}</span>
            <span class="meta-pill">requested_mode: {self._escape_html(runtime.get("requested_mode", "auto"))}</span>
            <span class="meta-pill">effective_mode: {self._escape_html(runtime.get("effective_mode", "unknown"))}</span>
          </div>
          <div class="status-grid">{status_cards_html}</div>
        </div>
        <div class="hero-side">
          <div class="{action_class}">
            <h2>{self._escape_html(action_title)}</h2>
            <p>{self._escape_html(action_text)}</p>
            <p><strong>Primary action:</strong> {self._escape_html(primary_action)}</p>
            <p><strong>Next step:</strong> {self._escape_html(next_step)}</p>
          </div>
          <div class="panel">
            <h2>Runtime activation</h2>
            <p>Configured remote source types</p>
            <div class="tag-wrap">{self._render_dashboard_tag_list(configured_remote, empty_label="нет")}</div>
            <p style="margin-top: 16px;">Active remote source types</p>
            <div class="tag-wrap">{self._render_dashboard_tag_list(active_remote, empty_label="нет")}</div>
            <p style="margin-top: 16px;">Demo sources enabled: <strong>{self._escape_html("yes" if runtime.get("demo_sources_enabled") else "no")}</strong></p>
            <p>Human review required: <strong>{self._escape_html("yes" if review_required else "no")}</strong></p>
            <p>GitHub auth mode: <strong>{self._escape_html(github_auth_mode)}</strong></p>
            <p>Fallback strategy: <strong>{self._escape_html(governance_fallback)}</strong></p>
            <p style="margin-top: 16px;">Providers requiring attention</p>
            <div class="tag-wrap">{self._render_dashboard_tag_list(governance_attention, empty_label="none")}</div>
          </div>
        </div>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <h2>Pipeline cycle</h2>
        <p>Статусы показывают, где мы находимся после текущего запуска и требует ли система ручного шага перед следующим retrain.</p>
        <div class="step-grid">{step_cards_html}</div>
      </div>
      <div class="panel">
        <h2>Quality and review signals</h2>
        {warnings_html}
      </div>
    </section>

    <section class="layout" style="margin-top: 20px;">
      <div class="panel">
        <h2>HITL control center</h2>
        <p>Эта зона показывает, что именно должен сделать человек на текущем запуске, какие метки допустимы, и какой файл нужно вернуть обратно в pipeline.</p>
        {hitl_panel_html}
      </div>
      <div class="panel">
        <h2>LLM annotation center</h2>
        <p>Здесь видно, какая annotation path реально работала в текущем запуске: offline mock, Gemini или fallback-режим, плюс сколько строк ушло в low-confidence и fallback.</p>
        {llm_panel_html}
      </div>
    </section>

    <section class="panel" style="margin-top: 20px;">
      <h2>Settings and gate status</h2>
      <p>This compact view explains which env-backed provider path was active and whether collection ran openly or under an explicit approved_sources gate.</p>
      {settings_panel_html}
    </section>

    <section class="panel" style="margin-top: 20px;">
      <h2>Cleaned word cloud</h2>
      <p>Preview of the cleaned post-quality text after lowercasing, punctuation cleanup, and stop-word filtering. It helps quickly verify the current topic focus before HITL and retrain.</p>
      {word_cloud_html}
    </section>

    <section class="panel" style="margin-top: 20px;">
      <h2>Open first</h2>
      <p>Быстрые переходы к тем артефактам, которые чаще всего нужны на защите и в ручном workflow.</p>
      <div class="quick-links">{primary_links_html}</div>
    </section>

    <section class="panel" style="margin-top: 20px;">
      <h2>Artifacts by role</h2>
      <p>Здесь собраны human-facing отчёты, HITL-файлы, source/approval контекст и модельные артефакты. Ожидаемые входные файлы помечаются отдельно, если они ещё не созданы.</p>
      <div class="artifact-grid">{artifact_groups_html}</div>
    </section>
  </div>
</body>
</html>"""

        self.registry.save_text(Path(dashboard_path), html)
        return dashboard_path

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

    def _normalize_text(self, value: Any) -> str:
        """Normalize arbitrary values into stable strings for reporting."""

        if value is None:
            return ""
        return str(value).strip()

    def _truncate_text(self, value: Any, *, limit: int) -> str:
        """Trim long free-text fields so HTML tables stay readable."""

        text = self._normalize_text(value)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _coerce_float(self, value: Any) -> float:
        """Convert a value to float while tolerating missing confidences."""

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric != numeric:
            return 0.0
        return max(0.0, min(1.0, numeric))

    def _format_numeric(self, value: Any) -> str:
        """Format numeric discovery values without confidence-style clamping."""

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return self._normalize_text(value)

        if numeric != numeric:
            return "nan"

        if numeric.is_integer():
            return str(int(numeric))

        return f"{numeric:.3f}".rstrip("0").rstrip(".")

    def _normalize_artifact_reference(self, value: Any) -> str:
        """Normalize local artifact references into project-relative POSIX paths when possible."""

        text = self._normalize_text(value)
        if not text:
            return ""
        if "://" in text:
            return text

        candidate = Path(text)
        if not candidate.is_absolute():
            return Path(text.replace("\\", "/")).as_posix()

        root_dir = Path(self.ctx.paths.root_dir).resolve(strict=False)
        resolved_candidate = candidate.resolve(strict=False)
        try:
            return resolved_candidate.relative_to(root_dir).as_posix()
        except ValueError:
            return resolved_candidate.as_posix()

    def _artifact_reference_exists(self, value: Any) -> bool:
        """Check whether a local artifact path currently exists under the project root."""

        normalized = self._normalize_artifact_reference(value)
        if not normalized or "://" in normalized:
            return False
        try:
            return self.registry.exists(normalized)
        except Exception:
            return False

    def _load_json_artifact(self, value: Any) -> dict[str, Any]:
        """Load a local JSON artifact when available and keep dashboard rendering resilient."""

        normalized = self._normalize_artifact_reference(value)
        if not normalized or "://" in normalized:
            return {}

        try:
            payload = self.registry.load_json(normalized)
        except Exception:
            return {}

        return payload if isinstance(payload, dict) else {}

    def _load_dataframe_artifact_records(self, value: Any) -> list[dict[str, Any]]:
        """Load a local dataframe-like artifact and materialize it into record dictionaries."""

        normalized = self._normalize_artifact_reference(value)
        if not normalized or "://" in normalized:
            return []

        try:
            payload = self.registry.load_dataframe(normalized)
        except Exception:
            return []

        return self._to_records(payload)

    def _dashboard_href(self, dashboard_path: str, target_path: Any) -> str:
        """Build an HTML-friendly relative href from the dashboard to a local artifact."""

        normalized_target = self._normalize_artifact_reference(target_path)
        if not normalized_target:
            return ""
        if "://" in normalized_target:
            return normalized_target

        normalized_dashboard = self._normalize_artifact_reference(dashboard_path)
        start_dir = posixpath.dirname(normalized_dashboard) or "."
        return posixpath.relpath(normalized_target, start=start_dir)

    def _render_dashboard_tag_list(self, values: list[Any], *, empty_label: str) -> str:
        """Render a compact tag list for runtime-mode metadata."""

        cleaned = [self._normalize_text(value) for value in values if self._normalize_text(value)]
        if not cleaned:
            cleaned = [empty_label]
        return "".join(f'<span class="tag">{self._escape_html(value)}</span>' for value in cleaned)

    def _build_dashboard_pipeline_steps(self, summary: dict[str, Any]) -> list[dict[str, str]]:
        """Build a compact status view for the main pipeline cycle."""

        sources = summary.get("sources", {}) if isinstance(summary.get("sources"), dict) else {}
        quality = summary.get("quality", {}) if isinstance(summary.get("quality"), dict) else {}
        annotation = summary.get("annotation", {}) if isinstance(summary.get("annotation"), dict) else {}
        review = summary.get("review", {}) if isinstance(summary.get("review"), dict) else {}
        training = summary.get("training", {}) if isinstance(summary.get("training"), dict) else {}

        warnings = quality.get("warnings", []) if isinstance(quality.get("warnings"), list) else []
        review_required = bool(review.get("review_required"))
        review_status = self._normalize_text(review.get("status")) or "unknown"
        review_needs_action = review_required and review_status == "skipped_missing_corrected_queue"
        training_detail = "accuracy: {accuracy}, f1: {f1}".format(
            accuracy=self._format_numeric(training.get("accuracy", "n/a")),
            f1=self._format_numeric(training.get("f1", "n/a")),
        )

        return [
            {
                "name": "Discovery",
                "status": "complete",
                "badge": "ready",
                "detail": f"Shortlist built: {self._format_numeric(sources.get('n_candidates', 0))} candidate(s).",
            },
            {
                "name": "Collection + Quality",
                "status": "complete",
                "badge": "ready",
                "detail": f"Quality warnings: {self._format_numeric(len(warnings))}.",
            },
            {
                "name": "Annotation",
                "status": "complete",
                "badge": "ready",
                "detail": "Low-confidence rows: {count} at threshold {threshold}.".format(
                    count=self._format_numeric(annotation.get("n_low_confidence", 0)),
                    threshold=self._format_numeric(annotation.get("confidence_threshold", 0)),
                ),
            },
            {
                "name": "Human Review",
                "status": "attention" if review_needs_action else "complete",
                "badge": "action" if review_needs_action else "ready",
                "detail": self._normalize_text(review.get("next_step"))
                or ("HITL step completed for current run." if not review_required else "Manual review still expected."),
            },
            {
                "name": "Active Learning",
                "status": "complete",
                "badge": "ready",
                "detail": "Active-learning cycle finished for the current reviewed dataset.",
            },
            {
                "name": "Training + Reporting",
                "status": "complete",
                "badge": "ready",
                "detail": training_detail,
            },
        ]

    def _render_pipeline_step_card(self, step: dict[str, Any]) -> str:
        """Render one pipeline-step card for the operator dashboard."""

        status = self._normalize_text(step.get("status")) or "complete"
        badge = self._normalize_text(step.get("badge")) or status
        badge_label = "needs action" if badge == "action" else "ready"
        return (
            f'<article class="step-card {self._escape_html(status)}">'
            f'<span class="step-badge">{self._escape_html(badge_label)}</span>'
            f'<div class="step-title">{self._escape_html(step.get("name"))}</div>'
            f'<div class="step-detail">{self._escape_html(step.get("detail"))}</div>'
            "</article>"
        )

    def _render_dashboard_hitl_panel(
        self,
        dashboard_path: str,
        review: dict[str, Any],
        review_queue_context: dict[str, Any],
        review_queue_rows: list[dict[str, Any]],
        agreement: dict[str, Any],
        training_comparison: dict[str, Any],
    ) -> str:
        """Render the human-in-the-loop control center inside the operator dashboard."""

        review_required = bool(review.get("review_required"))
        review_status = self._normalize_text(review.get("status")) or "unknown"
        corrected_queue_path = (
            self._normalize_artifact_reference(review_queue_context.get("expected_corrected_queue_path"))
            or "data/interim/review_queue_corrected.csv"
        )
        label_options = (
            review_queue_context.get("label_options")
            if isinstance(review_queue_context.get("label_options"), list)
            else []
        )
        preview_rows = review_queue_rows[:4]
        hitl_waiting = review_required and review_status == "skipped_missing_corrected_queue"
        queue_rows = self._format_numeric(review.get("review_queue_rows", len(review_queue_rows)))
        threshold = self._format_numeric(review_queue_context.get("confidence_threshold", "n/a"))
        compared_rows = self._format_numeric(agreement.get("compared_rows", 0))
        label_changes = self._format_numeric(training_comparison.get("n_effect_label_changes", 0))

        status_note = (
            "Reviewer action is currently blocking the next clean retrain."
            if hitl_waiting
            else (
                "Corrected labels have already been returned to the pipeline for this run."
                if review_status in {"merged", "merged_no_changes"}
                else "This run does not currently require manual correction."
            )
        )
        callout_class = "callout warning" if hitl_waiting else "callout"
        checklist_items = (
            [
                "Open the review workspace or the review_queue.csv file.",
                "Fill reviewed_effect_label only with one of the allowed effect labels.",
                "Optionally add review_comment and human_verified for confirmed rows.",
                "Save the result as review_queue_corrected.csv and rerun the same pipeline config.",
                "Then inspect review_merge_report.md, review_agreement_report.md, and training_comparison_report.md.",
            ]
            if hitl_waiting
            else (
                [
                    "Inspect review_merge_report.md to confirm how the corrected labels were applied.",
                    "Check review_agreement_report.md to see agreement and kappa on the reviewed subset.",
                    "Use training_comparison_report.md to confirm whether HITL improved the final metrics.",
                ]
                if review_status in {"merged", "merged_no_changes"}
                else [
                    "Manual review is not required for this run because the low-confidence queue is empty.",
                    "You can still inspect the review workspace and queue context for auditability.",
                ]
            )
        )
        checklist_html = "".join(f"<li>{self._escape_html(item)}</li>" for item in checklist_items)

        quick_links = [
            {
                "label": "Open review workspace",
                "path": review.get("review_workspace_path"),
                "description": "Main HTML interface for the reviewer.",
                "expected": False,
            },
            {
                "label": "Open review queue CSV",
                "path": review_queue_context.get("input_queue_path") or "data/interim/review_queue.csv",
                "description": "Raw queue exported from low-confidence annotation rows.",
                "expected": False,
            },
            {
                "label": "Open corrected queue CSV",
                "path": corrected_queue_path,
                "description": "Expected reviewer output file that should be returned to the pipeline.",
                "expected": True,
            },
            {
                "label": "Open merge report",
                "path": review.get("review_merge_report_path"),
                "description": "Summary of what happened after reviewer edits were merged back in.",
                "expected": False,
            },
        ]
        quick_links_html = "".join(
            self._render_dashboard_link_tile(
                dashboard_path,
                item["label"],
                item["path"],
                item["description"],
                expected=bool(item["expected"]),
            )
            for item in quick_links
        )

        if preview_rows:
            body_html = "".join(
                "<tr>"
                f"<td>{self._escape_html(self._normalize_text(row.get('id')))}</td>"
                f"<td>{self._escape_html(self._normalize_text(row.get('effect_label')))}</td>"
                f"<td>{self._escape_html(self._format_numeric(row.get('confidence')))}</td>"
                f"<td>{self._escape_html(self._truncate_text(row.get('text'), limit=110))}</td>"
                "</tr>"
                for row in preview_rows
            )
            preview_html = (
                '<div class="mini-table-wrap"><table class="mini-table">'
                "<thead><tr><th>id</th><th>effect_label</th><th>confidence</th><th>text</th></tr></thead>"
                f"<tbody>{body_html}</tbody></table></div>"
            )
        else:
            preview_html = '<p class="muted" style="margin-top: 14px;">Review queue preview is empty for this run.</p>'

        return (
            f'<div class="{self._escape_html(callout_class)}">{self._escape_html(status_note)}</div>'
            '<div class="sub-metric-grid">'
            f'<div class="sub-metric"><div class="metric-label">Queue rows</div><div class="metric-value">{self._escape_html(queue_rows)}</div></div>'
            f'<div class="sub-metric"><div class="metric-label">Threshold</div><div class="metric-value">{self._escape_html(threshold)}</div></div>'
            f'<div class="sub-metric"><div class="metric-label">Compared rows</div><div class="metric-value">{self._escape_html(compared_rows)}</div></div>'
            f'<div class="sub-metric"><div class="metric-label">Effect-label changes</div><div class="metric-value">{self._escape_html(label_changes)}</div></div>'
            "</div>"
            '<div class="stack">'
            f'<p><strong>Review status:</strong> {self._escape_html(review_status)}</p>'
            f'<p><strong>Next step:</strong> {self._escape_html(self._normalize_text(review.get("next_step")) or "inspect review artifacts")}</p>'
            '<div><p><strong>Allowed effect labels</strong></p>'
            f'<div class="tag-wrap">{self._render_dashboard_tag_list(label_options, empty_label="not set")}</div></div>'
            "<div><p><strong>Reviewer checklist</strong></p>"
            f'<ol class="checklist">{checklist_html}</ol></div>'
            "</div>"
            f'<div class="quick-links">{quick_links_html}</div>'
            f"{preview_html}"
        )

    def _render_dashboard_llm_panel(
        self,
        dashboard_path: str,
        annotation: dict[str, Any],
        annotation_trace: dict[str, Any],
    ) -> str:
        """Render the LLM/annotation control center inside the operator dashboard."""

        use_llm_requested = bool(annotation.get("use_llm_requested"))
        requested_provider = self._normalize_text(annotation.get("requested_provider")) or "disabled"
        resolved_provider = self._normalize_text(annotation.get("resolved_provider")) or "disabled"
        provider_status = self._normalize_text(annotation.get("provider_status")) or "unknown"
        llm_mode = self._normalize_text(annotation.get("llm_mode")) or self._normalize_text(annotation_trace.get("llm_mode")) or "unknown"
        n_low_confidence = self._format_numeric(annotation.get("n_low_confidence", 0))
        n_fallback_rows = self._format_numeric(annotation.get("n_fallback_rows", 0))
        effect_labels = annotation.get("effect_labels") if isinstance(annotation.get("effect_labels"), list) else []
        if not effect_labels:
            prompt_contract = annotation_trace.get("prompt_contract", {}) if isinstance(annotation_trace.get("prompt_contract"), dict) else {}
            effect_labels = prompt_contract.get("effect_labels", []) if isinstance(prompt_contract.get("effect_labels"), list) else []
        fallback_reason_counts = (
            annotation.get("fallback_reason_counts")
            if isinstance(annotation.get("fallback_reason_counts"), dict)
            else {}
        )
        if not fallback_reason_counts:
            parser_contract = annotation_trace.get("parser_contract", {}) if isinstance(annotation_trace.get("parser_contract"), dict) else {}
            fallback_reason_counts = (
                parser_contract.get("fallback_reason_counts")
                if isinstance(parser_contract.get("fallback_reason_counts"), dict)
                else {}
            )
        parse_status_counts = {}
        parser_contract = annotation_trace.get("parser_contract", {}) if isinstance(annotation_trace.get("parser_contract"), dict) else {}
        if isinstance(parser_contract.get("parse_status_counts"), dict):
            parse_status_counts = parser_contract.get("parse_status_counts")

        if provider_status == "gemini_requested_but_mock_fallback_active":
            llm_note = "Gemini was requested, but the run used the offline-safe mock fallback. This usually means GEMINI_API_KEY was missing."
            callout_class = "callout warning"
        elif resolved_provider == "gemini":
            llm_note = "Gemini annotation is active for this run, and the trace/report pair can be used to inspect the prompt contract."
            callout_class = "callout"
        elif resolved_provider == "mock":
            llm_note = "The deterministic mock LLM path is active. This is the safest offline/demo mode and keeps annotation reproducible."
            callout_class = "callout"
        elif not use_llm_requested:
            llm_note = "LLM assistance is disabled in config, so annotation stays on the deterministic offline path."
            callout_class = "callout"
        else:
            llm_note = "A custom or unknown annotation provider was resolved. Inspect the trace report before relying on the output."
            callout_class = "callout warning"

        parse_tags = [
            f"{self._normalize_text(key)}: {self._format_numeric(value)}"
            for key, value in parse_status_counts.items()
        ]
        fallback_tags = [
            f"{self._normalize_text(key)}: {self._format_numeric(value)}"
            for key, value in fallback_reason_counts.items()
        ]

        quick_links = [
            {
                "label": "Open annotation report",
                "path": annotation.get("annotation_report_path"),
                "description": "Human-facing summary of confidence and label distribution.",
                "expected": False,
            },
            {
                "label": "Open annotation trace",
                "path": annotation.get("annotation_trace_report_path"),
                "description": "Prompt contract, parser contract, and fallback trace for the current run.",
                "expected": False,
            },
        ]
        quick_links_html = "".join(
            self._render_dashboard_link_tile(
                dashboard_path,
                item["label"],
                item["path"],
                item["description"],
                expected=False,
            )
            for item in quick_links
        )

        return (
            f'<div class="{self._escape_html(callout_class)}">{self._escape_html(llm_note)}</div>'
            '<div class="sub-metric-grid">'
            f'<div class="sub-metric"><div class="metric-label">LLM requested</div><div class="metric-value">{self._escape_html("yes" if use_llm_requested else "no")}</div></div>'
            f'<div class="sub-metric"><div class="metric-label">Resolved provider</div><div class="metric-value">{self._escape_html(resolved_provider)}</div></div>'
            f'<div class="sub-metric"><div class="metric-label">LLM mode</div><div class="metric-value">{self._escape_html(llm_mode)}</div></div>'
            f'<div class="sub-metric"><div class="metric-label">Fallback rows</div><div class="metric-value">{self._escape_html(n_fallback_rows)}</div></div>'
            "</div>"
            '<div class="stack">'
            f'<p><strong>Requested provider:</strong> {self._escape_html(requested_provider)}</p>'
            f'<p><strong>Provider status:</strong> {self._escape_html(provider_status)}</p>'
            f'<p><strong>Low-confidence rows:</strong> {self._escape_html(n_low_confidence)}</p>'
            '<div><p><strong>Effect labels in prompt contract</strong></p>'
            f'<div class="tag-wrap">{self._render_dashboard_tag_list(effect_labels, empty_label="not set")}</div></div>'
            '<div><p><strong>Parse status counts</strong></p>'
            f'<div class="tag-wrap">{self._render_dashboard_tag_list(parse_tags, empty_label="not observed")}</div></div>'
            '<div><p><strong>Fallback reasons</strong></p>'
            f'<div class="tag-wrap">{self._render_dashboard_tag_list(fallback_tags, empty_label="none")}</div></div>'
            "</div>"
            f'<div class="quick-links">{quick_links_html}</div>'
        )

    def _render_dashboard_settings_panel(
        self,
        dashboard_path: str,
        settings: dict[str, Any],
        approval: dict[str, Any],
    ) -> str:
        """Render a compact settings panel with env/key status and approval semantics."""

        resolved_provider = self._normalize_text(settings.get("resolved_provider")) or "disabled"
        provider_status = self._normalize_text(settings.get("provider_status")) or "unknown"
        gemini_status = self._normalize_text(settings.get("gemini_api_key_status")) or "unknown"
        github_status = self._normalize_text(settings.get("github_token_status")) or "unknown"
        github_auth_mode = self._normalize_text(settings.get("github_auth_mode")) or "not_used"
        approval_gate_status = self._normalize_text(settings.get("approval_gate_status") or approval.get("approval_gate_status")) or "unknown"
        effective_scope = self._normalize_text(settings.get("effective_collection_scope") or approval.get("effective_collection_scope")) or "unknown"
        effective_source_count = self._format_numeric(
            settings.get("effective_source_count")
            if settings.get("effective_source_count") is not None
            else approval.get("effective_source_count", 0)
        )
        gate_note = self._normalize_text(settings.get("gate_note") or approval.get("gate_note"))

        if "missing" in gemini_status or "missing" in github_status or approval_gate_status == "restricted_empty_subset":
            callout_class = "callout warning"
        else:
            callout_class = "callout"

        quick_links = [
            {
                "label": "Open runtime settings",
                "path": settings.get("settings_workspace_path"),
                "description": "Detailed key status, onboarding commands, and gate semantics for this run.",
            },
            {
                "label": "Open source approval workspace",
                "path": settings.get("source_approval_workspace_path") or approval.get("source_approval_workspace_path"),
                "description": "Adjust approved_sources.json before the next rerun if you want to change collection scope.",
            },
        ]
        quick_links_html = "".join(
            self._render_dashboard_link_tile(dashboard_path, item["label"], item["path"], item["description"], expected=False)
            for item in quick_links
        )

        return (
            f'<div class="{self._escape_html(callout_class)}">{self._escape_html(gate_note or "Inspect runtime_settings.html before the next rerun if you need to change provider keys or source gate behavior.")}</div>'
            '<div class="sub-metric-grid">'
            f'<div class="sub-metric"><div class="metric-label">Resolved provider</div><div class="metric-value">{self._escape_html(resolved_provider)}</div></div>'
            f'<div class="sub-metric"><div class="metric-label">Gemini key</div><div class="metric-value">{self._escape_html(gemini_status)}</div></div>'
            f'<div class="sub-metric"><div class="metric-label">GitHub token</div><div class="metric-value">{self._escape_html(github_status)}</div></div>'
            f'<div class="sub-metric"><div class="metric-label">Approval gate</div><div class="metric-value">{self._escape_html(approval_gate_status)}</div></div>'
            "</div>"
            '<div class="stack">'
            f'<p><strong>Provider status:</strong> {self._escape_html(provider_status)}</p>'
            f'<p><strong>GitHub auth mode:</strong> {self._escape_html(github_auth_mode)}</p>'
            f'<p><strong>Effective collection scope:</strong> {self._escape_html(effective_scope)}</p>'
            f'<p><strong>Effective source count:</strong> {self._escape_html(effective_source_count)}</p>'
            "</div>"
            f'<div class="quick-links">{quick_links_html}</div>'
        )

    def _build_dashboard_artifact_groups(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        """Group the main run artifacts by the job they play for the operator."""

        dashboard = summary.get("dashboard", {}) if isinstance(summary.get("dashboard"), dict) else {}
        sources = summary.get("sources", {}) if isinstance(summary.get("sources"), dict) else {}
        online_governance = summary.get("online_governance", {}) if isinstance(summary.get("online_governance"), dict) else {}
        eda = summary.get("eda", {}) if isinstance(summary.get("eda"), dict) else {}
        annotation = summary.get("annotation", {}) if isinstance(summary.get("annotation"), dict) else {}
        review = summary.get("review", {}) if isinstance(summary.get("review"), dict) else {}
        agreement = summary.get("agreement", {}) if isinstance(summary.get("agreement"), dict) else {}
        settings = summary.get("settings", {}) if isinstance(summary.get("settings"), dict) else {}
        approval = summary.get("approval", {}) if isinstance(summary.get("approval"), dict) else {}
        active_learning = summary.get("active_learning", {}) if isinstance(summary.get("active_learning"), dict) else {}
        training_comparison = summary.get("training_comparison", {}) if isinstance(summary.get("training_comparison"), dict) else {}
        artifacts = summary.get("artifacts", {}) if isinstance(summary.get("artifacts"), dict) else {}

        return [
            {
                "title": "Human-facing reports",
                "items": [
                    {"label": "Final report", "path": dashboard.get("final_report_path"), "note": "Главный markdown summary по текущему запуску."},
                    {"label": "Runtime settings workspace", "path": settings.get("settings_workspace_path"), "note": "Env key status, provider resolution, onboarding commands and approval gate semantics."},
                    {"label": "Source shortlist", "path": sources.get("source_report_path"), "note": "Shortlist источников и approval guidance."},
                    {"label": "Source approval workspace", "path": approval.get("source_approval_workspace_path"), "note": "Interactive approval page for selecting allowed sources and exporting approved_sources.json."},
                    {"label": "Online governance report", "path": online_governance.get("governance_report_path"), "note": "Remote provider limits, auth mode, fallback behavior and operator guidance."},
                    {"label": "EDA markdown", "path": eda.get("eda_report_path"), "note": "Подробный EDA для README/demo narrative."},
                    {"label": "EDA HTML", "path": eda.get("eda_html_report_path"), "note": "Наглядный HTML-отчёт для показа преподавателю."},
                    {"label": "Annotation report", "path": annotation.get("annotation_report_path"), "note": "Сводка по effect labels и confidence."},
                    {"label": "Annotation trace report", "path": annotation.get("annotation_trace_report_path"), "note": "Prompt/parser contract и fallback trace."},
                    {"label": "Review guide", "path": review.get("review_queue_report_path"), "note": "Инструкция по HITL и объяснение следующего шага."},
                    {"label": "Review merge report", "path": review.get("review_merge_report_path"), "note": "Результат ручного merge и post-review status."},
                    {"label": "Active learning report", "path": active_learning.get("al_report_path"), "note": "История AL-итераций после review."},
                    {"label": "AL comparison report", "path": active_learning.get("al_comparison_report_path"), "note": "Сравнение стратегий entropy и random в одном offline AL цикле."},
                    {"label": "Review agreement report", "path": agreement.get("agreement_report_path"), "note": "Auto-vs-human agreement and Cohen's kappa for the reviewed subset."},
                    {"label": "Training comparison report", "path": training_comparison.get("comparison_report_path"), "note": "Сравнение baseline auto-label metrics и retrain после review/HITL."},
                ],
            },
            {
                "title": "Human review files",
                "items": [
                    {"label": "Review queue CSV", "path": "data/interim/review_queue.csv", "note": "CSV для ручной проверки low-confidence строк."},
                    {"label": "Review queue context", "path": review.get("review_queue_context_path"), "note": "Машиночитаемый контекст для reviewer tooling."},
                    {"label": "Corrected queue CSV", "path": "data/interim/review_queue_corrected.csv", "note": "Заполняется человеком и подаётся обратно в pipeline.", "expected": True},
                    {"label": "Review merge context", "path": review.get("review_merge_context_path"), "note": "Итог merge в JSON-виде для проверки согласованности."},
                    {"label": "Review agreement context", "path": agreement.get("agreement_context_path"), "note": "Machine-readable agreement metrics for the reviewed subset."},
                ],
            },
            {
                "title": "Source and approval context",
                "items": [
                    {"label": "Discovered sources", "path": "data/raw/discovered_sources.json", "note": "Полный сериализованный shortlist discovery stage."},
                    {"label": "Approval candidates", "path": "data/raw/approval_candidates.json", "note": "Упрощённый helper JSON для approval flow."},
                    {"label": "Source approval workspace", "path": approval.get("source_approval_workspace_path"), "note": "HTML approval entry point for choosing allowed sources before the next rerun."},
                    {"label": "Online governance context", "path": online_governance.get("governance_context_path"), "note": "Machine-readable remote-provider status, auth mode and fallback summary."},
                    {"label": "Approved sources input", "path": approval.get("approved_sources_path"), "note": "Опциональный input-файл для ручного approval.", "expected": True},
                ],
            },
            {
                "title": "Model and machine-readable artifacts",
                "items": [
                    {"label": "EDA context", "path": eda.get("eda_context_path"), "note": "JSON summary для EDA/HITL layer."},
                    {"label": "Annotation trace context", "path": annotation.get("annotation_trace_context_path"), "note": "JSON trace annotation contract."},
                    {"label": "AL comparison context", "path": active_learning.get("al_comparison_context_path"), "note": "Machine-readable entropy-vs-random AL comparison payload."},
                    {"label": "Training comparison context", "path": training_comparison.get("comparison_context_path"), "note": "Machine-readable baseline-vs-reviewed retrain summary."},
                    {"label": "Model metrics", "path": artifacts.get("metrics_path"), "note": "Финальные метрики baseline-модели."},
                    {"label": "Model artifact", "path": artifacts.get("model_path"), "note": "Сериализованный классификатор TF-IDF + LogReg."},
                    {"label": "Vectorizer artifact", "path": artifacts.get("vectorizer_path"), "note": "Сериализованный TF-IDF vectorizer."},
                ],
            },
        ]

    def _render_dashboard_artifact_group(self, dashboard_path: str, group: dict[str, Any]) -> str:
        """Render one grouped artifact block for the operator dashboard."""

        items_html = "".join(
            self._render_dashboard_artifact_item(dashboard_path, item)
            for item in group.get("items", [])
            if self._normalize_artifact_reference(item.get("path"))
        )
        return (
            '<section class="artifact-group">'
            f'<h3>{self._escape_html(group.get("title"))}</h3>'
            f'<div class="artifact-list">{items_html}</div>'
            "</section>"
        )

    def _render_dashboard_artifact_item(self, dashboard_path: str, item: dict[str, Any]) -> str:
        """Render one linkable artifact row with readiness status."""

        path = self._normalize_artifact_reference(item.get("path"))
        note = self._normalize_text(item.get("note"))
        expected = bool(item.get("expected"))
        exists = self._artifact_reference_exists(path)
        href = self._dashboard_href(dashboard_path, path)
        status_label = "ready" if exists else ("expected input" if expected else "missing")
        status_class = "artifact-status" if exists else "artifact-status expected"
        label_html = (
            f'<a href="{self._escape_html(href)}">{self._escape_html(item.get("label"))}</a>'
            if href
            else self._escape_html(item.get("label"))
        )
        return (
            '<div class="artifact-item">'
            f'<div>{label_html}</div>'
            f'<div class="{status_class}">{self._escape_html(status_label)}</div>'
            f'<div class="artifact-path">{self._escape_html(path)}</div>'
            f'<div class="artifact-note">{self._escape_html(note)}</div>'
            "</div>"
        )

    def _render_dashboard_link_tile(
        self,
        dashboard_path: str,
        label: str,
        path: Any,
        description: str,
        *,
        expected: bool,
    ) -> str:
        """Render a prominent quick-link tile for the dashboard hero area."""

        normalized_path = self._normalize_artifact_reference(path)
        href = self._dashboard_href(dashboard_path, normalized_path)
        if not normalized_path:
            return ""

        exists = self._artifact_reference_exists(normalized_path)
        status_label = "ready" if exists else ("expected input" if expected else "missing")
        return (
            f'<a class="link-tile" href="{self._escape_html(href)}">'
            f'<div class="title">{self._escape_html(label)}</div>'
            f'<div class="path">{self._escape_html(normalized_path)}</div>'
            f'<div class="description">{self._escape_html(description)}</div>'
            f'<div class="artifact-status{" expected" if not exists else ""}" style="margin-top: 12px;">{self._escape_html(status_label)}</div>'
            "</a>"
        )

    def _format_compact_metadata(self, metadata: Any) -> str:
        """Render a short metadata summary that stays readable in markdown reports."""

        if not isinstance(metadata, dict) or not metadata:
            return ""

        excluded_keys = set(COMPLIANCE_KEYS) | {"html"}
        preferred_keys = ["source_kind", "api_kind", "demo_mode", "topic", "web_url", "downloads", "likes", "tags", "stars", "language"]
        parts: list[str] = []
        seen_keys: set[str] = set()

        for key in preferred_keys:
            if key in metadata and key not in excluded_keys:
                parts.append(f"{key}={metadata[key]}")
                seen_keys.add(key)

        for key, value in metadata.items():
            if key in seen_keys or key in excluded_keys:
                continue
            if len(parts) >= 8:
                break
            parts.append(f"{key}={value}")

        return ", ".join(parts)

    def _source_compliance_payload(self, candidate: Any, metadata: Any) -> dict[str, str]:
        """Return the approval-facing compliance payload for a shortlist candidate."""

        normalized_metadata = metadata if isinstance(metadata, dict) else {}
        source_type = self._normalize_text(getattr(candidate, "source_type", ""))
        uri = self._normalize_text(getattr(candidate, "uri", ""))
        return build_candidate_compliance_metadata(source_type, uri, metadata=normalized_metadata)

    def _build_eda_summary(self, df_like: Any) -> dict[str, Any]:
        """Summarize the post-quality dataframe-like input without inventing values."""

        rows = self._to_records(df_like)
        columns = self._collect_columns(rows)
        if not columns:
            columns = self._collect_input_columns(df_like)
        columns_for_summary = columns or self._collect_input_columns(df_like)
        notes: list[str] = []

        if not rows:
            notes.append("Датасет пустой, поэтому статистика ограничена структурой входа.")

        source_distribution = self._build_distribution_summary(rows, "source", columns_for_summary)
        effect_label_distribution = self._build_distribution_summary(rows, "effect_label", columns_for_summary)
        rating_summary = self._build_numeric_summary(rows, "rating", columns_for_summary)
        text_length_summary = self._build_text_length_summary(rows, "text", columns_for_summary)
        missing_values_summary = self._build_missing_values_summary(
            rows,
            ["source", "effect_label", "rating", "text"],
            columns_for_summary,
        )

        if not columns:
            notes.append("Колонки не были переданы в dataframe-like input или не удалось извлечь записи.")

        return {
            "n_rows": len(rows),
            "columns": columns,
            "source_distribution": source_distribution,
            "effect_label_distribution": effect_label_distribution,
            "rating_summary": rating_summary,
            "text_length_summary": text_length_summary,
            "missing_values_summary": missing_values_summary,
            "notes": notes,
        }

    def _build_extended_eda_summary(
        self,
        df_like: Any,
        *,
        raw_df_like: Any | None = None,
        quality_report: Any | None = None,
    ) -> dict[str, Any]:
        """Build a richer EDA payload for markdown, HTML, and helper artifacts."""

        summary = self._build_eda_summary(df_like)
        rows = self._to_records(df_like)
        columns = summary.get("columns", []) or self._collect_input_columns(df_like)
        raw_rows = self._to_records(raw_df_like) if raw_df_like is not None else []

        summary["column_count"] = len(columns)
        summary["duplicate_summary"] = self._build_duplicate_summary(rows, columns)
        summary["raw_vs_cleaned"] = self._build_raw_vs_cleaned_summary(raw_rows, rows)
        summary["rating_distribution"] = self._build_rating_distribution(rows, "rating", columns)
        summary["text_length_buckets"] = self._build_text_length_buckets(rows, "text", columns)
        summary["cleaned_word_cloud"] = self._build_cleaned_word_cloud(rows, "text", columns)
        summary["quality_warnings"] = self._extract_quality_warnings(quality_report)

        notes = list(summary.get("notes", []))
        if summary["duplicate_summary"].get("available") and summary["duplicate_summary"].get("duplicate_rows", 0) > 0:
            notes.append("В cleaned датасете ещё есть повторяющиеся строки, это стоит проверить перед финальным обучением.")

        raw_vs_cleaned = summary["raw_vs_cleaned"]
        if raw_vs_cleaned.get("available") and raw_vs_cleaned.get("dropped_rows", 0) > 0:
            notes.append(
                "После quality stage часть строк была удалена или отфильтрована. Это полезно сравнить с логикой чистки и approval flow."
            )

        rating_distribution = summary["rating_distribution"]
        if rating_distribution.get("available") and len(rating_distribution.get("counts", {})) <= 2:
            notes.append("Распределение rating выглядит узким. Для анализа выбросов и дисбаланса стоит смотреть не только на среднее.")

        summary["notes"] = notes
        return summary

    def _build_cleaned_word_cloud(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a compact word-cloud payload from cleaned text values."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        if not rows:
            return {"available": False, "reason": "no_text_values"}

        token_counts: Counter[str] = Counter()
        valid_text_rows = 0
        for row in rows:
            tokens = self._extract_word_cloud_tokens(row.get(column_name))
            if not tokens:
                continue
            valid_text_rows += 1
            token_counts.update(tokens)

        if not token_counts:
            return {"available": False, "reason": "no_terms_after_cleaning"}

        top_terms = token_counts.most_common(18)
        max_count = max(count for _, count in top_terms)
        min_count = min(count for _, count in top_terms)
        terms: list[dict[str, Any]] = []
        for term, count in top_terms:
            emphasis = 1.0 if max_count == min_count else (count - min_count) / (max_count - min_count)
            terms.append(
                {
                    "term": term,
                    "count": count,
                    "font_size": 20 + round(emphasis * 16),
                    "opacity": round(0.68 + emphasis * 0.32, 3),
                }
            )

        return {
            "available": True,
            "column": column_name,
            "valid_text_rows": valid_text_rows,
            "token_count": sum(token_counts.values()),
            "unique_terms": len(token_counts),
            "terms": terms,
        }

    def _extract_word_cloud_tokens(self, value: Any) -> list[str]:
        """Tokenize cleaned text for a lightweight word-cloud preview."""

        if self._is_missing_value(value):
            return []

        text = self._normalize_text(value).lower()
        raw_tokens = re.findall(r"[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё'’_-]*", text)
        tokens: list[str] = []
        for raw_token in raw_tokens:
            token = raw_token.strip("'’_-")
            if len(token) < 3:
                continue
            if any(character.isdigit() for character in token):
                continue
            if token in WORD_CLOUD_STOP_WORDS:
                continue
            tokens.append(token)
        return tokens

    def _build_duplicate_summary(
        self,
        rows: list[dict[str, Any]],
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate duplicate rows directly from the cleaned dataframe-like input."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if not columns:
            return {"available": False, "reason": "column_absent"}

        if not rows:
            return {"available": False, "reason": "no_values"}

        seen: set[str] = set()
        duplicate_rows = 0
        for row in rows:
            fingerprint = repr({column: row.get(column) for column in columns})
            if fingerprint in seen:
                duplicate_rows += 1
                continue
            seen.add(fingerprint)

        return {"available": True, "duplicate_rows": duplicate_rows}

    def _build_raw_vs_cleaned_summary(
        self,
        raw_rows: list[dict[str, Any]],
        cleaned_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare raw and cleaned row counts when the raw frame is available."""

        if not raw_rows:
            return {"available": False, "reason": "no_values"}

        raw_count = len(raw_rows)
        cleaned_count = len(cleaned_rows)
        return {
            "available": True,
            "raw_rows": raw_count,
            "cleaned_rows": cleaned_count,
            "dropped_rows": max(0, raw_count - cleaned_count),
            "kept_fraction": (cleaned_count / raw_count) if raw_count else 0.0,
        }

    def _build_rating_distribution(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a compact distribution for rating values."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        counts: dict[str, int] = {}
        for row in rows:
            value = row.get(column_name)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_value != numeric_value:
                continue

            bucket = self._format_numeric(numeric_value)
            counts[bucket] = counts.get(bucket, 0) + 1

        if not counts:
            return {"available": False, "reason": "no_numeric_values"}

        return {"available": True, "counts": counts}

    def _build_text_length_buckets(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build coarse text-length buckets for a quick EDA view."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        counts = {"0-49": 0, "50-99": 0, "100-199": 0, "200+": 0}
        has_values = False
        for row in rows:
            value = row.get(column_name)
            if self._is_missing_value(value):
                continue
            length = len(self._normalize_text(value))
            has_values = True
            if length < 50:
                counts["0-49"] += 1
            elif length < 100:
                counts["50-99"] += 1
            elif length < 200:
                counts["100-199"] += 1
            else:
                counts["200+"] += 1

        if not has_values:
            return {"available": False, "reason": "no_text_values"}

        return {"available": True, "counts": counts}

    def _extract_quality_warnings(self, quality_report: Any) -> list[str]:
        """Extract warning text from the optional quality-report payload."""

        if quality_report is None:
            return []

        if hasattr(quality_report, "as_dict"):
            payload = quality_report.as_dict()
        elif isinstance(quality_report, dict):
            payload = quality_report
        else:
            return []

        warnings = payload.get("warnings", [])
        if not isinstance(warnings, list):
            return []
        return [self._normalize_text(item) for item in warnings if self._normalize_text(item)]

    def _collect_columns(self, rows: list[dict[str, Any]]) -> list[str]:
        """Collect columns in first-seen order from row dictionaries."""

        columns: list[str] = []
        for row in rows:
            for key in row.keys():
                normalized_key = self._normalize_text(key)
                if normalized_key and normalized_key not in columns:
                    columns.append(normalized_key)
        return columns

    def _collect_input_columns(self, df_like: Any) -> list[str]:
        """Collect column names from dataframe-like inputs when row materialization is empty."""

        raw_columns = getattr(df_like, "columns", None)
        if raw_columns is None:
            return []

        columns: list[str] = []
        for column_name in list(raw_columns):
            normalized_column = self._normalize_text(column_name)
            if normalized_column and normalized_column not in columns:
                columns.append(normalized_column)
        return columns

    def _build_distribution_summary(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Count categorical values only when the requested column is present."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        if not rows:
            return {"available": False, "reason": "no_values"}

        counts: dict[str, int] = {}
        for row in rows:
            value = self._normalize_text(row.get(column_name))
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1

        if not counts:
            return {"available": False, "reason": "no_values"}

        return {"available": True, "column": column_name, "counts": counts}

    def _build_numeric_summary(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Summarize numeric values without clamping or synthetic defaults."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        if not rows:
            return {"available": False, "reason": "no_numeric_values"}

        values: list[float] = []
        for row in rows:
            value = row.get(column_name)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_value != numeric_value:
                continue
            values.append(numeric_value)

        if not values:
            return {"available": False, "reason": "no_numeric_values"}

        return {
            "available": True,
            "column": column_name,
            "valid_count": len(values),
            "missing_or_invalid_count": len(rows) - len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    def _build_text_length_summary(
        self,
        rows: list[dict[str, Any]],
        column_name: str,
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Summarize text length in characters for the existing text column."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        if column_name not in columns:
            return {"available": False, "reason": "column_absent"}

        if not rows:
            return {"available": False, "reason": "no_text_values"}

        lengths: list[int] = []
        for row in rows:
            value = row.get(column_name)
            if self._is_missing_value(value):
                continue

            normalized_value = self._normalize_text(value)

            lengths.append(len(normalized_value))

        if not lengths:
            return {"available": False, "reason": "no_text_values"}

        return {
            "available": True,
            "column": column_name,
            "valid_count": len(lengths),
            "missing_or_invalid_count": len(rows) - len(lengths),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
            "mean_chars": sum(lengths) / len(lengths),
        }

    def _build_missing_values_summary(
        self,
        rows: list[dict[str, Any]],
        key_columns: list[str],
        available_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Summarize missing values for the columns the report cares about most."""

        columns = available_columns if available_columns is not None else self._collect_columns(rows)
        summary: dict[str, Any] = {}
        for column_name in key_columns:
            if column_name not in columns:
                summary[column_name] = {"available": False, "reason": "column_absent"}
                continue

            missing_count = 0
            for row in rows:
                if self._is_missing_value(row.get(column_name)):
                    missing_count += 1

            summary[column_name] = {
                "available": True,
                "column": column_name,
                "missing_count": missing_count,
                "missing_fraction": missing_count / len(rows) if rows else 0.0,
            }

        return summary

    def _is_missing_value(self, value: Any) -> bool:
        """Detect missing textual values in a dataframe-friendly way."""

        if value is None:
            return True

        try:
            if value != value:
                return True
        except Exception:
            return True

        return not self._normalize_text(value)

    def _format_count_map(self, counts: dict[str, int]) -> str:
        """Render a short value-count list for markdown reports."""

        if not counts:
            return "нет значений"
        return ", ".join(f"{key}: {value}" for key, value in counts.items())

    def _describe_absence(self, payload: dict[str, Any]) -> str:
        """Explain why a metric is unavailable in a concise Russian phrase."""

        reason = payload.get("reason", "unknown")
        if reason == "column_absent":
            return "колонка отсутствует"
        if reason == "no_values":
            return "колонка есть, но значений нет"
        if reason == "no_numeric_values":
            return "числовые значения не найдены"
        if reason == "no_text_values":
            return "текстовые значения не найдены"
        if reason == "no_terms_after_cleaning":
            return "no tokens left after cleaning"
        return f"недоступно: {reason}"

    def _render_dashboard_word_cloud(self, payload: dict[str, Any]) -> str:
        """Render a lightweight cleaned word cloud for the dashboard."""

        if not payload.get("available"):
            return f'<p class="muted">{self._escape_html(self._describe_absence(payload or {}))}</p>'

        terms = payload.get("terms", [])
        if not isinstance(terms, list) or not terms:
            return '<p class="muted">No cleaned tokens available for preview.</p>'

        chips: list[str] = []
        for term_payload in terms:
            term = self._normalize_text(term_payload.get("term"))
            if not term:
                continue

            try:
                font_size = int(term_payload.get("font_size", 22))
            except (TypeError, ValueError):
                font_size = 22

            try:
                opacity = float(term_payload.get("opacity", 0.82))
            except (TypeError, ValueError):
                opacity = 0.82

            chips.append(
                '<span class="word-chip" style="font-size: {font_size}px; opacity: {opacity};">'
                '<span>{term}</span><small>{count}</small></span>'.format(
                    font_size=max(18, min(font_size, 42)),
                    opacity=max(0.55, min(opacity, 1.0)),
                    term=self._escape_html(term),
                    count=self._escape_html(term_payload.get("count")),
                )
            )

        if not chips:
            return '<p class="muted">No cleaned tokens available for preview.</p>'

        meta = (
            "Text rows with tokens: {rows}. Tokens after cleaning: {tokens}. Unique terms: {unique}.".format(
                rows=self._format_numeric(payload.get("valid_text_rows", 0)),
                tokens=self._format_numeric(payload.get("token_count", 0)),
                unique=self._format_numeric(payload.get("unique_terms", 0)),
            )
        )
        return (
            '<div class="word-cloud-shell">'
            f'<div class="word-cloud-meta">{self._escape_html(meta)}</div>'
            f'<div class="word-cloud">{"".join(chips)}</div>'
            "</div>"
        )

    def _html_metric_block(self, title: str, body: str) -> str:
        """Render one HTML metric section for the EDA dashboard."""

        return f'<section class="section"><h3>{self._escape_html(title)}</h3>{body}</section>'

    def _escape_html(self, value: Any) -> str:
        """Escape text for safe inline HTML rendering."""

        text = self._normalize_text(value)
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _format_distribution_html(self, payload: dict[str, Any]) -> str:
        """Render a small HTML table or an absence description for distributions."""

        if not payload.get("available"):
            return f'<p class="muted">{self._escape_html(self._describe_absence(payload))}</p>'

        counts = payload.get("counts", {})
        rows = "".join(
            f"<tr><td>{self._escape_html(key)}</td><td>{self._escape_html(value)}</td></tr>"
            for key, value in counts.items()
        )
        return f"<table><tbody>{rows}</tbody></table>"

    def _format_summary_dict_html(self, payload: dict[str, Any]) -> str:
        """Render a compact HTML view for summary dictionaries."""

        if not payload.get("available"):
            return f'<p class="muted">{self._escape_html(self._describe_absence(payload))}</p>'

        rows = "".join(
            f"<tr><td>{self._escape_html(key)}</td><td>{self._escape_html(self._format_numeric(value) if isinstance(value, (int, float)) else value)}</td></tr>"
            for key, value in payload.items()
            if key not in {"available", "column"}
        )
        return f"<table><tbody>{rows}</tbody></table>"

    def _format_missing_values_html(self, payload: dict[str, Any]) -> str:
        """Render missing-value summaries for the HTML report."""

        if not payload:
            return '<p class="muted">Нет данных.</p>'

        rows: list[str] = []
        for column_name, column_summary in payload.items():
            if column_summary.get("available"):
                rows.append(
                    f"<tr><td>{self._escape_html(column_name)}</td><td>{self._escape_html(column_summary.get('missing_count'))}</td><td>{self._escape_html(self._format_numeric(column_summary.get('missing_fraction')))}</td></tr>"
                )
            else:
                rows.append(
                    f"<tr><td>{self._escape_html(column_name)}</td><td colspan=\"2\">{self._escape_html(self._describe_absence(column_summary))}</td></tr>"
                )

        return "<table><thead><tr><th>Column</th><th>Missing</th><th>Fraction</th></tr></thead><tbody>{rows}</tbody></table>".format(
            rows="".join(rows)
        )

    def _format_raw_vs_cleaned(self, payload: dict[str, Any]) -> str:
        """Render raw-vs-cleaned comparison for the HTML report."""

        if not payload.get("available"):
            return f'<p class="muted">{self._escape_html(self._describe_absence(payload))}</p>'

        rows = "".join(
            f"<tr><td>{self._escape_html(key)}</td><td>{self._escape_html(self._format_numeric(value) if isinstance(value, (int, float)) else value)}</td></tr>"
            for key, value in payload.items()
            if key != "available"
        )
        return f"<table><tbody>{rows}</tbody></table>"

    def _build_eda_plotly_html(self, summary: dict[str, Any]) -> str:
        """Render inline Plotly charts when available, otherwise fall back to static HTML."""

        try:
            import plotly.graph_objects as go  # type: ignore[import-not-found]
        except Exception:
            return "<p class=\"muted\">Plotly не установлен, поэтому HTML-отчет показывает summary без интерактивных графиков.</p>"

        figures: list[str] = []
        chart_specs = [
            ("Source distribution", summary.get("source_distribution", {})),
            ("Effect label distribution", summary.get("effect_label_distribution", {})),
            ("Rating distribution", summary.get("rating_distribution", {})),
            ("Text length buckets", summary.get("text_length_buckets", {})),
        ]

        for index, (title, payload) in enumerate(chart_specs):
            if not payload.get("available"):
                continue
            counts = payload.get("counts", {})
            figure = go.Figure(
                data=[
                    go.Bar(
                        x=list(counts.keys()),
                        y=list(counts.values()),
                        marker_color="#b7791f",
                    )
                ]
            )
            figure.update_layout(
                title=title,
                template="plotly_white",
                margin=dict(l=24, r=24, t=56, b=24),
                height=360,
            )
            figures.append(
                figure.to_html(
                    full_html=False,
                    include_plotlyjs="inline" if index == 0 else False,
                )
            )

        if not figures:
            return "<p class=\"muted\">Для текущего датасета недостаточно значений, чтобы построить графики.</p>"
        return "".join(figures)

    def _candidate_to_approval_record(self, candidate: Any) -> dict[str, Any]:
        """Convert a shortlist candidate into a stable helper artifact row.

        The helper artifact is intentionally simple so a human can inspect the markdown report
        while an approval workflow can read the JSON shortlist without extra parsing logic.
        """

        metadata = getattr(candidate, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        compliance = self._source_compliance_payload(candidate, metadata)

        return {
            "source_id": self._normalize_text(getattr(candidate, "source_id", "")),
            "source_type": self._normalize_text(getattr(candidate, "source_type", "")),
            "title": self._normalize_text(getattr(candidate, "title", "")),
            "uri": self._normalize_text(getattr(candidate, "uri", "")),
            "score": getattr(candidate, "score", 0.0),
            "license": compliance["license"],
            "license_status": compliance["license_status"],
            "robots_txt_status": compliance["robots_txt_status"],
            "robots_txt_url": compliance["robots_txt_url"],
            "approval_notes": compliance["approval_notes"],
            "metadata": dict(metadata),
        }
