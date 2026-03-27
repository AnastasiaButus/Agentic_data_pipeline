# Universal Agentic Data Pipeline

## Project Overview

This repository contains a modular Python pipeline for text-based review data. The current baseline is runnable in offline demo mode and is intentionally kept narrow: it supports local demo execution, canonical normalization, annotation, human review, active learning, training, and reporting.

The project is not presented as production-ready. The real online discovery path is still incomplete, and the current working baseline relies on offline demo payloads for repeatable local runs.

## Pipeline Stages

- Source discovery
- Data collection
- Data quality checks
- Annotation
- Human review queue export and merge
- Active learning
- Training
- Reporting

The main training target is `text -> effect_label`.

## Discovery Modes

The repository currently supports two discovery modes:

- Offline demo mode for the persistent fitness and minecraft configs. This path is deterministic and designed for reproducible local coursework runs.
- Emerging online discovery mode for non-demo configs. The online discovery path now includes a Hugging Face datasets discovery MVP plus a narrow GitHub repository discovery MVP that both search by the current request topic without auto-adding artificial API, GitHub, or scrape stubs.

The online path is discovery-only at this stage. It does not replace the offline demo flow and should not be read as a fully production-ready online agent.

## Human-in-the-Loop Point

Human-in-the-loop review happens after annotation. Low-confidence rows are exported to a review queue, corrected labels can be merged back by canonical `id`, and downstream stages continue from the reviewed data.

Label Studio export is supported for annotated batches.

The source shortlist report is now written in Russian and is meant for human review before approval. It lists discovered candidates in a compact, readable format and also saves a machine-readable `data/raw/approval_candidates.json` helper artifact for review tooling.

`data/raw/approved_sources.json` remains the separate human-edited approval input.

After auto-annotation, the pipeline also writes `data/interim/review_queue.csv` for manual checking, a Russian `reports/review_queue_report.md` for the reviewer, and a machine-readable `data/interim/review_queue_context.json` helper artifact for tooling or lightweight review UI support.

The corrected review queue is still edited by a human separately in `data/interim/review_queue_corrected.csv`.

## Repository Structure

- `src/` - pipeline implementation, agents, services, providers, and ML helpers
- `tests/` - unit, integration, and end-to-end tests
- `configs/` - demo configs for offline runs
- `data/` - generated artifacts from local runs

## How to Run Offline Demos

Run the pipeline from the repository root with one of the persistent demo configs:

```bash
python run_pipeline.py --config configs/demo_fitness.yaml
python run_pipeline.py --config configs/demo_minecraft.yaml
```

These demo paths are designed to work offline and do not require network access.

## Hugging Face Discovery MVP

For non-demo configs, source discovery can now query the public Hugging Face datasets search API using the request topic. This is a narrow discovery and shortlisting step only.

For real Hugging Face candidates, discovery stores the canonical dataset id in `uri` and keeps the dataset page URL in metadata when relevant.

If the online lookup fails, the service falls back safely instead of breaking the pipeline. The offline demo path remains unchanged and still uses the local deterministic payloads.

## GitHub Discovery MVP

For non-demo configs, source discovery can also query the public GitHub repository search API using the current request topic.

This is a discovery-only MVP. It maps real repository results into shortlist candidates, falls back safely on lookup failures, and does not add any GitHub collection logic.

## Hugging Face Collection MVP

The online capability now also includes a narrow Hugging Face collection path for shortlisted datasets. The collection loader accepts either a Hugging Face dataset id or a Hugging Face dataset URL and normalizes it before loading.

This is still not a full online pipeline. It is a focused discovery-and-collection MVP, while the offline demo mode remains the main reproducible coursework path.

## Source Approval Gate MVP

After discovery, shortlist candidates can be filtered through a minimal approval gate before any future collection step.

The MVP uses a simple `data/raw/approved_sources.json` file containing a JSON list of approved `source_id` strings. If the file is missing, the shortlist helper returns the original shortlist unchanged.

This is a discovery-side approval checkpoint only. It is not a UI, not a production approval workflow, and it does not change the offline demo mode.

## Approval-Aware Collection MVP

The orchestration layer now applies the approval helper between discovery and collection. In practice, the pipeline discovers sources, filters them through `approved_sources.json` when present, and then passes the approved subset to collection.

The approval-aware path now exposes simple file-based status semantics such as missing-file, applied, and empty-subset cases. This remains a file-based MVP, not a full approval subsystem, and it is intentionally narrow so the offline demo baseline stays intact.

## Artifacts Produced

A successful demo run produces artifacts such as:

- `final_report.md`
- `data/interim/model_metrics.json`
- `data/interim/review_queue.csv`
- `data/raw/discovered_sources.json`
- `data/raw/merged_raw.parquet`

## Current Limitations

- Online discovery is not complete.
- The online capability currently starts with Hugging Face datasets discovery and does not cover the full collection pipeline.
- The online capability now includes Hugging Face discovery, GitHub repository discovery MVP, and a minimal Hugging Face collection MVP, but it is still not a full production-ready online pipeline.
- The approval gate MVP is file-based and intentionally minimal; it only filters shortlist candidates by approved `source_id` values.
- The source shortlist report is a Russian MVP for human review, not a full approval UI.
- The `approval_candidates.json` artifact is a helper shortlist for review automation, not an approval decision file.
- `review_queue.csv` is the manual review queue, `review_queue_report.md` is the Russian reviewer guide, and `review_queue_context.json` is a helper artifact for tooling.
- The demo datasets are intentionally small and synthetic/local.
- The offline demo path is meant for reproducible coursework-style runs, not for production use.
- Some stages are intentionally deterministic to keep the baseline stable for local execution.
