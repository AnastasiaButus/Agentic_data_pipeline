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

## Human-in-the-Loop Point

Human-in-the-loop review happens after annotation. Low-confidence rows are exported to a review queue, corrected labels can be merged back by canonical `id`, and downstream stages continue from the reviewed data.

Label Studio export is supported for annotated batches.

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

## Artifacts Produced

A successful demo run produces artifacts such as:

- `final_report.md`
- `data/interim/model_metrics.json`
- `data/interim/review_queue.csv`
- `data/raw/discovered_sources.json`
- `data/raw/merged_raw.parquet`

## Current Limitations

- Online discovery is not complete.
- The demo datasets are intentionally small and synthetic/local.
- The offline demo path is meant for reproducible coursework-style runs, not for production use.
- Some stages are intentionally deterministic to keep the baseline stable for local execution.
