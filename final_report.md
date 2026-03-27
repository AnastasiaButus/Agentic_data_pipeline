# Final Report

## Sources

- n_candidates: 1
- source_report_path: reports/source_report.md

## Quality

- quality_report_path: reports/quality_report.md
- warnings: ['missing values detected']

## EDA

- eda_report_path: reports\eda_report.md
- eda_html_report_path: reports\eda_report.html
- eda_context_path: data\interim\eda_context.json
- n_rows: 3

## Annotation

- annotation_report_path: reports/annotation_report.md
- annotation_trace_report_path: reports\annotation_trace_report.md
- annotation_trace_context_path: data\interim\annotation_trace.json
- confidence_threshold: 0.6
- n_low_confidence: 0

## Review

- status: skipped_missing_corrected_queue
- review_queue_rows: 0
- review_required: False
- reviewer_action: review queue already processed or not required
- next_step: active learning and training completed for current run
- review_queue_report_path: reports\review_queue_report.md
- review_queue_context_path: data\interim\review_queue_context.json
- review_merge_report_path: reports\review_merge_report.md
- review_merge_context_path: data\interim\review_merge_context.json

## Approval

- approved_sources_path: data/raw/approved_sources.json
- n_approved_sources: 1
- approval_status: skipped_missing_file

## Active Learning

- al_report_path: reports/al_report.md
- history: [{'iteration': 1, 'n_labeled': 2, 'accuracy': 0.0, 'f1': 0.0}, {'iteration': 2, 'n_labeled': 3, 'accuracy': 0.0, 'f1': 0.0}]

## Training

- accuracy: 1.0
- f1: 1.0

## Artifacts

- model_path: D:\Projects\universal-agentic-data-pipeline\data\interim\model_artifact.pkl
- vectorizer_path: D:\Projects\universal-agentic-data-pipeline\data\interim\vectorizer_artifact.pkl
- metrics_path: D:\Projects\universal-agentic-data-pipeline\data\interim\model_metrics.json
