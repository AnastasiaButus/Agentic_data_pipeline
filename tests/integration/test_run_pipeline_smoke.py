"""Smoke test for the CLI entry point and orchestration layer."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.config import load_config


def _runtime_config_text(template_text: str, tmp_path: Path, provider: str) -> str:
    """Materialize a local runtime config while overriding the annotation provider flag."""

    return (
        template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}")
        .replace("llm_provider: mock", f"llm_provider: {provider}")
    )


def test_run_pipeline_smoke_creates_final_report_and_metrics(monkeypatch, tmp_path: Path) -> None:
    """The CLI should run end-to-end on the persistent fitness demo config without monkeypatch."""

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    assert "fitness supplements" in template_text

    config_path = tmp_path / "demo_fitness.runtime.yaml"
    config_path.write_text(
        template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}"),
        encoding="utf-8",
    )

    loaded_config = load_config(config_path)
    assert loaded_config.request.topic == "fitness supplements"
    assert loaded_config.annotation.use_llm is True
    assert loaded_config.annotation.llm_provider == "mock"
    assert loaded_config.runtime.mode == "offline_demo"

    from run_pipeline import main

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert (tmp_path / "final_report.md").exists()
    final_report = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert "## Runtime" in final_report
    assert "## Dashboard" in final_report
    assert "effective_mode: offline_demo" in final_report
    assert "dashboard_path: reports/run_dashboard.html" in final_report
    assert "## Online Governance" in final_report
    assert "## Approval" in final_report
    assert "## EDA" in final_report
    assert "## Annotation" in final_report
    assert "## Agreement" in final_report
    assert "eda_report_path" in final_report
    assert "eda_html_report_path" in final_report
    assert "eda_context_path" in final_report
    assert "annotation_trace_report_path" in final_report
    assert "annotation_trace_context_path" in final_report
    assert "approval_status: skipped_missing_file" in final_report
    assert "governance_report_path: reports/online_governance_report.md" in final_report
    assert "github_auth_mode: not_used" in final_report
    assert "review_workspace_path: reports/review_workspace.html" in final_report
    assert "agreement_report_path: reports/review_agreement_report.md" in final_report
    assert "review_required:" in final_report
    assert "next_step:" in final_report
    approval_candidates = json.loads((tmp_path / "data" / "raw" / "approval_candidates.json").read_text(encoding="utf-8"))
    assert isinstance(approval_candidates, list)
    assert len(approval_candidates) == 1
    assert approval_candidates[0]["source_id"] == "demo_fitness_scrape"
    assert approval_candidates[0]["title"] == "Fitness Supplements Offline Demo"
    assert approval_candidates[0]["license"] == "offline_demo_fixture"
    assert approval_candidates[0]["license_status"] == "demo_fixture"
    assert approval_candidates[0]["robots_txt_status"] == "not_applicable_local_demo"
    review_queue_report = (tmp_path / "reports" / "review_queue_report.md").read_text(encoding="utf-8")
    assert "# Очередь ручной проверки" in review_queue_report
    assert "ручной проверки после авторазметки" in review_queue_report
    assert "## Reviewer guide" in review_queue_report
    assert "## Next step" in review_queue_report
    review_workspace = (tmp_path / "reports" / "review_workspace.html").read_text(encoding="utf-8")
    assert "HITL Review Workspace" in review_workspace
    assert "review_queue_corrected.csv" in review_workspace
    assert "reviewed_effect_label" in review_workspace
    agreement_report = (tmp_path / "reports" / "review_agreement_report.md").read_text(encoding="utf-8")
    agreement_context = json.loads((tmp_path / "data" / "interim" / "review_agreement_context.json").read_text(encoding="utf-8"))
    assert "Review agreement report" in agreement_report
    assert "compared_rows: 0" in agreement_report
    assert agreement_context["compared_rows"] == 0
    assert agreement_context["kappa_status"] == "not_available_no_compared_rows"
    review_queue_context = json.loads((tmp_path / "data" / "interim" / "review_queue_context.json").read_text(encoding="utf-8"))
    assert review_queue_context["confidence_threshold"] == loaded_config.annotation.confidence_threshold
    assert review_queue_context["n_rows"] >= 0
    assert review_queue_context["label_options"] == loaded_config.annotation.effect_labels
    assert "review_required" in review_queue_context
    assert review_queue_context["current_stage"] == "human_review"
    assert "next_step" in review_queue_context
    annotation_trace_report = (tmp_path / "reports" / "annotation_trace_report.md").read_text(encoding="utf-8")
    assert "Трассировка annotation contract" in annotation_trace_report
    assert "Верни только JSON" in annotation_trace_report
    annotation_trace_context = json.loads((tmp_path / "data" / "interim" / "annotation_trace.json").read_text(encoding="utf-8"))
    assert "prompt_contract" in annotation_trace_context
    assert "parser_contract" in annotation_trace_context
    assert annotation_trace_context["llm_mode"] == "classify_effect"
    assert annotation_trace_context["n_rows"] >= 0
    eda_report = (tmp_path / "reports" / "eda_report.md").read_text(encoding="utf-8")
    eda_html_report = (tmp_path / "reports" / "eda_report.html").read_text(encoding="utf-8")
    assert "EDA-пакет" in eda_report
    assert "Это расширенный честный EDA-отчет" in eda_report
    assert "Raw vs cleaned" in eda_report
    assert "Дубликаты" in eda_report
    assert "source distribution" not in eda_report.lower()
    assert "<html" in eda_html_report.lower()
    assert "EDA Report" in eda_html_report
    eda_context = json.loads((tmp_path / "data" / "interim" / "eda_context.json").read_text(encoding="utf-8"))
    assert "n_rows" in eda_context
    assert "column_count" in eda_context
    assert "columns" in eda_context
    assert "raw_vs_cleaned" in eda_context
    assert "duplicate_summary" in eda_context
    assert "source_distribution" in eda_context
    assert "effect_label_distribution" in eda_context
    assert "rating_summary" in eda_context
    assert "rating_distribution" in eda_context
    assert "text_length_summary" in eda_context
    assert "text_length_buckets" in eda_context
    assert "cleaned_word_cloud" in eda_context
    assert "missing_values_summary" in eda_context
    assert "quality_warnings" in eda_context
    source_report = (tmp_path / "reports" / "source_report.md").read_text(encoding="utf-8")
    assert "Короткий shortlist источников" in source_report
    assert "ручного просмотра и одобрения" in source_report
    assert "Fitness Supplements Offline Demo" in source_report
    assert "score:" in source_report
    assert "license: offline_demo_fixture" in source_report
    assert "robots_txt_status: not_applicable_local_demo" in source_report
    governance_report = (tmp_path / "reports" / "online_governance_report.md").read_text(encoding="utf-8")
    governance_context = json.loads((tmp_path / "data" / "raw" / "online_governance_summary.json").read_text(encoding="utf-8"))
    assert "Online governance and fallback" in governance_report
    assert "provider_id: huggingface_datasets" in governance_report
    assert "configured_but_inactive_for_runtime" in governance_report
    assert governance_context["github_auth_mode"] == "not_used"
    dashboard_html = (tmp_path / "reports" / "run_dashboard.html").read_text(encoding="utf-8")
    assert "Pipeline Operator Dashboard" in dashboard_html
    assert "effective_mode: offline_demo" in dashboard_html
    assert "../final_report.md" in dashboard_html
    assert "review_agreement_report.md" in dashboard_html
    assert "review_workspace.html" in dashboard_html
    assert "online_governance_report.md" in dashboard_html
    assert "Cleaned word cloud" in dashboard_html
    assert "review_queue_report.md" in dashboard_html
    assert "review_queue_corrected.csv" in dashboard_html
    assert (tmp_path / "data" / "interim" / "review_queue.csv").exists()
    assert (tmp_path / "data" / "interim" / "model_metrics.json").exists()
    assert (tmp_path / "data" / "interim" / "review_queue.csv").exists()
    assert "Fitness Supplements Offline Demo" in (tmp_path / "data" / "raw" / "discovered_sources.json").read_text(encoding="utf-8")


def test_eda_pack_handles_empty_quality_output(monkeypatch, tmp_path: Path) -> None:
    """EDA reporting should stay truthful and stable when quality output is empty."""

    import pandas as pd

    from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
    from src.core.context import PipelineContext
    from src.services.pipeline_controller import PipelineController

    config = AppConfig(
        project=ProjectConfig(name="eda-empty-demo", root_dir=tmp_path),
        source=SourceConfig(),
        annotation=AnnotationConfig(confidence_threshold=0.0, effect_labels=[]),
        request=RequestConfig(topic="fitness supplements"),
    )
    context = PipelineContext.from_config(config)

    class FakeRegistry:
        def exists(self, path):
            return False

        def save_json(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return target

        def save_markdown(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
            return target

    class FakeDiscoveryService:
        def __init__(self) -> None:
            self.registry = FakeRegistry()

        def run(self):
            return []

        def load_approved_candidates(self, sources):
            return list(sources)

    class FakeCollectionAgent:
        def run(self, sources):
            return pd.DataFrame(columns=["id", "source", "text", "rating"])

    class FakeQualityAgent:
        def detect_issues(self, collected):
            return {"warnings": []}

        def run(self, collected):
            return collected

    class FakeAnnotationAgent:
        def auto_label(self, df):
            return []

        def check_quality(self, annotated):
            return {"confidence_threshold": 0.6, "n_low_confidence": 0, "n_rows": 0}

        def _effect_label_vocabulary(self):
            return ["energy", "side_effects", "other"]

        def get_annotation_trace(self):
            return {
                "prompt_contract": {
                    "language": "ru",
                    "task": "auto_annotation",
                    "input_fields": ["text", "rating"],
                    "output_fields": ["effect_label", "sentiment_label", "confidence"],
                    "sentiment_labels": ["negative", "neutral", "positive"],
                    "effect_labels": ["energy", "side_effects", "other"],
                    "prompt_preview": "Ты разметчик отзывов о пищевых добавках. Верни только JSON без пояснений, markdown и лишнего текста.",
                    "expected_output_example": {"effect_label": "other", "sentiment_label": "positive", "confidence": 0.5},
                },
                "parser_contract": {
                    "preferred_format": "json",
                    "accepted_fallbacks": ["key_value", "partial_json", "deterministic_fallback"],
                    "parse_status_counts": {},
                    "fallback_reason_counts": {},
                },
                "llm_mode": "unknown",
                "n_rows": 0,
                "n_fallback_rows": 0,
                "fallback_rows": [],
            }

    class FakeReviewQueueService:
        def export_low_confidence_queue(self, df, threshold=0.7):
            target = tmp_path / "data" / "interim" / "review_queue.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("id\n", encoding="utf-8")
            return []

        def load_corrected_queue(self):
            raise FileNotFoundError

        def merge_reviewed_labels(self, original_df, corrected_df):
            return original_df

    class FakeActiveLearningAgent:
        def run_cycle(self, reviewed, strategy, seed_size, n_iterations, batch_size):
            return [], reviewed

    class FakeTrainingService:
        def train(self, df):
            return ({"model_path": str(tmp_path / "model.pkl")}, {"accuracy": 1.0, "f1": 1.0})

    controller = PipelineController(
        context,
        discovery_service=FakeDiscoveryService(),
        collection_agent=FakeCollectionAgent(),
        quality_agent=FakeQualityAgent(),
        annotation_agent=FakeAnnotationAgent(),
        review_queue_service=FakeReviewQueueService(),
        active_learning_agent=FakeActiveLearningAgent(),
        training_service=FakeTrainingService(),
    )

    result = controller.run()

    eda_report = (tmp_path / "reports" / "eda_report.md").read_text(encoding="utf-8")
    eda_context = json.loads((tmp_path / "data" / "interim" / "eda_context.json").read_text(encoding="utf-8"))
    final_report = (tmp_path / "final_report.md").read_text(encoding="utf-8")

    assert result["review_status"] == "skipped_missing_corrected_queue"
    assert "Датасет пустой" in eda_report
    annotation_trace_report = (tmp_path / "reports" / "annotation_trace_report.md").read_text(encoding="utf-8")
    annotation_trace_context = json.loads((tmp_path / "data" / "interim" / "annotation_trace.json").read_text(encoding="utf-8"))
    assert "Трассировка annotation contract" in annotation_trace_report
    assert annotation_trace_context["n_rows"] == 0
    assert annotation_trace_context["n_fallback_rows"] == 0
    assert (tmp_path / "reports" / "annotation_trace_report.md").exists()
    assert (tmp_path / "data" / "interim" / "annotation_trace.json").exists()
    assert eda_context["n_rows"] == 0
    assert eda_context["column_count"] == 4
    assert eda_context["columns"] == ["id", "source", "text", "rating"]
    assert eda_context["raw_vs_cleaned"]["available"] is False
    assert eda_context["duplicate_summary"]["available"] is False
    assert eda_context["source_distribution"]["available"] is False
    assert eda_context["source_distribution"]["reason"] == "no_values"
    assert eda_context["effect_label_distribution"]["available"] is False
    assert eda_context["effect_label_distribution"]["reason"] == "column_absent"
    assert eda_context["rating_summary"]["available"] is False
    assert eda_context["rating_summary"]["reason"] == "no_numeric_values"
    assert eda_context["rating_distribution"]["available"] is False
    assert eda_context["rating_distribution"]["reason"] == "no_numeric_values"
    assert eda_context["text_length_summary"]["available"] is False
    assert eda_context["text_length_summary"]["reason"] == "no_text_values"
    assert eda_context["text_length_buckets"]["available"] is False
    assert eda_context["text_length_buckets"]["reason"] == "no_text_values"
    assert eda_context["missing_values_summary"]["source"]["available"] is True
    assert eda_context["missing_values_summary"]["source"]["missing_count"] == 0
    assert eda_context["missing_values_summary"]["source"]["missing_fraction"] == 0.0
    assert "eda_report_path" in final_report
    assert "eda_html_report_path" in final_report
    assert "eda_context_path" in final_report


def test_eda_pack_counts_blank_text_as_missing(monkeypatch, tmp_path: Path) -> None:
    """Blank, whitespace-only, None and NaN text values should count as missing."""

    import pandas as pd

    from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
    from src.core.context import PipelineContext
    from src.services.pipeline_controller import PipelineController

    config = AppConfig(
        project=ProjectConfig(name="eda-text-empty-demo", root_dir=tmp_path),
        source=SourceConfig(),
        annotation=AnnotationConfig(confidence_threshold=0.0, effect_labels=[]),
        request=RequestConfig(topic="fitness supplements"),
    )
    context = PipelineContext.from_config(config)

    class FakeRegistry:
        def exists(self, path):
            return False

        def save_json(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return target

        def save_markdown(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
            return target

    class FakeDiscoveryService:
        def __init__(self) -> None:
            self.registry = FakeRegistry()

        def run(self):
            return []

        def load_approved_candidates(self, sources):
            return list(sources)

    class FakeCollectionAgent:
        def run(self, sources):
            return pd.DataFrame(
                [
                    {"id": "1", "source": "demo", "text": "Useful insight", "rating": 5},
                    {"id": "2", "source": "demo", "text": "", "rating": 3},
                    {"id": "3", "source": "demo", "text": "   ", "rating": 1},
                    {"id": "4", "source": "demo", "text": None, "rating": 4},
                    {"id": "5", "source": "demo", "text": float("nan"), "rating": 2},
                ]
            )

    class FakeQualityAgent:
        def detect_issues(self, collected):
            return {"warnings": []}

        def run(self, collected):
            return collected

    class FakeAnnotationAgent:
        def auto_label(self, df):
            return df

        def check_quality(self, annotated):
            return {"confidence_threshold": 0.6, "n_low_confidence": 0, "n_rows": 5}

        def _effect_label_vocabulary(self):
            return ["energy", "side_effects", "other"]

        def get_annotation_trace(self):
            return {
                "prompt_contract": {"language": "ru", "task": "auto_annotation", "input_fields": ["text", "rating"], "output_fields": ["effect_label", "sentiment_label", "confidence"], "sentiment_labels": ["negative", "neutral", "positive"], "effect_labels": ["energy", "side_effects", "other"], "prompt_preview": "Ты разметчик отзывов о пищевых добавках.", "expected_output_example": {"effect_label": "other", "sentiment_label": "positive", "confidence": 0.5}},
                "parser_contract": {"preferred_format": "json", "accepted_fallbacks": ["key_value", "partial_json", "deterministic_fallback"], "parse_status_counts": {}, "fallback_reason_counts": {}},
                "llm_mode": "unknown",
                "n_rows": 5,
                "n_fallback_rows": 0,
                "fallback_rows": [],
            }

    class FakeReviewQueueService:
        def export_low_confidence_queue(self, df, threshold=0.7):
            target = tmp_path / "data" / "interim" / "review_queue.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("id\n", encoding="utf-8")
            return []

        def load_corrected_queue(self):
            raise FileNotFoundError

        def merge_reviewed_labels(self, original_df, corrected_df):
            return original_df

    class FakeActiveLearningAgent:
        def run_cycle(self, reviewed, strategy, seed_size, n_iterations, batch_size):
            return [], reviewed

    class FakeTrainingService:
        def train(self, df):
            return ({"model_path": str(tmp_path / "model.pkl")}, {"accuracy": 1.0, "f1": 1.0})

    controller = PipelineController(
        context,
        discovery_service=FakeDiscoveryService(),
        collection_agent=FakeCollectionAgent(),
        quality_agent=FakeQualityAgent(),
        annotation_agent=FakeAnnotationAgent(),
        review_queue_service=FakeReviewQueueService(),
        active_learning_agent=FakeActiveLearningAgent(),
        training_service=FakeTrainingService(),
    )

    controller.run()

    eda_context = json.loads((tmp_path / "data" / "interim" / "eda_context.json").read_text(encoding="utf-8"))
    text_summary = eda_context["text_length_summary"]
    missing_values_summary = eda_context["missing_values_summary"]

    assert eda_context["n_rows"] == 5
    assert eda_context["columns"] == ["id", "source", "text", "rating"]
    assert text_summary["available"] is True
    assert text_summary["valid_count"] == 1
    assert text_summary["missing_or_invalid_count"] == 4
    assert text_summary["min_chars"] == 14
    assert text_summary["max_chars"] == 14
    assert text_summary["mean_chars"] == 14
    assert missing_values_summary["text"]["available"] is True
    assert missing_values_summary["text"]["missing_count"] == 4


def test_run_pipeline_smoke_uses_gemini_provider_when_api_key_present(monkeypatch, tmp_path: Path) -> None:
    """When the API key is present, the controller should use the Gemini provider path."""

    import pandas as pd

    from src.domain import SourceCandidate
    from src.providers.llm.gemini_client import GeminiClient
    from src.providers.llm.mock_llm import MockLLM
    from src.services import source_discovery_service as discovery_module
    from src.agents import data_collection_agent as data_collection_module

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    config_path = tmp_path / "demo_fitness.runtime.yaml"
    config_path.write_text(_runtime_config_text(template_text, tmp_path, "gemini"), encoding="utf-8")

    loaded_config = load_config(config_path)
    assert loaded_config.annotation.use_llm is True
    assert loaded_config.annotation.llm_provider == "gemini"

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    monkeypatch.setattr(MockLLM, "classify_effect", lambda self, text, labels: (_ for _ in ()).throw(AssertionError("MockLLM must not be used when GEMINI_API_KEY is present")))
    monkeypatch.setattr(
        GeminiClient,
        "generate",
        lambda self, prompt: '{"effect_label": "energy", "sentiment_label": "positive", "confidence": 0.88}',
    )

    monkeypatch.setattr(
        discovery_module.SourceDiscoveryService,
        "run",
        lambda self: [
            SourceCandidate(
                "demo_fitness_scrape",
                "offline_demo",
                "Fitness Supplements Offline Demo",
                "demo_fitness_scrape",
                score=1.0,
                metadata={"web_url": "local://demo"},
            )
        ],
    )

    def fake_collect(self, sources):
        return pd.DataFrame(
            [
                {
                    "id": "1",
                    "source": "demo",
                    "text": "This supplement gives me more energy.",
                    "label": None,
                    "rating": 5,
                    "created_at": "now",
                    "split": None,
                    "meta_json": "{}",
                    "sentiment_label": None,
                    "effect_label": "energy",
                    "confidence": 1.0,
                }
            ]
        )

    monkeypatch.setattr(data_collection_module.DataCollectionAgent, "run", fake_collect)

    from src.services import training_service as training_module

    monkeypatch.setattr(
        training_module.TrainingService,
        "train",
        lambda self, df: (
            {
                "model_path": str(tmp_path / "model.pkl"),
                "vectorizer_path": str(tmp_path / "vectorizer.pkl"),
                "metrics_path": str(tmp_path / "metrics.json"),
            },
            {"accuracy": 1.0, "f1": 1.0},
        ),
    )

    from src.services import reporting_service as reporting_module

    captured_report: dict[str, object] = {}
    original_write_final_report = reporting_module.ReportingService.write_final_report

    def capture_final_report(self, summary):
        captured_report["summary"] = summary
        return original_write_final_report(self, summary)

    monkeypatch.setattr(reporting_module.ReportingService, "write_final_report", capture_final_report)

    from run_pipeline import main

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    annotation_trace_context = json.loads((tmp_path / "data" / "interim" / "annotation_trace.json").read_text(encoding="utf-8"))
    assert annotation_trace_context["llm_mode"] == "generate_parse"
    assert annotation_trace_context["n_fallback_rows"] == 0
    assert annotation_trace_context["parser_contract"]["parse_status_counts"]["parsed"] == 1
    assert Path(captured_report["summary"]["annotation"]["annotation_trace_context_path"]).as_posix() == "data/interim/annotation_trace.json"


def test_run_pipeline_smoke_uses_mock_provider_when_api_key_present(monkeypatch, tmp_path: Path) -> None:
    """An explicit mock provider should ignore GEMINI_API_KEY and stay on the offline path."""

    from src.providers.llm.gemini_client import GeminiClient

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    config_path = tmp_path / "demo_fitness.runtime.yaml"
    config_path.write_text(_runtime_config_text(template_text, tmp_path, "mock"), encoding="utf-8")

    loaded_config = load_config(config_path)
    assert loaded_config.annotation.use_llm is True
    assert loaded_config.annotation.llm_provider == "mock"

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        GeminiClient,
        "generate",
        lambda self, prompt: (_ for _ in ()).throw(AssertionError("GeminiClient must not be used when llm_provider is mock")),
    )

    from run_pipeline import main

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    annotation_trace_context = json.loads((tmp_path / "data" / "interim" / "annotation_trace.json").read_text(encoding="utf-8"))
    assert annotation_trace_context["llm_mode"] == "classify_effect"
    assert annotation_trace_context["n_fallback_rows"] == 0


def test_run_pipeline_smoke_falls_back_to_mock_when_gemini_key_missing(monkeypatch, tmp_path: Path) -> None:
    """A Gemini provider without a key should fall back safely to MockLLM."""

    from src.providers.llm.gemini_client import GeminiClient

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    config_path = tmp_path / "demo_fitness.runtime.yaml"
    config_path.write_text(_runtime_config_text(template_text, tmp_path, "gemini"), encoding="utf-8")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        GeminiClient,
        "generate",
        lambda self, prompt: (_ for _ in ()).throw(AssertionError("GeminiClient must not be called without GEMINI_API_KEY")),
    )

    from run_pipeline import main

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    annotation_trace_context = json.loads((tmp_path / "data" / "interim" / "annotation_trace.json").read_text(encoding="utf-8"))
    assert annotation_trace_context["llm_mode"] == "classify_effect"
    assert annotation_trace_context["n_fallback_rows"] == 0


def test_run_pipeline_smoke_uses_only_approved_sources_when_approval_file_exists(monkeypatch, tmp_path: Path) -> None:
    """An approval file should narrow the collection inputs to the approved shortlist only."""

    import pandas as pd

    from src.domain import SourceCandidate
    from src.services import source_discovery_service as discovery_module
    from src.agents import data_collection_agent as data_collection_module

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    config_path = tmp_path / "demo_fitness.runtime.yaml"
    config_path.write_text(
        template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}"),
        encoding="utf-8",
    )

    loaded_config = load_config(config_path)

    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    approved_source_id = "hf_fitness_supplements_reviews"
    (tmp_path / "data" / "raw" / "approved_sources.json").write_text(f'["{approved_source_id}"]', encoding="utf-8")

    captured_sources: dict[str, list[object]] = {}
    captured_report: dict[str, object] = {}

    monkeypatch.setattr(
        discovery_module.SourceDiscoveryService,
        "run",
        lambda self: [
            SourceCandidate(
                approved_source_id,
                "hf_dataset",
                "Approved HF",
                approved_source_id,
                score=17.5,
                metadata={"web_url": f"https://huggingface.co/datasets/{approved_source_id}"},
            ),
            SourceCandidate(
                "hf-unapproved",
                "hf_dataset",
                "Unapproved HF",
                "hf-unapproved",
                score=0.5,
                metadata={"web_url": "https://huggingface.co/datasets/hf-unapproved"},
            ),
        ],
    )

    def fake_collect(self, sources):
        captured_sources["sources"] = list(sources)
        return pd.DataFrame(
            [
                {
                    "id": "1",
                    "source": "approved-demo",
                    "text": "Great product",
                    "label": None,
                    "rating": 5,
                    "created_at": "now",
                    "split": None,
                    "meta_json": "{}",
                    "sentiment_label": None,
                    "effect_label": "energy",
                    "confidence": 1.0,
                },
                {
                    "id": "2",
                    "source": "approved-demo",
                    "text": "Too sweet",
                    "label": None,
                    "rating": 1,
                    "created_at": "now",
                    "split": None,
                    "meta_json": "{}",
                    "sentiment_label": None,
                    "effect_label": "side_effects",
                    "confidence": 1.0,
                },
            ]
        )

    monkeypatch.setattr(data_collection_module.DataCollectionAgent, "run", fake_collect)

    from src.services import training_service as training_module

    monkeypatch.setattr(
        training_module.TrainingService,
        "train",
        lambda self, df: (
            {
                "model_path": str(tmp_path / "model.pkl"),
                "vectorizer_path": str(tmp_path / "vectorizer.pkl"),
                "metrics_path": str(tmp_path / "metrics.json"),
            },
            {"accuracy": 1.0, "f1": 1.0},
        ),
    )

    from src.services import reporting_service as reporting_module

    original_write_final_report = reporting_module.ReportingService.write_final_report

    def capture_final_report(self, summary):
        captured_report["summary"] = summary
        return original_write_final_report(self, summary)

    monkeypatch.setattr(reporting_module.ReportingService, "write_final_report", capture_final_report)

    from run_pipeline import main

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert [source.source_id for source in captured_sources["sources"]] == [approved_source_id]
    assert captured_report["summary"]["approval"]["approval_status"] == "applied"
    assert (tmp_path / "final_report.md").exists()
    final_report = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert "## Approval" in final_report
    assert "approval_status: applied" in final_report
    approval_candidates = json.loads((tmp_path / "data" / "raw" / "approval_candidates.json").read_text(encoding="utf-8"))
    assert isinstance(approval_candidates, list)
    assert [row["source_id"] for row in approval_candidates] == [approved_source_id, "hf-unapproved"]
    assert approval_candidates[0]["score"] == 17.5
    assert approval_candidates[0]["license"] == "unknown"
    assert approval_candidates[0]["robots_txt_status"] == "not_applicable_api"
    review_queue_report = (tmp_path / "reports" / "review_queue_report.md").read_text(encoding="utf-8")
    assert "# Очередь ручной проверки" in review_queue_report
    assert "ручной проверки после авторазметки" in review_queue_report
    review_queue_context = json.loads((tmp_path / "data" / "interim" / "review_queue_context.json").read_text(encoding="utf-8"))
    assert review_queue_context["confidence_threshold"] == loaded_config.annotation.confidence_threshold
    assert review_queue_context["n_rows"] >= 0
    assert review_queue_context["label_options"] == loaded_config.annotation.effect_labels
    source_report = (tmp_path / "reports" / "source_report.md").read_text(encoding="utf-8")
    assert "Короткий shortlist источников" in source_report
    assert "ручного просмотра и одобрения" in source_report
    assert approved_source_id in source_report
    assert "score: 17.5" in source_report
    assert "robots_txt_status: not_applicable_api" in source_report


def test_run_pipeline_smoke_reports_applied_empty_subset(monkeypatch, tmp_path: Path) -> None:
    """An approval file with no matching ids should set applied_empty_subset truthfully."""

    import pandas as pd

    from src.domain import SourceCandidate
    from src.services import source_discovery_service as discovery_module
    from src.agents import data_collection_agent as data_collection_module

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    config_path = tmp_path / "demo_fitness.runtime.yaml"
    config_path.write_text(
        template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}"),
        encoding="utf-8",
    )

    loaded_config = load_config(config_path)

    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "raw" / "approved_sources.json").write_text('["missing-approved-id"]', encoding="utf-8")

    captured_report: dict[str, object] = {}

    monkeypatch.setattr(
        discovery_module.SourceDiscoveryService,
        "run",
        lambda self: [
            SourceCandidate(
                "unapproved-source",
                "hf_dataset",
                "Unapproved HF",
                "unapproved-source",
                score=0.5,
                metadata={"web_url": "https://huggingface.co/datasets/unapproved-source"},
            )
        ],
    )

    def fake_collect(self, sources):
        return pd.DataFrame(
            [
                {
                    "id": "1",
                    "source": "demo",
                    "text": "Great product",
                    "label": None,
                    "rating": 5,
                    "created_at": "now",
                    "split": None,
                    "meta_json": "{}",
                    "sentiment_label": None,
                    "effect_label": "energy",
                    "confidence": 1.0,
                }
            ]
        )

    monkeypatch.setattr(data_collection_module.DataCollectionAgent, "run", fake_collect)

    from src.services import training_service as training_module

    monkeypatch.setattr(
        training_module.TrainingService,
        "train",
        lambda self, df: (
            {
                "model_path": str(tmp_path / "model.pkl"),
                "vectorizer_path": str(tmp_path / "vectorizer.pkl"),
                "metrics_path": str(tmp_path / "metrics.json"),
            },
            {"accuracy": 1.0, "f1": 1.0},
        ),
    )

    from src.services import reporting_service as reporting_module

    original_write_final_report = reporting_module.ReportingService.write_final_report

    def capture_final_report(self, summary):
        captured_report["summary"] = summary
        return original_write_final_report(self, summary)

    monkeypatch.setattr(reporting_module.ReportingService, "write_final_report", capture_final_report)

    from run_pipeline import main

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert captured_report["summary"]["approval"]
    assert captured_report["summary"]["approval"]["approval_status"] == "applied_empty_subset"
    final_report = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert "approval_status: applied_empty_subset" in final_report
    assert (tmp_path / "reports" / "review_merge_report.md").exists()


def test_approval_flow_passes_empty_sources_when_no_match_and_reports_applied_empty_subset(monkeypatch, tmp_path: Path) -> None:
    """When approved_sources.json doesn't match any discovered source, collection must receive empty sources list."""

    import pandas as pd

    from src.domain import SourceCandidate
    from src.services import source_discovery_service as discovery_module
    from src.agents import data_collection_agent as data_collection_module

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_fitness.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    config_path = tmp_path / "demo_fitness.runtime.yaml"
    config_path.write_text(
        template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}"),
        encoding="utf-8",
    )

    loaded_config = load_config(config_path)

    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    # approved_sources.json contains a source_id that doesn't exist in discovered sources
    (tmp_path / "data" / "raw" / "approved_sources.json").write_text('["nonexistent_approved_source_id"]', encoding="utf-8")

    captured_sources: dict[str, list[object]] = {}
    captured_report: dict[str, object] = {}

    # Discovery returns one source that doesn't match the approved_sources.json
    monkeypatch.setattr(
        discovery_module.SourceDiscoveryService,
        "run",
        lambda self: [
            SourceCandidate(
                "discovered_but_not_approved",
                "hf_dataset",
                "Discovered Source",
                "discovered_but_not_approved",
                score=5.0,
                metadata={"web_url": "https://example.com/source"},
            )
        ],
    )

    def fake_collect(self, sources):
        # Capture the sources that were passed to collection
        captured_sources["sources"] = list(sources)
        return pd.DataFrame(
            [
                {
                    "id": "1",
                    "source": "demo",
                    "text": "Great product",
                    "label": None,
                    "rating": 5,
                    "created_at": "now",
                    "split": None,
                    "meta_json": "{}",
                    "sentiment_label": None,
                    "effect_label": "energy",
                    "confidence": 1.0,
                }
            ]
        )

    monkeypatch.setattr(data_collection_module.DataCollectionAgent, "run", fake_collect)

    from src.services import training_service as training_module

    monkeypatch.setattr(
        training_module.TrainingService,
        "train",
        lambda self, df: (
            {
                "model_path": str(tmp_path / "model.pkl"),
                "vectorizer_path": str(tmp_path / "vectorizer.pkl"),
                "metrics_path": str(tmp_path / "metrics.json"),
            },
            {"accuracy": 1.0, "f1": 1.0},
        ),
    )

    from src.services import reporting_service as reporting_module

    original_write_final_report = reporting_module.ReportingService.write_final_report

    def capture_final_report(self, summary):
        captured_report["summary"] = summary
        return original_write_final_report(self, summary)

    monkeypatch.setattr(reporting_module.ReportingService, "write_final_report", capture_final_report)

    from run_pipeline import main

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    # Verify that collection received an empty sources list
    assert len(captured_sources["sources"]) == 0
    assert [source.source_id for source in captured_sources["sources"]] == []
    # Verify approval_status is applied_empty_subset
    assert captured_report["summary"]["approval"]["approval_status"] == "applied_empty_subset"
    assert captured_report["summary"]["approval"]["n_approved_sources"] == 0
    # Verify final report reflects the honest status
    final_report = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert "approval_status: applied_empty_subset" in final_report
    assert "## Approval" in final_report


def test_review_pack_aligns_with_annotation_summary_and_vocabulary(monkeypatch, tmp_path: Path) -> None:
    """Review pack should follow AnnotationAgent summary threshold and vocabulary semantics."""

    import pandas as pd

    from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
    from src.core.context import PipelineContext
    from src.services.pipeline_controller import PipelineController

    config = AppConfig(
        project=ProjectConfig(name="review-pack-demo", root_dir=tmp_path),
        source=SourceConfig(),
        annotation=AnnotationConfig(confidence_threshold=0.0, effect_labels=[]),
        request=RequestConfig(topic="fitness supplements"),
    )
    context = PipelineContext.from_config(config)

    class FakeRegistry:
        def exists(self, path):
            return False

        def save_json(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return target

        def save_markdown(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
            return target

    class FakeDiscoveryService:
        def __init__(self) -> None:
            self.registry = FakeRegistry()

        def run(self):
            return []

        def load_approved_candidates(self, sources):
            return list(sources)

    class FakeCollectionAgent:
        def run(self, sources):
            return pd.DataFrame([{"id": "1", "text": "Great product", "rating": 5, "effect_label": "energy", "confidence": 1.0}])

    class FakeQualityAgent:
        def detect_issues(self, collected):
            return {"warnings": []}

        def run(self, collected):
            return collected

    class FakeAnnotationAgent:
        def auto_label(self, df):
            return [{"id": "1", "text": "Great product", "rating": 5, "effect_label": "energy", "confidence": 0.4}]

        def check_quality(self, annotated):
            return {"confidence_threshold": 0.6, "n_low_confidence": 1, "n_rows": 1}

        def _effect_label_vocabulary(self):
            return ["energy", "side_effects", "other"]

    class FakeReviewQueueService:
        def export_low_confidence_queue(self, df, threshold=0.7):
            records = [row for row in df if row["confidence"] < threshold]
            target = tmp_path / "data" / "interim" / "review_queue.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("id\n1\n" if records else "id\n", encoding="utf-8")
            return records

        def load_corrected_queue(self):
            raise FileNotFoundError

        def merge_reviewed_labels(self, original_df, corrected_df):
            return original_df

    class FakeActiveLearningAgent:
        def run_cycle(self, reviewed, strategy, seed_size, n_iterations, batch_size):
            return [], reviewed

    class FakeTrainingService:
        def train(self, df):
            return ({"model_path": str(tmp_path / "model.pkl")}, {"accuracy": 1.0, "f1": 1.0})

    controller = PipelineController(
        context,
        discovery_service=FakeDiscoveryService(),
        collection_agent=FakeCollectionAgent(),
        quality_agent=FakeQualityAgent(),
        annotation_agent=FakeAnnotationAgent(),
        review_queue_service=FakeReviewQueueService(),
        active_learning_agent=FakeActiveLearningAgent(),
        training_service=FakeTrainingService(),
    )

    result = controller.run()

    review_queue_context = json.loads((tmp_path / "data" / "interim" / "review_queue_context.json").read_text(encoding="utf-8"))
    assert review_queue_context["confidence_threshold"] == 0.6
    assert review_queue_context["label_options"] == ["energy", "side_effects", "other"]
    assert review_queue_context["n_rows"] == 1
    assert review_queue_context["review_required"] is True
    assert review_queue_context["next_step"] == "fill_corrected_queue_and_rerun"
    assert result["approval_status"] == "skipped_missing_file"


def test_review_merge_report_marks_changed_effect_labels(monkeypatch, tmp_path: Path) -> None:
    """A corrected queue should produce a truthful merge report with changed effect_label counts."""

    import pandas as pd

    from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
    from src.core.context import PipelineContext
    from src.services.pipeline_controller import PipelineController

    config = AppConfig(
        project=ProjectConfig(name="review-merge-demo", root_dir=tmp_path),
        source=SourceConfig(),
        annotation=AnnotationConfig(confidence_threshold=0.0, effect_labels=[]),
        request=RequestConfig(topic="fitness supplements"),
    )
    context = PipelineContext.from_config(config)

    class FakeRegistry:
        def exists(self, path):
            return False

        def save_json(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return target

        def save_markdown(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
            return target

    class FakeDiscoveryService:
        def __init__(self) -> None:
            self.registry = FakeRegistry()

        def run(self):
            return []

        def load_approved_candidates(self, sources):
            return list(sources)

    class FakeCollectionAgent:
        def run(self, sources):
            return pd.DataFrame(
                [
                    {"id": "1", "text": "Great product", "rating": 5, "effect_label": "energy", "confidence": 0.4},
                    {"id": "2", "text": "Too sweet", "rating": 1, "effect_label": "side_effects", "confidence": 0.4},
                ]
            )

    class FakeQualityAgent:
        def detect_issues(self, collected):
            return {"warnings": []}

        def run(self, collected):
            return collected

    class FakeAnnotationAgent:
        def auto_label(self, df):
            return [
                {"id": "1", "text": "Great product", "rating": 5, "effect_label": "energy", "confidence": 0.4},
                {"id": "2", "text": "Too sweet", "rating": 1, "effect_label": "side_effects", "confidence": 0.4},
            ]

        def check_quality(self, annotated):
            return {"confidence_threshold": 0.6, "n_low_confidence": 2, "n_rows": 2}

        def _effect_label_vocabulary(self):
            return ["energy", "side_effects", "other"]

    class FakeReviewQueueService:
        def export_low_confidence_queue(self, df, threshold=0.7):
            target = tmp_path / "data" / "interim" / "review_queue.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("id,reviewed_effect_label\n1,side_effects\n2,\n", encoding="utf-8")
            return [
                {"id": "1", "effect_label": "energy", "reviewed_effect_label": "side_effects", "confidence": 0.4, "text": "Great product", "source": "demo"},
                {"id": "2", "effect_label": "side_effects", "reviewed_effect_label": "", "confidence": 0.4, "text": "Too sweet", "source": "demo"},
            ]

        def load_corrected_queue(self):
            target = tmp_path / "data" / "interim" / "review_queue_corrected.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "id,reviewed_effect_label,review_comment,human_verified\n1,side_effects,,true\n2,,,false\n",
                encoding="utf-8",
            )
            return [
                {"id": "1", "reviewed_effect_label": "side_effects", "review_comment": "", "human_verified": True},
                {"id": "2", "reviewed_effect_label": "", "review_comment": "", "human_verified": False},
            ]

        def merge_reviewed_labels(self, original_df, corrected_df):
            return [
                {"id": "1", "effect_label": "side_effects", "confidence": 1.0},
                {"id": "2", "effect_label": "side_effects", "confidence": 0.4},
            ]

    class FakeActiveLearningAgent:
        def run_cycle(self, reviewed, strategy, seed_size, n_iterations, batch_size):
            return [], reviewed

    class FakeTrainingService:
        def train(self, df):
            return ({"model_path": str(tmp_path / "model.pkl")}, {"accuracy": 1.0, "f1": 1.0})

    controller = PipelineController(
        context,
        discovery_service=FakeDiscoveryService(),
        collection_agent=FakeCollectionAgent(),
        quality_agent=FakeQualityAgent(),
        annotation_agent=FakeAnnotationAgent(),
        review_queue_service=FakeReviewQueueService(),
        active_learning_agent=FakeActiveLearningAgent(),
        training_service=FakeTrainingService(),
    )

    result = controller.run()

    assert result["review_status"] == "merged"
    merge_report = (tmp_path / "reports" / "review_merge_report.md").read_text(encoding="utf-8")
    assert "# Результат ручного merge" in merge_report
    assert "corrected queue" in merge_report.lower() or "corrected_queue_found" in merge_report
    assert "n_effect_label_changes: 1" in merge_report
    assert "## Next step" in merge_report
    merge_context = json.loads((tmp_path / "data" / "interim" / "review_merge_context.json").read_text(encoding="utf-8"))
    assert merge_context["corrected_queue_found"] is True
    assert merge_context["n_corrected_rows"] == 2
    assert merge_context["n_rows_with_reviewed_effect_label"] == 1
    assert merge_context["n_effect_label_changes"] == 1
    assert merge_context["review_status"] == "merged"
    agreement_context = json.loads((tmp_path / "data" / "interim" / "review_agreement_context.json").read_text(encoding="utf-8"))
    assert agreement_context["compared_rows"] == 1
    assert agreement_context["agreement"] == 0.0
    assert agreement_context["kappa"] is None
    final_report = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert "review_merge_report_path" in final_report
    assert "agreement_report_path: reports/review_agreement_report.md" in final_report


def test_review_merge_summary_ignores_missing_ids(monkeypatch, tmp_path: Path) -> None:
    """Missing or blank ids should not be coerced into a literal 'None' merge key."""

    import pandas as pd

    from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
    from src.core.context import PipelineContext
    from src.services.pipeline_controller import PipelineController

    config = AppConfig(
        project=ProjectConfig(name="review-merge-id-demo", root_dir=tmp_path),
        source=SourceConfig(),
        annotation=AnnotationConfig(confidence_threshold=0.0, effect_labels=[]),
        request=RequestConfig(topic="fitness supplements"),
    )
    context = PipelineContext.from_config(config)

    class FakeRegistry:
        def exists(self, path):
            return False

        def save_json(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return target

        def save_markdown(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
            return target

    class FakeDiscoveryService:
        def __init__(self) -> None:
            self.registry = FakeRegistry()

        def run(self):
            return []

        def load_approved_candidates(self, sources):
            return list(sources)

    class FakeCollectionAgent:
        def run(self, sources):
            return pd.DataFrame([{"id": "valid-1", "text": "Great product", "rating": 5, "effect_label": "energy", "confidence": 0.4}])

    class FakeQualityAgent:
        def detect_issues(self, collected):
            return {"warnings": []}

        def run(self, collected):
            return collected

    class FakeAnnotationAgent:
        def auto_label(self, df):
            return [{"id": "valid-1", "text": "Great product", "rating": 5, "effect_label": "energy", "confidence": 0.4}]

        def check_quality(self, annotated):
            return {"confidence_threshold": 0.6, "n_low_confidence": 1, "n_rows": 1}

        def _effect_label_vocabulary(self):
            return ["energy", "side_effects", "other"]

    class FakeReviewQueueService:
        def export_low_confidence_queue(self, df, threshold=0.7):
            target = tmp_path / "data" / "interim" / "review_queue.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("id,reviewed_effect_label\nvalid-1,side_effects\n,energy\n\t,other\n", encoding="utf-8")
            return [
                {"id": "valid-1", "effect_label": "energy", "reviewed_effect_label": "side_effects", "confidence": 0.4},
                {"id": None, "effect_label": "energy", "reviewed_effect_label": "energy", "confidence": 0.4},
                {"id": "", "effect_label": "energy", "reviewed_effect_label": "other", "confidence": 0.4},
            ]

        def load_corrected_queue(self):
            target = tmp_path / "data" / "interim" / "review_queue_corrected.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "\n".join(
                    [
                        "id,reviewed_effect_label,review_comment,human_verified",
                        "valid-1,side_effects,,true",
                        ",,,",
                        ",energy,,false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return [
                {"id": "valid-1", "reviewed_effect_label": "side_effects", "review_comment": "", "human_verified": True},
                {"id": None, "reviewed_effect_label": "energy", "review_comment": "", "human_verified": False},
                {"id": "", "reviewed_effect_label": "other", "review_comment": "", "human_verified": False},
            ]

        def merge_reviewed_labels(self, original_df, corrected_df):
            return [{"id": "valid-1", "effect_label": "side_effects", "confidence": 1.0}]

    class FakeActiveLearningAgent:
        def run_cycle(self, reviewed, strategy, seed_size, n_iterations, batch_size):
            return [], reviewed

    class FakeTrainingService:
        def train(self, df):
            return ({"model_path": str(tmp_path / "model.pkl")}, {"accuracy": 1.0, "f1": 1.0})

    controller = PipelineController(
        context,
        discovery_service=FakeDiscoveryService(),
        collection_agent=FakeCollectionAgent(),
        quality_agent=FakeQualityAgent(),
        annotation_agent=FakeAnnotationAgent(),
        review_queue_service=FakeReviewQueueService(),
        active_learning_agent=FakeActiveLearningAgent(),
        training_service=FakeTrainingService(),
    )

    result = controller.run()

    merge_context = json.loads((tmp_path / "data" / "interim" / "review_merge_context.json").read_text(encoding="utf-8"))
    assert merge_context["corrected_queue_found"] is True
    assert merge_context["n_corrected_rows"] == 1
    assert merge_context["n_rows_with_reviewed_effect_label"] == 1
    assert merge_context["n_effect_label_changes"] == 1
    assert merge_context["reviewed_effect_labels"] == ["side_effects"]
    assert result["review_status"] == "merged"


def test_review_merge_status_merged_no_changes_when_corrected_queue_has_no_effect_changes(tmp_path: Path) -> None:
    """Review status should be 'merged_no_changes' when corrected_queue exists but has no effect_label changes."""

    import json
    import pandas as pd

    from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
    from src.core.context import PipelineContext
    from src.services.pipeline_controller import PipelineController

    config = AppConfig(
        project=ProjectConfig(name="review-merge-no-changes-demo", root_dir=tmp_path),
        source=SourceConfig(),
        annotation=AnnotationConfig(confidence_threshold=0.0, effect_labels=[]),
        request=RequestConfig(topic="fitness supplements"),
    )
    context = PipelineContext.from_config(config)

    class FakeRegistry:
        def exists(self, path):
            return False

        def save_json(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return target

        def save_markdown(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
            return target

    class FakeDiscoveryService:
        def __init__(self) -> None:
            self.registry = FakeRegistry()

        def run(self):
            return []

        def load_approved_candidates(self, sources):
            return list(sources)

    class FakeCollectionAgent:
        def run(self, sources):
            return pd.DataFrame(
                [
                    {"id": "1", "text": "Great product", "rating": 5, "effect_label": "energy", "confidence": 0.4},
                    {"id": "2", "text": "Too sweet", "rating": 1, "effect_label": "side_effects", "confidence": 0.4},
                ]
            )

    class FakeQualityAgent:
        def detect_issues(self, collected):
            return {"warnings": []}

        def run(self, collected):
            return collected

    class FakeAnnotationAgent:
        def auto_label(self, df):
            return [
                {"id": "1", "text": "Great product", "rating": 5, "effect_label": "energy", "confidence": 0.4},
                {"id": "2", "text": "Too sweet", "rating": 1, "effect_label": "side_effects", "confidence": 0.4},
            ]

        def check_quality(self, annotated):
            return {"confidence_threshold": 0.6, "n_low_confidence": 2, "n_rows": 2}

        def _effect_label_vocabulary(self):
            return ["energy", "side_effects", "other"]

    class FakeReviewQueueService:
        def export_low_confidence_queue(self, df, threshold=0.7):
            target = tmp_path / "data" / "interim" / "review_queue.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("id,reviewed_effect_label\n1,\n2,\n", encoding="utf-8")
            return [
                {"id": "1", "effect_label": "energy", "reviewed_effect_label": "", "confidence": 0.4, "text": "Great product", "source": "demo"},
                {"id": "2", "effect_label": "side_effects", "reviewed_effect_label": "", "confidence": 0.4, "text": "Too sweet", "source": "demo"},
            ]

        def load_corrected_queue(self):
            target = tmp_path / "data" / "interim" / "review_queue_corrected.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            # Corrected queue exists but has no reviewed_effect_label values
            target.write_text(
                "id,reviewed_effect_label,review_comment,human_verified\n1,,,false\n2,,,false\n",
                encoding="utf-8",
            )
            return [
                {"id": "1", "reviewed_effect_label": "", "review_comment": "", "human_verified": False},
                {"id": "2", "reviewed_effect_label": "", "review_comment": "", "human_verified": False},
            ]

        def merge_reviewed_labels(self, original_df, corrected_df):
            # No actual changes - return same labels as input
            return [
                {"id": "1", "effect_label": "energy", "confidence": 0.4},
                {"id": "2", "effect_label": "side_effects", "confidence": 0.4},
            ]

    class FakeActiveLearningAgent:
        def run_cycle(self, reviewed, strategy, seed_size, n_iterations, batch_size):
            return [], reviewed

    class FakeTrainingService:
        def train(self, df):
            return ({"model_path": str(tmp_path / "model.pkl")}, {"accuracy": 1.0, "f1": 1.0})

    controller = PipelineController(
        context,
        discovery_service=FakeDiscoveryService(),
        collection_agent=FakeCollectionAgent(),
        quality_agent=FakeQualityAgent(),
        annotation_agent=FakeAnnotationAgent(),
        review_queue_service=FakeReviewQueueService(),
        active_learning_agent=FakeActiveLearningAgent(),
        training_service=FakeTrainingService(),
    )

    result = controller.run()

    assert result["review_status"] == "merged_no_changes"
    merge_report = (tmp_path / "reports" / "review_merge_report.md").read_text(encoding="utf-8")
    assert "# Результат ручного merge" in merge_report
    assert "review_status: merged_no_changes" in merge_report
    assert "## Next step" in merge_report
    merge_context = json.loads((tmp_path / "data" / "interim" / "review_merge_context.json").read_text(encoding="utf-8"))
    assert merge_context["corrected_queue_found"] is True
    assert merge_context["n_corrected_rows"] == 2
    assert merge_context["n_rows_with_reviewed_effect_label"] == 0
    assert merge_context["n_effect_label_changes"] == 0
    assert merge_context["review_status"] == "merged_no_changes"
    agreement_context = json.loads((tmp_path / "data" / "interim" / "review_agreement_context.json").read_text(encoding="utf-8"))
    assert agreement_context["compared_rows"] == 0
    assert agreement_context["agreement"] is None
    assert agreement_context["kappa_status"] == "not_available_no_compared_rows"
    final_report = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert "review_merge_report_path" in final_report
    assert "status: merged_no_changes" in final_report


def test_review_merge_status_stays_merged_when_corrected_queue_has_effect_changes(tmp_path: Path) -> None:
    """Review status should be 'merged' when corrected_queue exists and has effect_label changes."""

    import json
    import pandas as pd

    from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, RequestConfig, SourceConfig
    from src.core.context import PipelineContext
    from src.services.pipeline_controller import PipelineController

    config = AppConfig(
        project=ProjectConfig(name="review-merge-with-changes-demo", root_dir=tmp_path),
        source=SourceConfig(),
        annotation=AnnotationConfig(confidence_threshold=0.0, effect_labels=[]),
        request=RequestConfig(topic="fitness supplements"),
    )
    context = PipelineContext.from_config(config)

    class FakeRegistry:
        def exists(self, path):
            return False

        def save_json(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return target

        def save_markdown(self, path, payload):
            target = tmp_path / Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
            return target

    class FakeDiscoveryService:
        def __init__(self) -> None:
            self.registry = FakeRegistry()

        def run(self):
            return []

        def load_approved_candidates(self, sources):
            return list(sources)

    class FakeCollectionAgent:
        def run(self, sources):
            return pd.DataFrame(
                [
                    {"id": "1", "text": "Great product", "rating": 5, "effect_label": "energy", "confidence": 0.4},
                    {"id": "2", "text": "Too sweet", "rating": 1, "effect_label": "side_effects", "confidence": 0.4},
                ]
            )

    class FakeQualityAgent:
        def detect_issues(self, collected):
            return {"warnings": []}

        def run(self, collected):
            return collected

    class FakeAnnotationAgent:
        def auto_label(self, df):
            return [
                {"id": "1", "text": "Great product", "rating": 5, "effect_label": "energy", "confidence": 0.4},
                {"id": "2", "text": "Too sweet", "rating": 1, "effect_label": "side_effects", "confidence": 0.4},
            ]

        def check_quality(self, annotated):
            return {"confidence_threshold": 0.6, "n_low_confidence": 2, "n_rows": 2}

        def _effect_label_vocabulary(self):
            return ["energy", "side_effects", "other"]

    class FakeReviewQueueService:
        def export_low_confidence_queue(self, df, threshold=0.7):
            target = tmp_path / "data" / "interim" / "review_queue.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("id,reviewed_effect_label\n1,side_effects\n2,\n", encoding="utf-8")
            return [
                {"id": "1", "effect_label": "energy", "reviewed_effect_label": "side_effects", "confidence": 0.4, "text": "Great product", "source": "demo"},
                {"id": "2", "effect_label": "side_effects", "reviewed_effect_label": "", "confidence": 0.4, "text": "Too sweet", "source": "demo"},
            ]

        def load_corrected_queue(self):
            target = tmp_path / "data" / "interim" / "review_queue_corrected.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "id,reviewed_effect_label,review_comment,human_verified\n1,side_effects,,true\n2,,,false\n",
                encoding="utf-8",
            )
            return [
                {"id": "1", "reviewed_effect_label": "side_effects", "review_comment": "", "human_verified": True},
                {"id": "2", "reviewed_effect_label": "", "review_comment": "", "human_verified": False},
            ]

        def merge_reviewed_labels(self, original_df, corrected_df):
            return [
                {"id": "1", "effect_label": "side_effects", "confidence": 1.0},
                {"id": "2", "effect_label": "side_effects", "confidence": 0.4},
            ]

    class FakeActiveLearningAgent:
        def run_cycle(self, reviewed, strategy, seed_size, n_iterations, batch_size):
            return [], reviewed

    class FakeTrainingService:
        def train(self, df):
            return ({"model_path": str(tmp_path / "model.pkl")}, {"accuracy": 1.0, "f1": 1.0})

    controller = PipelineController(
        context,
        discovery_service=FakeDiscoveryService(),
        collection_agent=FakeCollectionAgent(),
        quality_agent=FakeQualityAgent(),
        annotation_agent=FakeAnnotationAgent(),
        review_queue_service=FakeReviewQueueService(),
        active_learning_agent=FakeActiveLearningAgent(),
        training_service=FakeTrainingService(),
    )

    result = controller.run()

    assert result["review_status"] == "merged"
    merge_report = (tmp_path / "reports" / "review_merge_report.md").read_text(encoding="utf-8")
    assert "# Результат ручного merge" in merge_report
    assert "review_status: merged" in merge_report
    assert "n_effect_label_changes: 1" in merge_report
    assert "## Next step" in merge_report
    merge_context = json.loads((tmp_path / "data" / "interim" / "review_merge_context.json").read_text(encoding="utf-8"))
    assert merge_context["corrected_queue_found"] is True
    assert merge_context["n_corrected_rows"] == 2
    assert merge_context["n_rows_with_reviewed_effect_label"] == 1
    assert merge_context["n_effect_label_changes"] == 1
    assert merge_context["review_status"] == "merged"
    agreement_context = json.loads((tmp_path / "data" / "interim" / "review_agreement_context.json").read_text(encoding="utf-8"))
    assert agreement_context["compared_rows"] == 1
    assert agreement_context["agreement"] == 0.0
    final_report = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert "review_merge_report_path" in final_report
    assert "status: merged" in final_report
