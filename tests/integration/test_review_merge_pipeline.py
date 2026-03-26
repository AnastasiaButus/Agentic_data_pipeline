"""Integration coverage for the human review queue and merge round-trip."""

from __future__ import annotations

from pathlib import Path

from src.core.config import AnnotationConfig, AppConfig, ProjectConfig, SourceConfig
from src.core.context import PipelineContext
from src.services.review_queue_service import ReviewQueueService


class _Frame:
    """Tiny dataframe-like helper used to keep the integration test independent of pandas."""

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


class StubRegistry:
    """Persist review queue artifacts directly to the temporary workspace."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def save_dataframe(self, path: str | Path, df: object) -> Path:
        target_path = self.root_dir / Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        rows = df.to_dict(orient="records")
        columns = list(df.columns)

        import csv
        import json

        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if columns:
                writer.writeheader()
            for row in rows:
                writer.writerow({column: json.dumps(row.get(column)) for column in columns})

        return target_path

    def load_dataframe(self, path: str | Path) -> object:
        target_path = self.root_dir / Path(path)
        import csv
        import json

        with target_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [{column: self._decode_csv_value(value) for column, value in row.items()} for row in reader]
        return _Frame(rows)

    def _decode_csv_value(self, value: str) -> object:
        """Decode JSON-encoded CSV cells while tolerating plain strings in fixtures."""

        import json

        try:
            return json.loads(value)
        except Exception:
            return value

    def exists(self, path: str | Path) -> bool:
        return (self.root_dir / Path(path)).exists()


def _make_context(tmp_path: Path) -> PipelineContext:
    """Build a minimal pipeline context for the integration test."""

    config = AppConfig(
        project=ProjectConfig(name="fitness-demo", root_dir=tmp_path),
        source=SourceConfig(use_huggingface=True),
        annotation=AnnotationConfig(),
    )
    return PipelineContext.from_config(config)


def test_review_queue_round_trip_export_load_and_merge(tmp_path: Path) -> None:
    """The review queue should export, load corrected rows, and merge them back by canonical id."""

    registry = StubRegistry(tmp_path)
    service = ReviewQueueService(_make_context(tmp_path), registry=registry)

    original = _Frame(
        [
            {
                "id": "1",
                "source": "HF",
                "text": "Great product",
                "label": None,
                "rating": 5,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "positive",
                "effect_label": "energy",
                "confidence": 0.4,
            },
            {
                "id": "2",
                "source": "Web",
                "text": "Too many side effects",
                "label": None,
                "rating": 1,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "negative",
                "effect_label": "side_effects",
                "confidence": float("nan"),
            },
            {
                "id": "3",
                "source": "Web",
                "text": "Looks fine",
                "label": None,
                "rating": 3,
                "created_at": "now",
                "split": None,
                "meta_json": "{}",
                "sentiment_label": "neutral",
                "effect_label": "other",
                "confidence": 0.95,
            },
        ]
    )

    queue = service.export_low_confidence_queue(original, threshold=0.7)
    queue_path = tmp_path / "data" / "interim" / "review_queue.csv"
    assert queue_path.exists()
    assert [row["id"] for row in queue.to_dict(orient="records")] == ["1", "2"]

    corrected_path = tmp_path / "data" / "interim" / "review_queue_corrected.csv"
    corrected_path.write_text((Path(__file__).resolve().parents[1] / "fixtures" / "sample_review_queue_corrected.csv").read_text(encoding="utf-8"), encoding="utf-8")

    corrected = service.load_corrected_queue()
    merged = service.merge_reviewed_labels(original, corrected)
    merged_rows = merged.to_dict(orient="records")

    assert corrected_path.exists()
    assert merged_rows[0]["effect_label"] == "other"
    assert merged_rows[0]["confidence"] == 1.0
    assert merged_rows[0]["human_verified"] is True
    assert merged_rows[1]["effect_label"] == "side_effects"