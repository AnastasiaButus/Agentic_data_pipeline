"""End-to-end config switch test for the minecraft demo theme."""

from __future__ import annotations

from pathlib import Path

from src.core.config import load_config


def test_demo_minecraft_config_switch_runs_end_to_end(tmp_path: Path) -> None:
    """Load the minecraft config through the existing CLI and verify the pipeline still runs."""

    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "configs" / "demo_minecraft.yaml"
    template_text = template_path.read_text(encoding="utf-8")

    assert "minecraft instructions" in template_text

    runtime_config_path = tmp_path / "demo_minecraft.runtime.yaml"
    runtime_config_path.write_text(
        template_text.replace("root_dir: .", f"root_dir: {tmp_path.as_posix()}"),
        encoding="utf-8",
    )

    loaded_config = load_config(runtime_config_path)
    assert loaded_config.request.topic == "minecraft instructions"
    assert loaded_config.annotation.use_llm is True

    from run_pipeline import main

    exit_code = main(["--config", str(runtime_config_path)])

    assert exit_code == 0
    assert (tmp_path / "final_report.md").exists()
    assert (tmp_path / "data" / "interim" / "model_metrics.json").exists()
    assert (tmp_path / "data" / "interim" / "review_queue.csv").exists()
    assert "Minecraft Instructions Offline Demo" in (tmp_path / "data" / "raw" / "discovered_sources.json").read_text(encoding="utf-8")
