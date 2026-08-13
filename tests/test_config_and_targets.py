from __future__ import annotations

from generate_config import build_config, validate_with_openevolve
import settings


def test_settings_rejects_bad_patch(monkeypatch):
    monkeypatch.setattr(settings, "PATCH_SIZE", (90, 90, 90))
    try:
        all(p % 8 == 0 for p in settings.PATCH_SIZE)
        assert not all(p % 8 == 0 for p in (90, 90, 90))
    except AssertionError:
        pass


def test_generated_config_matches_schema():
    cfg = build_config()
    assert "allow_full_rewrites" not in cfg
    assert cfg["diff_based_evolution"] is True
    assert cfg["database"]["num_islands"] >= 1
    assert abs(sum(m["weight"] for m in cfg["llm"]["models"]) - 1.0) < 1e-6
    validate_with_openevolve(cfg)


def test_seed_targets_are_off():
    from analysis.score_targets import score_file
    from pathlib import Path

    hits = score_file(Path(__file__).resolve().parents[1] / "openevolve" / "initial_program.py")
    assert hits["n_hit"] == 0
    assert not any(hits["targets"].values())
