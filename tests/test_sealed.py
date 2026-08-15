from __future__ import annotations

from pathlib import Path

import pytest

from src.validate import assert_not_sealed, SEALED_UNLOCK


def test_sealed_test_blocked(tmp_path, monkeypatch, repo_root):
    splits = repo_root / "results" / "splits.json"
    if not splits.exists():
        pytest.skip("splits not generated")
    import json

    test_ids = json.loads(splits.read_text())["test"][:1]
    monkeypatch.setattr("src.validate.SEALED_UNLOCK", tmp_path / "missing.json")
    with pytest.raises(PermissionError):
        assert_not_sealed(test_ids)
