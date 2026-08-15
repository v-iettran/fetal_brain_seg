from __future__ import annotations

from importlib.metadata import version


def test_pinned_openevolve_patch_applies() -> None:
    assert version("openevolve") == "0.3.2"

    from openevolve import evaluator, process_parallel
    from patches.parent_lr import apply

    apply()

    assert evaluator.Evaluator._feta_parent_lr_patched is True
    assert process_parallel._feta_parent_lr_patched is True
