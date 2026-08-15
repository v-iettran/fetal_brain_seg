"""Version-checked OpenEvolve 0.3.2 patch: pass parent best_lr to the evaluator.

Upstream evaluate(program_path) cannot see the parent program record. This module
writes <temp_program>.parent.json beside the candidate file before evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_VERSION = "0.3.2"


def apply() -> None:
    import openevolve
    from openevolve import evaluator as ev

    version = getattr(openevolve, "__version__", None)
    if version is None:
        from importlib.metadata import version as package_version

        version = package_version("openevolve")
    if version != REQUIRED_VERSION:
        raise RuntimeError(
            f"parent_lr patch is pinned to OpenEvolve {REQUIRED_VERSION}, found {version}"
        )

    if getattr(ev.Evaluator, "_feta_parent_lr_patched", False):
        return

    original = ev.Evaluator.evaluate_program

    async def evaluate_program(self, program_code, program_id=""):
        # The original writes a NamedTemporaryFile then calls cascade/direct evaluate.
        # We wrap _cascade_evaluate / _direct_evaluate to drop a sidecar first.
        orig_cascade = self._cascade_evaluate
        orig_direct = self._direct_evaluate
        parent_lr = None
        db = getattr(self, "database", None)
        if db is not None and program_id:
            # During initial evaluation there is no parent. Workers set FETA_PARENT_BEST_LR.
            pass

        async def with_sidecar(program_path: str, fn):
            sidecar = Path(str(program_path) + ".parent.json")
            import os

            lr = os.environ.get("FETA_PARENT_BEST_LR")
            payload = {}
            if lr:
                payload["best_lr"] = float(lr)
            sidecar.write_text(json.dumps(payload))
            try:
                return await fn(program_path)
            finally:
                if sidecar.exists():
                    # leave it for the worker evaluate() call; cleaned with the temp file
                    pass

        async def cascade(program_path):
            return await with_sidecar(program_path, orig_cascade)

        async def direct(program_path):
            return await with_sidecar(program_path, orig_direct)

        self._cascade_evaluate = cascade
        self._direct_evaluate = direct
        try:
            return await original(self, program_code, program_id)
        finally:
            self._cascade_evaluate = orig_cascade
            self._direct_evaluate = orig_direct

    ev.Evaluator.evaluate_program = evaluate_program
    ev.Evaluator._feta_parent_lr_patched = True
    _patch_worker()


def _patch_worker() -> None:
    """Inject parent best_lr into the worker process environment."""
    from openevolve import process_parallel as pp

    if getattr(pp, "_feta_parent_lr_patched", False):
        return
    original = pp._run_iteration_worker

    def wrapped(iteration, db_snapshot, parent_id, inspiration_ids):
        import os

        programs = db_snapshot.get("programs", {})
        parent = programs.get(parent_id, {})
        metrics = parent.get("metrics") or {}
        lr = metrics.get("best_lr")
        if lr is not None:
            os.environ["FETA_PARENT_BEST_LR"] = str(lr)
        return original(iteration, db_snapshot, parent_id, inspiration_ids)

    pp._run_iteration_worker = wrapped
    pp._feta_parent_lr_patched = True


def sha256() -> str:
    import hashlib

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
