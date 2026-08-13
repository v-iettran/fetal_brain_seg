"""Isolated candidate execution. OpenEvolve's thread timeout cannot kill CUDA work."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

TRACK_B = Path(__file__).resolve().parent
REPO_ROOT = TRACK_B.parent
WORKER = TRACK_B / "candidate_worker.py"


def run_isolated(
    program_path: str,
    stage: str,
    extra_env: dict[str, str] | None = None,
    timeout: int = 7200,
) -> dict:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TRACK_B), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    if extra_env:
        env.update(extra_env)
    with tempfile.TemporaryDirectory(prefix="feta_eval_") as scratch:
        result_path = Path(scratch) / "result.json"
        cmd = [
            sys.executable,
            str(WORKER),
            "--program",
            str(program_path),
            "--stage",
            stage,
            "--result",
            str(result_path),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(REPO_ROOT),
                start_new_session=True,
            )
        except OSError as exc:
            return {"ok": False, "reason": f"failed to spawn worker: {exc}"}
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            return {
                "ok": False,
                "reason": f"worker timed out after {timeout}s",
                "stderr": "timeout",
            }
        payload = {}
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                payload = {"ok": False, "reason": "invalid worker JSON"}
        else:
            payload = {
                "ok": False,
                "reason": f"worker exit {proc.returncode}",
                "stdout": stdout.decode("utf-8", errors="replace")[-4000:],
                "stderr": stderr.decode("utf-8", errors="replace")[-4000:],
            }
        payload.setdefault("stderr", stderr.decode("utf-8", errors="replace")[-4000:])
        return payload
