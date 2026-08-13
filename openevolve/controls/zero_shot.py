"""Zero-shot LLM control. Run before the main evolution. Costs LLM tokens."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TRACK_B = Path(__file__).resolve().parent
REPO_ROOT = TRACK_B.parent
sys.path.insert(0, str(TRACK_B))
sys.path.insert(0, str(REPO_ROOT))

import bootstrap  # noqa: E402

bootstrap.ensure_paths()

from require_review import require_prompt_review  # noqa: E402
import settings  # noqa: E402

PROMPT = (
    "Improve this 3D U-Net for segmentation of a small (n≈40) 3D medical imaging dataset.\n\n"
    "Return a complete Python module that preserves the six contract signatures and "
    "the EVOLVE-BLOCK markers. Make changes only inside the evolve block.\n\n"
    "SEED PROGRAM:\n"
)


def _claude(system: str, user: str, model: str, timeout: int = 300) -> str:
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--no-session-persistence",
        "--output-format",
        "text",
        "--system-prompt",
        system,
        "--max-budget-usd",
        "2.0",
        user,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr[:500] or "empty Claude response")
    return result.stdout.strip()


def extract_python(text: str) -> str:
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip() + "\n"
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip() + "\n"
    return text if text.lstrip().startswith('"""') or "def build_model" in text else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--execute", action="store_true", help="Call the LLM. Costs money.")
    parser.add_argument("--skip-review-gate", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "zero_shot")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to call the LLM without --execute.")
    if not args.skip_review_gate:
        require_prompt_review()
    seed = (TRACK_B / "initial_program.py").read_text()
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    model = settings.LLM_MODELS[0]["name"]
    for i in range(args.n):
        print(f"zero-shot sample {i+1}/{args.n}")
        text = _claude(
            "You are an expert in deep learning for 3D medical image segmentation.",
            PROMPT + seed,
            model=model,
        )
        code = extract_python(text)
        path = args.out / f"sample_{i:02d}.py"
        path.write_text(code)
        index.append({"i": i, "path": str(path), "unix": time.time(), "n_chars": len(code)})
    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    from analysis.score_targets import score_file

    scores = [score_file(Path(item["path"])) for item in index]
    (REPO_ROOT / "results" / "zero_shot_targets.json").write_text(json.dumps(scores, indent=2))
    print(f"Wrote {args.out} and results/zero_shot_targets.json")


if __name__ == "__main__":
    main()
