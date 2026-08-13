"""Require human leakage sign-off before any paid LLM experiment."""

from __future__ import annotations

from pathlib import Path

REVIEW = Path(__file__).resolve().parent / "prompts" / "LEAKAGE_REVIEW.md"


def require_prompt_review() -> None:
    text = REVIEW.read_text()
    if "SIGNED_OFF=YES" not in text:
        raise SystemExit(
            f"{REVIEW} is not signed off. A reviewer who has not read the target "
            "list must set SIGNED_OFF=YES before zero-shot, dry-run, or evolution."
        )
