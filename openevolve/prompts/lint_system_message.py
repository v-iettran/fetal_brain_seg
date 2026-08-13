"""Fail the system message if it names pre-registered targets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROMPT_PATH = Path(__file__).with_name("system_message.txt")

FORBIDDEN = [
    (r"instance\s*norm", "instance normalisation"),
    (r"GroupNorm", "GroupNorm"),
    (r"InstanceNorm", "InstanceNorm"),
    (r"leaky\s*relu", "LeakyReLU"),
    (r"\bELU\b", "ELU"),
    (r"\bGELU\b", "GELU"),
    (r"deep\s+supervision", "deep supervision"),
    (r"auxiliary\s+head", "auxiliary heads"),
    (r"soft\s*dice", "soft Dice"),
    (r"dice\s+loss", "Dice loss"),
    (r"\bSGD\b", "SGD"),
    (r"nesterov", "Nesterov"),
    (r"poly(nomial)?\s+(lr|schedule|decay)", "polynomial schedule"),
    (r"foreground\s+(over)?sampl", "foreground oversampling"),
    (r"forced\s+foreground", "forced foreground patches"),
    (r"elastic\s+deform", "elastic deformation"),
]


def lint(text: str) -> list[str]:
    hits = []
    for pattern, label in FORBIDDEN:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(label)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=PROMPT_PATH)
    args = parser.parse_args()
    text = args.path.read_text()
    hits = lint(text)
    if hits:
        raise SystemExit(f"system message leaks targets: {hits}")
    print(f"OK: no forbidden target strings in {args.path}")


if __name__ == "__main__":
    main()
