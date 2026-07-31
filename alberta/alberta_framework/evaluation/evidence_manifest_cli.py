"""CLI for the versioned, fail-closed evidence-claim manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from alberta_framework.evaluation.evidence_manifest import (
    REPO_ROOT,
    build_evidence_manifest,
    evidence_manifest_exit_code,
    evidence_manifest_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every registered evidence claim and distinguish accepted, "
            "valid-rejection, not-run, invalid, and nonpromoting evidence."
        )
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="optionally write the versioned JSON manifest as well as printing it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate registered artifacts without running scientific protocols."""

    args = _parser().parse_args(argv)
    manifest = build_evidence_manifest(args.root)
    serialized = evidence_manifest_json(manifest)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return evidence_manifest_exit_code(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
