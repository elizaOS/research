"""Console entry points for the Step 1/2 smoke kernels and evidence-status alias.

``alberta-step1-smoke`` and ``alberta-step2-smoke`` run the seeded Step 1
(optimizer/normalizer) and Step 2 (UPGD) production kernels for a short
horizon and exit nonzero unless every reported metric is finite; they are
integration probes, not scientific evidence. ``alberta-evidence-gate``
is a deprecated compatibility alias for ``alberta-evidence-status`` — see
:func:`evidence_gate_main`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import cast

from alberta_framework.steps.step1 import (
    Step1KernelConfig,
    Step1NormalizerName,
    Step1OptimizerName,
    run_step1_smoke,
)
from alberta_framework.steps.step2 import (
    Step2KernelConfig,
    Step2StreamName,
    run_step2_smoke,
)


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def step1_smoke_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``alberta-step1-smoke``."""
    parser = argparse.ArgumentParser(description="Run a Step 1 kernel smoke test.")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=64)
    parser.add_argument(
        "--optimizer",
        choices=(
            "lms",
            "idbd",
            "autostep",
            "autostep_gtd",
            "adagain",
            "adam",
            "rmsprop",
            "nadaline",
        ),
        default="autostep",
    )
    parser.add_argument(
        "--normalizer",
        choices=("none", "ema", "welford", "streaming_batch"),
        default="ema",
    )
    args = parser.parse_args(argv)
    result = run_step1_smoke(
        Step1KernelConfig(
            optimizer=cast(Step1OptimizerName, args.optimizer),
            normalizer=cast(Step1NormalizerName, args.normalizer),
        ),
        steps=args.steps,
        seed=args.seed,
        final_window=args.final_window,
    )
    _print_json(result.to_dict())
    return 0 if result.finite else 1


def step2_smoke_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``alberta-step2-smoke``."""
    parser = argparse.ArgumentParser(description="Run a Step 2 UPGD kernel smoke test.")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=32)
    parser.add_argument(
        "--stream",
        choices=("polynomial", "frequency", "compositional"),
        default="polynomial",
    )
    parser.add_argument("--n-heads", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=8)
    args = parser.parse_args(argv)
    result = run_step2_smoke(
        Step2KernelConfig(
            stream=cast(Step2StreamName, args.stream),
            n_heads=args.n_heads,
            feature_dim=args.feature_dim,
        ),
        steps=args.steps,
        seed=args.seed,
        final_window=args.final_window,
    )
    _print_json(result.to_dict())
    return 0 if result.finite else 1


_EVIDENCE_GATE_DEPRECATION = (
    "alberta-evidence-gate is deprecated; use alberta-evidence-status. "
    "Delegating to the versioned evidence registry.\n"
)
_EVIDENCE_GATE_STEP_ERROR = (
    "error: --step belonged to the retired Step 1/2 file-availability check "
    "and has no modern registry equivalent; use alberta-evidence-status "
    "without --step.\n"
)


def evidence_gate_main(argv: Sequence[str] | None = None) -> int:
    """Delegate the deprecated command to the versioned evidence registry.

    The former Step 1/2 availability check depended on unshipped upstream
    experiment trees and accepted arbitrary parseable JSON. No current
    scientific contract can preserve its ``--step`` selector, so that option
    is rejected rather than silently mapped to unrelated registered claims.
    """

    resolved_argv = tuple(sys.argv[1:] if argv is None else argv)
    sys.stderr.write(_EVIDENCE_GATE_DEPRECATION)
    if any(arg == "--step" or arg.startswith("--step=") for arg in resolved_argv):
        sys.stderr.write(_EVIDENCE_GATE_STEP_ERROR)
        return 2

    # Import lazily so the lightweight Step 1/2 smoke commands do not import
    # every scientific validator merely because they share this module.
    from alberta_framework.evaluation.evidence_manifest_cli import (
        main as evidence_status_main,
    )

    return evidence_status_main(resolved_argv)
