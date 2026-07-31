"""Build the FOV paired-comparison artifact: Alberta (matrix eval) vs official DQN.

Run from the frozen source copy so library imports match the code that produced
the Alberta runs:
    cd outputs/forager/frozen_src_20260731 && .venv/bin/python <this script>
"""

import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

from alberta_framework.benchmarks import forager_matrix as fm
from alberta_framework.benchmarks.forager import (
    ForagerRunResult,
    paper_protocol,
    paper_reference_targets,
    summarize_forager_runs,
)
from alberta_framework.benchmarks.forager_results import (
    import_official_foragax_npz,
    paired_forager_comparison,
)
from alberta_framework.benchmarks.official_foragax import (
    official_foragax_batch_run_specs_from_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "outputs/forager/fov_eval_500k_seeds0_29"
OFFICIAL_MANIFEST = ROOT / "outputs/forager/official_dqn_fov_500k_seeds0_29/manifest.json"
OUTPUT = ROOT / "outputs/forager/fov_paired_comparison_dqn_seeds0_29.json"

METRIC = "fov_last_10pct_ema_auc"
PROTOCOL = paper_protocol("field_of_view")


def run_from_payload(payload: dict) -> ForagerRunResult:
    kwargs = dict(payload)
    for name in ("curve_steps", "curve_ewm_reward", "curve_window_reward"):
        kwargs[name] = tuple(payload[name])
    for name in ("mean_biome_regret", "final_biome_regret"):
        value = payload[name]
        kwargs[name] = math.nan if value is None else float(value)
    return ForagerRunResult(**kwargs)


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main() -> int:
    report_path = EVAL_DIR / "report.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report["status"] != "complete":
        raise SystemExit("Alberta evaluation matrix report is not complete")
    conformance = report["protocol_conformance"]
    if conformance["full_paper_protocol_conformant"] is not True:
        raise SystemExit("Alberta evaluation is not full-paper-protocol conformant")

    alberta_runs: list[ForagerRunResult] = []
    for relative in sorted(report["batch_artifacts"]):
        batch = fm._load_canonical_artifact(
            EVAL_DIR / relative, description=f"batch artifact {relative}"
        )
        alberta_runs.extend(run_from_payload(item) for item in batch["runs"])
    alberta_runs.sort(key=lambda run: run.seed)

    specs = official_foragax_batch_run_specs_from_manifest(OFFICIAL_MANIFEST)
    dqn_runs = [
        import_official_foragax_npz(
            spec,
            ewm_decay=PROTOCOL.ewm_decay,
            record_every=1_000,
            final_window=PROTOCOL.final_window_steps,
        )
        for spec in specs
    ]
    dqn_runs.sort(key=lambda run: run.seed)

    comparison = paired_forager_comparison(
        alberta_runs,
        dqn_runs,
        metric=METRIC,
        confidence=0.95,
        bootstrap_resamples=10_000,
        bootstrap_seed=0,
    )
    summaries = {
        "alberta_horde_ac": summarize_forager_runs(
            alberta_runs, metric=METRIC, confidence=0.95,
            bootstrap_resamples=10_000, bootstrap_seed=0,
        ),
        "DQN": summarize_forager_runs(
            dqn_runs, metric=METRIC, confidence=0.95,
            bootstrap_resamples=10_000, bootstrap_seed=0,
        ),
    }

    def summary_payload(summary) -> dict:
        return {
            "agent": summary.agent,
            "privileged": summary.privileged,
            "seeds": list(summary.seeds),
            "metric": summary.metric,
            "mean": summary.mean,
            "ci_low": summary.ci_low,
            "ci_high": summary.ci_high,
            "confidence": summary.confidence,
            "per_seed_values": {
                str(run.seed): float(getattr(run, METRIC)) for run in summary.runs
            },
        }

    official_manifest = json.loads(OFFICIAL_MANIFEST.read_bytes())
    payload = {
        "schema_version": "1.0",
        "artifact_type": "alberta_forager_fov_paired_comparison",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "preset": "field_of_view",
        "metric": METRIC,
        "metric_definition": PROTOCOL.metric_definition,
        "paper_protocol": PROTOCOL.to_dict(),
        "protocol_conformance": conformance,
        "candidate": {
            "name": "alberta_horde_ac",
            "source": "alberta_forager_matrix evaluation report",
            "report_path": str(report_path.relative_to(ROOT)),
            "report_file_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "report_payload_sha256": report["payload_sha256"],
            "selected_config": report["matrix_config"]["variants"],
            "tuning_selection": report["matrix_config"]["tuning_selection"],
        },
        "baseline": {
            "name": "DQN",
            "source": "official continual-foragax-agents NPZ batch (corrected official reproduction)",
            "manifest_path": str(OFFICIAL_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": official_manifest["manifest_sha256"],
            "classification": official_manifest["claim"]["classification"],
            "execution_commit": official_manifest["claim"]["execution_commit"],
            "config_commit": official_manifest["claim"]["config_commit"],
            "config_path": official_manifest["source"]["config_path"],
            "protocol_attested": all(
                run.agent_metadata.get("protocol_attested") is True for run in dqn_runs
            ),
        },
        "summaries": {name: summary_payload(s) for name, s in summaries.items()},
        "paired_comparison": comparison.to_dict(),
        "paper_figure_digitized_targets": [
            target.to_dict() for target in paper_reference_targets("field_of_view")
        ],
        "notes": [
            "Both methods executed against the same verified foragax 0.55.0 install "
            "tree on identical seeds 0-29 with identical 500k-step budgets and "
            "identical metric contracts; the paired bootstrap uses per-seed "
            "differences.",
            "Digitized paper targets are figure-read orientation values, not "
            "acceptance thresholds.",
            "This is a Foragax 0.55.0 reproduction, not the paper-time NumPy "
            "Forager environment.",
        ],
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")

    print(f"wrote {OUTPUT}")
    print(f"alberta mean {summaries['alberta_horde_ac'].mean:.4f} "
          f"[{summaries['alberta_horde_ac'].ci_low:.4f}, {summaries['alberta_horde_ac'].ci_high:.4f}]")
    print(f"dqn mean {summaries['DQN'].mean:.4f} "
          f"[{summaries['DQN'].ci_low:.4f}, {summaries['DQN'].ci_high:.4f}]")
    print(f"paired diff {comparison.mean_difference:.4f} "
          f"[{comparison.ci_low:.4f}, {comparison.ci_high:.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
