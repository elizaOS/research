"""Capability-gated operational evaluator for the one-shot v3 panel.

This module is intentionally the post-consumption boundary.  A future
pure-stdlib bootstrap must durably consume the attempt before importing this
module, then pass its opaque process-attempt capability, an authorizer owned by
that bootstrap, and the independently measured execution bindings below.
The authorizer receives every named boundary; at ``closure-postflight`` the
bootstrap is expected to revalidate its declared-loader and execution-closure
bytes before returning exact ``True``.

The capability check is a practical Python call boundary, not a sandbox or a
security boundary: Python callers that control this process can forge objects,
callbacks, or imports.  The evaluator itself cannot issue a root, initialize a
ledger, retry an attempt, write output, define a threshold, search, tune, select
a winner/default, authorize evidence, or promote a scientific claim.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from typing import Any, Final, cast

from alberta_framework.core.compositional_features import CompositionalFeatureLearner
from alberta_framework.evaluation import (
    _compositional_future_utility_calibration_engine as engine,
)
from alberta_framework.evaluation import (
    _compositional_future_utility_state_gate as state_gate,
)
from alberta_framework.evaluation import (
    _compositional_future_utility_v3_report_gate as report_gate,
)
from alberta_framework.evaluation import (
    _compositional_future_utility_v3_reward_counts as reward_gate,
)
from alberta_framework.evaluation import compositional_control_life_development as control
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v3_protocol as v3_protocol,
)
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v3_source as v3_source,
)

AttemptAuthorizer = Callable[
    [object, str, report_gate.ExpectedExecutionBindings],
    bool,
]

DEVELOPMENT_ONLY: Final = True
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
RETRY_OR_RECOVERY_AUTHORIZED: Final = False
SOURCE_ARM_NAME: Final = v3_protocol.LEFT_PACK_SOURCE_ARM
LEARNER_CONFIG_SHA256S: Final = (
    "5bca00ecc8a3c14dff9eb1afbd7af2e0d6cfc371e80fad21da4a5239af7548e7",
    "34d98992313753d1e810a22714cd22bf4199cfcdb9359eff1b4e887564ca1392",
    "590a9e5f757cffcc9ca8aac120a57b34ebf7ffce53f57b96974433f3e9c1778f",
    "f1ddcfde6a7d3ed6cf5f238afa95e1846bf2367315c112e5b9cc811d3590a269",
    "defe82edf61c6e7fbbd3f5dce7c4353738bfead2f5e13858245c9ecd393dc12e",
)

_ENTRY_AUTHORIZATION_STAGE: Final = "entry-preflight"
_CLOSURE_POSTFLIGHT_STAGE: Final = "closure-postflight"
_COMPLETION_AUTHORIZATION_STAGE: Final = "completion-postflight"


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_active_attempt(
    capability: object,
    authorizer: AttemptAuthorizer,
    bindings: report_gate.ExpectedExecutionBindings,
    *,
    stage: str,
) -> None:
    """Require the bootstrap-owned opaque capability at one named boundary."""

    if capability is None:
        raise TypeError("attempt capability must be a non-None opaque object")
    if not callable(authorizer):
        raise TypeError("attempt authorizer must be callable")
    if type(bindings) is not report_gate.ExpectedExecutionBindings:
        raise TypeError("expected bindings must be exact ExpectedExecutionBindings")
    if type(stage) is not str or not stage:
        raise TypeError("authorization stage must be a nonempty exact string")
    authorized = authorizer(capability, stage, bindings)
    if type(authorized) is not bool:
        raise TypeError("attempt authorizer must return an exact bool")
    if not authorized:
        raise PermissionError(f"operational attempt is not active at {stage}")


@dataclasses.dataclass(frozen=True, slots=True)
class V3EvaluatorProgressSnapshot:
    """Immutable bootstrap-facing progress and failure metadata."""

    entered: bool
    stage: str
    current_arm: str | None
    scans_completed: int
    arm_records_completed: int
    panel_completed: bool
    succeeded: bool
    failed: bool
    failure_type: str | None
    failure_message: str | None

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.entered,
                self.panel_completed,
                self.succeeded,
                self.failed,
            )
        ):
            raise TypeError("progress flags must be exact booleans")
        if type(self.stage) is not str or not self.stage:
            raise TypeError("progress stage must be a nonempty exact string")
        if self.current_arm is not None and (
            type(self.current_arm) is not str
            or self.current_arm not in v3_protocol.ARM_NAMES
        ):
            raise ValueError("progress current arm is not declared")
        for field_name, value in (
            ("scans_completed", self.scans_completed),
            ("arm_records_completed", self.arm_records_completed),
        ):
            if type(value) is not int or not 0 <= value <= len(v3_protocol.ARM_NAMES):
                raise ValueError(f"progress {field_name} is outside the panel")
        if self.arm_records_completed > self.scans_completed:
            raise ValueError("arm records cannot outnumber completed scans")
        if self.panel_completed is not (
            self.scans_completed == len(v3_protocol.ARM_NAMES)
        ):
            raise ValueError("panel completion must mean all five scans returned")
        if self.succeeded and (
            self.failed
            or not self.panel_completed
            or self.arm_records_completed != len(v3_protocol.ARM_NAMES)
        ):
            raise ValueError("successful progress does not close the full panel")
        if self.failed is not (self.failure_type is not None):
            raise ValueError("failure type does not match the failure flag")
        if self.failed is not (self.failure_message is not None):
            raise ValueError("failure message does not match the failure flag")
        if self.succeeded and self.stage != "completed":
            raise ValueError("successful progress must be at the completed stage")

    def to_config(self) -> dict[str, object]:
        """Return a fresh strict-JSON progress record for terminal sealing."""

        return dataclasses.asdict(self)


class V3EvaluatorProgress:
    """Single-use in-memory tracker retained by the bootstrap on failure."""

    __slots__ = (
        "_arm_records_completed",
        "_current_arm",
        "_entered",
        "_failed",
        "_failure_message",
        "_failure_type",
        "_scans_completed",
        "_stage",
        "_succeeded",
    )

    def __init__(self) -> None:
        self._entered = False
        self._stage = "not-entered"
        self._current_arm: str | None = None
        self._scans_completed = 0
        self._arm_records_completed = 0
        self._succeeded = False
        self._failed = False
        self._failure_type: str | None = None
        self._failure_message: str | None = None

    def snapshot(self) -> V3EvaluatorProgressSnapshot:
        """Return the current immutable progress record without doing work."""

        return V3EvaluatorProgressSnapshot(
            entered=self._entered,
            stage=self._stage,
            current_arm=self._current_arm,
            scans_completed=self._scans_completed,
            arm_records_completed=self._arm_records_completed,
            panel_completed=self._scans_completed == len(v3_protocol.ARM_NAMES),
            succeeded=self._succeeded,
            failed=self._failed,
            failure_type=self._failure_type,
            failure_message=self._failure_message,
        )

    def _enter(self) -> None:
        if self.snapshot() != V3EvaluatorProgressSnapshot(
            entered=False,
            stage="not-entered",
            current_arm=None,
            scans_completed=0,
            arm_records_completed=0,
            panel_completed=False,
            succeeded=False,
            failed=False,
            failure_type=None,
            failure_message=None,
        ):
            raise RuntimeError("evaluator progress is single-use")
        self._entered = True
        self._stage = "protocol-preflight"

    def _set_stage(self, stage: str) -> None:
        if not self._entered or self._succeeded or self._failed:
            raise RuntimeError("progress cannot advance outside an active evaluation")
        if type(stage) is not str or not stage:
            raise TypeError("progress stage must be a nonempty exact string")
        self._stage = stage

    def _start_arm(self, arm_name: str) -> None:
        expected = v3_protocol.ARM_NAMES[self._scans_completed]
        if arm_name != expected or self._arm_records_completed != self._scans_completed:
            raise RuntimeError("arm execution order or completion closure drifted")
        self._current_arm = arm_name
        self._set_stage(f"scan:{arm_name}")

    def _complete_scan(self, arm_name: str) -> None:
        if self._current_arm != arm_name or self._stage != f"scan:{arm_name}":
            raise RuntimeError("scan completion does not match the active arm")
        self._scans_completed += 1
        self._set_stage(f"post-scan-gates:{arm_name}")

    def _complete_arm_record(self, arm_name: str) -> None:
        if (
            self._current_arm != arm_name
            or self._stage != f"post-scan-gates:{arm_name}"
            or self._arm_records_completed + 1 != self._scans_completed
        ):
            raise RuntimeError("arm record completion does not close its scan")
        self._arm_records_completed += 1
        self._current_arm = None
        self._set_stage("between-arms")

    def _complete(self) -> None:
        if (
            self._scans_completed != len(v3_protocol.ARM_NAMES)
            or self._arm_records_completed != len(v3_protocol.ARM_NAMES)
            or self._current_arm is not None
        ):
            raise RuntimeError("cannot complete an unfinished panel")
        self._stage = "completed"
        self._succeeded = True

    def _record_failure(self, error: BaseException) -> None:
        if self._succeeded:
            self._succeeded = False
            self._stage = "result-construction-failed"
        self._failed = True
        self._failure_type = type(error).__name__
        self._failure_message = str(error)
        if not self._entered:
            self._stage = "entry-authorization-failed"


@dataclasses.dataclass(frozen=True, slots=True)
class V3OperationalEvaluationResult:
    """Frozen in-memory result; accessing ``report`` returns a fresh copy."""

    canonical_report_json: str
    report_sha256: str
    progress: V3EvaluatorProgressSnapshot
    development_only: bool = True
    output_writes_allowed: bool = False
    evidence_authorized: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        if type(self.canonical_report_json) is not str:
            raise TypeError("canonical report JSON must be an exact string")
        try:
            report = json.loads(self.canonical_report_json)
        except (TypeError, ValueError) as error:
            raise ValueError("canonical report JSON cannot be decoded") from error
        if report_gate.canonical_json(report) != self.canonical_report_json:
            raise ValueError("result report text is not canonical JSON")
        if not _is_sha256(self.report_sha256):
            raise ValueError("result report SHA-256 is invalid")
        if type(report) is not dict or report.get("report_sha256") != self.report_sha256:
            raise ValueError("result report hash does not match the sealed report")
        body = {key: value for key, value in report.items() if key != "report_sha256"}
        if report_gate.canonical_json_sha256(body) != self.report_sha256:
            raise ValueError("result report SHA-256 does not reconstruct")
        if type(self.progress) is not V3EvaluatorProgressSnapshot:
            raise TypeError("result progress must be an exact progress snapshot")
        if not self.progress.succeeded:
            raise ValueError("a result requires successful full-panel progress")
        if (
            self.development_only is not True
            or self.output_writes_allowed is not False
            or self.evidence_authorized is not False
            or self.scientific_promotion_allowed is not False
        ):
            raise ValueError("operational result acquired forbidden authority")

    @property
    def report(self) -> dict[str, object]:
        """Return a fresh strict-JSON copy of the validated report."""

        value = json.loads(self.canonical_report_json)
        if type(value) is not dict:
            raise RuntimeError("validated result report is no longer a JSON object")
        return cast(dict[str, object], value)


@dataclasses.dataclass(frozen=True, slots=True)
class _PreparedArm:
    specification: engine.FutureUtilityArmSpec
    learner: CompositionalFeatureLearner
    learner_config: dict[str, Any]
    learner_config_sha256: str


def _arm_specifications(
    declaration: v3_protocol.CompositionalFutureUtilityCalibrationV3Protocol,
) -> tuple[engine.FutureUtilityArmSpec, ...]:
    specifications = tuple(
        engine.FutureUtilityArmSpec(
            name=arm.name,
            role=arm.role,
            mix=arm.future_utility_mix,
            trace_decay=arm.future_utility_trace_decay,
            normalization=arm.future_utility_normalization,
        )
        for arm in declaration.arms
    )
    if tuple(specification.name for specification in specifications) != (
        v3_protocol.ARM_NAMES
    ):
        raise RuntimeError("v3 engine arm order differs from the frozen protocol")
    return specifications


def _validate_report_gate_preflight() -> None:
    """Require all public report-schema geometry before the first scan."""

    exact_sequences = (
        (v3_protocol.ARM_NAMES, report_gate.ARM_ORDER, "arm order"),
        (v3_protocol.PHASE_ORDER, report_gate.PHASE_ORDER, "phase order"),
        (v3_protocol.PHASE_LENGTHS, report_gate.PHASE_LENGTHS, "phase lengths"),
        (
            v3_protocol.PHASE_BOUNDARIES,
            report_gate.PHASE_BOUNDARIES,
            "phase boundaries",
        ),
        (v3_protocol.TARGET_NAMES, report_gate.TARGET_NAMES, "target order"),
        (
            engine.PRIMARY_ENDPOINT_NAMES,
            report_gate.PRIMARY_ENDPOINT_ORDER,
            "primary endpoint order",
        ),
    )
    for observed, expected, label in exact_sequences:
        if observed != expected:
            raise RuntimeError(f"engine/protocol and report-gate {label} differ")
    exact_scalars = (
        (v3_protocol.TOTAL_STEPS, report_gate.TOTAL_STEPS, "total steps"),
        (
            v3_protocol.TOTAL_CURATION_OPPORTUNITIES,
            report_gate.TOTAL_CURATION_OPPORTUNITIES,
            "curation opportunities",
        ),
        (
            v3_protocol.CURATION_INTERVAL,
            report_gate.CURATION_INTERVAL,
            "curation interval",
        ),
        (v3_protocol.ENTRY_WINDOW, report_gate.ENTRY_WINDOW, "entry window"),
        (v3_protocol.TAIL_WINDOW, report_gate.TAIL_WINDOW, "tail window"),
        (v3_protocol.RAW_DIM, report_gate.RAW_DIM, "raw dimension"),
        (v3_protocol.ACTIVE_SLOTS, report_gate.ACTIVE_SLOTS, "active slots"),
        (
            v3_protocol.CANDIDATE_SLOTS,
            report_gate.CANDIDATE_SLOTS,
            "candidate slots",
        ),
        (v3_protocol.ACTION_HEADS, report_gate.ACTION_HEADS, "action heads"),
        (SOURCE_ARM_NAME, report_gate.SOURCE_ARM_NAME, "source arm"),
        (
            control.ARM_EXECUTION_RECEIPT_SCHEMA,
            report_gate.ARM_EXECUTION_RECEIPT_SCHEMA,
            "execution receipt schema",
        ),
        (
            state_gate.STATE_GATE_SCHEMA,
            report_gate.STATE_GATE_SCHEMA,
            "state-gate schema",
        ),
        (
            reward_gate.REWARD_COUNT_SCHEMA,
            report_gate.REWARD_COUNT_SCHEMA,
            "reward-count schema",
        ),
    )
    for scalar_observed, scalar_expected, scalar_label in exact_scalars:
        if scalar_observed != scalar_expected:
            raise RuntimeError(
                f"engine/protocol and report-gate {scalar_label} differ"
            )


def _prepare_arms(
    declaration: v3_protocol.CompositionalFutureUtilityCalibrationV3Protocol,
    specifications: tuple[engine.FutureUtilityArmSpec, ...],
) -> tuple[_PreparedArm, ...]:
    matching_source_arms = tuple(
        arm for arm in control.CONTROL_LIFE_ARMS if arm.name == SOURCE_ARM_NAME
    )
    if (
        len(matching_source_arms) != 1
        or matching_source_arms[0].to_config()
        != dict(v3_protocol.SOURCE_ARM_CONFIG)
        or matching_source_arms[0].composed_readout_enabled is not True
    ):
        raise RuntimeError("control-life source-arm semantics differ from v3")
    historical_base = control.learner_config_for_arm(declaration.left_pack_source_arm)
    configurations: dict[str, dict[str, Any]] = {}
    prepared: list[_PreparedArm] = []
    for declared_arm, specification in zip(
        declaration.arms,
        specifications,
        strict=True,
    ):
        config = engine.build_future_utility_learner_config(
            historical_base,
            specification,
        )
        reconstructed = v3_protocol.reconstruct_arm_learner_config(declared_arm)
        if report_gate.canonical_json(config) != report_gate.canonical_json(
            reconstructed
        ):
            raise RuntimeError(
                f"{specification.name} engine and protocol configs disagree"
            )
        learner = CompositionalFeatureLearner.from_config(config)
        if report_gate.canonical_json(learner.to_config()) != report_gate.canonical_json(
            config
        ):
            raise RuntimeError(
                f"{specification.name} learner config roundtrip does not close"
            )
        configurations[specification.name] = config
        prepared.append(
            _PreparedArm(
                specification=specification,
                learner=learner,
                learner_config=config,
                learner_config_sha256=report_gate.canonical_json_sha256(config),
            )
        )
    engine.validate_future_utility_arm_contrasts(
        historical_base,
        specifications,
        configurations,
    )
    observed_hashes = tuple(arm.learner_config_sha256 for arm in prepared)
    if observed_hashes != LEARNER_CONFIG_SHA256S:
        raise RuntimeError("v3 learner config hashes differ from their frozen pins")
    return tuple(prepared)


def _curation_totals(
    analysis: control.CompositionalControlLifeArmAnalysisReceipt,
) -> dict[str, int]:
    payload = analysis.to_config().get("curation_totals")
    if type(payload) is not dict or any(
        type(name) is not str or type(value) is not int
        for name, value in payload.items()
    ):
        raise RuntimeError("public arm analysis returned invalid curation totals")
    return cast(dict[str, int], payload)


def _require_engine_gate_receipts(
    endpoint_geometry: engine.FutureUtilityEndpointGeometry,
    bound: v3_source.BoundV3Source,
    execution: control.CompositionalControlLifeArmExecution,
    curation_totals: Mapping[str, int],
) -> None:
    shapes = engine.validate_future_utility_trace_shapes(
        endpoint_geometry,
        execution.events,
    )
    if not shapes or any(shape[0] != endpoint_geometry.total_steps for shape in shapes.values()):
        raise RuntimeError("future-utility trace-shape receipt is incomplete")
    experience = engine.validate_future_utility_experience_semantics(
        endpoint_geometry,
        execution.events,
        observations=bound.source.observations,
        phase_indices=bound.source.phase_indices,
        exploration_mask=bound.source.exploration_mask,
        random_actions=bound.source.random_actions,
        phase_target_raw_indices=v3_protocol.PHASE_TARGET_RAW_INDICES,
        action_reward_multipliers=(-1.0, 1.0),
        composed_readout_enabled=True,
    )
    if (
        type(experience) is not dict
        or experience.get("all_experience_semantics_match") is not True
        or experience.get("steps") != endpoint_geometry.total_steps
        or experience.get("composed_readout_enabled") is not True
    ):
        raise RuntimeError("future-utility experience receipt is not exact")
    eventwise = engine.validate_future_utility_eventwise_curation_semantics(
        endpoint_geometry,
        execution.events,
    )
    if (
        type(eventwise) is not dict
        or eventwise.get("all_eventwise_curation_semantics_match") is not True
    ):
        raise RuntimeError("future-utility eventwise receipt is not accepted")
    cadence = engine.future_utility_cadence_audit_from_events(
        endpoint_geometry,
        execution.events,
        pinned_due_mask=bound.source.curation_due_mask,
    )
    if cadence.due_opportunity_count != v3_protocol.TOTAL_CURATION_OPPORTUNITIES:
        raise RuntimeError("future-utility cadence receipt differs from v3")
    count_closure = engine.validate_future_utility_curation_count_closure(
        cadence,
        curation_totals,
    )
    if (
        type(count_closure) is not dict
        or count_closure.get("all_checked_counts_close") is not True
        or count_closure.get("curation_due_count")
        != v3_protocol.TOTAL_CURATION_OPPORTUNITIES
    ):
        raise RuntimeError("future-utility curation counts do not close")


def _run_record(
    prepared: _PreparedArm,
    analysis: control.CompositionalControlLifeArmAnalysisReceipt,
    state_receipt: state_gate.FutureUtilityStateGateReceipt,
    endpoints: dict[str, object],
    reward_counts: reward_gate.ExactRewardCountProjection,
) -> dict[str, object]:
    body: dict[str, object] = {
        "arm": prepared.specification.name,
        "source_arm_name": SOURCE_ARM_NAME,
        "learner_config_sha256": prepared.learner_config_sha256,
        "execution_receipt": analysis.execution_receipt.to_config(),
        "state_gate_receipt": state_receipt.to_config(),
        "primary_endpoints": endpoints,
        "reward_counts": reward_counts.to_config(),
    }
    return {
        **body,
        "arm_record_sha256": report_gate.canonical_json_sha256(body),
    }


def _bindings_record(
    declaration: v3_protocol.CompositionalFutureUtilityCalibrationV3Protocol,
    bound: v3_source.BoundV3Source,
    expected: report_gate.ExpectedExecutionBindings,
) -> dict[str, object]:
    return {
        "development_root": declaration.development_root,
        "development_root_hex": declaration.development_root_hex,
        "protocol_config_sha256": v3_protocol.PROTOCOL_CONFIG_SHA256,
        "control_protocol_config_sha256": bound.control_protocol_config_sha256,
        "runtime_config_sha256": bound.runtime_config_sha256,
        "consumed_history_sha256": bound.consumed_history_sha256,
        "key_manifest_sha256": bound.key_manifest_sha256,
        "stream_sha256": bound.stream_sha256,
        "cadence_bound_stream_sha256": bound.cadence_bound_stream_sha256,
        "source_envelope_sha256": bound.stream_envelope_sha256,
        **expected.to_config(),
    }


def _build_report(
    declaration: v3_protocol.CompositionalFutureUtilityCalibrationV3Protocol,
    bound: v3_source.BoundV3Source,
    expected_bindings: report_gate.ExpectedExecutionBindings,
    runs: tuple[dict[str, object], ...],
    work_contract: dict[str, object],
) -> dict[str, object]:
    if len(runs) != len(v3_protocol.ARM_NAMES):
        raise RuntimeError("a completed report requires exactly five run records")
    execution_receipt = cast(dict[str, object], runs[0]["execution_receipt"])
    state_receipt = cast(dict[str, object], runs[0]["state_gate_receipt"])
    body: dict[str, object] = {
        "schema": report_gate.REPORT_SCHEMA,
        "status": report_gate.REPORT_STATUS,
        "bindings": _bindings_record(declaration, bound, expected_bindings),
        "execution": {
            "attempt_index": 1,
            "attempts_authorized": 1,
            "attempts_consumed": 1,
            "root_consumed": True,
            "attempt_consumed_before_evaluator_import": True,
            "retry_or_recovery_authorized": False,
            "panel_completed": True,
            "arm_count": len(runs),
        },
        "authority": {
            "development_only": True,
            "descriptive_result_available": True,
            "scientific_promotion_allowed": False,
            "evidence_authorized": False,
            "experiment_output_writes_allowed": False,
            "artifact_authorized": False,
            "threshold_defined_or_applied": False,
            "winner_or_default_selected": False,
            "search_or_tuning_performed": False,
            "retry_or_recovery_authorized": False,
        },
        "arm_order": list(v3_protocol.ARM_NAMES),
        "primary_endpoint_order": list(engine.PRIMARY_ENDPOINT_NAMES),
        "reward_metric_order": list(report_gate.REWARD_RECORD_FIELDS),
        "runs": list(runs),
        "cross_arm_contract": {
            "shared_initial_state_sha256": execution_receipt["initial_state_sha256"],
            "shared_initial_subset_sha256": state_receipt["initial_subset_sha256"],
            "shared_protocol_source_and_genesis": True,
            "shared_base_logical_work_matched": True,
            "stream_shapes_and_update_opportunities_matched": True,
            "persistent_shapes_and_bytes_matched": True,
            "intervention_specific_logical_work_matched": False,
            "total_named_logical_work_equivalence_claimed": False,
            "behavior_dependent_branch_work_equivalence_claimed": False,
            "behavioral_experience_matching_claimed": False,
            "compiled_flop_equivalence_claimed": False,
            "work_resource_contract_embedded": True,
            "work_resource_contract_sha256_bound": True,
        },
        "work_resource_contract": work_contract,
        "work_resource_contract_sha256": report_gate.WORK_RESOURCE_CONTRACT_SHA256,
    }
    return {**body, "report_sha256": report_gate.canonical_json_sha256(body)}


def _postflight(
    declaration: v3_protocol.CompositionalFutureUtilityCalibrationV3Protocol,
    bound: v3_source.BoundV3Source,
    prepared_arms: tuple[_PreparedArm, ...],
) -> None:
    v3_source.validate_protocol_and_source_constants(
        observed_protocol_config_sha256=v3_protocol.protocol_config_sha256(
            declaration
        )
    )
    if v3_source.validate_bound_v3_source(bound) is not bound:
        raise RuntimeError("bound v3 source postflight changed object identity")
    reconstructed = v3_protocol.reconstruct_protocol(declaration.to_config())
    if reconstructed != declaration:
        raise RuntimeError("v3 protocol postflight reconstruction differs")
    if tuple(arm.specification.name for arm in prepared_arms) != v3_protocol.ARM_NAMES:
        raise RuntimeError("v3 arm order changed during execution")
    for prepared in prepared_arms:
        if report_gate.canonical_json(prepared.learner.to_config()) != (
            report_gate.canonical_json(prepared.learner_config)
        ):
            raise RuntimeError(
                f"{prepared.specification.name} learner config changed during execution"
            )
        if report_gate.canonical_json_sha256(prepared.learner_config) != (
            prepared.learner_config_sha256
        ):
            raise RuntimeError(
                f"{prepared.specification.name} learner config hash changed"
            )


def evaluate_v3_operational_panel(
    *,
    attempt_capability: object,
    attempt_authorizer: AttemptAuthorizer,
    expected_bindings: report_gate.ExpectedExecutionBindings,
    progress: V3EvaluatorProgress,
) -> V3OperationalEvaluationResult:
    """Execute the sole preconsumed v3 panel and return an in-memory report.

    This entry point must never be called before the future bootstrap has
    durably created and validated ``started.json``.  It deliberately has no
    convenience defaults and no path that constructs a capability or ledger.
    """

    if type(progress) is not V3EvaluatorProgress:
        raise TypeError("progress must be an exact V3EvaluatorProgress")
    try:
        if type(expected_bindings) is not report_gate.ExpectedExecutionBindings:
            raise TypeError(
                "expected_bindings must be exact ExpectedExecutionBindings"
            )
        _require_active_attempt(
            attempt_capability,
            attempt_authorizer,
            expected_bindings,
            stage=_ENTRY_AUTHORIZATION_STAGE,
        )
        progress._enter()

        declaration = v3_protocol.CompositionalFutureUtilityCalibrationV3Protocol()
        if (
            v3_protocol.reconstruct_protocol(declaration.to_config()) != declaration
            or v3_protocol.protocol_config_sha256(declaration)
            != v3_protocol.PROTOCOL_CONFIG_SHA256
        ):
            raise RuntimeError("frozen v3 protocol did not reconstruct at preflight")
        v3_source.validate_protocol_and_source_constants(
            observed_protocol_config_sha256=v3_protocol.PROTOCOL_CONFIG_SHA256
        )
        bound = v3_source.build_bound_v3_source()
        if v3_source.validate_bound_v3_source(bound) is not bound:
            raise RuntimeError("bound v3 source validation changed object identity")

        progress._set_stage("geometry-and-arm-preflight")
        _validate_report_gate_preflight()
        endpoint_geometry = engine.FutureUtilityEndpointGeometry(
            phase_order=declaration.phase_order,
            phase_lengths=declaration.phase_lengths,
            target_names=declaration.target_names,
            curation_interval=declaration.curation_interval,
        )
        work_geometry = engine.FutureUtilityWorkGeometry(
            steps=endpoint_geometry.total_steps,
            curation_interval=declaration.curation_interval,
            active_slots=v3_protocol.ACTIVE_SLOTS,
            candidate_slots=v3_protocol.CANDIDATE_SLOTS,
            action_heads=v3_protocol.ACTION_HEADS,
        )
        if (
            endpoint_geometry.total_steps != v3_protocol.TOTAL_STEPS
            or endpoint_geometry.phase_boundaries != v3_protocol.PHASE_BOUNDARIES
            or bound.control_protocol.phase_lengths != endpoint_geometry.phase_lengths
            or bound.control_protocol.total_steps != work_geometry.steps
        ):
            raise RuntimeError("v3 endpoint, work, and bound-source geometry differ")
        specifications = _arm_specifications(declaration)
        prepared_arms = _prepare_arms(declaration, specifications)
        work_contract = engine.work_resource_contract(work_geometry, specifications)
        if (
            report_gate.canonical_json(work_contract)
            != report_gate.canonical_json(report_gate.work_resource_contract_config())
            or report_gate.canonical_json_sha256(work_contract)
            != report_gate.WORK_RESOURCE_CONTRACT_SHA256
        ):
            raise RuntimeError("engine and report-gate work contracts disagree")

        runs: list[dict[str, object]] = []
        for prepared in prepared_arms:
            arm_name = prepared.specification.name
            progress._start_arm(arm_name)
            _require_active_attempt(
                attempt_capability,
                attempt_authorizer,
                expected_bindings,
                stage=f"before-scan:{arm_name}",
            )
            execution = control.execute_compositional_control_life_arm(
                bound.control_protocol,
                prepared.learner,
                bound.source.learner_key,
                bound.source.observations,
                bound.source.phase_indices,
                bound.source.exploration_mask,
                bound.source.random_actions,
                composed_readout_enabled=True,
            )
            progress._complete_scan(arm_name)
            _require_active_attempt(
                attempt_capability,
                attempt_authorizer,
                expected_bindings,
                stage=f"after-scan:{arm_name}",
            )

            analysis = control.analyze_compositional_control_life_arm_execution(
                bound.control_protocol,
                execution,
                curation_geometry_arm_name=SOURCE_ARM_NAME,
                pinned_curation_due_mask=bound.source.curation_due_mask,
            )
            state_receipt = state_gate.validate_future_utility_state_gate(
                execution,
                future_utility_mix=prepared.specification.mix,
                future_utility_trace_decay=prepared.specification.trace_decay,
                future_utility_normalization=prepared.specification.normalization,
            )
            reward_counts = reward_gate.project_v3_exact_reward_counts(
                bound,
                execution.events,
            )
            totals = _curation_totals(analysis)
            _require_engine_gate_receipts(
                endpoint_geometry,
                bound,
                execution,
                totals,
            )
            active_trajectories = analysis.active_structural_trajectories
            target_trajectories = {
                name: cast(Mapping[str, object], active_trajectories[name])
                for name in declaration.target_names
            }
            endpoints = engine.build_future_utility_primary_endpoints(
                endpoint_geometry,
                execution.events,
                active_trajectories=target_trajectories,
                curation_totals=totals,
                curation_audit=analysis.curation_decision_audit,
                pinned_due_mask=bound.source.curation_due_mask,
            )
            if endpoints.get("endpoint_order") != list(engine.PRIMARY_ENDPOINT_NAMES):
                raise RuntimeError("primary endpoint order differs from its engine pin")
            runs.append(
                _run_record(
                    prepared,
                    analysis,
                    state_receipt,
                    endpoints,
                    reward_counts,
                )
            )
            progress._complete_arm_record(arm_name)

        progress._set_stage("report-gate")
        report = _build_report(
            declaration,
            bound,
            expected_bindings,
            tuple(runs),
            work_contract,
        )
        canonical = report_gate.serialize_v3_descriptive_report(
            report,
            expected_bindings,
        )

        progress._set_stage("source-and-loader-postflight")
        _postflight(declaration, bound, prepared_arms)
        _require_active_attempt(
            attempt_capability,
            attempt_authorizer,
            expected_bindings,
            stage=_CLOSURE_POSTFLIGHT_STAGE,
        )
        if (
            report_gate.serialize_v3_descriptive_report(report, expected_bindings)
            != canonical
        ):
            raise RuntimeError("validated report changed across source postflight")
        _require_active_attempt(
            attempt_capability,
            attempt_authorizer,
            expected_bindings,
            stage=_COMPLETION_AUTHORIZATION_STAGE,
        )
        progress._complete()
        report_sha256 = report["report_sha256"]
        if not _is_sha256(report_sha256):
            raise RuntimeError("validated report lacks an exact report SHA-256")
        return V3OperationalEvaluationResult(
            canonical_report_json=canonical,
            report_sha256=cast(str, report_sha256),
            progress=progress.snapshot(),
        )
    except BaseException as error:
        progress._record_failure(error)
        raise


__all__ = [
    "AttemptAuthorizer",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "OUTPUT_WRITES_ALLOWED",
    "RETRY_OR_RECOVERY_AUTHORIZED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SOURCE_ARM_NAME",
    "V3EvaluatorProgress",
    "V3EvaluatorProgressSnapshot",
    "V3OperationalEvaluationResult",
    "evaluate_v3_operational_panel",
]
