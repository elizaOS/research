"""One-shot future-utility calibration v2 for compositional control life.

This module declares five causal future-utility configurations on one newly
issued, deterministic 8,998-step development stream.  The configurations are
fixed before execution and differ only in mix, trace decay, and normalization.
The full panel is process-local, at-most-once, and in-memory.  It has no output
writer, evidence authority, promotion path, threshold, winner rule, rerun, or
tuning mechanism.

The common learner base is the control-life left-pack arm with the legacy
ranking compatibility correction: ``legacy`` scoring, zero novelty admission,
contribution traces, and zero rare-task weighting.  Rewards are secondary;
the primary readout is the exact curation and structural-retention path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import struct
import sys
import threading
from collections.abc import Callable, Mapping
from importlib.metadata import version as package_version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import alberta_framework.core.compositional_features as compositional_core
import alberta_framework.core.future_utility as future_utility_core
import alberta_framework.evaluation.compositional_control_life_development as control_life
import alberta_framework.evaluation.generated_birth_identity_scrub_epoch as birth_identity_scrub
import alberta_framework.evaluation.generated_class_lifecycle_scrub as lifecycle_scrub
from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)

PROTOCOL_NAMESPACE: Final = "alberta.compositional-future-utility-calibration-v2"
PROTOCOL_NAMESPACE_SHA256: Final = (
    "72b0a3f637872fbdaa750423a784367d5940004aad60e3d53bf81b60fa062217"
)
PROTOCOL_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v2-development.protocol.v1"
)
REPORT_SCHEMA: Final = (
    "alberta.compositional-future-utility-calibration-v2-development.report.v1"
)
STATUS: Final = "DEVELOPMENT_FUTURE_UTILITY_CALIBRATION_V2_NOT_ASSESSED"
ASSESSMENT_STATUS: Final = "not-assessed"
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EXECUTION_ATTEMPTS_AUTHORIZED: Final = 1
EXECUTION_ATTEMPTS_CONSUMED: Final = 1
EXECUTION_ATTEMPTS_REMAINING: Final = 0
EXECUTION_OUTCOME: Final = "failed-invalid-evaluator-cadence-invariant"

DEVELOPMENT_ROOT_HEX: Final = "0x72B0A3F6"
DEVELOPMENT_ROOT: Final = 1_924_178_934
PHASE_ORDER: Final = ("A", "B", "A", "D", "A", "C", "A", "B", "C", "A")
PHASE_LENGTHS: Final = (797, 829, 857, 883, 911, 941, 971, 1009, 1031, 769)
TOTAL_STEPS: Final = 8_998
CURATION_INTERVAL: Final = 32
PHASE_BOUNDARIES: Final = (0, 797, 1626, 2483, 3366, 4277, 5218, 6189, 7198, 8229, 8998)
CURATION_OPPORTUNITIES_PER_PHASE: Final = (24, 26, 27, 28, 28, 30, 30, 31, 33, 24)

KEY_MANIFEST: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {
        "root": (0, 1_924_178_934),
        "observations": (1_189_056_302, 2_383_774_845),
        "exploration": (3_352_410_003, 3_947_271_724),
        "random_actions": (3_382_640_669, 4_117_898_437),
        "learner_genesis": (2_592_838_183, 3_227_537_730),
    }
)
STREAM_SHA256: Final = (
    "bb741db073a13026425d2cc98cce93a1af1d1b65f2abf24ebc97e43b61abd39c"
)

LONG_TRACE_DECAY_F32_BITS: Final = "3f7fcc93"
LONG_TRACE_DECAY: Final = 0.999215304851532
NORMALIZATION_DECAY: Final = 0.99
LEFT_PACK_SOURCE_ARM: Final = "dovetail_coverage_ancestor_headroom_leftpack"

ARM_NAMES: Final = (
    "current_mix0_decay095_none",
    "future_mix1_decay095_none",
    "calibrated_mix05_decay095_none",
    "normalized_mix1_decay095_uncertainty_age",
    "horizon_mix1_decay883_uncertainty_age",
)
_ARM_PARAMETERS: Final[Mapping[str, tuple[float, float, str]]] = MappingProxyType(
    {
        ARM_NAMES[0]: (0.0, 0.95, "none"),
        ARM_NAMES[1]: (1.0, 0.95, "none"),
        ARM_NAMES[2]: (0.5, 0.95, "none"),
        ARM_NAMES[3]: (1.0, 0.95, "uncertainty_age"),
        ARM_NAMES[4]: (1.0, LONG_TRACE_DECAY, "uncertainty_age"),
    }
)
_ARM_ROLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        ARM_NAMES[0]: "current-utility reference with contribution traces retained",
        ARM_NAMES[1]: "unscaled future-utility endpoint",
        ARM_NAMES[2]: "equal current/future mixture calibration",
        ARM_NAMES[3]: "causally age-and-uncertainty-normalized future utility",
        ARM_NAMES[4]: "long-horizon normalized future utility (about 883-step half-life)",
    }
)
_EXPECTED_INTERVENTION_FIELDS: Final = (
    "future_utility_mix",
    "future_utility_trace_decay",
    "future_utility_normalization",
)
_COMMON_DEPARTURE_FIELDS: Final = (
    "candidate_scoring_mode",
    "candidate_novelty_admission_bonus",
    "future_utility_trace_mode",
    "future_utility_rare_task_power",
)
_FUTURE_TRACE_FIELDS: Final = (
    "utility_contribution_trace",
    "utility_error_trace",
    "utility_feature_trace",
    "utility_feature_energy_trace",
    "candidate_utility_contribution_trace",
    "candidate_utility_feature_trace",
    "candidate_utility_feature_energy_trace",
)
_TARGET_NAMES: Final = ("A", "B", "C")

PRIMARY_ENDPOINTS: Final = (
    "margin_passes",
    "promotions",
    "candidate_refreshes",
    "cascade_losses",
    "target_admission_loss_end",
    "pre_recurrence_presence",
    "target_occupancy",
    "pre_recurrence_ranks",
)
SECONDARY_ENDPOINTS: Final = ("lifetime_reward", "phase_reward")
RUNTIME_IDENTITY_SCOPE: Final = (
    "selected Python, NumPy, JAX, backend, x64, and device fields; not an "
    "environment, accelerator-driver, XLA-flag, or compiler closure"
)
SOURCE_MANIFEST_SCOPE: Final = "selected-direct-files-not-transitive-closure"
RESOURCE_ACCOUNTING_SCOPE: Final = (
    "exact persistent learner-state bytes, matched shared-base logical cell/update "
    "counts, intervention-specific named cell counts, and measured curation-audit "
    "arrays; excludes behavior-dependent branch work, source arrays, full scan "
    "telemetry, compiler workspaces, and compiled FLOPs"
)
INTERPRETATION: Final = (
    "One-shot development calibration of five predeclared causal future-utility "
    "settings. Results remain descriptive and cannot select a winner, tune a setting, "
    "promote a claim, or establish Alberta Plan completion."
)
LIMITATIONS: Final = (
    "one newly issued development root with no held-out confirmation",
    "finite synthetic product grammar and iid contextual-bandit observations",
    "immediate selected-action reward prediction, not TD control or world modeling",
    "legacy scoring and zero novelty bonus depart from the historical left-pack base",
    "structural signature presence is not authenticated cross-birth identity",
    "selected source files are bound, not a transitive source closure",
    "selected runtime fields are bound, not an environment or compiler closure",
    "resource accounting omits behavior-dependent branch work, source arrays, full "
    "telemetry, and compiler workspaces",
    "exogenous streams and genesis are paired, but behavioral experience is not matched",
    "a process-local latch cannot prevent a fresh Python process from replaying the panel",
    "the long decay represents about 882.986 steps and does not span every recurrence gap",
    "no writer, artifact, threshold, winner, rerun, tuning, evidence, or promotion path",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_clone(value: object) -> object:
    return json.loads(_canonical_json(value))


def _f32_hex(value: float) -> str:
    return struct.pack(">f", value).hex()


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalFutureUtilityCalibrationV2Protocol:
    """The exact newly issued development declaration; customization is forbidden."""

    schema_version: str = PROTOCOL_SCHEMA
    namespace: str = PROTOCOL_NAMESPACE
    namespace_sha256: str = PROTOCOL_NAMESPACE_SHA256
    development_root: int = DEVELOPMENT_ROOT
    phase_order: tuple[str, ...] = PHASE_ORDER
    phase_lengths: tuple[int, ...] = PHASE_LENGTHS
    epsilon: float = 0.1
    entry_window: int = 64
    tail_window: int = 64
    curation_interval: int = CURATION_INTERVAL
    left_pack_source_arm: str = LEFT_PACK_SOURCE_ARM

    def __post_init__(self) -> None:
        exact = (
            self.schema_version == PROTOCOL_SCHEMA
            and self.namespace == PROTOCOL_NAMESPACE
            and self.namespace_sha256 == PROTOCOL_NAMESPACE_SHA256
            and type(self.development_root) is int
            and self.development_root == DEVELOPMENT_ROOT
            and type(self.phase_order) is tuple
            and self.phase_order == PHASE_ORDER
            and type(self.phase_lengths) is tuple
            and self.phase_lengths == PHASE_LENGTHS
            and type(self.epsilon) is float
            and self.epsilon == 0.1
            and type(self.entry_window) is int
            and self.entry_window == 64
            and type(self.tail_window) is int
            and self.tail_window == 64
            and type(self.curation_interval) is int
            and self.curation_interval == CURATION_INTERVAL
            and self.left_pack_source_arm == LEFT_PACK_SOURCE_ARM
        )
        if not exact:
            raise ValueError("the calibration-v2 development protocol is frozen")

    @property
    def total_steps(self) -> int:
        return sum(self.phase_lengths)

    def control_life_protocol(self) -> control_life.CompositionalControlLifeProtocol:
        return control_life.CompositionalControlLifeProtocol(
            phase_lengths=self.phase_lengths,
            epsilon=self.epsilon,
            entry_window=self.entry_window,
            tail_window=self.tail_window,
        )

    def to_config(self) -> dict[str, object]:
        return {
            "schema": self.schema_version,
            "namespace": self.namespace,
            "namespace_sha256": self.namespace_sha256,
            "root_derivation": "first four SHA-256 bytes, unsigned big-endian",
            "development_root": self.development_root,
            "development_root_hex": DEVELOPMENT_ROOT_HEX,
            "new_development_root_issued_once": True,
            "root_and_schedule_consumed_by_source_declaration": True,
            "consumed_regardless_of_panel_success_or_failure": True,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "evidence_authorized": False,
            "output_writer_available": False,
            "phase_order": list(self.phase_order),
            "phase_lengths": list(self.phase_lengths),
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "entry_window": self.entry_window,
            "tail_window": self.tail_window,
            "curation_interval": self.curation_interval,
            "phase_boundaries": list(PHASE_BOUNDARIES),
            "curation_opportunities_per_phase": list(
                CURATION_OPPORTUNITIES_PER_PHASE
            ),
            "left_pack_source_arm": self.left_pack_source_arm,
            "raw_dim": control_life.RAW_DIM,
            "active_slots": control_life.ACTIVE_SLOTS,
            "candidate_slots": control_life.CANDIDATE_SLOTS,
            "action_heads": control_life.ACTION_HEADS,
            "allocated_max_depth": control_life.ALLOCATED_MAX_DEPTH,
            "learner_observation_fields": ["raw_rademacher_values"],
            "learner_feedback_fields": ["selected_action_reward"],
            "evaluator_only_fields": [
                "phase_name",
                "phase_boundary",
                "target_expression",
                "counterfactual_action_reward",
            ],
            "resets_allowed": False,
            "search_performed": False,
            "winner_selection_allowed": False,
            "rerun_allowed": False,
            "tuning_allowed": False,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> CompositionalFutureUtilityCalibrationV2Protocol:
        canonical = cls()
        if set(payload) != set(canonical.to_config()):
            raise ValueError("protocol fields do not match the frozen schema")
        if dict(payload) != canonical.to_config():
            raise ValueError("protocol payload does not match the frozen declaration")
        return canonical


@dataclasses.dataclass(frozen=True, slots=True)
class BoundSourceArrays:
    """The exact pinned exogenous stream and typed learner-genesis key."""

    key_manifest: dict[str, list[int]]
    observations: Array
    phase_indices: Array
    exploration_mask: Array
    random_actions: Array
    learner_key: Array
    stream_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalFutureUtilityCalibrationV2Validation:
    """Fail-closed validation result for an in-memory report."""

    valid: bool
    errors: tuple[str, ...]


class _ProcessAttemptLatch:
    """Run one capability-bound builder once, sealing success or failure."""

    def __init__(self, builder: Callable[[object], str]) -> None:
        if not callable(builder):
            raise TypeError("process-attempt builder must be callable")
        self._builder = builder
        self._lock = threading.Lock()
        self._attempted = False
        self._active_capability: object | None = None
        self._value: str | None = None
        self._failure: BaseException | None = None

    def get(self) -> str:
        """Return the sole value; concurrent callers serialize behind its attempt."""

        with self._lock:
            if self._attempted:
                if self._failure is not None:
                    raise RuntimeError(
                        "the process-local calibration-v2 panel is sealed after failure"
                    ) from self._failure
                if self._value is None:
                    raise RuntimeError("the process-local attempt latch is internally invalid")
                return self._value
            self._attempted = True
            capability = object()
            self._active_capability = capability
            try:
                value = self._builder(capability)
                if type(value) is not str:
                    raise TypeError("process-attempt builder must return an exact string")
            except BaseException as error:
                self._failure = error
                raise
            finally:
                self._active_capability = None
            self._value = value
            return value

    def authorizes(self, capability: object) -> bool:
        """Return whether ``capability`` belongs to the active build attempt."""

        return self._active_capability is capability

    def completed_value(self) -> str | None:
        """Return a completed success without starting or waiting for an attempt.

        A concurrent build owns ``_lock`` for its full duration.  Non-blocking
        acquisition therefore makes validation fail closed while an attempt is
        in progress instead of turning validation into an implicit wait/run path.
        Failed and never-attempted latches both have no completed value.
        """

        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._value
        finally:
            self._lock.release()


def _source_arm() -> control_life.CompositionalControlLifeArm:
    matches = tuple(
        arm for arm in control_life.CONTROL_LIFE_ARMS if arm.name == LEFT_PACK_SOURCE_ARM
    )
    if len(matches) != 1:
        raise RuntimeError("historical left-pack source arm is not uniquely declared")
    return matches[0]


def _arm_learner_config(name: str) -> dict[str, Any]:
    """Return one exact arm on the corrected common base."""

    if type(name) is not str or name not in _ARM_PARAMETERS:
        raise ValueError("unknown predeclared calibration-v2 arm")
    mix, decay, normalization = _ARM_PARAMETERS[name]
    config = control_life.learner_config_for_arm(LEFT_PACK_SOURCE_ARM)
    config.update(
        {
            "candidate_scoring_mode": "legacy",
            "candidate_novelty_admission_bonus": 0.0,
            "future_utility_trace_mode": "contribution",
            "future_utility_mix": mix,
            "future_utility_trace_decay": decay,
            "future_utility_normalization": normalization,
            "future_utility_normalization_decay": NORMALIZATION_DECAY,
            "future_utility_rare_task_power": 0.0,
        }
    )
    return config


def _build_arm_learner(name: str) -> CompositionalFeatureLearner:
    return CompositionalFeatureLearner.from_config(_arm_learner_config(name))


def _key_manifest_record() -> dict[str, list[int]]:
    return {name: list(words) for name, words in KEY_MANIFEST.items()}


def _source_arrays_bound(
    protocol: CompositionalFutureUtilityCalibrationV2Protocol,
) -> BoundSourceArrays:
    """Generate then fail closed against the pre-run key and stream pins."""

    if type(protocol) is not CompositionalFutureUtilityCalibrationV2Protocol:
        raise TypeError("protocol must be the exact calibration-v2 protocol")
    (
        key_manifest,
        observations,
        phase_indices,
        exploration_mask,
        random_actions,
        stream_sha256,
    ) = control_life._stream_arrays(
        protocol.control_life_protocol(),
        protocol.development_root,
    )
    if key_manifest != _key_manifest_record():
        raise RuntimeError("development key manifest does not match its frozen pin")
    if stream_sha256 != STREAM_SHA256:
        raise RuntimeError("development stream does not match its frozen digest")
    learner_key = jr.wrap_key_data(
        jnp.asarray(key_manifest["learner_genesis"], dtype=jnp.uint32),
        impl="threefry2x32",
    )
    return BoundSourceArrays(
        key_manifest=key_manifest,
        observations=observations,
        phase_indices=phase_indices,
        exploration_mask=exploration_mask,
        random_actions=random_actions,
        learner_key=learner_key,
        stream_sha256=stream_sha256,
    )


def logical_work_per_arm(
    protocol: CompositionalFutureUtilityCalibrationV2Protocol,
) -> dict[str, object]:
    """Declare the fixed shared-base logical work for all five arms."""

    if type(protocol) is not CompositionalFutureUtilityCalibrationV2Protocol:
        raise TypeError("protocol must be the exact calibration-v2 protocol")
    steps = protocol.total_steps
    active = control_life.ACTIVE_SLOTS
    candidates = control_life.CANDIDATE_SLOTS
    heads = control_life.ACTION_HEADS
    return {
        "learner_updates": steps,
        "curation_update_opportunities": steps // protocol.curation_interval,
        "behavior_active_feature_value_cells": steps * active,
        "learner_update_active_feature_value_cells": steps * active,
        "total_active_feature_value_cells": steps * active * 2,
        "learner_update_candidate_feature_value_cells": steps * candidates,
        "evaluator_full_q_dot_products": steps,
        "evaluator_raw_q_dot_products": steps,
        "learner_prediction_q_dot_products": steps,
        "full_and_raw_q_dot_products": steps * 2,
        "total_q_dot_products": steps * 3,
        "total_q_head_scalar_outputs": steps * heads * 3,
        "ranking_diagnostic_calls": steps + 1,
        "active_future_reduction_cells": steps * heads * active,
        "candidate_future_reduction_cells": steps * heads * candidates,
        "future_contribution_trace_cells": steps * heads * (active + candidates),
        "future_feature_energy_trace_cells": steps * (active + candidates),
        "persistent_candidate_active_correlation_cells": active * candidates,
        "candidate_active_correlation_statistical_accumulation_cells": 0,
        "candidate_active_correlation_reset_mask_cells": steps * active * candidates,
        "ranking_candidate_active_correlation_cells": (steps + 1) * active * candidates,
        "persistent_state_nbytes": (
            control_life.compositional_control_state_nbytes_formula(
                active_slots=active,
                candidate_slots=candidates,
                action_heads=heads,
            )
        ),
        "persistent_search_archive_entries": 0,
        "keys_stream_shapes_and_update_opportunities_matched": True,
        "behavioral_experience_matching_claimed": False,
        "compiled_flop_equivalence_claimed": False,
    }


def _intervention_work_for_arm(
    protocol: CompositionalFutureUtilityCalibrationV2Protocol,
    name: str,
) -> dict[str, int]:
    """Count logical cells whose execution is conditional on this intervention."""

    if type(protocol) is not CompositionalFutureUtilityCalibrationV2Protocol:
        raise TypeError("protocol must be the exact calibration-v2 protocol")
    if type(name) is not str or name not in _ARM_PARAMETERS:
        raise ValueError("unknown predeclared calibration-v2 arm")
    mix, _decay, normalization = _ARM_PARAMETERS[name]
    steps = protocol.total_steps
    active_cells = steps * control_life.ACTIVE_SLOTS
    candidate_cells = steps * control_life.CANDIDATE_SLOTS
    normalized = normalization == "uncertainty_age"
    return {
        "utility_mixture_cells": 0 if mix == 0.0 else active_cells + candidate_cells,
        "active_second_moment_cells": active_cells if normalized else 0,
        "candidate_second_moment_cells": candidate_cells if normalized else 0,
        "active_age_debias_cells": active_cells if normalized else 0,
        "candidate_age_debias_cells": candidate_cells if normalized else 0,
        "active_uncertainty_normalization_cells": active_cells if normalized else 0,
        "candidate_uncertainty_normalization_cells": (
            candidate_cells if normalized else 0
        ),
    }


def _work_resource_contract(
    protocol: CompositionalFutureUtilityCalibrationV2Protocol,
) -> dict[str, object]:
    """Separate matched shared-base work from deliberately unequal interventions."""

    if type(protocol) is not CompositionalFutureUtilityCalibrationV2Protocol:
        raise TypeError("protocol must be the exact calibration-v2 protocol")
    shared_base = logical_work_per_arm(protocol)
    intervention_specific = {
        name: _intervention_work_for_arm(protocol, name) for name in ARM_NAMES
    }
    intervention_equal = len(
        {_canonical_json(value) for value in intervention_specific.values()}
    ) == 1
    if intervention_equal:
        raise RuntimeError("calibration interventions unexpectedly have equal named work")
    return {
        "selected_arm_count": len(ARM_NAMES),
        "per_arm_shared_base": shared_base,
        "intervention_specific_per_arm": intervention_specific,
        "panel_learner_updates": protocol.total_steps * len(ARM_NAMES),
        "panel_curation_update_opportunities": (
            protocol.total_steps // protocol.curation_interval
        )
        * len(ARM_NAMES),
        "aggregate_arm_state_byte_equivalent": cast(
            int, shared_base["persistent_state_nbytes"]
        )
        * len(ARM_NAMES),
        "aggregate_arm_state_byte_equivalent_is_peak_memory": False,
        "accounting_scope": RESOURCE_ACCOUNTING_SCOPE,
        "shared_base_logical_work_matched": True,
        "stream_shapes_and_update_opportunities_matched": True,
        "intervention_specific_logical_work_matched": False,
        "total_named_logical_work_equivalence_claimed": False,
        "behavior_dependent_branch_work_equivalence_claimed": False,
        "persistent_shapes_matched": True,
        "source_array_bytes_included": False,
        "full_scan_telemetry_bytes_included": False,
        "compiler_workspace_bytes_included": False,
        "compiled_flop_equivalence_claimed": False,
        "behavioral_experience_matching_claimed": False,
    }


def _selected_source_manifest() -> dict[str, str]:
    files = {
        "evaluation_module_sha256": _sha256_file(Path(__file__).resolve()),
        "control_life_v1_sha256": _sha256_file(Path(control_life.__file__).resolve()),
        "compositional_core_sha256": _sha256_file(
            Path(compositional_core.__file__).resolve()
        ),
        "future_utility_core_sha256": _sha256_file(
            Path(future_utility_core.__file__).resolve()
        ),
        "birth_identity_scrub_sha256": _sha256_file(
            Path(birth_identity_scrub.__file__).resolve()
        ),
        "lifecycle_state_size_sha256": _sha256_file(
            Path(lifecycle_scrub.__file__).resolve()
        ),
    }
    return {**files, "manifest_sha256": _json_sha256(files)}


def _runtime_identity() -> dict[str, object]:
    devices = jax.devices()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax": jax.__version__,
        "jaxlib": package_version("jaxlib"),
        "numpy": np.__version__,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_device_count": len(devices),
        "jax_device_kinds": [device.device_kind for device in devices],
    }


def _trace_record(state: CompositionalFeatureState) -> dict[str, object]:
    fields: dict[str, object] = {}
    arrays: list[Array] = []
    for name in _FUTURE_TRACE_FIELDS:
        value = cast(Array, getattr(state, name))
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"future-utility trace {name} is not finite")
        arrays.append(value)
        fields[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": control_life._array_tree_sha256(value),
            "nonzero_count": int(np.count_nonzero(array)),
            "l1_norm": float(np.sum(np.abs(array), dtype=np.float64)),
            "l2_norm": float(np.linalg.norm(array.astype(np.float64).ravel())),
            "max_abs": float(np.max(np.abs(array), initial=0.0)),
        }
    return {
        "field_order": list(_FUTURE_TRACE_FIELDS),
        "combined_sha256": control_life._array_tree_sha256(tuple(arrays)),
        "fields": fields,
    }


def _arm_definition(name: str) -> dict[str, object]:
    mix, decay, normalization = _ARM_PARAMETERS[name]
    return {
        "name": name,
        "role": _ARM_ROLES[name],
        "future_utility_mix": mix,
        "future_utility_trace_decay": decay,
        "future_utility_trace_decay_f32_bits": _f32_hex(decay),
        "future_utility_trace_mode": "contribution",
        "future_utility_normalization": normalization,
        "future_utility_normalization_decay": NORMALIZATION_DECAY,
        "future_utility_rare_task_power": 0.0,
        "candidate_scoring_mode": "legacy",
        "candidate_novelty_admission_bonus": 0.0,
    }


def _differing_fields(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[str, ...]:
    if tuple(left) != tuple(right):
        raise RuntimeError("arm learner-config schemas differ")
    return tuple(
        field
        for field in left
        if _canonical_json(left[field]) != _canonical_json(right[field])
    )


def _arm_configuration_audit(
    configs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Validate exact interventions, common correction, and isolated contrasts."""

    if tuple(configs) != ARM_NAMES:
        raise RuntimeError("calibration-v2 arm configs are not in frozen order")
    first_fields = tuple(configs[ARM_NAMES[0]])
    if any(tuple(configs[name]) != first_fields for name in ARM_NAMES[1:]):
        raise RuntimeError("calibration-v2 learner-config schemas differ")
    varying: dict[str, dict[str, object]] = {}
    for field in first_fields:
        values = {name: cast(object, configs[name][field]) for name in ARM_NAMES}
        if len({_canonical_json(value) for value in values.values()}) > 1:
            varying[field] = values
    if tuple(varying) != _EXPECTED_INTERVENTION_FIELDS:
        raise RuntimeError("arms differ outside the exact three-field intervention")

    historical = control_life.learner_config_for_arm(LEFT_PACK_SOURCE_ARM)
    if tuple(historical) != first_fields:
        raise RuntimeError("corrected and historical config schemas differ")
    historical_expected: dict[str, object] = {
        "candidate_scoring_mode": "energy_novelty",
        "candidate_novelty_admission_bonus": 1.0,
        "future_utility_trace_mode": "marginal",
        "future_utility_rare_task_power": 0.0,
    }
    corrected_expected: dict[str, object] = {
        "candidate_scoring_mode": "legacy",
        "candidate_novelty_admission_bonus": 0.0,
        "future_utility_trace_mode": "contribution",
        "future_utility_rare_task_power": 0.0,
    }
    common_departures: dict[str, dict[str, object]] = {}
    for field in _COMMON_DEPARTURE_FIELDS:
        if historical[field] != historical_expected[field]:
            raise RuntimeError(f"historical common-base field {field} drifted")
        corrected = configs[ARM_NAMES[0]][field]
        if corrected != corrected_expected[field] or any(
            configs[name][field] != corrected for name in ARM_NAMES
        ):
            raise RuntimeError(f"corrected common-base field {field} is not paired")
        common_departures[field] = {
            "historical_left_pack": historical[field],
            "corrected_common_base": corrected,
            "is_value_departure": historical[field] != corrected,
        }

    declared_fields = {*_COMMON_DEPARTURE_FIELDS, *_EXPECTED_INTERVENTION_FIELDS}
    for name in ARM_NAMES:
        undeclared = tuple(
            field
            for field in first_fields
            if field not in declared_fields
            and _canonical_json(configs[name][field])
            != _canonical_json(historical[field])
        )
        if undeclared:
            raise RuntimeError(f"arm {name} changes undeclared fields: {undeclared}")
        expected = _ARM_PARAMETERS[name]
        actual = (
            configs[name]["future_utility_mix"],
            configs[name]["future_utility_trace_decay"],
            configs[name]["future_utility_normalization"],
        )
        if _canonical_json(actual) != _canonical_json(expected):
            raise RuntimeError(f"arm {name} does not match its frozen tuple")
        if (
            configs[name]["future_utility_normalization_decay"]
            != NORMALIZATION_DECAY
            or configs[name]["future_utility_rare_task_power"] != 0.0
        ):
            raise RuntimeError("normalization decay or rare-task power drifted")

    isolated_contrasts = {
        "current_to_future": _differing_fields(configs[ARM_NAMES[0]], configs[ARM_NAMES[1]]),
        "calibrated_to_future": _differing_fields(
            configs[ARM_NAMES[2]], configs[ARM_NAMES[1]]
        ),
        "normalized_to_future": _differing_fields(
            configs[ARM_NAMES[3]], configs[ARM_NAMES[1]]
        ),
        "horizon_to_normalized": _differing_fields(
            configs[ARM_NAMES[4]], configs[ARM_NAMES[3]]
        ),
    }
    expected_contrasts = {
        "current_to_future": ("future_utility_mix",),
        "calibrated_to_future": ("future_utility_mix",),
        "normalized_to_future": ("future_utility_normalization",),
        "horizon_to_normalized": ("future_utility_trace_decay",),
    }
    if isolated_contrasts != expected_contrasts:
        raise RuntimeError("the four calibration contrasts are not isolated")
    if not _source_arm().composed_readout_enabled:
        raise RuntimeError("historical left-pack composed readout is disabled")
    return varying, common_departures


def _decay_zero_formula_witness() -> dict[str, object]:
    """Bind the exact sign-degenerate decay-zero signal to the production core."""

    step_size = 0.01
    active_count = 1.0
    task_head_count = 2.0
    errors = (-2.0, -1.0, 0.5, 1.0, 2.0)
    feature_values = jnp.asarray((-1.0, 1.0), dtype=jnp.float32)
    active_mask = jnp.asarray((True, False), dtype=jnp.bool_)
    rows: list[dict[str, object]] = []
    for error in errors:
        error_values = jnp.asarray((error, 0.0), dtype=jnp.float32)
        reductions, new_contribution, new_energy = (
            future_utility_core.contribution_trace_output_loss_reduction(
                error_values,
                feature_values,
                active_mask,
                step_size,
                active_count,
                jnp.zeros((2, 2), dtype=jnp.float32),
                jnp.zeros((2,), dtype=jnp.float32),
                0.0,
            )
        )
        one_step = future_utility_core.one_step_output_loss_reduction(
            error_values,
            feature_values,
            active_mask,
            step_size,
            active_count,
        )
        reductions_host = np.asarray(reductions, dtype=np.float32)
        one_step_host = np.asarray(one_step, dtype=np.float32)
        contribution_host = np.asarray(new_contribution, dtype=np.float32)
        energy_host = np.asarray(new_energy, dtype=np.float32)
        expected_contribution = np.asarray(
            ((-error, error), (0.0, 0.0)), dtype=np.float32
        )
        if (
            not np.array_equal(reductions_host, one_step_host)
            or not np.array_equal(contribution_host, expected_contribution)
            or not np.array_equal(energy_host, np.ones((2,), dtype=np.float32))
        ):
            raise RuntimeError("production decay-zero trace does not reduce to one-step")
        per_active_values = [
            float(reductions_host[0, 0]),
            float(reductions_host[0, 1]),
        ]
        slot_values_array = np.mean(reductions_host, axis=0, dtype=np.float32)
        slot_values = [float(slot_values_array[0]), float(slot_values_array[1])]
        expected_per_active = (
            step_size * error * error / active_count
            - 0.5 * (step_size * error / active_count) ** 2
        )
        expected_slot = expected_per_active / task_head_count
        if (
            _f32_uint_bits(per_active_values[0])
            != _f32_uint_bits(per_active_values[1])
            or _f32_uint_bits(slot_values[0]) != _f32_uint_bits(slot_values[1])
            or not np.isclose(per_active_values[0], expected_per_active, rtol=1e-6)
            or not np.isclose(slot_values[0], expected_slot, rtol=1e-6)
        ):
            raise RuntimeError("decay-zero sign-degeneracy witness failed")
        rows.append(
            {
                "error": error,
                "per_active_head_feature_minus_one": per_active_values[0],
                "per_active_head_feature_plus_one": per_active_values[1],
                "mean_slot_feature_minus_one": slot_values[0],
                "mean_slot_feature_plus_one": slot_values[1],
                "per_active_head_f32_bits": _f32_uint_bits(per_active_values[0]),
                "mean_slot_f32_bits": _f32_uint_bits(slot_values[0]),
                "per_active_head_formula": "0.00995 * error**2",
                "mean_slot_formula": "0.004975 * error**2",
                "exactly_equal": True,
            }
        )
    return {
        "trace_decay": 0.0,
        "active_target_count": 1,
        "task_head_count": 2,
        "features": [-1.0, 1.0],
        "rows": rows,
        "sign_can_change_decay_zero_ranking": False,
        "production_core_bound": True,
        "production_core_functions": [
            "contribution_trace_output_loss_reduction",
            "one_step_output_loss_reduction",
        ],
        "scope": "production-core decay-zero witness only; no panel executed",
    }


def _static_preflight() -> dict[str, object]:
    """Prove configuration reachability and exact contrasts without running a scan."""

    if hashlib.sha256(PROTOCOL_NAMESPACE.encode("ascii")).hexdigest() != (
        PROTOCOL_NAMESPACE_SHA256
    ):
        raise RuntimeError("protocol namespace digest drifted")
    if int(PROTOCOL_NAMESPACE_SHA256[:8], 16) != DEVELOPMENT_ROOT:
        raise RuntimeError("protocol root derivation drifted")
    if sum(PHASE_LENGTHS) != TOTAL_STEPS:
        raise RuntimeError("phase lengths no longer total 8,998")
    if _f32_hex(LONG_TRACE_DECAY) != LONG_TRACE_DECAY_F32_BITS:
        raise RuntimeError("long trace decay does not have its frozen float32 bits")

    configs = {name: _arm_learner_config(name) for name in ARM_NAMES}
    varying, common_departures = _arm_configuration_audit(configs)
    positive_banks = all(
        type(configs[name]["n_features"]) is int
        and configs[name]["n_features"] > 0
        and type(configs[name]["candidate_count"]) is int
        and configs[name]["candidate_count"] > 0
        for name in ARM_NAMES
    )
    overwrite_disabled = all(
        configs[name]["candidate_scoring_mode"] == "legacy" for name in ARM_NAMES
    )
    contribution_enabled = all(
        configs[name]["future_utility_trace_mode"] == "contribution"
        for name in ARM_NAMES
    )
    mix_reachable = (
        positive_banks
        and overwrite_disabled
        and configs[ARM_NAMES[0]]["future_utility_mix"] == 0.0
        and configs[ARM_NAMES[1]]["future_utility_mix"] == 1.0
        and configs[ARM_NAMES[2]]["future_utility_mix"] == 0.5
    )
    normalization_reachable = (
        positive_banks
        and overwrite_disabled
        and configs[ARM_NAMES[1]]["future_utility_normalization"] == "none"
        and configs[ARM_NAMES[3]]["future_utility_normalization"]
        == "uncertainty_age"
    )
    horizon_reachable = (
        contribution_enabled
        and configs[ARM_NAMES[3]]["future_utility_trace_decay"] == 0.95
        and _f32_hex(configs[ARM_NAMES[4]]["future_utility_trace_decay"])
        == LONG_TRACE_DECAY_F32_BITS
    )
    if not (mix_reachable and normalization_reachable and horizon_reachable):
        raise RuntimeError("one or more predeclared causal interventions is unreachable")
    return {
        "static_audit_performed": True,
        "panel_executed_during_preflight": False,
        "namespace_root_derivation_valid": True,
        "long_trace_decay_f32_bits": LONG_TRACE_DECAY_F32_BITS,
        "positive_active_and_candidate_banks": positive_banks,
        "mixed_utility_overwrite_disabled": overwrite_disabled,
        "contribution_trace_path_enabled": contribution_enabled,
        "mix_intervention_reaches_ranking": mix_reachable,
        "mix_intervention_reaches_active_ranking": mix_reachable,
        "mix_intervention_reaches_candidate_ranking": mix_reachable,
        "normalization_intervention_reaches_ranking": normalization_reachable,
        "normalization_reaches_active_second_moment_and_ranking": (
            normalization_reachable
        ),
        "normalization_reaches_candidate_second_moment_and_ranking": (
            normalization_reachable
        ),
        "horizon_intervention_reaches_contribution_traces": horizon_reachable,
        "horizon_reaches_active_contribution_trace": horizon_reachable,
        "horizon_reaches_candidate_contribution_trace": horizon_reachable,
        "only_varying_config_fields": varying,
        "common_base_corrections": common_departures,
        "decay_zero_formula_witness": _decay_zero_formula_witness(),
    }


def _f32_uint_bits(value: float) -> int:
    return int(struct.unpack(">I", struct.pack(">f", value))[0])


def _bank_descending_rank(
    mask: object,
    scores: object,
    *,
    start: int,
    stop: int,
    matching_slots_field: str,
) -> dict[str, object]:
    """Return a tie-aware f32 rank for matching slots in one exact bank slice."""

    selected_all = np.asarray(mask, dtype=np.bool_)
    values_all = np.asarray(scores, dtype=np.float32)
    if (
        selected_all.ndim != 1
        or values_all.ndim != 1
        or selected_all.shape != values_all.shape
        or start < 0
        or stop <= start
        or stop > selected_all.shape[0]
    ):
        raise RuntimeError("rank mask, scores, or bank slice is invalid")
    selected = selected_all[start:stop]
    values = values_all[start:stop]
    finite = np.isfinite(values)
    matching = selected & finite
    matching_slots = [start + int(index) for index in np.flatnonzero(selected)]
    if not np.any(matching):
        return {
            "present": False,
            matching_slots_field: matching_slots,
            "matching_score_f32_bits": [],
            "best_score_f32_bits": None,
            "descending_rank_interval": None,
        }
    best_value = np.max(values[matching])
    strictly_greater = int(np.count_nonzero(finite & (values > best_value)))
    equal = int(np.count_nonzero(finite & (values == best_value)))
    return {
        "present": True,
        matching_slots_field: matching_slots,
        "matching_score_f32_bits": [
            _f32_uint_bits(float(value)) for value in values[matching]
        ],
        "best_score_f32_bits": _f32_uint_bits(float(best_value)),
        "descending_rank_interval": [
            1 + strictly_greater,
            strictly_greater + equal,
        ],
    }


def _descending_rank(mask: object, scores: object) -> dict[str, object]:
    """Rank the best matching signature among composed active-bank slots only."""

    return _bank_descending_rank(
        mask,
        scores,
        start=control_life.RAW_DIM,
        stop=control_life.ACTIVE_SLOTS,
        matching_slots_field="matching_composed_slots",
    )


def _candidate_descending_rank(mask: object, scores: object) -> dict[str, object]:
    """Rank the best matching signature among all candidate-bank slots."""

    return _bank_descending_rank(
        mask,
        scores,
        start=0,
        stop=control_life.CANDIDATE_SLOTS,
        matching_slots_field="matching_candidate_slots",
    )


def _pre_recurrence_records(events: object) -> list[dict[str, object]]:
    scan = cast(Any, events)
    starts: list[int] = []
    cursor = 0
    for length in PHASE_LENGTHS:
        starts.append(cursor)
        cursor += length
    seen: dict[str, int] = {}
    records: list[dict[str, object]] = []
    post_slots = np.asarray(scan.post_active_signature_slots, dtype=np.bool_)
    post_candidate_slots = np.asarray(
        scan.post_candidate_signature_slots, dtype=np.bool_
    )
    direct = np.asarray(scan.direct_active_scores, dtype=np.float32)
    backed = np.asarray(scan.backed_active_scores, dtype=np.float32)
    candidate_direct = np.asarray(scan.direct_candidate_scores, dtype=np.float32)
    candidate_augmented = np.asarray(
        scan.augmented_candidate_scores, dtype=np.float32
    )
    for phase_index, (name, start) in enumerate(zip(PHASE_ORDER, starts, strict=True)):
        occurrence = seen.get(name, 0) + 1
        seen[name] = occurrence
        if name not in _TARGET_NAMES or occurrence == 1:
            continue
        event_index = start - 1
        signature_index = control_life.SIGNATURE_NAMES.index(name)
        signature_mask = post_slots[event_index, :, signature_index]
        candidate_mask = post_candidate_slots[event_index, :, signature_index]
        records.append(
            {
                "target": name,
                "occurrence": occurrence,
                "recurrence_phase_index": phase_index,
                "pre_recurrence_post_step": start,
                "active_present": bool(np.any(signature_mask)),
                "candidate_present": bool(np.any(candidate_mask)),
                "active_slot_count": int(np.count_nonzero(signature_mask)),
                "candidate_slot_count": int(np.count_nonzero(candidate_mask)),
                "matching_active_slots": [
                    int(slot) for slot in np.flatnonzero(signature_mask)
                ],
                "matching_candidate_slots": [
                    int(slot) for slot in np.flatnonzero(candidate_mask)
                ],
                "direct_rank": _descending_rank(signature_mask, direct[event_index]),
                "ancestor_backed_rank": _descending_rank(
                    signature_mask, backed[event_index]
                ),
                "candidate_direct_rank": _candidate_descending_rank(
                    candidate_mask, candidate_direct[event_index]
                ),
                "candidate_augmented_rank": _candidate_descending_rank(
                    candidate_mask, candidate_augmented[event_index]
                ),
            }
        )
    return records


def _primary_endpoint_record(
    protocol: CompositionalFutureUtilityCalibrationV2Protocol,
    events: object,
    trajectories: Mapping[str, Mapping[str, object]],
    curation_totals: Mapping[str, int],
    curation_audit: Mapping[str, object],
) -> dict[str, object]:
    scan = cast(Any, events)
    trace = scan.curation_trace
    margin_pass_mask = np.asarray(trace.decision_margin_passed, dtype=np.bool_)
    all_step_margin_passes = int(np.count_nonzero(margin_pass_mask))
    due_mask = (
        np.arange(1, protocol.total_steps + 1, dtype=np.int64)
        % protocol.curation_interval
        == 0
    )
    margin_passes_due = int(
        np.count_nonzero(margin_pass_mask & due_mask)
    )
    margin_passes_off_opportunity = all_step_margin_passes - margin_passes_due
    candidate_destination_margin_pairs = int(
        np.count_nonzero(
            np.asarray(trace.decision_candidate_margin_eligible, dtype=np.bool_)
            & due_mask[:, None, None]
        )
    )
    promotion_events = int(np.count_nonzero(np.asarray(trace.promotion_applied)))
    if promotion_events != curation_totals["promotion"]:
        raise RuntimeError("promotion event and curation counts disagree")
    ordinary_refresh_slots = int(
        np.count_nonzero(np.asarray(trace.ordinary_candidate_refresh_mask))
    )
    post_promotion_refresh_slots = int(
        np.count_nonzero(np.asarray(trace.post_promotion_candidate_refresh_mask))
    )
    total_refresh_slots = int(np.count_nonzero(np.asarray(trace.candidate_refresh_mask)))
    if (
        ordinary_refresh_slots != curation_totals["ordinary_candidate_refresh"]
        or post_promotion_refresh_slots
        != curation_totals["post_promotion_candidate_refresh"]
        or total_refresh_slots != curation_totals["candidate_refresh"]
    ):
        raise RuntimeError("candidate refresh masks and curation counts disagree")

    transitions = cast(
        Mapping[str, Mapping[str, object]],
        curation_audit["active_signature_transition_causes"],
    )
    cascade_losses: dict[str, object] = {}
    target_lifecycle: dict[str, Mapping[str, object]] = {}
    counts = np.asarray(scan.active_signature_counts, dtype=np.int64)
    candidate_counts = np.asarray(scan.candidate_signature_counts, dtype=np.int64)
    occupancy: dict[str, object] = {}
    outcome_counts = cast(
        Mapping[str, Mapping[str, int]], curation_audit["target_outcome_counts"]
    )
    for name in _TARGET_NAMES:
        transition = transitions[name]
        loss_causes = cast(Mapping[str, int], transition["loss_slot_cause_counts"])
        cascade_losses[name] = {
            "loss_episode_count": transition["loss_episode_count"],
            "root_replacement_lost_slot_count": loss_causes[
                "promotion_root_replacement"
            ],
            "cascade_dependency_refill_lost_slot_count": loss_causes[
                "cascade_dependency_refill"
            ],
            "all_changed_slots_accounted": transition["all_changed_slots_accounted"],
        }
        trajectory = trajectories[name]
        target_lifecycle[name] = {
            "direct_candidate_admission_count": outcome_counts[name].get(
                "admitted", 0
            ),
            "admission_episode_count": trajectory["acquisition_episode_count"],
            "loss_episode_count": trajectory["loss_episode_count"],
            "present_at_end": trajectory["present_at_end"],
            "structural_reacquisition_count": trajectory["structural_reacquisition_count"],
        }
        signature_index = control_life.SIGNATURE_NAMES.index(name)
        values = counts[:, signature_index]
        candidate_values = candidate_counts[:, signature_index]
        occupancy[name] = {
            "active_present_post_steps": int(np.count_nonzero(values > 0)),
            "active_presence_fraction": float(
                np.mean(values > 0, dtype=np.float64)
            ),
            "active_slot_step_cells": int(np.sum(values, dtype=np.int64)),
            "candidate_present_post_steps": int(
                np.count_nonzero(candidate_values > 0)
            ),
            "candidate_presence_fraction": float(
                np.mean(candidate_values > 0, dtype=np.float64)
            ),
            "candidate_slot_step_cells": int(
                np.sum(candidate_values, dtype=np.int64)
            ),
        }
    pre_recurrence = _pre_recurrence_records(scan)
    coexistence = control_life._active_target_coexistence_record(
        scan.active_signature_counts,
        start_post_step=0,
    )
    return {
        "endpoint_order": list(PRIMARY_ENDPOINTS),
        "margin_passes": {
            "selected_strict_margin_pass_count": margin_passes_due,
            "selected_strict_margin_all_step_diagnostic_count": (
                all_step_margin_passes
            ),
            "selected_strict_margin_off_opportunity_diagnostic_count": (
                margin_passes_off_opportunity
            ),
            "candidate_destination_strict_margin_pair_count": (
                candidate_destination_margin_pairs
            ),
            "due_curation_event_count": curation_audit["due_curation_event_count"],
        },
        "promotions": {"event_count": promotion_events},
        "cascade_refill_slot_count": curation_totals["cascade_refill"],
        "candidate_refreshes": {
            "decision_should_refresh_event_count": int(
                np.count_nonzero(np.asarray(trace.decision_should_refresh))
            ),
            "ordinary_refreshed_slot_count": ordinary_refresh_slots,
            "post_promotion_refreshed_slot_count": post_promotion_refresh_slots,
            "total_refreshed_slot_count": total_refresh_slots,
        },
        "cascade_losses": cascade_losses,
        "cascade_loss_definition": (
            "target-signature lost slots whose exact decision audit cause is "
            "cascade_dependency_refill"
        ),
        "target_admission_loss_end": target_lifecycle,
        "pre_recurrence_presence": [
            {
                "target": record["target"],
                "occurrence": record["occurrence"],
                "pre_recurrence_post_step": record["pre_recurrence_post_step"],
                "active_present": record["active_present"],
                "candidate_present": record["candidate_present"],
                "active_slot_count": record["active_slot_count"],
                "candidate_slot_count": record["candidate_slot_count"],
            }
            for record in pre_recurrence
        ],
        "a_retention": {
            "pre_recurrence_phase_indices": [2, 4, 6, 9],
            "pre_recurrence_presence": [
                record["active_present"]
                for record in pre_recurrence
                if record["target"] == "A"
            ],
            "present_at_end": target_lifecycle["A"]["present_at_end"],
        },
        "target_occupancy": {
            "post_update_state_count": protocol.total_steps,
            "per_target": occupancy,
            "coexistence": coexistence,
            "steps_by_distinct_active_target_count": coexistence[
                "steps_by_active_target_count"
            ],
            "maximum_distinct_active_target_count": coexistence[
                "maximum_active_target_count"
            ],
            "final_active_targets": coexistence["active_targets_at_end"],
        },
        "pre_recurrence_ranks": {
            "active_definition": (
                "best matching target slot among composed slots RAW_DIM:ACTIVE_SLOTS; "
                "tie-aware descending rank interval, with rank 1 highest"
            ),
            "candidate_definition": (
                "best matching target slot among all candidate slots; direct and "
                "novelty-augmented scores each use a tie-aware descending rank interval, "
                "with rank 1 highest"
            ),
            "records": pre_recurrence,
        },
        "identity_reacquisition_claimed": False,
    }


def _run_arm(
    protocol: CompositionalFutureUtilityCalibrationV2Protocol,
    source: BoundSourceArrays,
    name: str,
    *,
    _execution_capability: object,
) -> dict[str, object]:
    if not _FULL_REPORT_ATTEMPT.authorizes(_execution_capability):
        raise RuntimeError("arm scans require the active one-shot panel capability")
    learner = _build_arm_learner(name)
    state = cast(
        CompositionalFeatureState,
        learner.init(control_life.RAW_DIM, source.learner_key).replace(  # type: ignore[attr-defined]
            birth_timestamp=0.0,
            uptime_s=0.0,
        ),
    )
    initial_diagnostics = learner.ranking_diagnostics(state, control_life.RAW_DIM)
    if not bool(initial_diagnostics.contract_valid):
        raise RuntimeError("initial ranking contract is invalid")
    (
        initial_active_counts,
        initial_candidate_counts,
        initial_active_pair_counts,
        initial_candidate_pair_counts,
    ) = control_life._product_signature_counts(state)
    if bool(jnp.any(initial_active_counts)) or bool(jnp.any(initial_candidate_counts)):
        raise RuntimeError("a target or useful intermediate is prewired at genesis")

    expected_nbytes = cast(int, logical_work_per_arm(protocol)["persistent_state_nbytes"])
    initial_nbytes = lifecycle_scrub.persistent_compositional_state_nbytes(state)
    if initial_nbytes != expected_nbytes:
        raise RuntimeError("initial state violates the fixed byte formula")
    initial_state_sha256 = (
        birth_identity_scrub.generated_birth_identity_scrub_epoch_core_state_sha256(state)
    )
    initial_traces = _trace_record(state)
    final_state, device_events = control_life._run_compiled_scan(
        learner,
        state,
        True,
        source.observations,
        source.phase_indices,
        source.exploration_mask,
        source.random_actions,
    )
    final_state_sha256 = (
        birth_identity_scrub.generated_birth_identity_scrub_epoch_core_state_sha256(
            final_state
        )
    )
    final_traces = _trace_record(final_state)
    events = jax.device_get(device_events)
    final_nbytes = lifecycle_scrub.persistent_compositional_state_nbytes(final_state)
    if final_nbytes != expected_nbytes:
        raise RuntimeError("final state violates the fixed byte formula")
    if tuple(int(value) for value in np.asarray(final_state.step_words)) != (
        0,
        protocol.total_steps,
    ):
        raise RuntimeError("final exact lifetime clock does not match 8,998 steps")

    initial_ranking = control_life._initial_ranking_record(state, initial_diagnostics)
    curation_totals_array = np.sum(
        np.asarray(events.curation_counts, dtype=np.int64), axis=0
    )
    curation_totals = {
        key: int(value)
        for key, value in zip(
            control_life.CURATION_COUNT_NAMES,
            curation_totals_array,
            strict=True,
        )
    }
    active_trajectories: dict[str, Mapping[str, object]] = {}
    candidate_trajectories: dict[str, Mapping[str, object]] = {}
    for index, signature_name in enumerate(control_life.SIGNATURE_NAMES):
        active_trajectories[signature_name] = control_life._structural_trajectory(
            int(initial_active_counts[index]),
            events.active_signature_counts[:, index],
        )
        candidate_trajectories[signature_name] = control_life._structural_trajectory(
            int(initial_candidate_counts[index]),
            events.candidate_signature_counts[:, index],
        )
    v1_protocol = protocol.control_life_protocol()
    curation_audit, audit_elements, audit_nbytes = control_life._curation_decision_audit(
        v1_protocol, events
    )
    if curation_audit["shared_p45_active_bank_loss_count"] != active_trajectories[
        "shared_p45"
    ]["loss_episode_count"]:
        raise RuntimeError("active shared-p45 loss audit does not close")
    if curation_audit["shared_p45_candidate_bank_loss_count"] != candidate_trajectories[
        "shared_p45"
    ]["loss_episode_count"]:
        raise RuntimeError("candidate shared-p45 loss audit does not close")
    transition_causes = cast(
        Mapping[str, Mapping[str, object]],
        curation_audit["active_signature_transition_causes"],
    )
    for signature_name in control_life.AUDITED_ADMISSION_SIGNATURE_NAMES:
        trajectory = active_trajectories[signature_name]
        initial_acquisition = int(bool(trajectory["initially_present"]))
        if (
            transition_causes[signature_name]["acquisition_episode_count"]
            != cast(int, trajectory["acquisition_episode_count"])
            - initial_acquisition
            or transition_causes[signature_name]["loss_episode_count"]
            != trajectory["loss_episode_count"]
            or transition_causes[signature_name]["all_changed_slots_accounted"]
            is not True
        ):
            raise RuntimeError(
                f"active {signature_name} transition-cause audit does not close"
            )

    primary = _primary_endpoint_record(
        protocol,
        events,
        active_trajectories,
        curation_totals,
        curation_audit,
    )
    lifetime = control_life._window_metrics(events, 0, protocol.total_steps)
    phase_metrics = control_life._phase_records(
        v1_protocol,
        initial_ranking,
        initial_active_counts,
        initial_candidate_counts,
        events,
    )
    learner_config = learner.to_config()
    return {
        "arm": name,
        "role": _ARM_ROLES[name],
        "arm_definition": _arm_definition(name),
        "learner_config": learner_config,
        "learner_config_sha256": _json_sha256(learner_config),
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "trace_sha256": control_life._array_tree_sha256(events),
        "future_utility_traces": {
            "initial": initial_traces,
            "final": final_traces,
            "advanced": initial_traces["combined_sha256"]
            != final_traces["combined_sha256"],
        },
        "initial_persistent_state_nbytes": initial_nbytes,
        "final_persistent_state_nbytes": final_nbytes,
        "expected_persistent_state_nbytes": expected_nbytes,
        "final_step_count_telemetry": int(final_state.step_count),
        "final_step_words_uint32": [
            int(value) for value in np.asarray(final_state.step_words)
        ],
        "final_replacement_phase": int(final_state.replacement_phase),
        "initial_state_finite": control_life._state_is_finite(state),
        "final_state_finite": control_life._state_is_finite(final_state),
        "all_lifetime_counters_valid": bool(
            np.all(np.asarray(events.lifetime_counter_valid))
        ),
        "all_lifetime_capacity_available": bool(
            np.all(np.asarray(events.lifetime_capacity_available))
        ),
        "all_ranking_contracts_valid": bool(
            np.all(np.asarray(events.ranking_contract_valid))
        ),
        "all_core_predictions_match_full_q": bool(
            np.all(np.asarray(events.core_prediction_matches_full_q))
        ),
        "primary_endpoints": primary,
        "secondary_reward_endpoints": {
            "endpoint_order": list(SECONDARY_ENDPOINTS),
            "lifetime_reward": lifetime,
            "phase_reward": phase_metrics,
        },
        "curation_totals": curation_totals,
        "active_structural_trajectories": active_trajectories,
        "candidate_structural_trajectories": candidate_trajectories,
        "curation_decision_audit": curation_audit,
        "curation_decision_audit_resources": {
            "events": cast(int, curation_audit["due_curation_event_count"]),
            "ephemeral_array_elements": audit_elements,
            "ephemeral_array_bytes": audit_nbytes,
            "report_json_bytes": cast(int, curation_audit["records_canonical_json_bytes"]),
        },
        "raw_pair_coverage": control_life._raw_pair_coverage_record(
            initial_active_pair_counts,
            initial_candidate_pair_counts,
            events.active_raw_pair_counts,
            events.candidate_raw_pair_counts,
        ),
        "raw_pair_reachability": control_life._raw_pair_reachability_record(
            _source_arm(),
            bool(
                jnp.any(
                    (state.depth >= 1)
                    & (state.depth + 1 <= _source_arm().effective_max_depth)
                )
            ),
            events,
            curation_totals["cascade_refill"],
        ),
        "initial_ranking": initial_ranking,
        "final_ranking": control_life._event_ranking_record(
            events, protocol.total_steps - 1
        ),
        "shared_base_work": logical_work_per_arm(protocol),
        "intervention_specific_work": _intervention_work_for_arm(protocol, name),
    }


def _arm_comparison(runs: list[dict[str, object]]) -> dict[str, object]:
    configs = {
        cast(str, run["arm"]): cast(dict[str, Any], run["learner_config"])
        for run in runs
    }
    varying, common_departures = _arm_configuration_audit(configs)
    shared_base_work_equal = all(
        run["shared_base_work"] == runs[0]["shared_base_work"] for run in runs
    )
    work_records = [
        cast(Mapping[str, object], run["shared_base_work"]) for run in runs
    ]
    update_opportunity_fields = (
        "learner_updates",
        "curation_update_opportunities",
        "keys_stream_shapes_and_update_opportunities_matched",
    )
    stream_update_opportunities_equal = all(
        tuple(record[field] for field in update_opportunity_fields)
        == tuple(work_records[0][field] for field in update_opportunity_fields)
        for record in work_records
    )
    intervention_work_equal = len(
        {
            _canonical_json(run["intervention_specific_work"])
            for run in runs
        }
    ) == 1
    if intervention_work_equal:
        raise RuntimeError("calibration intervention-specific work unexpectedly matches")
    return {
        "initial_states_equal": len({run["initial_state_sha256"] for run in runs}) == 1,
        "shared_base_logical_work_equal": shared_base_work_equal,
        "stream_shapes_and_update_opportunities_equal": stream_update_opportunities_equal,
        "intervention_specific_logical_work_equal": False,
        "total_named_logical_work_equivalence_claimed": False,
        "behavior_dependent_branch_work_equivalence_claimed": False,
        "persistent_state_shapes_equal": len(
            {run["expected_persistent_state_nbytes"] for run in runs}
        )
        == 1,
        "exogenous_stream_genesis_and_update_opportunities_paired": True,
        "behavioral_experience_matching_claimed": False,
        "behavioral_experience_note": (
            "actions, rewards, errors, and subsequent learner states may diverge causally "
            "after the shared genesis"
        ),
        "only_varying_config_fields": varying,
        "expected_varying_config_fields": list(_EXPECTED_INTERVENTION_FIELDS),
        "common_base_corrections": common_departures,
        "descriptive_primary_endpoints": {
            cast(str, run["arm"]): run["primary_endpoints"] for run in runs
        },
        "descriptive_secondary_rewards": {
            cast(str, run["arm"]): run["secondary_reward_endpoints"] for run in runs
        },
        "winner_selected": False,
        "threshold_applied": False,
        "rerun_or_tuning_authorized": False,
        "historical_result_reconstruction_claimed": False,
        "compiled_flop_equivalence_claimed": False,
    }


def _execution_attempts_remaining() -> int:
    """Return the operational budget without narrowing it to a literal."""

    return EXECUTION_ATTEMPTS_REMAINING


def _build_report(_execution_capability: object) -> dict[str, object]:
    if not _FULL_REPORT_ATTEMPT.authorizes(_execution_capability):
        raise RuntimeError("full report construction requires the active one-shot capability")
    if _execution_attempts_remaining() != 1:
        raise RuntimeError(
            "the calibration-v2 development root was consumed by its failed attempt"
        )
    protocol = CompositionalFutureUtilityCalibrationV2Protocol()
    source_pre = _selected_source_manifest()
    if source_pre != dict(_IMPORT_TIME_SELECTED_SOURCE_HASHES):
        raise RuntimeError("selected source files changed since module import")
    runtime_pre = _runtime_identity()
    static_preflight = _static_preflight()
    source = _source_arrays_bound(protocol)
    runs = [
        _run_arm(
            protocol,
            source,
            name,
            _execution_capability=_execution_capability,
        )
        for name in ARM_NAMES
    ]
    source_post = _selected_source_manifest()
    runtime_post = _runtime_identity()
    if (
        source_post != source_pre
        or source_post != dict(_IMPORT_TIME_SELECTED_SOURCE_HASHES)
    ):
        raise RuntimeError("selected source files changed during the full panel")
    if runtime_post != runtime_pre:
        raise RuntimeError("selected runtime identity changed during the full panel")
    if len({run["initial_state_sha256"] for run in runs}) != 1:
        raise RuntimeError("paired arms do not share the exact genesis state")
    if not all(
        run["shared_base_work"] == runs[0]["shared_base_work"] for run in runs
    ):
        raise RuntimeError("paired arms do not have identical shared-base logical work")

    body: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": STATUS,
        "assessment_status": ASSESSMENT_STATUS,
        "development_only": DEVELOPMENT_ONLY,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "evidence_authorized": EVIDENCE_AUTHORIZED,
        "output_writes_allowed": OUTPUT_WRITES_ALLOWED,
        "output_writer_available": False,
        "artifact_available": False,
        "artifact_bytes_written": 0,
        "interpretation": INTERPRETATION,
        "limitations": list(LIMITATIONS),
        "protocol": protocol.to_config(),
        "protocol_sha256": _json_sha256(protocol.to_config()),
        "source_manifest_import_snapshot": dict(_IMPORT_TIME_SELECTED_SOURCE_HASHES),
        "source_manifest_live_pre": source_pre,
        "source_manifest_live_post": source_post,
        "source_manifest_pre_post_import_equal": True,
        "source_manifest_scope": SOURCE_MANIFEST_SCOPE,
        "transitive_source_closure_claimed": False,
        "runtime_identity_pre": runtime_pre,
        "runtime_identity_post": runtime_post,
        "runtime_identity_pre_post_equal": True,
        "runtime_identity_scope": RUNTIME_IDENTITY_SCOPE,
        "environment_or_compiler_closure_claimed": False,
        "resource_accounting_scope": RESOURCE_ACCOUNTING_SCOPE,
        "development_root": protocol.development_root,
        "development_root_hex": DEVELOPMENT_ROOT_HEX,
        "root_role": "newly-issued-development-nonpromoting-one-shot",
        "root_and_schedule_consumed_by_source_declaration": True,
        "root_consumed_regardless_of_panel_success_or_failure": True,
        "process_local_latch_prevents_cross_process_replay": False,
        "key_manifest": source.key_manifest,
        "stream_sha256": source.stream_sha256,
        "key_manifest_and_stream_verified_before_first_arm": True,
        "static_preflight": static_preflight,
        "primary_endpoint_order": list(PRIMARY_ENDPOINTS),
        "secondary_endpoint_order": list(SECONDARY_ENDPOINTS),
        "arm_order": list(ARM_NAMES),
        "arm_definitions": [_arm_definition(name) for name in ARM_NAMES],
        "runs": runs,
        "arm_comparison": _arm_comparison(runs),
        "winner_or_default_selected": False,
        "threshold_defined_or_applied": False,
        "search_performed": False,
        "rerun_or_tuning_authorized": False,
        "identity_tracking": {
            "birth_ledger_integrated": False,
            "retained_identity_assessed": False,
            "reported_reacquisition_kind": "bank-level-algebraic-structural-only",
        },
        "work_resource_contract": _work_resource_contract(protocol),
    }
    return cast(dict[str, object], _json_clone({**body, "report_sha256": _json_sha256(body)}))


# This snapshot is captured at import, before any full-panel attempt can begin.
_IMPORT_TIME_SELECTED_SOURCE_HASHES: Final[Mapping[str, str]] = MappingProxyType(
    _selected_source_manifest()
)

_FULL_REPORT_ATTEMPT = _ProcessAttemptLatch(
    lambda capability: _canonical_json(_build_report(capability))
)


def _expected_report_json() -> str:
    """Return the only process-local full build, sealing success or failure."""

    return _FULL_REPORT_ATTEMPT.get()


def validate_compositional_future_utility_calibration_v2_report(
    report: Mapping[str, object],
) -> CompositionalFutureUtilityCalibrationV2Validation:
    """Validate only against an already completed cached reconstruction.

    Validation is intentionally incapable of starting or waiting for the
    one-shot panel.  Before successful completion it returns a closed result.
    """

    try:
        candidate = cast(dict[str, object], _json_clone(dict(report)))
    except (TypeError, ValueError) as error:
        return CompositionalFutureUtilityCalibrationV2Validation(
            False, (f"report is not canonical JSON: {error}",)
        )
    errors: list[str] = []
    completed_json = _FULL_REPORT_ATTEMPT.completed_value()
    if completed_json is None:
        errors.append("the one-shot panel has not completed successfully")
    else:
        expected = cast(dict[str, object], json.loads(completed_json))
        if candidate != expected:
            errors.append("report does not match the cached one-shot reconstruction")
    body = {key: value for key, value in candidate.items() if key != "report_sha256"}
    if candidate.get("report_sha256") != _json_sha256(body):
        errors.append("report_sha256 does not reconstruct")
    return CompositionalFutureUtilityCalibrationV2Validation(not errors, tuple(errors))


def run_compositional_future_utility_calibration_v2_development() -> dict[str, object]:
    """Execute the only full panel and return its validated in-memory report."""

    report = cast(dict[str, object], json.loads(_expected_report_json()))
    validation = validate_compositional_future_utility_calibration_v2_report(report)
    if not validation.valid:
        raise RuntimeError(
            "internally generated calibration-v2 report is invalid: "
            + "; ".join(validation.errors)
        )
    return report


def compositional_future_utility_calibration_v2_report_json(
    report: Mapping[str, object],
) -> str:
    """Serialize a validated report in memory; no output writer is exposed."""

    validation = validate_compositional_future_utility_calibration_v2_report(report)
    if not validation.valid:
        raise ValueError("invalid calibration-v2 report: " + "; ".join(validation.errors))
    return _canonical_json(dict(report))


__all__ = [
    "ARM_NAMES",
    "ASSESSMENT_STATUS",
    "CURATION_INTERVAL",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_ROOT",
    "DEVELOPMENT_ROOT_HEX",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_ATTEMPTS_AUTHORIZED",
    "EXECUTION_ATTEMPTS_CONSUMED",
    "EXECUTION_ATTEMPTS_REMAINING",
    "EXECUTION_OUTCOME",
    "KEY_MANIFEST",
    "LONG_TRACE_DECAY",
    "LONG_TRACE_DECAY_F32_BITS",
    "OUTPUT_WRITES_ALLOWED",
    "PHASE_LENGTHS",
    "PHASE_ORDER",
    "PRIMARY_ENDPOINTS",
    "PROTOCOL_NAMESPACE",
    "PROTOCOL_NAMESPACE_SHA256",
    "PROTOCOL_SCHEMA",
    "REPORT_SCHEMA",
    "RESOURCE_ACCOUNTING_SCOPE",
    "RUNTIME_IDENTITY_SCOPE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SECONDARY_ENDPOINTS",
    "STATUS",
    "STREAM_SHA256",
    "TOTAL_STEPS",
    "BoundSourceArrays",
    "CompositionalFutureUtilityCalibrationV2Protocol",
    "CompositionalFutureUtilityCalibrationV2Validation",
    "compositional_future_utility_calibration_v2_report_json",
    "logical_work_per_arm",
    "run_compositional_future_utility_calibration_v2_development",
    "validate_compositional_future_utility_calibration_v2_report",
]
