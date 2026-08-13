"""Consumed-root future-utility comparison for compositional control life.

This sibling evaluator deliberately preserves the v1 compositional control-life
implementation.  It reuses that lane's exact 8,998-step stream and compiled
scan, changing only the three predeclared future-utility configurations.  The
result is a deterministic in-memory development diagnostic: it has no writer,
threshold, winner-selection rule, evidence authority, or promotion path.

The historical left-pack configuration cannot be used verbatim.  Its
``energy_novelty`` branch overwrites the mixed future-utility signal, and its
positive novelty-admission bonus is invalid under ``legacy`` scoring.  The
corrected common base therefore records both compatibility departures before
varying only future-utility mix and trace decay.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
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
import alberta_framework.evaluation.compositional_discovery_development as compositional_discovery
import alberta_framework.evaluation.generated_birth_identity_scrub_epoch as birth_identity_scrub
import alberta_framework.evaluation.generated_class_lifecycle_scrub as lifecycle_scrub
from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)

PROTOCOL_SCHEMA: Final = "alberta.compositional-future-utility-development.protocol.v1"
REPORT_SCHEMA: Final = "alberta.compositional-future-utility-development.report.v1"
STATUS: Final = "DEVELOPMENT_COMPOSITIONAL_FUTURE_UTILITY_NOT_ASSESSED"
ASSESSMENT_STATUS: Final = "not-assessed"
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False

CONSUMED_KEY_MANIFEST: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {
        "exploration": (2_227_216_649, 3_977_711_669),
        "learner_genesis": (2_002_082_676, 3_427_004_161),
        "observations": (2_316_273_231, 3_036_545_927),
        "random_actions": (382_045_127, 333_255_797),
        "root": (0, 329_631_721),
    }
)
CONSUMED_STREAM_SHA256: Final = (
    "02fd5efbbb304b624fcfd29e259c361d5048233817e896300057d8e36f3fc036"
)
RUNTIME_IDENTITY_SCOPE: Final = (
    "selected Python, NumPy, JAX, backend, x64, and device fields; not an "
    "environment, accelerator-driver, XLA-flag, or compiler closure"
)
RESOURCE_ACCOUNTING_SCOPE: Final = (
    "exact persistent learner-state bytes and named logical cell/update counts, plus "
    "measured curation-audit arrays; excludes source arrays, full scan telemetry, "
    "compiler workspaces, and compiled FLOPs"
)

LEFT_PACK_SOURCE_ARM: Final = "dovetail_coverage_ancestor_headroom_leftpack"
ARM_NAMES: Final = (
    "contribution_mix0_decay095",
    "contribution_mix1_decay0",
    "contribution_mix1_decay095",
)

_ARM_PARAMETERS: Final = {
    ARM_NAMES[0]: (0.0, 0.95),
    ARM_NAMES[1]: (1.0, 0.0),
    ARM_NAMES[2]: (1.0, 0.95),
}
_ARM_ROLES: Final = {
    ARM_NAMES[0]: "future term disabled with contribution-trace work retained",
    ARM_NAMES[1]: "one-step contribution future-utility endpoint",
    ARM_NAMES[2]: "decay-0.95 contribution future-utility endpoint",
}
_EXPECTED_INTERVENTION_FIELDS: Final = (
    "future_utility_mix",
    "future_utility_trace_decay",
)
_COMMON_DEPARTURE_FIELDS: Final = (
    "candidate_scoring_mode",
    "candidate_novelty_admission_bonus",
    "future_utility_trace_mode",
)
_COMPATIBILITY_DEPARTURE_FIELDS: Final = (
    "candidate_scoring_mode",
    "candidate_novelty_admission_bonus",
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

INTERPRETATION: Final = (
    "Development-only paired comparison of three predeclared causal future-utility "
    "settings on one already-consumed compositional control life. Exact observations "
    "are descriptive and do not select a winner or establish Alberta Plan completion."
)
LIMITATIONS: Final = (
    "one already-consumed development root with no held-out confirmation",
    "finite synthetic product grammar and iid contextual-bandit observations",
    "immediate selected-action reward prediction, not TD control or world modeling",
    "legacy scoring and zero novelty bonus depart from the historical left-pack base",
    "the disabled arm is an internal comparator, not a historical-result reconstruction",
    "the source manifest binds selected direct files, not a transitive source closure",
    "runtime identity binds selected process/JAX fields, not an environment or compiler closure",
    "resource accounting excludes source arrays, full scan telemetry, and compiler workspaces",
    "no thresholds, search, writer, artifact, evidence authority, or promotion path",
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


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalFutureUtilityProtocol:
    """The exact consumed 8,998-step declaration; customization is forbidden."""

    schema_version: str = PROTOCOL_SCHEMA
    development_seed: int = control_life.DEFAULT_CONSUMED_SEED
    phase_order: tuple[str, ...] = control_life.PHASE_ORDER
    phase_lengths: tuple[int, ...] = control_life.DEFAULT_PHASE_LENGTHS
    epsilon: float = 0.1
    entry_window: int = 64
    tail_window: int = 64
    curation_interval: int = control_life.CURATION_INTERVAL
    left_pack_source_arm: str = LEFT_PACK_SOURCE_ARM

    def __post_init__(self) -> None:
        expected = (
            self.schema_version == PROTOCOL_SCHEMA
            and type(self.development_seed) is int
            and self.development_seed == control_life.DEFAULT_CONSUMED_SEED
            and type(self.phase_order) is tuple
            and self.phase_order == control_life.PHASE_ORDER
            and type(self.phase_lengths) is tuple
            and self.phase_lengths == control_life.DEFAULT_PHASE_LENGTHS
            and type(self.epsilon) is float
            and self.epsilon == 0.1
            and type(self.entry_window) is int
            and self.entry_window == 64
            and type(self.tail_window) is int
            and self.tail_window == 64
            and type(self.curation_interval) is int
            and self.curation_interval == control_life.CURATION_INTERVAL
            and self.left_pack_source_arm == LEFT_PACK_SOURCE_ARM
        )
        if not expected:
            raise ValueError("the compositional future-utility protocol is frozen")

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
            "status": STATUS,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "evidence_authorized": False,
            "writer_available": False,
            "development_seed": self.development_seed,
            "development_root_already_consumed": True,
            "new_seed_or_initialization_drawn": False,
            "seed_role": "consumed_development_nonpromoting",
            "phase_order": list(self.phase_order),
            "phase_lengths": list(self.phase_lengths),
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "entry_window": self.entry_window,
            "tail_window": self.tail_window,
            "curation_interval": self.curation_interval,
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
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> CompositionalFutureUtilityProtocol:
        canonical = cls()
        if set(payload) != set(canonical.to_config()):
            raise ValueError("protocol fields do not match the frozen schema")
        if dict(payload) != canonical.to_config():
            raise ValueError("protocol payload does not match the frozen declaration")
        return canonical


@dataclasses.dataclass(frozen=True, slots=True)
class BoundSourceArrays:
    """Exact paired stream and learner-genesis key derived from the consumed root."""

    key_manifest: dict[str, list[int]]
    observations: Array
    phase_indices: Array
    exploration_mask: Array
    random_actions: Array
    learner_key: Array
    stream_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionalFutureUtilityValidation:
    """Fail-closed validation result for an in-memory report."""

    valid: bool
    errors: tuple[str, ...]


class _ProcessAttemptLatch:
    """Run one string builder at most once, sealing success or BaseException failure."""

    def __init__(self, builder: Callable[[], str]) -> None:
        if not callable(builder):
            raise TypeError("process-attempt builder must be callable")
        self._builder = builder
        self._lock = threading.Lock()
        self._attempted = False
        self._value: str | None = None
        self._failure: BaseException | None = None

    def get(self) -> str:
        """Return the sole result; concurrent callers serialize behind the first attempt."""

        with self._lock:
            if self._attempted:
                if self._failure is not None:
                    raise RuntimeError(
                        "the process-local panel attempt is sealed after failure"
                    ) from self._failure
                if self._value is None:
                    raise RuntimeError("the process-local panel latch is internally invalid")
                return self._value

            self._attempted = True
            try:
                value = self._builder()
                if type(value) is not str:
                    raise TypeError("process-attempt builder must return an exact string")
            except BaseException as error:
                self._failure = error
                raise
            self._value = value
            return value


def _source_arm() -> control_life.CompositionalControlLifeArm:
    matches = tuple(
        arm for arm in control_life.CONTROL_LIFE_ARMS if arm.name == LEFT_PACK_SOURCE_ARM
    )
    if len(matches) != 1:
        raise RuntimeError("historical left-pack source arm is not uniquely declared")
    return matches[0]


def _arm_learner_config(name: str) -> dict[str, Any]:
    """Return one corrected predeclared config; no additional arms are accepted."""

    if type(name) is not str or name not in _ARM_PARAMETERS:
        raise ValueError("unknown predeclared compositional future-utility arm")
    mix, decay = _ARM_PARAMETERS[name]
    config = control_life.learner_config_for_arm(LEFT_PACK_SOURCE_ARM)
    config.update(
        {
            "candidate_scoring_mode": "legacy",
            "candidate_novelty_admission_bonus": 0.0,
            "future_utility_trace_mode": "contribution",
            "future_utility_mix": mix,
            "future_utility_trace_decay": decay,
            "future_utility_normalization": "none",
            "future_utility_rare_task_power": 0.0,
        }
    )
    return config


def _build_arm_learner(name: str) -> CompositionalFeatureLearner:
    return CompositionalFeatureLearner.from_config(_arm_learner_config(name))


def _source_arrays_bound(
    protocol: CompositionalFutureUtilityProtocol,
) -> BoundSourceArrays:
    """Bind the v1 stream and learner key without drawing a new root."""

    if type(protocol) is not CompositionalFutureUtilityProtocol:
        raise TypeError("protocol must be an exact CompositionalFutureUtilityProtocol")
    (
        key_manifest,
        observations,
        phase_indices,
        exploration_mask,
        random_actions,
        stream_sha256,
    ) = control_life._stream_arrays(
        protocol.control_life_protocol(),
        protocol.development_seed,
    )
    expected_key_manifest = {
        name: list(words) for name, words in CONSUMED_KEY_MANIFEST.items()
    }
    if key_manifest != expected_key_manifest:
        raise RuntimeError("consumed development key manifest does not match its frozen pin")
    if stream_sha256 != CONSUMED_STREAM_SHA256:
        raise RuntimeError("consumed development stream does not match its frozen digest")
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
    protocol: CompositionalFutureUtilityProtocol,
) -> dict[str, object]:
    """Declare fixed logical work shared by all three corrected arms."""

    if type(protocol) is not CompositionalFutureUtilityProtocol:
        raise TypeError("protocol must be an exact CompositionalFutureUtilityProtocol")
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
        "compiled_flop_equivalence_claimed": False,
    }


def _source_manifest() -> dict[str, str]:
    files = {
        "evaluation_module_sha256": _sha256_file(Path(__file__).resolve()),
        "control_life_v1_sha256": _sha256_file(Path(control_life.__file__).resolve()),
        "compositional_core_sha256": _sha256_file(
            Path(compositional_core.__file__).resolve()
        ),
        "future_utility_core_sha256": _sha256_file(
            Path(future_utility_core.__file__).resolve()
        ),
        "consumed_seed_declaration_sha256": _sha256_file(
            Path(compositional_discovery.__file__).resolve()
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
    mix, decay = _ARM_PARAMETERS[name]
    return {
        "name": name,
        "role": _ARM_ROLES[name],
        "future_utility_mix": mix,
        "future_utility_trace_decay": decay,
        "future_utility_trace_mode": "contribution",
        "future_utility_normalization": "none",
        "future_utility_rare_task_power": 0.0,
        "candidate_scoring_mode": "legacy",
        "candidate_novelty_admission_bonus": 0.0,
    }


def _arm_configuration_audit(
    configs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Validate the complete intervention and historical common-base departures."""

    if tuple(configs) != ARM_NAMES:
        raise RuntimeError("future-utility arm configs are not in the frozen order")
    first_fields = tuple(configs[ARM_NAMES[0]])
    if any(tuple(configs[name]) != first_fields for name in ARM_NAMES[1:]):
        raise RuntimeError("future-utility arm config schemas do not match exactly")

    varying: dict[str, dict[str, object]] = {}
    for field in first_fields:
        values = {name: cast(object, configs[name][field]) for name in ARM_NAMES}
        if len({_canonical_json(value) for value in values.values()}) > 1:
            varying[field] = values
    if tuple(varying) != _EXPECTED_INTERVENTION_FIELDS:
        raise RuntimeError("the three arms differ outside the predeclared intervention")

    historical = control_life.learner_config_for_arm(LEFT_PACK_SOURCE_ARM)
    if tuple(historical) != first_fields:
        raise RuntimeError("corrected and historical learner config schemas do not match")
    historical_expected: dict[str, object] = {
        "candidate_scoring_mode": "energy_novelty",
        "candidate_novelty_admission_bonus": 1.0,
        "future_utility_trace_mode": "marginal",
    }
    corrected_expected: dict[str, object] = {
        "candidate_scoring_mode": "legacy",
        "candidate_novelty_admission_bonus": 0.0,
        "future_utility_trace_mode": "contribution",
    }
    common_departures: dict[str, dict[str, object]] = {}
    for field in _COMMON_DEPARTURE_FIELDS:
        if historical[field] != historical_expected[field]:
            raise RuntimeError(f"historical left-pack field {field} changed before the panel")
        corrected = configs[ARM_NAMES[0]][field]
        if corrected != corrected_expected[field] or any(
            configs[name][field] != corrected for name in ARM_NAMES
        ):
            raise RuntimeError(f"corrected common-base field {field} is not exact and paired")
        if historical[field] == corrected:
            raise RuntimeError(f"declared common-base departure {field} is not a departure")
        common_departures[field] = {
            "historical_left_pack": historical[field],
            "corrected_common_base": corrected,
        }

    declared_departure_fields = {
        *_COMMON_DEPARTURE_FIELDS,
        *_EXPECTED_INTERVENTION_FIELDS,
    }
    for name in ARM_NAMES:
        undeclared = tuple(
            field
            for field in first_fields
            if field not in declared_departure_fields
            and _canonical_json(configs[name][field])
            != _canonical_json(historical[field])
        )
        if undeclared:
            raise RuntimeError(
                f"future-utility arm {name} changes undeclared common fields: {undeclared}"
            )

    for name, parameters in _ARM_PARAMETERS.items():
        actual = (
            configs[name]["future_utility_mix"],
            configs[name]["future_utility_trace_decay"],
        )
        if _canonical_json(actual) != _canonical_json(parameters):
            raise RuntimeError(f"future-utility arm {name} does not match its frozen parameters")
        if (
            configs[name]["future_utility_normalization"] != "none"
            or configs[name]["future_utility_rare_task_power"] != 0.0
        ):
            raise RuntimeError("future-utility normalization or rare-task weighting drifted")
    if not _source_arm().composed_readout_enabled:
        raise RuntimeError("historical left-pack composed readout is no longer enabled")
    return varying, common_departures


def _static_preflight() -> dict[str, object]:
    """Prove the fixed config intervention is reachable before any scan executes."""

    configs = {name: _arm_learner_config(name) for name in ARM_NAMES}
    varying, common_departures = _arm_configuration_audit(configs)
    historical = control_life.learner_config_for_arm(LEFT_PACK_SOURCE_ARM)
    declared_departure_fields = {
        *_COMMON_DEPARTURE_FIELDS,
        *_EXPECTED_INTERVENTION_FIELDS,
    }
    core_constraint_changed = any(
        _canonical_json(configs[name][field]) != _canonical_json(historical[field])
        for name in ARM_NAMES
        for field in historical
        if field not in declared_departure_fields
    )
    active_overwrite_disabled = all(
        configs[name]["candidate_scoring_mode"] == "legacy" for name in ARM_NAMES
    )
    candidate_overwrite_disabled = active_overwrite_disabled
    positive_active_and_candidate_banks = all(
        type(configs[name]["n_features"]) is int
        and configs[name]["n_features"] > 0
        and type(configs[name]["candidate_count"]) is int
        and configs[name]["candidate_count"] > 0
        for name in ARM_NAMES
    )
    future_utility_reaches_ranking = (
        active_overwrite_disabled
        and candidate_overwrite_disabled
        and positive_active_and_candidate_banks
        and configs[ARM_NAMES[0]]["future_utility_mix"] == 0.0
        and configs[ARM_NAMES[2]]["future_utility_mix"] == 1.0
    )
    trace_decay_reaches_contribution_traces = (
        positive_active_and_candidate_banks
        and all(
            configs[name]["future_utility_trace_mode"] == "contribution"
            for name in ARM_NAMES
        )
        and configs[ARM_NAMES[1]]["future_utility_mix"] == 1.0
        and configs[ARM_NAMES[2]]["future_utility_mix"] == 1.0
        and configs[ARM_NAMES[1]]["future_utility_trace_decay"] == 0.0
        and configs[ARM_NAMES[2]]["future_utility_trace_decay"] == 0.95
    )
    if core_constraint_changed:
        raise RuntimeError("the corrected panel changes an undeclared core constraint")
    if not future_utility_reaches_ranking:
        raise RuntimeError("future utility cannot reach both active and candidate ranking")
    if not trace_decay_reaches_contribution_traces:
        raise RuntimeError("trace decay cannot reach active and candidate contribution traces")
    return {
        "static_audit_performed": True,
        "panel_executed_during_preflight": False,
        "candidate_scoring_mode": "legacy",
        "candidate_novelty_admission_bonus": 0.0,
        "future_utility_trace_mode": "contribution",
        "future_utility_normalization": "none",
        "future_utility_rare_task_power": 0.0,
        "core_constraint_changed": core_constraint_changed,
        "active_mixed_utility_overwrite_disabled": active_overwrite_disabled,
        "candidate_mixed_utility_overwrite_disabled": candidate_overwrite_disabled,
        "future_utility_reaches_active_and_candidate_ranking": (
            future_utility_reaches_ranking
        ),
        "trace_decay_reaches_active_and_candidate_contribution_traces": (
            trace_decay_reaches_contribution_traces
        ),
        "only_varying_config_fields": varying,
        "expected_varying_config_fields": list(_EXPECTED_INTERVENTION_FIELDS),
        "common_base_departures_from_historical_left_pack": common_departures,
        "compatibility_departure_fields": list(_COMPATIBILITY_DEPARTURE_FIELDS),
    }


def _run_arm(
    protocol: CompositionalFutureUtilityProtocol,
    source: BoundSourceArrays,
    name: str,
) -> dict[str, object]:
    learner = _build_arm_learner(name)
    state = cast(
        CompositionalFeatureState,
        learner.init(control_life.RAW_DIM, source.learner_key).replace(  # type: ignore[attr-defined]
            birth_timestamp=0.0,
            uptime_s=0.0,
        ),
    )
    initial_ranking_diagnostics = learner.ranking_diagnostics(
        state, control_life.RAW_DIM
    )
    if not bool(initial_ranking_diagnostics.contract_valid):
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
        birth_identity_scrub.generated_birth_identity_scrub_epoch_core_state_sha256(
            state
        )
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

    initial_ranking = control_life._initial_ranking_record(
        state, initial_ranking_diagnostics
    )
    curation_totals = np.sum(
        np.asarray(events.curation_counts, dtype=np.int64), axis=0
    )
    curation_total_record = {
        key: int(value)
        for key, value in zip(
            control_life.CURATION_COUNT_NAMES, curation_totals, strict=True
        )
    }
    active_trajectories: dict[str, object] = {}
    candidate_trajectories: dict[str, object] = {}
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
    curation_audit, audit_elements, audit_nbytes = (
        control_life._curation_decision_audit(v1_protocol, events)
    )
    if curation_audit["shared_p45_active_bank_loss_count"] != cast(
        Mapping[str, object], active_trajectories["shared_p45"]
    )["loss_episode_count"]:
        raise RuntimeError("active shared-p45 loss audit does not close")
    if curation_audit["shared_p45_candidate_bank_loss_count"] != cast(
        Mapping[str, object], candidate_trajectories["shared_p45"]
    )["loss_episode_count"]:
        raise RuntimeError("candidate shared-p45 loss audit does not close")

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
            "advanced": (
                initial_traces["combined_sha256"] != final_traces["combined_sha256"]
            ),
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
        "curation_totals": curation_total_record,
        "lifetime_metrics": control_life._window_metrics(
            events, 0, protocol.total_steps
        ),
        "phase_metrics": control_life._phase_records(
            v1_protocol,
            initial_ranking,
            initial_active_counts,
            initial_candidate_counts,
            events,
        ),
        "active_structural_trajectories": active_trajectories,
        "candidate_structural_trajectories": candidate_trajectories,
        "active_target_coexistence": (
            control_life._active_target_coexistence_record(
                events.active_signature_counts,
                start_post_step=0,
            )
        ),
        "curation_decision_audit": curation_audit,
        "curation_decision_audit_resources": {
            "events": cast(int, curation_audit["due_curation_event_count"]),
            "ephemeral_array_elements": audit_elements,
            "ephemeral_array_bytes": audit_nbytes,
            "report_json_bytes": cast(
                int, curation_audit["records_canonical_json_bytes"]
            ),
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
            curation_total_record["cascade_refill"],
        ),
        "initial_ranking": initial_ranking,
        "final_ranking": control_life._event_ranking_record(
            events, protocol.total_steps - 1
        ),
        "work": logical_work_per_arm(protocol),
    }


def _findings(run: Mapping[str, object]) -> dict[str, object]:
    coexistence = cast(Mapping[str, object], run["active_target_coexistence"])
    trajectories = cast(
        Mapping[str, Mapping[str, object]], run["active_structural_trajectories"]
    )
    lifetime = cast(Mapping[str, object], run["lifetime_metrics"])
    return {
        "lifetime_executed_reward": lifetime["executed_reward"],
        "lifetime_greedy_reward": lifetime["greedy_reward"],
        "maximum_active_target_count": coexistence["maximum_active_target_count"],
        "all_three_present_steps": coexistence["all_three_present_steps"],
        "active_targets_at_end": coexistence["active_targets_at_end"],
        "target_present_at_end": {
            name: trajectories[name]["present_at_end"] for name in ("A", "B", "C")
        },
        "obsolete_p12_present_at_end": trajectories["obsolete_p12"][
            "present_at_end"
        ],
        "shared_p45_present_at_end": trajectories["shared_p45"]["present_at_end"],
    }


def _arm_comparison(runs: list[dict[str, object]]) -> dict[str, object]:
    configs = {
        cast(str, run["arm"]): cast(dict[str, Any], run["learner_config"])
        for run in runs
    }
    varying, common_departures = _arm_configuration_audit(configs)

    disabled = _findings(runs[0])
    deltas: dict[str, object] = {}
    for run in runs[1:]:
        name = cast(str, run["arm"])
        finding = _findings(run)
        deltas[name] = {
            "lifetime_executed_reward": cast(float, finding["lifetime_executed_reward"])
            - cast(float, disabled["lifetime_executed_reward"]),
            "lifetime_greedy_reward": cast(float, finding["lifetime_greedy_reward"])
            - cast(float, disabled["lifetime_greedy_reward"]),
            "maximum_active_target_count": cast(
                int, finding["maximum_active_target_count"]
            )
            - cast(int, disabled["maximum_active_target_count"]),
            "all_three_present_steps": cast(int, finding["all_three_present_steps"])
            - cast(int, disabled["all_three_present_steps"]),
        }
    return {
        "initial_states_equal": len({run["initial_state_sha256"] for run in runs}) == 1,
        "logical_work_equal": all(run["work"] == runs[0]["work"] for run in runs),
        "stream_keys_shapes_cadence_and_update_opportunities_equal": True,
        "common_base_departures_from_historical_left_pack": common_departures,
        "compatibility_departure_fields": list(_COMPATIBILITY_DEPARTURE_FIELDS),
        "only_varying_config_fields": varying,
        "expected_varying_config_fields": list(_EXPECTED_INTERVENTION_FIELDS),
        "disabled_internal_comparator": ARM_NAMES[0],
        "historical_left_pack_result_reconstruction_claimed": False,
        "behavioral_experience_matching_claimed": False,
        "descriptive_deltas_from_internal_comparator": deltas,
        "winner_selected": False,
        "null_or_harm_results_must_stop_without_tuning": True,
        "compiled_flop_equivalence_claimed": False,
    }


def _build_report() -> dict[str, object]:
    protocol = CompositionalFutureUtilityProtocol()
    source_manifest = _source_manifest()
    runtime_identity = _runtime_identity()
    corrected_static_preflight = _static_preflight()
    source = _source_arrays_bound(protocol)
    runs = [_run_arm(protocol, source, name) for name in ARM_NAMES]
    if _source_manifest() != source_manifest:
        raise RuntimeError("selected source files changed during the future-utility panel")
    if _runtime_identity() != runtime_identity:
        raise RuntimeError("runtime identity changed during the future-utility panel")
    if len({run["initial_state_sha256"] for run in runs}) != 1:
        raise RuntimeError("paired arms do not have an identical genesis")
    if not all(run["work"] == runs[0]["work"] for run in runs):
        raise RuntimeError("paired arms do not have identical declared work")

    body: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": STATUS,
        "assessment_status": ASSESSMENT_STATUS,
        "development_only": DEVELOPMENT_ONLY,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "evidence_authorized": EVIDENCE_AUTHORIZED,
        "output_writes_allowed": OUTPUT_WRITES_ALLOWED,
        "artifact_bytes_written": 0,
        "interpretation": INTERPRETATION,
        "limitations": list(LIMITATIONS),
        "protocol": protocol.to_config(),
        "protocol_sha256": _json_sha256(protocol.to_config()),
        "source_manifest": source_manifest,
        "source_manifest_scope": "selected-direct-files-not-transitive-closure",
        "transitive_source_closure_claimed": False,
        "runtime_identity": runtime_identity,
        "runtime_identity_scope": RUNTIME_IDENTITY_SCOPE,
        "runtime_identity_bound_by_validation": True,
        "runtime_environment_or_compiler_closure_claimed": False,
        "resource_accounting_scope": RESOURCE_ACCOUNTING_SCOPE,
        "seed": protocol.development_seed,
        "seed_role": "consumed_development_nonpromoting",
        "key_manifest": source.key_manifest,
        "stream_sha256": source.stream_sha256,
        "failed_static_preflight": {
            "static_audit_performed": True,
            "panel_executed": False,
            "candidate_scoring_mode": "energy_novelty",
            "reason": (
                "energy_novelty replaces active and candidate mixed-utility signals, "
                "so future_utility_mix cannot reach ranking, curation, or behavior"
            ),
            "vacuous_panel_run": False,
            "outcome_claimed": False,
        },
        "corrected_static_preflight": corrected_static_preflight,
        "arm_order": list(ARM_NAMES),
        "arm_definitions": [_arm_definition(name) for name in ARM_NAMES],
        "runs": runs,
        "arm_comparison": _arm_comparison(runs),
        "consumed_findings": {
            cast(str, run["arm"]): _findings(run) for run in runs
        },
        "winner_or_default_selected": False,
        "search_performed": False,
        "rerun_or_tuning_authorized": False,
        "identity_tracking": {
            "v4_birth_ledger_integrated": False,
            "retained_identity_assessed": False,
            "reported_reacquisition_kind": "bank_level_algebraic_structural_only",
        },
        "work_resource_contract": {
            "selected_arm_count": len(ARM_NAMES),
            "per_arm": logical_work_per_arm(protocol),
            "accounting_scope": RESOURCE_ACCOUNTING_SCOPE,
            "logical_work_matched": True,
            "persistent_shapes_matched": True,
            "source_array_bytes_included": False,
            "full_scan_telemetry_bytes_included": False,
            "compiler_workspace_bytes_included": False,
            "compiled_flop_equivalence_claimed": False,
        },
    }
    return cast(
        dict[str, object],
        _json_clone({**body, "report_sha256": _json_sha256(body)}),
    )


_EXPECTED_REPORT_ATTEMPT = _ProcessAttemptLatch(
    lambda: _canonical_json(_build_report())
)


def _expected_report_json() -> str:
    """Return the sole process-local build result, sealing failures and concurrency."""

    return _EXPECTED_REPORT_ATTEMPT.get()


def validate_compositional_future_utility_report(
    report: Mapping[str, object],
) -> CompositionalFutureUtilityValidation:
    """Fail closed against the full cached deterministic reconstruction."""

    try:
        candidate = cast(dict[str, object], _json_clone(dict(report)))
    except (TypeError, ValueError) as error:
        return CompositionalFutureUtilityValidation(
            False, (f"report is not canonical JSON: {error}",)
        )
    expected = cast(dict[str, object], json.loads(_expected_report_json()))
    errors: list[str] = []
    if candidate != expected:
        errors.append("report does not match the deterministic consumed-root reconstruction")
    body = {key: value for key, value in candidate.items() if key != "report_sha256"}
    if candidate.get("report_sha256") != _json_sha256(body):
        errors.append("report_sha256 does not reconstruct")
    return CompositionalFutureUtilityValidation(not errors, tuple(errors))


def run_compositional_future_utility_development() -> dict[str, object]:
    """Execute the sole predeclared panel and return its validated in-memory report."""

    report = cast(dict[str, object], json.loads(_expected_report_json()))
    validation = validate_compositional_future_utility_report(report)
    if not validation.valid:
        raise RuntimeError(
            "internally generated future-utility report is invalid: "
            + "; ".join(validation.errors)
        )
    return report


def compositional_future_utility_report_json(report: Mapping[str, object]) -> str:
    """Serialize a valid report without exposing any output writer."""

    validation = validate_compositional_future_utility_report(report)
    if not validation.valid:
        raise ValueError("invalid future-utility report: " + "; ".join(validation.errors))
    return _canonical_json(dict(report))


__all__ = [
    "ARM_NAMES",
    "ASSESSMENT_STATUS",
    "CONSUMED_KEY_MANIFEST",
    "CONSUMED_STREAM_SHA256",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "OUTPUT_WRITES_ALLOWED",
    "PROTOCOL_SCHEMA",
    "REPORT_SCHEMA",
    "RESOURCE_ACCOUNTING_SCOPE",
    "RUNTIME_IDENTITY_SCOPE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "STATUS",
    "BoundSourceArrays",
    "CompositionalFutureUtilityProtocol",
    "CompositionalFutureUtilityValidation",
    "compositional_future_utility_report_json",
    "logical_work_per_arm",
    "run_compositional_future_utility_development",
    "validate_compositional_future_utility_report",
]
