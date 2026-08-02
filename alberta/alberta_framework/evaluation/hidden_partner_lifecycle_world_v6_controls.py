"""Nonexecuting development controls for the noisy-world v6 lifecycle design.

This module binds the mechanism configurations that already have exact source
contracts.  It does not create keys, choose seeds, construct a runner, write an
artifact, set thresholds, authorize execution, or authorize scientific
promotion.  Runner-binding readiness is deliberately separate from authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Literal, cast

from alberta_framework.core.grounded_joint_world_model import (
    GroundedJointWorldModelConfig,
)
from alberta_framework.core.integrated_hidden_partner import (
    DEPLOYED_FEATURE_DIM,
    N_ACTIONS,
    RAW_OBSERVATION_DIM,
    IntegratedHiddenPartnerAgent,
    IntegratedHiddenPartnerConfig,
)
from alberta_framework.core.representation_gradient_mixer import (
    GradientMixMode,
    RepresentationGradientMixerConfig,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCHEMA,
    OPEN_PREREQUISITES,
    PRIMARY_CONDITION_ORDER,
    PRIMARY_CONDITION_OVERRIDES,
    V6_INITIAL_ACTIVE_DESCRIPTORS,
)
from alberta_framework.evaluation.hidden_partner_world_online_bridge import (
    HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA,
    HiddenPartnerWorldOnlineBridge,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    OBSERVATION_FIELDS,
    HiddenPartnerWorldFeedbackConfig,
    HiddenPartnerWorldFeedbackWorld,
)

HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.controls-development.v4"
)
HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_BINDING_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.control-binding-development.v1"
)
HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_DESIGN_SUCCESSOR_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.design-successor-bindings-development.v1"
)
DEVELOPMENT_ONLY = True
EXECUTION_AUTHORIZED = False
EVIDENCE_AUTHORIZED = False
SCIENTIFIC_PROMOTION_ALLOWED = False

V6_TARGET_HEAD_ORDER: tuple[str, ...] = (*OBSERVATION_FIELDS, "reward", "discount")
V6_REPRESENTATION_LOSS_WEIGHTS: tuple[float, ...] = (
    0.0,
    10.0 / 3.0,
    0.0,
    0.0,
    0.0,
    0.0,
    10.0 / 3.0,
    10.0 / 3.0,
    0.0,
    0.0,
)

V6_DIAGNOSTIC_ORDER: tuple[str, ...] = (
    "uniform_action",
    "equal_cue",
    "row_bias",
)

FocalActionPolicy = Literal["agent", "balanced_external"]
ControlFamily = Literal["primary", "diagnostic"]

V6_MANIFEST_OVERRIDE_PATH_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("representation_gradient_mode", ("representation_gradient_mixer", "mode")),
)

V6_DESIGN_SUCCESSOR_CLOSED_BINDINGS: tuple[tuple[str, str], ...] = (
    (
        "core_custom_initial_descriptor_bank_config",
        "IntegratedHiddenPartnerConfig.initial_active_descriptors",
    ),
    (
        "matched_freeze_control",
        "IntegratedHiddenPartnerConfig.feature_lifecycle_enabled",
    ),
    (
        "grounded_world_row_bias_and_target_weights",
        (
            "GroundedJointWorldModelConfig.feature_path_mode+"
            "representation_loss_weights"
        ),
    ),
)
V6_DESIGN_SUCCESSOR_REMAINING_OPEN: tuple[tuple[str, str], ...] = (
    ("v6_runner", "runner_is_not_yet_source_bound_and_certified"),
    ("v6_strict_validator", "validator_is_not_yet_source_bound_and_certified"),
    (
        "development_threshold_calibration",
        "thresholds_are_not_yet_calibrated_frozen_and_certified",
    ),
    (
        "fresh_seed_namespace_and_seed_snapshot",
        "fresh_namespace_and_seed_snapshot_are_not_yet_reserved_and_certified",
    ),
)

if tuple(identifier for identifier, _ in OPEN_PREREQUISITES[:3]) != tuple(
    identifier for identifier, _ in V6_DESIGN_SUCCESSOR_CLOSED_BINDINGS
):
    raise RuntimeError("v6 successor mechanism bindings differ from the frozen design")
if OPEN_PREREQUISITES[3:] != V6_DESIGN_SUCCESSOR_REMAINING_OPEN:
    raise RuntimeError("v6 successor open prerequisites differ from the frozen design")


def _grounded_config(
    *,
    feature_path_mode: Literal["affine", "row_bias_only"] = "affine",
) -> GroundedJointWorldModelConfig:
    return GroundedJointWorldModelConfig(
        representation_dim=DEPLOYED_FEATURE_DIM,
        target_observation_dim=RAW_OBSERVATION_DIM,
        n_focal_actions=N_ACTIONS,
        n_partner_actions=N_ACTIONS,
        representation_loss_weights=V6_REPRESENTATION_LOSS_WEIGHTS,
        feature_path_mode=feature_path_mode,
    )


def _mixer_config(*, mode: GradientMixMode = "full") -> RepresentationGradientMixerConfig:
    return RepresentationGradientMixerConfig(
        representation_dim=DEPLOYED_FEATURE_DIM,
        mode=mode,
        behavior_weight=1.0,
        grounded_world_weight=1.0,
        behavior_normalization="unit_l2",
        grounded_world_normalization="unit_l2",
        normalization_epsilon=1e-8,
        final_clip_norm=5.0,
    )


def build_v6_full_agent_config() -> IntegratedHiddenPartnerConfig:
    """Return the explicit uncalibrated full-mechanism development candidate."""

    return IntegratedHiddenPartnerConfig(
        initial_active_descriptors=V6_INITIAL_ACTIVE_DESCRIPTORS,
        grounded_world_model=_grounded_config(),
        representation_gradient_mixer=_mixer_config(),
        grounded_world_learning_enabled=True,
        grounded_world_planning_enabled=True,
        epsilon=0.20,
        active_utility_retention_decay=0.9999,
        active_utility_retention_grace_steps=4_096,
        active_utility_evidence_threshold=0.10,
        retire_stale_features=True,
        candidate_promotion_floor=0.10,
        evidence_gated_feature_memory=True,
        feature_evidence_confirmation_steps=24,
        independent_relevance_probe=True,
        candidate_promotion_confirmation_steps=1,
        candidate_reacquisition_confirmation_steps=8,
        evidence_gated_consumer_memory=True,
        consumer_evidence_confirmation_steps=12,
        consumer_read_confirmation_steps=4,
        consumer_read_lease_steps=4,
    )


def _replace_mixer_mode(
    config: IntegratedHiddenPartnerConfig,
    mode: GradientMixMode,
) -> IntegratedHiddenPartnerConfig:
    mixer = config.representation_gradient_mixer
    if mixer is None:
        raise ValueError("v6 controls require the grounded representation mixer")
    return dataclasses.replace(
        config,
        representation_gradient_mixer=dataclasses.replace(mixer, mode=mode),
    )


def _primary_agent_config(
    name: str,
    full: IntegratedHiddenPartnerConfig,
) -> IntegratedHiddenPartnerConfig | None:
    if name == "full":
        return full
    if name == "grounded_model_frozen":
        return dataclasses.replace(full, grounded_world_learning_enabled=False)
    if name == "world_credit_off":
        return _replace_mixer_mode(full, "behavior_only")
    if name == "behavior_credit_off":
        return _replace_mixer_mode(full, "world_only")
    if name == "all_representation_credit_off":
        return _replace_mixer_mode(full, "discard")
    if name == "state_frozen":
        return dataclasses.replace(full, state_learning_enabled=False)
    if name == "recurrent_memory_masked":
        return dataclasses.replace(full, memory_masked=True)
    if name == "table_planner":
        return dataclasses.replace(full, grounded_world_planning_enabled=False)
    if name == "no_planning":
        return dataclasses.replace(full, planning_enabled=False)
    if name == "uniform_partner":
        return dataclasses.replace(full, uniform_partner_belief=True)
    if name == "lifecycle_frozen":
        return dataclasses.replace(full, feature_lifecycle_enabled=False)
    if name == "no_identity_carry":
        return dataclasses.replace(full, carry_survivors=False)
    if name == "no_retention_floor":
        return dataclasses.replace(full, active_utility_retention_decay=None)
    if name == "retirement_disabled":
        return dataclasses.replace(full, retire_stale_features=False)
    if name == "random_curation":
        return dataclasses.replace(full, random_feature_curation=True)
    raise ValueError(f"unknown v6 primary condition: {name!r}")


def _manifest_primary_agent_config(
    name: str,
    full: IntegratedHiddenPartnerConfig,
) -> IntegratedHiddenPartnerConfig:
    """Apply the frozen design override through the reviewed source-field mapping."""

    try:
        index = PRIMARY_CONDITION_ORDER.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown v6 primary condition: {name!r}") from exc
    config = full
    for manifest_field, value in PRIMARY_CONDITION_OVERRIDES[index]:
        if manifest_field == "representation_gradient_mode":
            if type(value) is not str or value not in (
                "full",
                "behavior_only",
                "world_only",
                "discard",
            ):
                raise RuntimeError("frozen representation-gradient override is invalid")
            config = _replace_mixer_mode(config, cast(GradientMixMode, value))
            continue
        if manifest_field not in {field.name for field in dataclasses.fields(config)}:
            raise RuntimeError(
                f"frozen v6 override has no source mapping: {manifest_field!r}"
            )
        config = dataclasses.replace(
            config,
            **cast(Any, {manifest_field: value}),
        )
    return config


def _expected_diagnostic_components(
    name: str,
    full: IntegratedHiddenPartnerConfig,
) -> tuple[
    IntegratedHiddenPartnerConfig,
    HiddenPartnerWorldFeedbackConfig,
    FocalActionPolicy,
]:
    if name == "uniform_action":
        return (
            dataclasses.replace(full, action_selection_mode="externally_forced"),
            HiddenPartnerWorldFeedbackConfig(),
            "balanced_external",
        )
    if name == "equal_cue":
        return (
            full,
            HiddenPartnerWorldFeedbackConfig(cue_flip_probabilities=(0.30, 0.30)),
            "agent",
        )
    if name == "row_bias":
        grounded = full.grounded_world_model
        if grounded is None:
            raise RuntimeError("v6 row-bias diagnostic requires the grounded model")
        return (
            dataclasses.replace(
                full,
                grounded_world_model=dataclasses.replace(
                    grounded,
                    feature_path_mode="row_bias_only",
                ),
            ),
            HiddenPartnerWorldFeedbackConfig(),
            "agent",
        )
    raise ValueError(f"unknown v6 diagnostic condition: {name!r}")


@dataclasses.dataclass(frozen=True)
class HiddenPartnerLifecycleWorldV6Control:
    """One nonexecuting primary or diagnostic mechanism binding."""

    name: str
    primary: bool
    agent_config: IntegratedHiddenPartnerConfig | None
    world_config: HiddenPartnerWorldFeedbackConfig
    focal_action_policy: FocalActionPolicy = "agent"
    initial_external_action: int = 0
    execution_blocker: str | None = None

    def __post_init__(self) -> None:
        if type(self) is not HiddenPartnerLifecycleWorldV6Control:
            raise TypeError("control must be an exact HiddenPartnerLifecycleWorldV6Control")
        if type(self.name) is not str:
            raise TypeError("control name must be an exact built-in str")
        if type(self.primary) is not bool:
            raise TypeError("control primary must be an exact built-in bool")
        allowed = PRIMARY_CONDITION_ORDER if self.primary else V6_DIAGNOSTIC_ORDER
        if self.name not in allowed:
            raise ValueError("control name is not in the declared v6 order")
        if self.agent_config is not None and type(self.agent_config) is not (
            IntegratedHiddenPartnerConfig
        ):
            raise TypeError(
                "agent_config must be an exact IntegratedHiddenPartnerConfig or None"
            )
        if type(self.world_config) is not HiddenPartnerWorldFeedbackConfig:
            raise TypeError("world_config must be an exact HiddenPartnerWorldFeedbackConfig")
        if type(self.focal_action_policy) is not str:
            raise TypeError("focal_action_policy must be an exact built-in str")
        if self.focal_action_policy not in ("agent", "balanced_external"):
            raise ValueError("focal_action_policy must be agent or balanced_external")
        if type(self.initial_external_action) is not int:
            raise TypeError("initial_external_action must be an exact built-in int")
        if self.initial_external_action != 0:
            raise ValueError("all live v6 controls require canonical initial_external_action 0")
        if self.agent_config is not None:
            if type(self.agent_config.action_selection_mode) is not str:
                raise TypeError("agent action_selection_mode must be an exact built-in str")
            required_action_selection_mode = (
                "externally_forced"
                if self.focal_action_policy == "balanced_external"
                else "agent"
            )
            if self.agent_config.action_selection_mode != required_action_selection_mode:
                raise ValueError(
                    "focal_action_policy and agent action_selection_mode must match exactly"
                )
        if self.execution_blocker is not None:
            if type(self.execution_blocker) is not str:
                raise TypeError("execution_blocker must be an exact built-in str or None")
            if not self.execution_blocker:
                raise ValueError("execution_blocker must be a non-empty string or None")
        if self.agent_config is None and self.execution_blocker is None:
            raise ValueError("a missing agent config must have an explicit blocker")
        _validate_v6_control_semantics(self)

    @property
    def execution_ready(self) -> bool:
        return self.agent_config is not None and self.execution_blocker is None

    def to_config(self) -> dict[str, object]:
        validated = validate_v6_control(self)
        return {
            "schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA,
            "development_only": DEVELOPMENT_ONLY,
            "execution_authorized": EXECUTION_AUTHORIZED,
            "evidence_authorized": EVIDENCE_AUTHORIZED,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "name": validated.name,
            "primary": validated.primary,
            "status": (
                "READY_FOR_RUNNER_BINDING" if validated.execution_ready else "BLOCKED"
            ),
            "execution_ready": validated.execution_ready,
            "execution_blocker": validated.execution_blocker,
            "focal_action_policy": validated.focal_action_policy,
            "initial_external_action": validated.initial_external_action,
            "world_config": validated.world_config.to_config(),
            "agent_config": (
                None
                if validated.agent_config is None
                else validated.agent_config.to_config()
            ),
        }


def _validate_v6_control_semantics(
    control: HiddenPartnerLifecycleWorldV6Control,
) -> None:
    full = build_v6_full_agent_config()
    if control.primary:
        expected_agent = _manifest_primary_agent_config(control.name, full)
        implementation_agent = _primary_agent_config(control.name, full)
        if implementation_agent != expected_agent:
            raise RuntimeError(
                f"v6 primary control {control.name!r} differs from its frozen override mapping"
            )
        expected_world = HiddenPartnerWorldFeedbackConfig()
        expected_policy: FocalActionPolicy = "agent"
    else:
        expected_agent, expected_world, expected_policy = _expected_diagnostic_components(
            control.name,
            full,
        )
    if control.world_config != expected_world:
        raise ValueError(f"v6 control {control.name!r} has noncanonical world semantics")
    if control.focal_action_policy != expected_policy:
        raise ValueError(f"v6 control {control.name!r} has noncanonical action policy")
    if control.initial_external_action != 0:
        raise ValueError(f"v6 control {control.name!r} has noncanonical action phase")
    if control.agent_config is not None and control.agent_config != expected_agent:
        raise ValueError(f"v6 control {control.name!r} has noncanonical agent semantics")


def validate_v6_control(
    control: HiddenPartnerLifecycleWorldV6Control,
) -> HiddenPartnerLifecycleWorldV6Control:
    """Accept one exact, semantically canonical v6 control binding."""

    if type(control) is not HiddenPartnerLifecycleWorldV6Control:
        raise TypeError("control must be an exact HiddenPartnerLifecycleWorldV6Control")
    # Re-run construction validation so frozen-instance tampering also fails closed.
    control.__post_init__()
    return control


def build_v6_primary_controls() -> tuple[HiddenPartnerLifecycleWorldV6Control, ...]:
    """Bind all fifteen primary mechanism controls exactly."""

    full = build_v6_full_agent_config()
    world = HiddenPartnerWorldFeedbackConfig()
    controls: list[HiddenPartnerLifecycleWorldV6Control] = []
    for name in PRIMARY_CONDITION_ORDER:
        controls.append(
            HiddenPartnerLifecycleWorldV6Control(
                name=name,
                primary=True,
                agent_config=_primary_agent_config(name, full),
                world_config=world,
            )
        )
    return tuple(controls)


def build_v6_diagnostic_controls() -> tuple[HiddenPartnerLifecycleWorldV6Control, ...]:
    """Bind all required diagnostics to their exact development mechanisms."""

    full = build_v6_full_agent_config()
    grounded = full.grounded_world_model
    if grounded is None:
        raise ValueError("v6 diagnostics require the grounded model")
    return (
        HiddenPartnerLifecycleWorldV6Control(
            name="uniform_action",
            primary=False,
            agent_config=dataclasses.replace(
                full,
                action_selection_mode="externally_forced",
            ),
            world_config=HiddenPartnerWorldFeedbackConfig(),
            focal_action_policy="balanced_external",
        ),
        HiddenPartnerLifecycleWorldV6Control(
            name="equal_cue",
            primary=False,
            agent_config=full,
            world_config=HiddenPartnerWorldFeedbackConfig(
                cue_flip_probabilities=(0.30, 0.30)
            ),
        ),
        HiddenPartnerLifecycleWorldV6Control(
            name="row_bias",
            primary=False,
            agent_config=dataclasses.replace(
                full,
                grounded_world_model=dataclasses.replace(
                    grounded,
                    feature_path_mode="row_bias_only",
                ),
            ),
            world_config=HiddenPartnerWorldFeedbackConfig(),
        ),
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("v6 controls must contain finite canonical JSON data") from exc


def _require_exact_sha256(value: object, *, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be one lowercase SHA-256 hex digest")


@dataclasses.dataclass(frozen=True)
class HiddenPartnerLifecycleWorldV6ControlBinding:
    """Versioned hashes for one exact control and constructible bridge config."""

    binding_schema: str
    family: ControlFamily
    name: str
    control_schema: str
    bridge_schema: str
    initial_external_action: int
    control_config_sha256: str
    bridge_config_sha256: str

    def __post_init__(self) -> None:
        validate_v6_control_binding(self)

    def to_config(self) -> dict[str, object]:
        validated = validate_v6_control_binding(self)
        return {
            "schema": validated.binding_schema,
            "development_only": DEVELOPMENT_ONLY,
            "execution_authorized": EXECUTION_AUTHORIZED,
            "evidence_authorized": EVIDENCE_AUTHORIZED,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "family": validated.family,
            "name": validated.name,
            "control_schema": validated.control_schema,
            "bridge_schema": validated.bridge_schema,
            "initial_external_action": validated.initial_external_action,
            "control_config_sha256": validated.control_config_sha256,
            "bridge_config_sha256": validated.bridge_config_sha256,
        }


def validate_v6_control_binding(
    binding: HiddenPartnerLifecycleWorldV6ControlBinding,
) -> HiddenPartnerLifecycleWorldV6ControlBinding:
    """Validate exact local types and version domains for one binding record."""

    if type(binding) is not HiddenPartnerLifecycleWorldV6ControlBinding:
        raise TypeError(
            "binding must be an exact HiddenPartnerLifecycleWorldV6ControlBinding"
        )
    for field in (
        "binding_schema",
        "family",
        "name",
        "control_schema",
        "bridge_schema",
    ):
        if type(getattr(binding, field)) is not str:
            raise TypeError(f"binding.{field} must be an exact built-in str")
    if binding.binding_schema != HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_BINDING_SCHEMA:
        raise ValueError("binding schema differs from the current v6 binding schema")
    if binding.control_schema != HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA:
        raise ValueError("binding control schema differs from the current v6 control schema")
    if binding.bridge_schema != HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA:
        raise ValueError("binding bridge schema differs from the current bridge schema")
    if binding.family not in ("primary", "diagnostic"):
        raise ValueError("binding family must be primary or diagnostic")
    allowed = PRIMARY_CONDITION_ORDER if binding.family == "primary" else V6_DIAGNOSTIC_ORDER
    if binding.name not in allowed:
        raise ValueError("binding name is not in its declared v6 family")
    if type(binding.initial_external_action) is not int:
        raise TypeError("binding.initial_external_action must be an exact built-in int")
    if binding.initial_external_action != 0:
        raise ValueError("v6 binding initial_external_action must be canonical zero")
    _require_exact_sha256(
        binding.control_config_sha256,
        name="binding.control_config_sha256",
    )
    _require_exact_sha256(
        binding.bridge_config_sha256,
        name="binding.bridge_config_sha256",
    )
    return binding


def _build_v6_control_binding(
    control: HiddenPartnerLifecycleWorldV6Control,
    *,
    family: ControlFamily,
) -> HiddenPartnerLifecycleWorldV6ControlBinding:
    validated = validate_v6_control(control)
    if validated.agent_config is None or not validated.execution_ready:
        raise ValueError(f"v6 control {validated.name!r} is not bridge-constructible")
    bridge = HiddenPartnerWorldOnlineBridge(
        world=HiddenPartnerWorldFeedbackWorld(validated.world_config),
        agent=IntegratedHiddenPartnerAgent(validated.agent_config),
        focal_action_policy=validated.focal_action_policy,
        initial_external_action=validated.initial_external_action,
    )
    bridge_config = bridge.to_config()
    if bridge_config.get("schema") != HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA:
        raise RuntimeError("constructed v6 bridge schema differs from its source constant")
    return HiddenPartnerLifecycleWorldV6ControlBinding(
        binding_schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_BINDING_SCHEMA,
        family=family,
        name=validated.name,
        control_schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA,
        bridge_schema=HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA,
        initial_external_action=validated.initial_external_action,
        control_config_sha256=hashlib.sha256(
            _canonical_json_bytes(validated.to_config())
        ).hexdigest(),
        bridge_config_sha256=hashlib.sha256(
            _canonical_json_bytes(bridge_config)
        ).hexdigest(),
    )


def build_v6_control_bindings() -> tuple[HiddenPartnerLifecycleWorldV6ControlBinding, ...]:
    """Construct and hash every currently ready live composition without executing it."""

    records: list[HiddenPartnerLifecycleWorldV6ControlBinding] = []
    for family, controls in (
        ("primary", build_v6_primary_controls()),
        ("diagnostic", build_v6_diagnostic_controls()),
    ):
        for control in controls:
            if control.execution_ready:
                records.append(
                    _build_v6_control_binding(
                        control,
                        family=cast(ControlFamily, family),
                    )
                )
    return tuple(records)


def build_v6_design_successor_mapping_config() -> dict[str, object]:
    """Map frozen historical prerequisites to current bindings without editing v6."""

    return {
        "schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_DESIGN_SUCCESSOR_SCHEMA,
        "frozen_design_schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCHEMA,
        "frozen_open_prerequisites_are_historical": True,
        "development_only": DEVELOPMENT_ONLY,
        "execution_authorized": EXECUTION_AUTHORIZED,
        "evidence_authorized": EVIDENCE_AUTHORIZED,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "mechanism_bindings_now_closed": [
            {"identifier": identifier, "source_binding": source_binding}
            for identifier, source_binding in V6_DESIGN_SUCCESSOR_CLOSED_BINDINGS
        ],
        "remaining_open_prerequisites": [
            {"identifier": identifier, "requirement": requirement}
            for identifier, requirement in V6_DESIGN_SUCCESSOR_REMAINING_OPEN
        ],
    }


def build_v6_control_matrix_config() -> dict[str, object]:
    """Return the complete JSON-compatible, nonauthorizing control matrix."""

    primary = build_v6_primary_controls()
    diagnostics = build_v6_diagnostic_controls()
    primary_ready = sum(control.execution_ready for control in primary)
    diagnostic_ready = sum(control.execution_ready for control in diagnostics)
    all_ready = primary_ready == len(primary) and diagnostic_ready == len(diagnostics)
    return {
        "schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA,
        "status": (
            "DEVELOPMENT_CONTROLS_READY_FOR_RUNNER_BINDING"
            if all_ready
            else "DEVELOPMENT_CONTROLS_PARTIALLY_BLOCKED"
        ),
        "development_only": DEVELOPMENT_ONLY,
        "execution_authorized": EXECUTION_AUTHORIZED,
        "evidence_authorized": EVIDENCE_AUTHORIZED,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "seed_namespace": None,
        "thresholds": None,
        "outcomes": None,
        "primary_ready_count": primary_ready,
        "primary_required_count": len(primary),
        "diagnostic_ready_count": diagnostic_ready,
        "diagnostic_required_count": len(diagnostics),
        "all_controls_execution_ready": all_ready,
        "binding_contract": {
            "schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_BINDING_SCHEMA,
            "bridge_schema": HIDDEN_PARTNER_WORLD_ONLINE_BRIDGE_CONFIG_SCHEMA,
            "initial_external_action": 0,
        },
        "design_successor": build_v6_design_successor_mapping_config(),
        "target_head_order": list(V6_TARGET_HEAD_ORDER),
        "representation_loss_weights": list(V6_REPRESENTATION_LOSS_WEIGHTS),
        "primary_controls": [control.to_config() for control in primary],
        "diagnostic_controls": [control.to_config() for control in diagnostics],
    }


def canonical_v6_control_matrix_json() -> str:
    """Serialize the current development matrix without freezing its contents."""

    return _canonical_json_bytes(build_v6_control_matrix_config()).decode("ascii")


def canonical_v6_control_matrix_sha256() -> str:
    """Hash the complete canonical matrix so readiness detects same-count drift."""

    return hashlib.sha256(canonical_v6_control_matrix_json().encode("ascii")).hexdigest()


__all__ = [
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_AUTHORIZED",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_BINDING_SCHEMA",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_DESIGN_SUCCESSOR_SCHEMA",
    "HiddenPartnerLifecycleWorldV6Control",
    "HiddenPartnerLifecycleWorldV6ControlBinding",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "V6_DIAGNOSTIC_ORDER",
    "V6_DESIGN_SUCCESSOR_CLOSED_BINDINGS",
    "V6_DESIGN_SUCCESSOR_REMAINING_OPEN",
    "V6_MANIFEST_OVERRIDE_PATH_MAP",
    "V6_REPRESENTATION_LOSS_WEIGHTS",
    "V6_TARGET_HEAD_ORDER",
    "build_v6_control_bindings",
    "build_v6_control_matrix_config",
    "build_v6_design_successor_mapping_config",
    "build_v6_diagnostic_controls",
    "build_v6_full_agent_config",
    "build_v6_primary_controls",
    "canonical_v6_control_matrix_json",
    "canonical_v6_control_matrix_sha256",
    "validate_v6_control",
    "validate_v6_control_binding",
]
