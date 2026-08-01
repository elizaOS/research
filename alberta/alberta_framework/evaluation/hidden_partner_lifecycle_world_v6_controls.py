"""Nonexecuting development controls for the noisy-world v6 lifecycle design.

This module binds the mechanism configurations that already have exact source
contracts.  It does not create keys, choose seeds, construct a runner, write an
artifact, set thresholds, or authorize scientific promotion.  A control that
cannot yet be implemented with exact matched-compute semantics remains
explicitly blocked instead of being replaced by a superficially similar arm.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Literal

from alberta_framework.core.grounded_joint_world_model import (
    GroundedJointWorldModelConfig,
)
from alberta_framework.core.integrated_hidden_partner import (
    DEPLOYED_FEATURE_DIM,
    N_ACTIONS,
    RAW_OBSERVATION_DIM,
    IntegratedHiddenPartnerConfig,
)
from alberta_framework.core.representation_gradient_mixer import (
    GradientMixMode,
    RepresentationGradientMixerConfig,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    PRIMARY_CONDITION_ORDER,
    V6_INITIAL_ACTIVE_DESCRIPTORS,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    OBSERVATION_FIELDS,
    HiddenPartnerWorldFeedbackConfig,
)

HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.controls-development.v2"
)
DEVELOPMENT_ONLY = True
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

_UNIFORM_ACTION_BLOCKER = (
    "the bridge does not yet provide a cache-safe exactly balanced external "
    "focal-action intervention"
)


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


@dataclasses.dataclass(frozen=True)
class HiddenPartnerLifecycleWorldV6Control:
    """One nonexecuting primary or diagnostic mechanism binding."""

    name: str
    primary: bool
    agent_config: IntegratedHiddenPartnerConfig | None
    world_config: HiddenPartnerWorldFeedbackConfig
    focal_action_policy: FocalActionPolicy = "agent"
    execution_blocker: str | None = None

    def __post_init__(self) -> None:
        allowed = PRIMARY_CONDITION_ORDER if self.primary else V6_DIAGNOSTIC_ORDER
        if self.name not in allowed:
            raise ValueError("control name is not in the declared v6 order")
        if self.agent_config is not None and not isinstance(
            self.agent_config, IntegratedHiddenPartnerConfig
        ):
            raise TypeError("agent_config must be an IntegratedHiddenPartnerConfig or None")
        if not isinstance(self.world_config, HiddenPartnerWorldFeedbackConfig):
            raise TypeError("world_config must be a HiddenPartnerWorldFeedbackConfig")
        if self.focal_action_policy not in ("agent", "balanced_external"):
            raise ValueError("focal_action_policy must be agent or balanced_external")
        if self.execution_blocker is not None and (
            not isinstance(self.execution_blocker, str) or not self.execution_blocker
        ):
            raise ValueError("execution_blocker must be a non-empty string or None")
        if self.agent_config is None and self.execution_blocker is None:
            raise ValueError("a missing agent config must have an explicit blocker")

    @property
    def execution_ready(self) -> bool:
        return self.agent_config is not None and self.execution_blocker is None

    def to_config(self) -> dict[str, object]:
        return {
            "schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA,
            "development_only": DEVELOPMENT_ONLY,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "name": self.name,
            "primary": self.primary,
            "status": "READY_FOR_RUNNER_BINDING" if self.execution_ready else "BLOCKED",
            "execution_ready": self.execution_ready,
            "execution_blocker": self.execution_blocker,
            "focal_action_policy": self.focal_action_policy,
            "world_config": self.world_config.to_config(),
            "agent_config": (
                None if self.agent_config is None else self.agent_config.to_config()
            ),
        }


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
    """Bind required diagnostics without pretending the missing action hook exists."""

    full = build_v6_full_agent_config()
    grounded = full.grounded_world_model
    if grounded is None:
        raise ValueError("v6 diagnostics require the grounded model")
    return (
        HiddenPartnerLifecycleWorldV6Control(
            name="uniform_action",
            primary=False,
            agent_config=full,
            world_config=HiddenPartnerWorldFeedbackConfig(),
            focal_action_policy="balanced_external",
            execution_blocker=_UNIFORM_ACTION_BLOCKER,
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


def build_v6_control_matrix_config() -> dict[str, object]:
    """Return the complete JSON-compatible, nonauthorizing control matrix."""

    primary = build_v6_primary_controls()
    diagnostics = build_v6_diagnostic_controls()
    primary_ready = sum(control.execution_ready for control in primary)
    diagnostic_ready = sum(control.execution_ready for control in diagnostics)
    return {
        "schema": HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA,
        "status": "DEVELOPMENT_CONTROLS_PARTIALLY_BLOCKED",
        "development_only": DEVELOPMENT_ONLY,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "seed_namespace": None,
        "thresholds": None,
        "outcomes": None,
        "primary_ready_count": primary_ready,
        "primary_required_count": len(primary),
        "diagnostic_ready_count": diagnostic_ready,
        "diagnostic_required_count": len(diagnostics),
        "all_controls_execution_ready": (
            primary_ready == len(primary) and diagnostic_ready == len(diagnostics)
        ),
        "target_head_order": list(V6_TARGET_HEAD_ORDER),
        "representation_loss_weights": list(V6_REPRESENTATION_LOSS_WEIGHTS),
        "primary_controls": [control.to_config() for control in primary],
        "diagnostic_controls": [control.to_config() for control in diagnostics],
    }


def canonical_v6_control_matrix_json() -> str:
    """Serialize the current development matrix without freezing its contents."""

    return json.dumps(
        build_v6_control_matrix_config(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "DEVELOPMENT_ONLY",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_SCHEMA",
    "HiddenPartnerLifecycleWorldV6Control",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "V6_DIAGNOSTIC_ORDER",
    "V6_REPRESENTATION_LOSS_WEIGHTS",
    "V6_TARGET_HEAD_ORDER",
    "build_v6_control_matrix_config",
    "build_v6_diagnostic_controls",
    "build_v6_full_agent_config",
    "build_v6_primary_controls",
    "canonical_v6_control_matrix_json",
]
