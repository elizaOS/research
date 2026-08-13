# mypy: disable-error-code="attr-defined,call-arg,no-any-return,unused-ignore"
"""End-to-end Alberta Plan Step 1-4 pipeline glue.

The production pipeline composes the existing packaged pieces conservatively:

1. Step 1 enters through the adaptive optimizers used by later learners.
2. Step 2 supplies feature augmentation in one of four modes: the lightweight
   temporal-context featurizer, the promoted nonlinear UPGD learner (whose
   penultimate hidden activations become the feature vector for downstream
   Step 3 and Step 4 learners), the associative-memory learner (whose
   next-token probability vector becomes the features), or raw identity
   passthrough.
3. Step 3 learns GVF/Horde predictions on those features. Cumulants are
   either supplied through a caller-provided callable or fall back to the
   observation-channel cumulant function used by the legacy smoke API.
4. Step 4 learns control on the same features, either as discrete SARSA
   (default) or as a Horde-backed actor-critic (``HordeActorCriticAgent``).

The API is intentionally narrow and transition-oriented.  It is suitable for
daemon smoke tests, downstream integration probes, and checkpointed online
state, while research-scale experiments should continue to use their dedicated
runners.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from alberta_framework.core.associative_memory import (
    AssociativeFeatureFamily,
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
    AssociativeMemoryState,
)
from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.horde import HordeLearner, HordeUpdateResult
from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticConfig,
    HordeActorCriticState,
    HordeActorCriticUpdateResult,
)
from alberta_framework.core.multi_head_learner import MultiHeadMLPState
from alberta_framework.core.optimizers import ObGDBounding
from alberta_framework.core.sarsa import SARSAState, SARSAUpdateResult
from alberta_framework.core.temporal_context import (
    TemporalContextConfig,
    TemporalContextFeaturizer,
    TemporalContextState,
    TemporalContextStepResult,
)
from alberta_framework.core.upgd import UPGDLearner, UPGDState
from alberta_framework.steps.step3 import (
    Step3HordeConfig,
    init_step3_state,
    make_step3_horde,
    step3_predict,
)
from alberta_framework.steps.step4 import (
    Step4SARSAConfig,
    init_step4_state,
    make_step4_sarsa_agent,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

ALBERTA_PIPELINE_CONFIG_SCHEMA = "alberta.step1-4-pipeline.config.v2"
ALBERTA_PIPELINE_STATE_SCHEMA = "alberta.step1-4-pipeline.state.v2"
ALBERTA_PIPELINE_CHECKPOINT_SCHEMA = "alberta.step1-4-pipeline.checkpoint.v2"
_LEGACY_ALBERTA_PIPELINE_CHECKPOINT_SCHEMA = (
    "alberta.step1-4-pipeline.checkpoint.v1"
)
ALBERTA_PIPELINE_LIFETIME_COUNTER_NBYTES = 12
ALBERTA_PIPELINE_LIFETIME_COUNTER_DELTA_NBYTES = 8

PIPELINE_REJECTION_NONE = 0
PIPELINE_REJECTION_STATE_INVALID = 1
PIPELINE_REJECTION_LIFETIME_EXHAUSTED = 2
PIPELINE_REJECTION_SOURCE_INVALID = 3
PIPELINE_REJECTION_STEP2_UNAVAILABLE = 4
PIPELINE_REJECTION_STEP3_REFUSED = 5
PIPELINE_REJECTION_CONTROL_REFUSED = 6
PIPELINE_REJECTION_CHILD_MISALIGNED = 7
PIPELINE_REJECTION_CANDIDATE_INVALID = 8
PIPELINE_REJECTION_REASON_NAMES = (
    "none",
    "state_invalid",
    "lifetime_exhausted",
    "source_invalid",
    "step2_unavailable",
    "step3_refused",
    "control_refused",
    "child_misaligned",
    "candidate_invalid",
)


def _checked_step_words_increment(step_words: Array) -> tuple[Array, Array]:
    """Return the next exact uint64 identity and whether it exists."""

    if getattr(step_words, "shape", None) != (2,):
        raise ValueError("pipeline step_words must have shape (2,)")
    if getattr(step_words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("pipeline step_words must have dtype uint32")
    maximum = jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(step_words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = step_words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    candidate = jnp.stack((step_words[0] + carry, low))
    return (
        jnp.where(capacity_available, candidate, step_words).astype(jnp.uint32),
        capacity_available,
    )


def _saturating_count_from_words(step_words: Array) -> Array:
    """Return int32 compatibility telemetry authenticated by exact words."""

    saturated = (step_words[0] != jnp.asarray(0, dtype=jnp.uint32)) | (
        step_words[1] >= jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        saturated,
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words[1].astype(jnp.int32),
    )


def _lifetime_counter_valid(step_words: Array, step_count: Array) -> Array:
    """Validate exact identity shape/dtype and its saturating telemetry."""

    _checked_step_words_increment(step_words)
    if getattr(step_count, "shape", None) != ():
        raise ValueError("pipeline step_count must be scalar")
    if getattr(step_count, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("pipeline step_count must have dtype int32")
    return step_count == _saturating_count_from_words(step_words)


def _tree_arrays_finite(tree: Any) -> Array:
    """Return whether every persistent floating/complex array is finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def _host_field_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Return a strict shallow mapping for explicit host migration."""

    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: getattr(value, item.name)
            for item in dataclasses.fields(value)
        }
    raise TypeError(f"{name} must be a mapping or dataclass")

Step2Mode = Literal["temporal_context", "upgd", "associative", "identity"]
Step2UPGDPreset = Literal["default", "strict_digit_readout"]
Step2UPGDReadoutMode = Literal[
    "linear_mse",
    "softmax_ce",
    "adaptive_simplex",
    "factorized_simplex",
    "adaptive_factorized_simplex",
    "two_timescale_simplex",
]
ControlMode = Literal["sarsa", "horde_ac"]

CumulantFn = Callable[[Array, Array, Array], Array]
"""Caller-supplied cumulant function.

Signature: ``(observation, reward, terminated) -> Array(n_demons,)``.
"""


@dataclass(frozen=True)
class Step2FeatureConfig:
    """Config for the lightweight temporal-context Step 2 layer.

    This is the historical "raw + EMA + delta + phase products" featurizer
    retained for back-compatibility. New deployments should consider
    :class:`Step2UPGDConfig` for the promoted nonlinear Step 2 path.
    """

    observation_dim: int = 4
    include_raw: bool = True
    include_ema: bool = True
    include_delta: bool = True
    include_phase_products: bool = False
    ema_decay: float = 0.95
    periods: tuple[float, ...] = (32.0, 64.0)

    def __post_init__(self) -> None:
        """Validate observation and feature settings."""
        if self.observation_dim < 1:
            msg = f"observation_dim must be positive, got {self.observation_dim}"
            raise ValueError(msg)
        if not (self.include_raw or self.include_ema or self.include_delta):
            msg = "at least one of include_raw/include_ema/include_delta is required"
            raise ValueError(msg)
        if not 0.0 <= self.ema_decay < 1.0:
            msg = f"ema_decay must be in [0, 1), got {self.ema_decay}"
            raise ValueError(msg)
        if any(period <= 0.0 for period in self.periods):
            msg = "all periods must be positive"
            raise ValueError(msg)

    @classmethod
    def identity(cls, observation_dim: int) -> Step2FeatureConfig:
        """Return a raw-observation feature config."""
        return cls(
            observation_dim=observation_dim,
            include_raw=True,
            include_ema=False,
            include_delta=False,
            periods=(),
        )

    def to_temporal_context_config(self) -> TemporalContextConfig:
        """Return the core Step 2 featurizer config."""
        return TemporalContextConfig(
            input_dim=self.observation_dim,
            include_raw=self.include_raw,
            include_ema=self.include_ema,
            include_delta=self.include_delta,
            include_phase_products=self.include_phase_products,
            ema_decay=self.ema_decay,
            periods=self.periods,
        )

    def output_dim(self) -> int:
        """Return the Step 2 feature dimensionality."""
        return self.to_temporal_context_config().output_dim()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["periods"] = list(self.periods)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step2FeatureConfig:
        """Reconstruct from :meth:`to_dict` output."""
        config = dict(payload)
        config["periods"] = tuple(cast(list[float], config.get("periods", [])))
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class Step2UPGDConfig:
    """Config for the promoted UPGD-backed Step 2 featurizer.

    The UPGD learner's penultimate hidden activations are exposed as the
    feature vector for downstream Step 3 and Step 4 learners. The number of
    UPGD heads is configurable; supervised targets may optionally be passed
    through :meth:`AlbertaPipeline.update` to drive UPGD learning. When no
    targets are supplied, callers may still use :meth:`AlbertaPipeline.predict`
    as a representation extractor, but an atomic pipeline *update* refuses:
    there is no authenticated Step 2 learning event to advance alongside
    Steps 3 and 4. Explicit all-NaN targets mean an intentional inactive-head
    UPGD event and retain UPGD's own perturbation semantics.
    """

    observation_dim: int = 4
    n_heads: int = 1
    hidden_sizes: tuple[int, ...] = (32,)
    step_size: float = 0.03
    sparsity: float = 0.5
    use_layer_norm: bool = True
    learner_preset: Step2UPGDPreset = "default"
    loss_normalization: Literal["target_structure", "target_density"] = (
        "target_structure"
    )
    readout_mode: Step2UPGDReadoutMode = "linear_mse"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.observation_dim < 1:
            msg = f"observation_dim must be positive, got {self.observation_dim}"
            raise ValueError(msg)
        if self.n_heads < 1:
            msg = f"n_heads must be positive, got {self.n_heads}"
            raise ValueError(msg)
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            msg = (
                "hidden_sizes must contain at least one positive size, "
                f"got {self.hidden_sizes!r}"
            )
            raise ValueError(msg)
        if self.step_size < 0.0:
            msg = f"step_size must be non-negative, got {self.step_size}"
            raise ValueError(msg)
        if not 0.0 <= self.sparsity <= 1.0:
            msg = f"sparsity must be in [0, 1], got {self.sparsity}"
            raise ValueError(msg)
        if self.learner_preset not in ("default", "strict_digit_readout"):
            msg = f"unknown learner_preset {self.learner_preset!r}"
            raise ValueError(msg)
        if self.loss_normalization not in ("target_structure", "target_density"):
            msg = f"unknown loss_normalization {self.loss_normalization!r}"
            raise ValueError(msg)
        valid_readouts = (
            "linear_mse",
            "softmax_ce",
            "adaptive_simplex",
            "factorized_simplex",
            "adaptive_factorized_simplex",
            "two_timescale_simplex",
        )
        if self.readout_mode not in valid_readouts:
            msg = f"unknown readout_mode {self.readout_mode!r}"
            raise ValueError(msg)
        if self.learner_preset == "strict_digit_readout" and (
            self.loss_normalization != "target_structure"
            or self.readout_mode != "two_timescale_simplex"
        ):
            msg = (
                "strict_digit_readout preset requires "
                "loss_normalization='target_structure' and "
                "readout_mode='two_timescale_simplex'"
            )
            raise ValueError(msg)
        if self.learner_preset == "strict_digit_readout" and (
            self.sparsity != 0.5 or not self.use_layer_norm
        ):
            msg = (
                "strict_digit_readout preset owns sparsity/use_layer_norm; "
                "use sparsity=0.5 and use_layer_norm=True"
            )
            raise ValueError(msg)

    @classmethod
    def strict_digit_readout(
        cls,
        *,
        observation_dim: int = 64,
        n_heads: int = 10,
        hidden_sizes: tuple[int, ...] = (64, 64),
        step_size: float = 0.018,
    ) -> Step2UPGDConfig:
        """Return the promoted strict digit/readout Step 2 config."""
        return cls(
            observation_dim=observation_dim,
            n_heads=n_heads,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            learner_preset="strict_digit_readout",
            readout_mode="two_timescale_simplex",
        )

    def output_dim(self) -> int:
        """Penultimate-layer dimensionality used as features."""
        return self.hidden_sizes[-1]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step2UPGDConfig:
        """Reconstruct from :meth:`to_dict` output."""
        config = dict(payload)
        config["hidden_sizes"] = tuple(cast(list[int], config["hidden_sizes"]))
        return cls(**cast(Any, config))


@dataclass(frozen=True)
class Step2AssociativePipelineConfig:
    """Config for associative Step 2 features in the end-to-end pipeline."""

    vocab_size: int = 16
    block_size: int = 8
    suffix_length: int = 4
    feature_family: AssociativeFeatureFamily = "token_suffix_pair"
    max_features: int = 512
    write_lr: float = 1.0
    retention: float = 0.80
    utility_lr: float = 0.10
    utility_decay: float = 0.995
    min_weight: float = 0.02
    max_weight: float = 8.0
    logit_scale: float = 4.0
    normalize_by_weight: bool = True
    adaptive_feature_family: bool = False
    adaptive_window: bool = False
    adaptive_budget: bool = False
    scope_lr: float = 0.05
    budget_lr: float = 0.05
    initial_budget_fraction: float = 0.5
    min_effective_budget: int = 1
    scope_logit_clip: float = 8.0

    def __post_init__(self) -> None:
        """Validate integer context settings."""
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.suffix_length < 2 or self.suffix_length > self.block_size:
            raise ValueError("suffix_length must be in [2, block_size]")
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.scope_lr < 0.0:
            raise ValueError("scope_lr must be non-negative")
        if self.budget_lr < 0.0:
            raise ValueError("budget_lr must be non-negative")
        if not 0.0 < self.initial_budget_fraction <= 1.0:
            raise ValueError("initial_budget_fraction must be in (0, 1]")
        if self.min_effective_budget < 1:
            raise ValueError("min_effective_budget must be positive")
        if self.min_effective_budget > self.max_features:
            raise ValueError("min_effective_budget must be <= max_features")
        if self.scope_logit_clip <= 0.0:
            raise ValueError("scope_logit_clip must be positive")

    def output_dim(self) -> int:
        """Return the associative probability-vector dimensionality."""
        return self.vocab_size

    def to_core_config(self) -> AssociativeMemoryConfig:
        """Return the core associative memory config."""
        return AssociativeMemoryConfig(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            suffix_length=self.suffix_length,
            feature_family=self.feature_family,
            max_features=self.max_features,
            write_lr=self.write_lr,
            retention=self.retention,
            utility_lr=self.utility_lr,
            utility_decay=self.utility_decay,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            logit_scale=self.logit_scale,
            normalize_by_weight=self.normalize_by_weight,
            adaptive_feature_family=self.adaptive_feature_family,
            adaptive_window=self.adaptive_window,
            adaptive_budget=self.adaptive_budget,
            scope_lr=self.scope_lr,
            budget_lr=self.budget_lr,
            initial_budget_fraction=self.initial_budget_fraction,
            min_effective_budget=self.min_effective_budget,
            scope_logit_clip=self.scope_logit_clip,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> Step2AssociativePipelineConfig:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**cast(Any, payload))


@dataclass(frozen=True)
class HordeActorCriticPipelineConfig:
    """Config wrapper for the Horde actor-critic Step 4 control."""

    n_actions: int = 2
    actor_step_size: float = 0.01
    actor_lamda: float = 0.9
    temperature: float = 1.0
    value_head_index: int = 0
    actor_obgd_kappa: float | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_actions < 1:
            msg = f"n_actions must be positive, got {self.n_actions}"
            raise ValueError(msg)
        if self.actor_step_size < 0.0:
            msg = f"actor_step_size must be non-negative, got {self.actor_step_size}"
            raise ValueError(msg)
        if not 0.0 <= self.actor_lamda <= 1.0:
            msg = f"actor_lamda must be in [0, 1], got {self.actor_lamda}"
            raise ValueError(msg)
        if self.temperature <= 0.0:
            msg = f"temperature must be positive, got {self.temperature}"
            raise ValueError(msg)
        if self.value_head_index < 0:
            msg = (
                "value_head_index must be non-negative, "
                f"got {self.value_head_index}"
            )
            raise ValueError(msg)

    def to_horde_actor_critic_config(self) -> HordeActorCriticConfig:
        """Return the core actor-critic config."""
        return HordeActorCriticConfig(
            n_actions=self.n_actions,
            actor_step_size=self.actor_step_size,
            actor_lamda=self.actor_lamda,
            temperature=self.temperature,
            value_head_index=self.value_head_index,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> HordeActorCriticPipelineConfig:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**cast(Any, payload))


@dataclass(frozen=True)
class AlbertaPipelineConfig:
    """Config for the Step 1-4 production pipeline.

    The ``step2`` and ``control`` fields select which Step 2 featurizer and
    Step 4 control mode the pipeline runs. Defaults preserve the legacy
    behavior (temporal-context features + SARSA control); set ``step2="upgd"``
    or ``control="horde_ac"`` to opt into the integrated Step 2/Step 4
    components.
    """

    features: Step2FeatureConfig = field(default_factory=Step2FeatureConfig)
    upgd: Step2UPGDConfig | None = None
    associative: Step2AssociativePipelineConfig | None = None
    horde: Step3HordeConfig = field(default_factory=Step3HordeConfig)
    control: Step4SARSAConfig = field(default_factory=Step4SARSAConfig)
    horde_ac: HordeActorCriticPipelineConfig | None = None
    step2: Step2Mode = "temporal_context"
    control_mode: ControlMode = "sarsa"

    def __post_init__(self) -> None:
        """Validate combinations of step2/control and required sub-configs."""
        if self.step2 not in ("temporal_context", "upgd", "associative", "identity"):
            msg = f"unknown step2 mode {self.step2!r}"
            raise ValueError(msg)
        if self.control_mode not in ("sarsa", "horde_ac"):
            msg = f"unknown control_mode {self.control_mode!r}"
            raise ValueError(msg)
        if self.step2 == "upgd" and self.upgd is None:
            msg = "upgd config is required when step2='upgd'"
            raise ValueError(msg)
        if self.step2 == "associative" and self.associative is None:
            msg = "associative config is required when step2='associative'"
            raise ValueError(msg)
        if self.control_mode == "horde_ac" and self.horde_ac is None:
            msg = "horde_ac config is required when control_mode='horde_ac'"
            raise ValueError(msg)
        if self.control_mode == "horde_ac":
            ac = cast(HordeActorCriticPipelineConfig, self.horde_ac)
            if ac.value_head_index >= self.horde.n_demons:
                msg = (
                    "horde_ac.value_head_index must reference an existing "
                    f"horde demon (got {ac.value_head_index}, n_demons="
                    f"{self.horde.n_demons})"
                )
                raise ValueError(msg)

    def feature_dim(self) -> int:
        """Return the feature dimensionality passed to Step 3 and Step 4."""
        if self.step2 == "upgd":
            return cast(Step2UPGDConfig, self.upgd).output_dim()
        if self.step2 == "associative":
            return cast(Step2AssociativePipelineConfig, self.associative).output_dim()
        if self.step2 == "identity":
            return self.features.observation_dim
        return self.features.output_dim()

    def to_dict(self) -> dict[str, object]:
        """Return the strict v2 JSON-serializable representation."""
        return {
            "type": "AlbertaPipelineConfig",
            "schema": ALBERTA_PIPELINE_CONFIG_SCHEMA,
            "state_schema": ALBERTA_PIPELINE_STATE_SCHEMA,
            "features": self.features.to_dict(),
            "upgd": self.upgd.to_dict() if self.upgd is not None else None,
            "associative": (
                self.associative.to_dict() if self.associative is not None else None
            ),
            "horde": self.horde.to_dict(),
            "control": self.control.to_dict(),
            "horde_ac": (
                self.horde_ac.to_dict() if self.horde_ac is not None else None
            ),
            "step2": self.step2,
            "control_mode": self.control_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> AlbertaPipelineConfig:
        """Strictly reconstruct a v2 pipeline configuration."""
        if type(payload) is not dict:
            raise TypeError("pipeline config must be an exact dict")
        expected = {
            "type",
            "schema",
            "state_schema",
            "features",
            "upgd",
            "associative",
            "horde",
            "control",
            "horde_ac",
            "step2",
            "control_mode",
        }
        if set(payload) != expected:
            if "schema" not in payload:
                raise ValueError(
                    "legacy pipeline config requires explicit migration"
                )
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ValueError(
                "pipeline config fields do not match v2; "
                f"missing={missing}, extra={extra}"
            )
        if payload["type"] != "AlbertaPipelineConfig":
            raise ValueError("unexpected pipeline config type")
        if payload["schema"] != ALBERTA_PIPELINE_CONFIG_SCHEMA:
            raise ValueError("pipeline config schema is unsupported")
        if payload["state_schema"] != ALBERTA_PIPELINE_STATE_SCHEMA:
            raise ValueError("pipeline state schema is unsupported")
        upgd_payload = payload["upgd"]
        associative_payload = payload["associative"]
        horde_ac_payload = payload["horde_ac"]
        return cls(
            features=Step2FeatureConfig.from_dict(
                cast(dict[str, object], payload["features"])
            ),
            upgd=Step2UPGDConfig.from_dict(cast(dict[str, object], upgd_payload))
            if upgd_payload is not None
            else None,
            associative=Step2AssociativePipelineConfig.from_dict(
                cast(dict[str, object], associative_payload)
            )
            if associative_payload is not None
            else None,
            horde=Step3HordeConfig.from_dict(
                cast(dict[str, object], payload["horde"])
            ),
            control=Step4SARSAConfig.from_dict(
                cast(dict[str, object], payload["control"])
            ),
            horde_ac=HordeActorCriticPipelineConfig.from_dict(
                cast(dict[str, object], horde_ac_payload)
            )
            if horde_ac_payload is not None
            else None,
            step2=cast(Step2Mode, payload["step2"]),
            control_mode=cast(ControlMode, payload["control_mode"]),
        )


def migrate_legacy_alberta_pipeline_config(
    payload: Mapping[str, object],
) -> AlbertaPipelineConfig:
    """Explicitly migrate the pre-v2 unversioned pipeline config."""

    fields = dict(payload)
    expected = {
        "features",
        "upgd",
        "associative",
        "horde",
        "control",
        "horde_ac",
        "step2",
        "control_mode",
    }
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise ValueError(
            "legacy pipeline config fields are unsupported; "
            f"missing={missing}, extra={extra}"
        )
    fields.update(
        {
            "type": "AlbertaPipelineConfig",
            "schema": ALBERTA_PIPELINE_CONFIG_SCHEMA,
            "state_schema": ALBERTA_PIPELINE_STATE_SCHEMA,
        }
    )
    return AlbertaPipelineConfig.from_dict(fields)


@chex.dataclass(frozen=True)
class AlbertaPipelineState:
    """Checkpoint-friendly immutable state for the Step 1-4 pipeline.

    ``feature_state`` stores the temporal-context state when ``step2`` is
    ``"temporal_context"``; otherwise it is None. ``upgd_state`` stores the
    UPGD learner state when ``step2`` is ``"upgd"``; otherwise it is None.
    ``associative_state`` stores the associative-memory state when ``step2``
    is ``"associative"``; otherwise it is None. ``control_state`` is either
    a SARSA state or a HordeActorCritic state depending on ``control_mode``.
    ``step_words`` is the authoritative big-endian uint32 pair for the finite
    pipeline event lifetime; ``step_count`` is saturating int32 telemetry.
    """

    feature_state: TemporalContextState | None
    upgd_state: UPGDState | None
    associative_state: AssociativeMemoryState | None
    horde_state: MultiHeadMLPState
    control_state: SARSAState | HordeActorCriticState
    last_features: Array
    step_count: Array
    step_words: Array


@chex.dataclass(frozen=True)
class AlbertaPipelineStepResult:
    """Result from one end-to-end transition update.

    ``q_values`` carries Q-values when ``control_mode == "sarsa"`` and the
    softmax policy when ``control_mode == "horde_ac"``. The ``action`` field
    is the action selected/sampled at the new observation. On refusal the
    complete state is the input state and emitted learning values are safe
    zeros; ``rejection_reason`` indexes
    :data:`PIPELINE_REJECTION_REASON_NAMES`, while the availability/applied
    booleans preserve the child-level verdicts that led to it.
    """

    state: AlbertaPipelineState
    features: Array
    horde_predictions: Array
    horde_td_errors: Array
    horde_td_targets: Array
    q_values: Array
    action: Array
    control_td_error: Array
    reward: Array
    pre_step_words: Array
    post_step_words: Array
    lifetime_counter_valid: Array
    lifetime_capacity_available: Array
    source_valid: Array
    state_valid: Array
    step2_contract_available: Array
    step2_update_applied: Array
    step3_contract_available: Array
    step3_update_applied: Array
    control_contract_available: Array
    control_update_applied: Array
    children_pre_aligned: Array
    children_post_aligned: Array
    candidate_state_valid: Array
    update_applied: Array
    update_rejected: Array
    rejection_reason: Array


@chex.dataclass(frozen=True)
class AlbertaPipelineArrayResult:
    """Result from scanning the end-to-end pipeline over arrays."""

    state: AlbertaPipelineState
    features: Array
    horde_predictions: Array
    horde_td_errors: Array
    q_values: Array
    actions: Array
    control_td_errors: Array
    update_applied: Array
    rejection_reasons: Array


@dataclass(frozen=True)
class AlbertaPipelineResourceBudget:
    """Exact persistent-array accounting for one pipeline state."""

    persistent_state_nbytes: int
    exact_pipeline_identity_nbytes: int = 8
    compatibility_telemetry_nbytes: int = 4

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable resource declaration."""

        return asdict(self)


def measure_alberta_pipeline_state_nbytes(state: AlbertaPipelineState) -> int:
    """Measure persistent JAX-array bytes in a concrete pipeline state."""

    def measure(value: Any) -> int:
        if isinstance(value, Array):
            return int(value.size) * int(value.dtype.itemsize)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return sum(
                measure(getattr(value, item.name))
                for item in dataclasses.fields(value)
                if item.name not in {"birth_timestamp", "uptime_s"}
            )
        if isinstance(value, Mapping):
            return sum(measure(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(measure(item) for item in value)
        return 0

    return measure(state)


@dataclass(frozen=True)
class AlbertaPipelineSmokeResult:
    """Summary returned by :func:`run_pipeline_smoke`."""

    config: AlbertaPipelineConfig
    steps: int
    seed: int
    feature_shape: tuple[int, ...]
    horde_predictions_shape: tuple[int, ...]
    q_values_shape: tuple[int, ...]
    actions_shape: tuple[int, ...]
    finite: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["feature_shape"] = list(self.feature_shape)
        payload["horde_predictions_shape"] = list(self.horde_predictions_shape)
        payload["q_values_shape"] = list(self.q_values_shape)
        payload["actions_shape"] = list(self.actions_shape)
        return payload


def observation_channel_cumulant_fn(
    n_demons: int, observation_dim: int
) -> CumulantFn:
    """Return a cumulant function that maps demons to observation channels."""
    if n_demons < 1:
        msg = f"n_demons must be positive, got {n_demons}"
        raise ValueError(msg)
    if observation_dim < 1:
        msg = f"observation_dim must be positive, got {observation_dim}"
        raise ValueError(msg)

    indices = jnp.arange(n_demons) % max(observation_dim, 1)

    def cumulant_fn(
        observation: Array, _reward: Array, _terminated: Array
    ) -> Array:
        obs_1d = jnp.atleast_1d(observation)
        return obs_1d[indices]

    return cumulant_fn


class AlbertaPipeline:
    """Composable Step 2 featurization + Step 3 Horde + Step 4 control.

    See :class:`AlbertaPipelineConfig` for selecting between temporal-context
    and UPGD Step 2 featurization, and between SARSA and HordeActorCritic
    Step 4 control. A caller-supplied ``cumulant_fn`` substitutes domain Step 3
    cumulants for the default observation-channel cumulants; passing
    ``cumulant_fn=None`` preserves the legacy smoke behavior for
    back-compatibility.
    """

    def __init__(
        self,
        config: AlbertaPipelineConfig | None = None,
        *,
        cumulant_fn: CumulantFn | None = None,
    ):
        """Construct all pipeline components from ``config``."""
        self._config = config or AlbertaPipelineConfig()

        if self._config.step2 == "temporal_context":
            self._featurizer: TemporalContextFeaturizer | None = (
                TemporalContextFeaturizer(
                    self._config.features.to_temporal_context_config()
                )
            )
        else:
            self._featurizer = None

        if self._config.step2 == "upgd":
            upgd_cfg = cast(Step2UPGDConfig, self._config.upgd)
            if upgd_cfg.learner_preset == "strict_digit_readout":
                self._upgd: UPGDLearner | None = (
                    UPGDLearner.step2_strict_digit_readout_default(
                        n_heads=upgd_cfg.n_heads,
                        hidden_sizes=upgd_cfg.hidden_sizes,
                        step_size=upgd_cfg.step_size,
                    )
                )
            else:
                self._upgd = UPGDLearner(
                    n_heads=upgd_cfg.n_heads,
                    hidden_sizes=upgd_cfg.hidden_sizes,
                    step_size=upgd_cfg.step_size,
                    bounder=ObGDBounding(kappa=0.5),
                    sparsity=upgd_cfg.sparsity,
                    use_layer_norm=upgd_cfg.use_layer_norm,
                    perturbation_sigma=1e-4,
                    perturbation_noise="rademacher",
                    utility_decay=0.995,
                    perturbation_beta=2.0,
                    perturbation_interval=16,
                    loss_normalization=upgd_cfg.loss_normalization,
                    readout_mode=upgd_cfg.readout_mode,
                    track_unit_utilities=False,
                    track_gradient_history=False,
                )
        else:
            self._upgd = None

        if self._config.step2 == "associative":
            assoc_cfg = cast(Step2AssociativePipelineConfig, self._config.associative)
            self._associative: AssociativeMemoryLearner | None = (
                AssociativeMemoryLearner(assoc_cfg.to_core_config())
            )
        else:
            self._associative = None

        self._horde = make_step3_horde(self._config.horde)

        self._control: HordeActorCriticAgent | Any
        if self._config.control_mode == "horde_ac":
            ac_cfg = cast(HordeActorCriticPipelineConfig, self._config.horde_ac)
            actor_bounder = (
                ObGDBounding(kappa=ac_cfg.actor_obgd_kappa)
                if ac_cfg.actor_obgd_kappa is not None
                else None
            )
            # HordeActorCritic requires the shared-trunk HordeLearner; the
            # mixed/independent routings are unsupported as a critic backend.
            if not isinstance(self._horde, HordeLearner):
                msg = (
                    "control_mode='horde_ac' requires Step 3 routing='shared'; "
                    f"got {type(self._horde).__name__}"
                )
                raise TypeError(msg)
            self._control = HordeActorCriticAgent(
                config=ac_cfg.to_horde_actor_critic_config(),
                critic=self._horde,
                actor_bounder=actor_bounder,
            )
        else:
            self._control = make_step4_sarsa_agent(
                self._config.control,
                prediction_demons=tuple(self._horde.horde_spec.demons),
            )

        observation_dim = self._observation_dim()
        self._cumulant_fn: CumulantFn = cumulant_fn or observation_channel_cumulant_fn(
            self._config.horde.n_demons, observation_dim
        )

    def _observation_dim(self) -> int:
        if self._config.step2 == "upgd":
            return cast(Step2UPGDConfig, self._config.upgd).observation_dim
        if self._config.step2 == "associative":
            return cast(Step2AssociativePipelineConfig, self._config.associative).block_size
        return self._config.features.observation_dim

    @property
    def config(self) -> AlbertaPipelineConfig:
        """Pipeline configuration."""
        return self._config

    @property
    def feature_dim(self) -> int:
        """Feature dimensionality emitted by Step 2."""
        return self._config.feature_dim()

    @property
    def featurizer(self) -> TemporalContextFeaturizer | None:
        """Underlying temporal-context featurizer if configured."""
        return self._featurizer

    @property
    def upgd(self) -> UPGDLearner | None:
        """Underlying UPGD learner if configured."""
        return self._upgd

    @property
    def associative(self) -> AssociativeMemoryLearner | None:
        """Underlying associative memory learner if configured."""
        return self._associative

    @property
    def horde(self) -> Any:
        """Underlying Step 3 Horde learner."""
        return self._horde

    @property
    def control(self) -> Any:
        """Underlying Step 4 control agent (SARSA or HordeActorCritic)."""
        return self._control

    @property
    def cumulant_fn(self) -> CumulantFn:
        """Cumulant function used by Step 3."""
        return self._cumulant_fn

    @staticmethod
    def _exact_child_aligned(child_state: Any, step_words: Array) -> Array:
        """Authenticate one exact-clock child against the pipeline identity."""

        child_words = getattr(child_state, "step_words", None)
        child_count = getattr(child_state, "step_count", None)
        if child_words is None or child_count is None:
            raise ValueError("configured child does not expose an exact clock")
        return _lifetime_counter_valid(child_words, child_count) & jnp.all(
            child_words == step_words
        )

    @staticmethod
    def _trees_equal(left: Any, right: Any) -> Array:
        """Return bitwise array equality for two statically identical trees."""

        left_leaves = jax.tree.leaves(left)
        right_leaves = jax.tree.leaves(right)
        if len(left_leaves) != len(right_leaves):
            raise ValueError("duplicated child trees have different structures")
        equal = jnp.asarray(True, dtype=jnp.bool_)
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
            if getattr(left_leaf, "shape", None) != getattr(right_leaf, "shape", None):
                raise ValueError("duplicated child leaves have different shapes")
            leaf_equal = left_leaf == right_leaf
            dtype = getattr(left_leaf, "dtype", None)
            if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
                leaf_equal = leaf_equal | (
                    jnp.isnan(left_leaf) & jnp.isnan(right_leaf)
                )
            equal = equal & jnp.all(leaf_equal)
        return equal

    def _validate_state_static_contract(self, state: AlbertaPipelineState) -> None:
        """Reject structural state mismatches before tracing child updates."""

        if not isinstance(state, AlbertaPipelineState):
            raise TypeError("state must be an AlbertaPipelineState")
        _lifetime_counter_valid(state.step_words, state.step_count)
        if getattr(state.last_features, "shape", None) != (self.feature_dim,):
            raise ValueError(
                f"last_features must have shape ({self.feature_dim},)"
            )
        if getattr(state.last_features, "dtype", None) != jnp.dtype(jnp.float32):
            raise TypeError("last_features must have dtype float32")
        if self._config.step2 == "temporal_context":
            if state.feature_state is None:
                raise ValueError("temporal-context state is missing")
            if state.upgd_state is not None or state.associative_state is not None:
                raise ValueError("inactive Step 2 state must be absent")
        elif self._config.step2 == "upgd":
            if state.upgd_state is None:
                raise ValueError("UPGD state is missing")
            if state.feature_state is not None or state.associative_state is not None:
                raise ValueError("inactive Step 2 state must be absent")
        elif self._config.step2 == "associative":
            if state.associative_state is None:
                raise ValueError("associative state is missing")
            if state.feature_state is not None or state.upgd_state is not None:
                raise ValueError("inactive Step 2 state must be absent")
        elif (
            state.feature_state is not None
            or state.upgd_state is not None
            or state.associative_state is not None
        ):
            raise ValueError("identity Step 2 must not carry learner state")

    def _children_aligned(self, state: AlbertaPipelineState) -> Array:
        """Return exact wrapper/child identity and route alignment."""

        outer_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        horde_aligned = self._exact_child_aligned(
            state.horde_state,
            state.step_words,
        )
        step2_aligned = jnp.asarray(True, dtype=jnp.bool_)
        if self._config.step2 == "temporal_context":
            feature_state = cast(TemporalContextState, state.feature_state)
            featurizer = cast(TemporalContextFeaturizer, self._featurizer)
            expected_words, offset_available = _checked_step_words_increment(
                state.step_words
            )
            step2_aligned = (
                offset_available
                & featurizer.state_valid(feature_state)
                & jnp.all(feature_state.step_words == expected_words)
            )
        elif self._config.step2 == "upgd":
            step2_aligned = self._exact_child_aligned(
                cast(UPGDState, state.upgd_state),
                state.step_words,
            )
        elif self._config.step2 == "associative":
            associative_state = cast(
                AssociativeMemoryState,
                state.associative_state,
            )
            if getattr(associative_state.step_count, "shape", None) != ():
                raise ValueError("associative step_count must be scalar")
            if getattr(associative_state.step_count, "dtype", None) != jnp.dtype(
                jnp.int32
            ):
                raise TypeError("associative step_count must have dtype int32")
            bounded_identity = (
                state.step_words[0] == jnp.asarray(0, dtype=jnp.uint32)
            ) & (
                state.step_words[1]
                <= jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
            )
            step2_aligned = bounded_identity & (
                associative_state.step_count
                == state.step_words[1].astype(jnp.int32)
            )
            step2_aligned = (
                step2_aligned
                & (associative_state.allocations >= 0)
                & (associative_state.replacements >= 0)
                & jnp.all(associative_state.counts >= 0.0)
                & jnp.all(associative_state.last_update >= 0)
                & jnp.all(
                    associative_state.last_update
                    <= associative_state.step_count
                )
            )

        control_state = state.control_state
        control_count = getattr(control_state, "step_count", None)
        if control_count is None or getattr(control_count, "shape", None) != ():
            raise ValueError("control step_count must be scalar")
        if getattr(control_count, "dtype", None) != jnp.dtype(jnp.int32):
            raise TypeError("control step_count must have dtype int32")
        control_aligned = control_count == _saturating_count_from_words(
            state.step_words
        )
        if self._config.control_mode == "horde_ac":
            ac_state = cast(HordeActorCriticState, control_state)
            control_aligned = (
                control_aligned
                & self._exact_child_aligned(ac_state.critic_state, state.step_words)
                & self._trees_equal(state.horde_state, ac_state.critic_state)
                & (ac_state.last_action >= 0)
                & (
                    ac_state.last_action
                    < cast(
                        HordeActorCriticPipelineConfig,
                        self._config.horde_ac,
                    ).n_actions
                )
            )
        else:
            sarsa_state = cast(SARSAState, control_state)
            control_aligned = control_aligned & self._exact_child_aligned(
                sarsa_state.learner_state,
                state.step_words,
            )
            control_aligned = control_aligned & (
                (sarsa_state.last_action >= 0)
                & (sarsa_state.last_action < self._config.control.n_actions)
            )

        return (
            outer_valid
            & horde_aligned
            & step2_aligned
            & control_aligned
        )

    def _state_valid(self, state: AlbertaPipelineState) -> Array:
        """Return dynamic integrity and exact child-alignment validity."""

        return (
            self._children_aligned(state)
            & _tree_arrays_finite(state)
            & jnp.all(jnp.isfinite(state.last_features))
        )

    def state_valid(self, state: AlbertaPipelineState) -> Array:
        """Return whether a state satisfies the complete pipeline contract."""

        self._validate_state_static_contract(state)
        return self._state_valid(state)

    def resource_budget(
        self,
        state: AlbertaPipelineState,
    ) -> AlbertaPipelineResourceBudget:
        """Return exact persistent-array accounting for ``state``."""

        self._validate_state_static_contract(state)
        return AlbertaPipelineResourceBudget(
            persistent_state_nbytes=measure_alberta_pipeline_state_nbytes(state)
        )

    def _features_from_observation(
        self,
        feature_state: TemporalContextState | None,
        upgd_state: UPGDState | None,
        associative_state: AssociativeMemoryState | None,
        observation: Array,
    ) -> tuple[
        TemporalContextState | None,
        UPGDState | None,
        AssociativeMemoryState | None,
        Array,
    ]:
        """Produce the Step 2 feature vector for an observation."""
        if self._config.step2 == "temporal_context":
            featurizer = cast(TemporalContextFeaturizer, self._featurizer)
            assert feature_state is not None
            new_feature_state, features = featurizer.step(feature_state, observation)
            return new_feature_state, upgd_state, associative_state, features
        if self._config.step2 == "upgd":
            upgd = cast(UPGDLearner, self._upgd)
            assert upgd_state is not None
            features = upgd._trunk_forward(  # noqa: SLF001
                upgd_state.trunk_params.weights,
                upgd_state.trunk_params.biases,
                observation,
                upgd._leaky_relu_slope,  # noqa: SLF001
                upgd._use_layer_norm,  # noqa: SLF001
            )
            return feature_state, upgd_state, associative_state, features
        if self._config.step2 == "associative":
            associative = cast(AssociativeMemoryLearner, self._associative)
            assert associative_state is not None
            prediction = associative.predict(
                associative_state,
                jnp.asarray(observation, dtype=jnp.int32),
            )
            return feature_state, upgd_state, associative_state, prediction.probabilities
        # identity
        return feature_state, upgd_state, associative_state, observation

    def init(self, key: Array, initial_observation: Array) -> AlbertaPipelineState:
        """Initialize learner state and prime control with the first observation."""
        expected_shape = (self._observation_dim(),)
        if getattr(initial_observation, "shape", None) != expected_shape:
            raise ValueError(
                f"initial_observation must have shape {expected_shape}"
            )
        expected_dtype = jnp.dtype(
            jnp.int32
            if self._config.step2 == "associative"
            else jnp.float32
        )
        if getattr(initial_observation, "dtype", None) != expected_dtype:
            raise TypeError(
                f"initial_observation must have dtype {expected_dtype.name}"
            )
        initial_observation = jnp.asarray(
            initial_observation,
            dtype=expected_dtype,
        )
        upgd_key, horde_key, control_key = jr.split(key, 3)

        feature_state: TemporalContextState | None = None
        upgd_state: UPGDState | None = None
        associative_state: AssociativeMemoryState | None = None
        observation_dim = self._observation_dim()

        if self._config.step2 == "temporal_context":
            featurizer = cast(TemporalContextFeaturizer, self._featurizer)
            feature_state, initial_features = featurizer.step(
                featurizer.init(),
                initial_observation,
            )
        elif self._config.step2 == "upgd":
            upgd = cast(UPGDLearner, self._upgd)
            upgd_state = upgd.init(observation_dim, upgd_key)
            initial_features = upgd._trunk_forward(  # noqa: SLF001
                upgd_state.trunk_params.weights,
                upgd_state.trunk_params.biases,
                initial_observation,
                upgd._leaky_relu_slope,  # noqa: SLF001
                upgd._use_layer_norm,  # noqa: SLF001
            )
        elif self._config.step2 == "associative":
            associative = cast(AssociativeMemoryLearner, self._associative)
            associative_state = associative.init()
            initial_features = associative.predict(
                associative_state,
                jnp.asarray(initial_observation, dtype=jnp.int32),
            ).probabilities
        else:
            initial_features = jnp.asarray(initial_observation, dtype=jnp.float32)

        initial_features = jnp.asarray(initial_features, dtype=jnp.float32)

        horde_state = init_step3_state(
            self._horde,
            feature_dim=self.feature_dim,
            key=horde_key,
        )

        control_state: SARSAState | HordeActorCriticState
        if self._config.control_mode == "horde_ac":
            ac = cast(HordeActorCriticAgent, self._control)
            ac_state = ac.init(self.feature_dim, control_key)
            ac_state, _action, _probs = ac.start(ac_state, initial_features)
            horde_state = ac_state.critic_state
            control_state = ac_state
        else:
            control_state = init_step4_state(
                self._control,
                feature_dim=self.feature_dim,
                key=control_key,
                initial_features=initial_features,
            )

        return AlbertaPipelineState(
            feature_state=feature_state,
            upgd_state=upgd_state,
            associative_state=associative_state,
            horde_state=horde_state,
            control_state=control_state,
            last_features=initial_features,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def predict(self, state: AlbertaPipelineState) -> tuple[Array, Array]:
        """Return Step 3 predictions and Step 4 control outputs.

        For SARSA control, the second element is the per-action Q-value
        vector. For HordeActorCritic control, it is the softmax action
        probability vector.
        """
        horde_predictions = step3_predict(
            self._horde,
            state.horde_state,
            state.last_features,
        )
        if self._config.control_mode == "horde_ac":
            ac = cast(HordeActorCriticAgent, self._control)
            ac_state = cast(HordeActorCriticState, state.control_state)
            policy = ac.policy(ac_state, state.last_features)
            return horde_predictions, policy
        sarsa_state = cast(SARSAState, state.control_state)
        q_values = self._control.horde.predict(
            sarsa_state.learner_state,
            state.last_features,
        )[: self._config.control.n_actions]
        return horde_predictions, q_values

    def update(
        self,
        state: AlbertaPipelineState,
        observation: Array,
        reward: Array,
        terminated: Array,
        horde_cumulants: Array | None = None,
        upgd_targets: Array | None = None,
        associative_label: Array | None = None,
    ) -> AlbertaPipelineStepResult:
        """Atomically advance every configured pipeline component.

        ``state.last_features`` represents the previous observation. The new
        raw ``observation`` is transformed by Step 2, then Step 3 and Step 4
        stage updates on the resulting transition. The staged state commits
        only if the source, exact pipeline identity, every child contract, and
        every persistent candidate are valid together. Any refusal preserves
        the complete input state, including RNG keys, traces, optimizer state,
        and all duplicate critic leaves.

        Args:
            state: Current pipeline state.
            observation: Next raw observation.
            reward: Scalar transition reward.
            terminated: Scalar termination flag (``0.0`` or ``1.0``).
            horde_cumulants: Optional explicit Step 3 cumulants of shape
                ``(n_demons,)``. When omitted, the configured cumulant
                function, or the default observation-channel cumulant function,
                is used.
            upgd_targets: Optional supervised targets of shape ``(n_heads,)``
                that drive UPGD learning when ``step2='upgd'``. NaN entries
                mark intentionally inactive heads. When omitted, UPGD stays
                frozen and the complete pipeline transaction refuses because
                Step 2 has no authenticated event to advance.
            associative_label: Optional integer next-token/class label that
                drives associative-memory writes when ``step2='associative'``.
                It is required for an associative pipeline transaction; its
                absence makes the bounded Step 2 contract unavailable and the
                whole pipeline update is rejected.
        """
        self._validate_state_static_contract(state)
        expected_observation_shape = (self._observation_dim(),)
        if getattr(observation, "shape", None) != expected_observation_shape:
            raise ValueError(
                f"observation must have shape {expected_observation_shape}"
            )
        expected_observation_dtype = jnp.dtype(
            jnp.int32
            if self._config.step2 == "associative"
            else jnp.float32
        )
        if getattr(observation, "dtype", None) != expected_observation_dtype:
            raise TypeError(
                f"observation must have dtype {expected_observation_dtype.name}"
            )
        observation = jnp.asarray(
            observation,
            dtype=expected_observation_dtype,
        )
        if getattr(reward, "dtype", None) != jnp.dtype(jnp.float32):
            raise TypeError("reward must have dtype float32")
        if getattr(terminated, "dtype", None) != jnp.dtype(jnp.float32):
            raise TypeError("terminated must have dtype float32")
        reward = jnp.asarray(reward, dtype=jnp.float32)
        terminated = jnp.asarray(terminated, dtype=jnp.float32)
        if reward.shape != ():
            raise ValueError("reward must be scalar")
        if terminated.shape != ():
            raise ValueError("terminated must be scalar")

        if horde_cumulants is None:
            horde_cumulants = self._cumulant_fn(observation, reward, terminated)
        if getattr(horde_cumulants, "dtype", None) != jnp.dtype(jnp.float32):
            raise TypeError("horde_cumulants must have dtype float32")
        horde_cumulants = jnp.asarray(horde_cumulants, dtype=jnp.float32)
        if horde_cumulants.shape != (self._config.horde.n_demons,):
            raise ValueError(
                "horde_cumulants must have shape "
                f"({self._config.horde.n_demons},)"
            )

        upgd_targets_array: Array | None = None
        upgd_targets_available = upgd_targets is not None
        if self._config.step2 == "upgd":
            n_heads = cast(Step2UPGDConfig, self._config.upgd).n_heads
            upgd_targets_array = (
                jnp.full((n_heads,), jnp.nan, dtype=jnp.float32)
                if upgd_targets is None
                else upgd_targets
            )
            if getattr(upgd_targets_array, "dtype", None) != jnp.dtype(jnp.float32):
                raise TypeError("upgd_targets must have dtype float32")
            upgd_targets_array = jnp.asarray(upgd_targets_array, dtype=jnp.float32)
            if upgd_targets_array.shape != (n_heads,):
                raise ValueError(f"upgd_targets must have shape ({n_heads},)")
        elif upgd_targets is not None:
            raise ValueError("upgd_targets require step2='upgd'")

        associative_label_array: Array | None = None
        if associative_label is not None:
            if self._config.step2 != "associative":
                raise ValueError("associative_label requires step2='associative'")
            if getattr(associative_label, "dtype", None) != jnp.dtype(jnp.int32):
                raise TypeError("associative_label must have dtype int32")
            associative_label_array = jnp.asarray(
                associative_label,
                dtype=jnp.int32,
            )
            if associative_label_array.shape != ():
                raise ValueError("associative_label must be scalar")

        proposed_step_words, lifetime_capacity_available = (
            _checked_step_words_increment(state.step_words)
        )
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        state_valid = self._state_valid(state)
        observation_valid = jnp.all(jnp.isfinite(observation))
        if self._config.step2 == "associative":
            associative_cfg = cast(
                Step2AssociativePipelineConfig,
                self._config.associative,
            )
            observation_valid = observation_valid & jnp.all(
                (observation >= 0) & (observation < associative_cfg.vocab_size)
            )
        terminated_valid = jnp.isfinite(terminated) & (
            (terminated == 0.0) | (terminated == 1.0)
        )
        cumulants_valid = jnp.all(jnp.isfinite(horde_cumulants))
        target_source_valid = jnp.asarray(True, dtype=jnp.bool_)
        if upgd_targets_array is not None:
            target_source_valid = jnp.all(
                jnp.isfinite(upgd_targets_array) | jnp.isnan(upgd_targets_array)
            )
        label_source_valid = jnp.asarray(True, dtype=jnp.bool_)
        if associative_label_array is not None:
            associative_cfg = cast(
                Step2AssociativePipelineConfig,
                self._config.associative,
            )
            label_source_valid = (
                (associative_label_array >= 0)
                & (associative_label_array < associative_cfg.vocab_size)
            )
        source_valid = (
            observation_valid
            & jnp.isfinite(reward)
            & terminated_valid
            & cumulants_valid
            & target_source_valid
            & label_source_valid
        )

        # Stage Step 2. Explicit all-NaN UPGD targets are an intentional exact
        # inactive-head event; an omitted target vector is unavailable and
        # leaves UPGD untouched. Temporal context owns an exact clock with a
        # one-event priming offset; associative memory remains explicitly
        # bounded by its legacy int32 identity.
        temporal_step_result: TemporalContextStepResult | None = None
        if self._config.step2 == "temporal_context":
            featurizer = cast(TemporalContextFeaturizer, self._featurizer)
            temporal_step_result = featurizer.step_result(
                cast(TemporalContextState, state.feature_state),
                observation,
            )
            new_feature_state = temporal_step_result.state
            new_upgd_state = state.upgd_state
            new_associative_state = state.associative_state
            features = temporal_step_result.features
        else:
            (
                new_feature_state,
                new_upgd_state,
                new_associative_state,
                features,
            ) = self._features_from_observation(
                state.feature_state,
                state.upgd_state,
                state.associative_state,
                observation,
            )
        step2_contract_available = jnp.asarray(True, dtype=jnp.bool_)
        step2_update_applied = jnp.asarray(True, dtype=jnp.bool_)
        if self._config.step2 == "temporal_context":
            assert temporal_step_result is not None
            temporal_update = temporal_step_result.update
            expected_pre_words, pre_offset_available = (
                _checked_step_words_increment(state.step_words)
            )
            expected_post_words, post_offset_available = (
                _checked_step_words_increment(proposed_step_words)
            )
            step2_contract_available = (
                pre_offset_available
                & post_offset_available
                & temporal_update.lifetime_capacity_available
            )
            step2_update_applied = (
                step2_contract_available
                & temporal_update.update_applied
                & temporal_update.state_valid
                & temporal_update.input_valid
                & temporal_update.candidate_state_finite
                & jnp.all(temporal_update.pre_step_words == expected_pre_words)
                & jnp.all(temporal_update.post_step_words == expected_post_words)
                & jnp.all(jnp.isfinite(features))
            )
        elif self._config.step2 == "upgd":
            step2_contract_available = jnp.asarray(
                upgd_targets_available,
                dtype=jnp.bool_,
            )
            if upgd_targets_available:
                upgd = cast(UPGDLearner, self._upgd)
                upgd_result = upgd.update(
                    cast(UPGDState, new_upgd_state),
                    observation,
                    cast(Array, upgd_targets_array),
                )
                new_upgd_state = upgd_result.state
                features = upgd._trunk_forward(  # noqa: SLF001
                    new_upgd_state.trunk_params.weights,
                    new_upgd_state.trunk_params.biases,
                    observation,
                    upgd._leaky_relu_slope,  # noqa: SLF001
                    upgd._use_layer_norm,  # noqa: SLF001
                )
                step2_contract_available = step2_contract_available & jnp.asarray(
                    upgd_result.lifetime_capacity_available,
                    dtype=jnp.bool_,
                )
                step2_update_applied = (
                    jnp.asarray(upgd_result.update_applied, dtype=jnp.bool_)
                    & jnp.all(upgd_result.pre_step_words == state.step_words)
                    & jnp.all(upgd_result.post_step_words == proposed_step_words)
                    & jnp.all(jnp.isfinite(features))
                )
            else:
                step2_update_applied = jnp.asarray(False, dtype=jnp.bool_)
        elif self._config.step2 == "associative":
            associative_state = cast(
                AssociativeMemoryState,
                state.associative_state,
            )
            associative = cast(AssociativeMemoryLearner, self._associative)
            maximum_row_events = associative.max_active_features
            diagnostic_counter_capacity = (
                associative_state.allocations
                <= jnp.asarray(
                    _INT32_MAX - maximum_row_events,
                    dtype=jnp.int32,
                )
            ) & (
                associative_state.replacements
                <= jnp.asarray(
                    _INT32_MAX - maximum_row_events,
                    dtype=jnp.int32,
                )
            )
            # Float32 row counts cease to represent every +1 event at 2**24.
            # The legacy child has no exact per-row identity, so refuse before
            # that estimator boundary instead of treating a rounded write as
            # authenticated progress.
            row_count_capacity = jnp.all(
                associative_state.counts
                < jnp.asarray(2**24, dtype=jnp.float32)
            )
            step2_contract_available = (
                associative_label_array is not None
            ) & (
                associative_state.step_count
                < jnp.asarray(_INT32_MAX, dtype=jnp.int32)
            ) & diagnostic_counter_capacity & row_count_capacity
            if associative_label_array is not None:
                assoc_result = associative.update(
                    cast(AssociativeMemoryState, new_associative_state),
                    observation,
                    associative_label_array,
                )
                new_associative_state = assoc_result.state
                features = assoc_result.predictions
                step2_update_applied = (
                    step2_contract_available
                    & (
                        new_associative_state.step_count
                        == associative_state.step_count
                        + jnp.asarray(1, dtype=jnp.int32)
                    )
                    & _tree_arrays_finite(new_associative_state)
                    & (
                        new_associative_state.allocations
                        >= associative_state.allocations
                    )
                    & (
                        new_associative_state.replacements
                        >= associative_state.replacements
                    )
                    & jnp.all(
                        new_associative_state.counts >= associative_state.counts
                    )
                    & jnp.all(jnp.isfinite(assoc_result.metrics))
                    & jnp.all(jnp.isfinite(features))
                )
            else:
                step2_update_applied = jnp.asarray(False, dtype=jnp.bool_)
        else:
            features = jnp.asarray(features, dtype=jnp.float32)
            step2_update_applied = jnp.all(jnp.isfinite(features))

        features = jnp.asarray(features, dtype=jnp.float32)

        horde_result: HordeUpdateResult
        new_control_state: SARSAState | HordeActorCriticState
        if self._config.control_mode == "horde_ac":
            ac = cast(HordeActorCriticAgent, self._control)
            ac_state = cast(HordeActorCriticState, state.control_state)
            ac_state = ac_state.replace(critic_state=state.horde_state)
            n_total_demons = self._horde.n_demons
            value_index = cast(
                HordeActorCriticPipelineConfig, self._config.horde_ac
            ).value_head_index
            aux_indices = jnp.array(
                [i for i in range(n_total_demons) if i != value_index],
                dtype=jnp.int32,
            )
            auxiliary_cumulants = horde_cumulants[aux_indices] if aux_indices.size else None
            value_gamma = self._horde.horde_spec.gammas[value_index]
            transition_discount = jnp.where(terminated != 0.0, 0.0, value_gamma)
            ac_result: HordeActorCriticUpdateResult = ac.update(
                ac_state,
                reward,
                features,
                auxiliary_cumulants=auxiliary_cumulants,
                discount=transition_discount,
            )
            new_control_state = ac_result.state
            q_values_or_policy = ac_result.policy
            action_out = ac_result.action
            control_td_error = ac_result.td_error
            reward_out = jnp.asarray(reward, dtype=jnp.float32)
            # The actor-critic update already updated the critic for us;
            # we override horde_state to keep them in sync.
            new_horde_state = ac_result.critic_result.state
            horde_predictions = ac_result.critic_result.predictions
            horde_td_errors = ac_result.critic_result.td_errors
            horde_td_targets = ac_result.critic_result.td_targets
            horde_result = ac_result.critic_result
            control_update_applied = jnp.asarray(
                ac_result.update_applied,
                dtype=jnp.bool_,
            )
        else:
            horde_result = self._horde.update(
                state.horde_state,
                state.last_features,
                horde_cumulants,
                features,
            )
            sarsa_state = cast(SARSAState, state.control_state)
            next_action, next_key = self._control.select_action(
                sarsa_state,
                features,
            )
            ready_sarsa_state = sarsa_state.replace(rng_key=next_key)
            control_result: SARSAUpdateResult = self._control.update(
                ready_sarsa_state,
                reward,
                features,
                terminated,
                next_action,
                prediction_cumulants=horde_cumulants,
            )
            new_control_state = control_result.state
            q_values_or_policy = control_result.q_values
            action_out = control_result.action
            control_td_error = control_result.td_error
            reward_out = control_result.reward
            new_horde_state = horde_result.state
            horde_predictions = horde_result.predictions
            horde_td_errors = horde_result.td_errors
            horde_td_targets = horde_result.td_targets
            control_update_applied = jnp.asarray(
                control_result.update_applied,
                dtype=jnp.bool_,
            )

        step3_pre_available = (
            horde_result.pre_step_words is not None
            and horde_result.post_step_words is not None
            and horde_result.update_applied is not None
            and horde_result.lifetime_capacity_available is not None
        )
        if step3_pre_available:
            step3_contract_available = jnp.asarray(
                horde_result.lifetime_capacity_available,
                dtype=jnp.bool_,
            )
            step3_update_applied = (
                jnp.asarray(horde_result.update_applied, dtype=jnp.bool_)
                & jnp.all(cast(Array, horde_result.pre_step_words) == state.step_words)
                & jnp.all(
                    cast(Array, horde_result.post_step_words)
                    == proposed_step_words
                )
            )
        else:
            step3_contract_available = jnp.asarray(False, dtype=jnp.bool_)
            step3_update_applied = jnp.asarray(False, dtype=jnp.bool_)

        control_contract_available = lifetime_capacity_available
        if self._config.control_mode == "horde_ac":
            control_contract_available = control_contract_available & jnp.asarray(
                ac_result.critic_update_applied,
                dtype=jnp.bool_,
            )
        else:
            control_contract_available = control_contract_available & jnp.asarray(
                control_result.horde_update_applied,
                dtype=jnp.bool_,
            )

        candidate_state = AlbertaPipelineState(
            feature_state=new_feature_state,
            upgd_state=new_upgd_state,
            associative_state=new_associative_state,
            horde_state=new_horde_state,
            control_state=new_control_state,
            last_features=features,
            step_count=_saturating_count_from_words(proposed_step_words),
            step_words=proposed_step_words,
        )
        children_pre_aligned = self._children_aligned(state)
        children_post_aligned = self._children_aligned(candidate_state)
        horde_outputs_valid = (
            jnp.all(jnp.isfinite(horde_predictions))
            & jnp.all(jnp.isfinite(horde_td_errors))
            & jnp.all(jnp.isfinite(horde_td_targets))
            & jnp.all(jnp.isfinite(horde_result.per_demon_metrics))
            & jnp.isfinite(horde_result.trunk_bounding_metric)
        )
        control_outputs_valid = (
            jnp.all(jnp.isfinite(q_values_or_policy))
            & jnp.isfinite(control_td_error)
            & jnp.isfinite(reward_out)
        )
        if self._config.control_mode == "horde_ac":
            control_outputs_valid = control_outputs_valid & jnp.isfinite(
                ac_result.bound_metric
            )
            action_valid = (action_out >= 0) & (
                action_out
                < cast(HordeActorCriticPipelineConfig, self._config.horde_ac).n_actions
            )
        else:
            action_valid = (action_out >= 0) & (
                action_out < self._config.control.n_actions
            )
        candidate_state_valid = (
            children_post_aligned
            & _tree_arrays_finite(candidate_state)
            & jnp.all(jnp.isfinite(features))
            & horde_outputs_valid
            & control_outputs_valid
            & action_valid
        )
        update_applied = (
            lifetime_counter_valid
            & lifetime_capacity_available
            & source_valid
            & state_valid
            & step2_contract_available
            & step2_update_applied
            & step3_contract_available
            & step3_update_applied
            & control_contract_available
            & control_update_applied
            & children_pre_aligned
            & children_post_aligned
            & candidate_state_valid
        )
        next_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )

        rejection_reason = jnp.asarray(
            PIPELINE_REJECTION_NONE,
            dtype=jnp.int32,
        )
        rejection_reason = jnp.where(
            ~candidate_state_valid,
            PIPELINE_REJECTION_CANDIDATE_INVALID,
            rejection_reason,
        )
        rejection_reason = jnp.where(
            ~(children_pre_aligned & children_post_aligned),
            PIPELINE_REJECTION_CHILD_MISALIGNED,
            rejection_reason,
        )
        rejection_reason = jnp.where(
            ~(control_contract_available & control_update_applied),
            PIPELINE_REJECTION_CONTROL_REFUSED,
            rejection_reason,
        )
        rejection_reason = jnp.where(
            ~(step3_contract_available & step3_update_applied),
            PIPELINE_REJECTION_STEP3_REFUSED,
            rejection_reason,
        )
        rejection_reason = jnp.where(
            ~(step2_contract_available & step2_update_applied),
            PIPELINE_REJECTION_STEP2_UNAVAILABLE,
            rejection_reason,
        )
        rejection_reason = jnp.where(
            ~source_valid,
            PIPELINE_REJECTION_SOURCE_INVALID,
            rejection_reason,
        )
        rejection_reason = jnp.where(
            ~lifetime_capacity_available,
            PIPELINE_REJECTION_LIFETIME_EXHAUSTED,
            rejection_reason,
        )
        rejection_reason = jnp.where(
            ~state_valid,
            PIPELINE_REJECTION_STATE_INVALID,
            rejection_reason,
        )
        rejection_reason = jnp.where(
            update_applied,
            PIPELINE_REJECTION_NONE,
            rejection_reason,
        ).astype(jnp.int32)

        safe_features = jnp.where(
            update_applied,
            features,
            jnp.zeros_like(features),
        )
        safe_horde_predictions = jnp.where(
            update_applied,
            horde_predictions,
            jnp.zeros_like(horde_predictions),
        )
        safe_horde_td_errors = jnp.where(
            update_applied,
            horde_td_errors,
            jnp.zeros_like(horde_td_errors),
        )
        safe_horde_td_targets = jnp.where(
            update_applied,
            horde_td_targets,
            jnp.zeros_like(horde_td_targets),
        )
        safe_q_values = jnp.where(
            update_applied,
            q_values_or_policy,
            jnp.zeros_like(q_values_or_policy),
        )
        safe_action = jnp.where(
            update_applied,
            action_out,
            jnp.asarray(0, dtype=jnp.int32),
        )
        safe_control_td_error = jnp.where(
            update_applied,
            control_td_error,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        safe_reward = jnp.where(
            update_applied,
            reward_out,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return AlbertaPipelineStepResult(
            state=next_state,
            features=safe_features,
            horde_predictions=safe_horde_predictions,
            horde_td_errors=safe_horde_td_errors,
            horde_td_targets=safe_horde_td_targets,
            q_values=safe_q_values,
            action=safe_action,
            control_td_error=safe_control_td_error,
            reward=safe_reward,
            pre_step_words=state.step_words,
            post_step_words=next_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            source_valid=source_valid,
            state_valid=state_valid,
            step2_contract_available=step2_contract_available,
            step2_update_applied=step2_update_applied,
            step3_contract_available=step3_contract_available,
            step3_update_applied=step3_update_applied,
            control_contract_available=control_contract_available,
            control_update_applied=control_update_applied,
            children_pre_aligned=children_pre_aligned,
            children_post_aligned=children_post_aligned,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
            update_rejected=~update_applied,
            rejection_reason=rejection_reason,
        )

    def run_arrays(
        self,
        state: AlbertaPipelineState,
        observations: Array,
        rewards: Array,
        terminated: Array,
        horde_cumulants: Array,
        upgd_targets: Array | None = None,
        associative_labels: Array | None = None,
    ) -> AlbertaPipelineArrayResult:
        """Scan the pipeline over transition arrays.

        ``state`` should be initialized with the observation that precedes the
        first row in ``observations``. ``horde_cumulants`` is required here
        (the per-step callable variant is :meth:`update`); array runs use a
        fully resolved cumulant table for ``jax.lax.scan`` compatibility.
        """
        use_upgd_targets = upgd_targets is not None
        if upgd_targets is None:
            steps = observations.shape[0]
            upgd_targets_array = jnp.full(
                (steps, self._config.upgd.n_heads if self._config.upgd else 1),
                jnp.nan,
                dtype=jnp.float32,
            )
        else:
            if getattr(upgd_targets, "dtype", None) != jnp.dtype(jnp.float32):
                raise TypeError("upgd_targets must have dtype float32")
            upgd_targets_array = jnp.asarray(upgd_targets, dtype=jnp.float32)
        if associative_labels is not None and getattr(
            associative_labels,
            "dtype",
            None,
        ) != jnp.dtype(jnp.int32):
            raise TypeError("associative_labels must have dtype int32")
        associative_labels_array = (
            jnp.asarray(associative_labels, dtype=jnp.int32)
            if associative_labels is not None
            else jnp.zeros((observations.shape[0],), dtype=jnp.int32)
        )
        use_associative_labels = (
            self._config.step2 == "associative" and associative_labels is not None
        )

        def step_fn(
            carry: AlbertaPipelineState,
            inputs: tuple[Array, Array, Array, Array, Array, Array],
        ) -> tuple[
            AlbertaPipelineState,
            tuple[Array, Array, Array, Array, Array, Array, Array, Array],
        ]:
            (
                obs_t,
                reward_t,
                terminated_t,
                cumulants_t,
                upgd_target_t,
                associative_label_t,
            ) = inputs
            result = self.update(
                carry,
                obs_t,
                reward_t,
                terminated_t,
                cumulants_t,
                (
                    upgd_target_t
                    if self._config.step2 == "upgd" and use_upgd_targets
                    else None
                ),
                associative_label_t if use_associative_labels else None,
            )
            return result.state, (
                result.features,
                result.horde_predictions,
                result.horde_td_errors,
                result.q_values,
                result.action,
                result.control_td_error,
                result.update_applied,
                result.rejection_reason,
            )

        final_state, outputs = jax.lax.scan(
            step_fn,
            state,
            (
                observations,
                rewards,
                terminated,
                horde_cumulants,
                upgd_targets_array,
                associative_labels_array,
            ),
        )
        (
            features,
            horde_predictions,
            horde_td_errors,
            q_values,
            actions,
            control_td_errors,
            update_applied,
            rejection_reasons,
        ) = outputs
        return AlbertaPipelineArrayResult(
            state=final_state,
            features=features,
            horde_predictions=horde_predictions,
            horde_td_errors=horde_td_errors,
            q_values=q_values,
            actions=actions,
            control_td_errors=control_td_errors,
            update_applied=update_applied,
            rejection_reasons=rejection_reasons,
        )


def migrate_legacy_alberta_pipeline_state(
    pipeline: AlbertaPipeline,
    legacy_state: Any,
) -> AlbertaPipelineState:
    """Migrate an unsaturated pre-v2 wrapper state on the host.

    This migration can authenticate only a legacy wrapper whose int32 clock
    never saturated and whose already-versioned exact children agree with that
    clock. Legacy child trees that lack their own exact identity require their
    component-specific migration first and are rejected here rather than
    assigned a guessed history.
    """

    fields = _host_field_mapping(legacy_state, name="legacy pipeline state")
    current_names = {
        item.name for item in dataclasses.fields(AlbertaPipelineState)
    }
    legacy_names = current_names - {"step_words"}
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(
            "legacy pipeline state fields are unsupported; "
            f"missing={missing}, extra={extra}"
        )
    legacy_step = fields["step_count"]
    if getattr(legacy_step, "shape", None) != ():
        raise ValueError("legacy pipeline step_count must be scalar")
    if getattr(legacy_step, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("legacy pipeline step_count must have dtype int32")
    host_step = int(legacy_step)
    if host_step < 0:
        raise ValueError("negative legacy pipeline step_count is invalid")
    if host_step >= _INT32_MAX:
        raise ValueError(
            "saturated legacy pipeline step_count cannot authenticate an exact lifetime"
        )
    if pipeline.config.step2 == "temporal_context" and not hasattr(
        fields["feature_state"],
        "step_words",
    ):
        raise ValueError(
            "legacy temporal-context child lacks exact step_words; migrate it "
            "with migrate_legacy_temporal_context_state before the pipeline wrapper"
        )
    fields["step_words"] = jnp.asarray((0, host_step), dtype=jnp.uint32)
    migrated = AlbertaPipelineState(**fields)
    pipeline._validate_state_static_contract(migrated)  # noqa: SLF001
    if not bool(pipeline._state_valid(migrated)):  # noqa: SLF001
        raise ValueError(
            "legacy pipeline children are not exact-clock aligned; migrate each "
            "bounded child before migrating the wrapper"
        )
    return migrated


def save_alberta_pipeline_checkpoint(
    pipeline: AlbertaPipeline,
    state: AlbertaPipelineState,
    path: str | Path,
) -> None:
    """Save only a structurally valid v2 exact-transaction checkpoint."""

    pipeline._validate_state_static_contract(state)  # noqa: SLF001
    if not bool(pipeline._state_valid(state)):  # noqa: SLF001
        raise ValueError("pipeline checkpoint state is invalid")
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": ALBERTA_PIPELINE_CHECKPOINT_SCHEMA,
            "state_schema": ALBERTA_PIPELINE_STATE_SCHEMA,
            "mechanism_status": "development_mechanism_only",
            "scientific_promotion_allowed": False,
            "pipeline_config": pipeline.config.to_dict(),
            "resource_budget": pipeline.resource_budget(state).to_dict(),
        },
    )


def load_alberta_pipeline_checkpoint(
    path: str | Path,
    *,
    cumulant_fn: CumulantFn | None = None,
) -> tuple[AlbertaPipeline, AlbertaPipelineState]:
    """Restore a strict v2 pipeline checkpoint and revalidate every child."""

    metadata = load_checkpoint_metadata(path)
    expected_metadata = {
        "schema",
        "state_schema",
        "mechanism_status",
        "scientific_promotion_allowed",
        "pipeline_config",
        "resource_budget",
    }
    if set(metadata) != expected_metadata:
        raise ValueError("pipeline checkpoint metadata fields are invalid")
    schema = metadata["schema"]
    if schema == _LEGACY_ALBERTA_PIPELINE_CHECKPOINT_SCHEMA:
        raise ValueError(
            "legacy pipeline checkpoint v1 lacks exact wrapper identity; "
            "decode it with its original schema and call "
            "migrate_legacy_alberta_pipeline_state explicitly"
        )
    if schema != ALBERTA_PIPELINE_CHECKPOINT_SCHEMA:
        raise ValueError("pipeline checkpoint schema is unsupported")
    if metadata["state_schema"] != ALBERTA_PIPELINE_STATE_SCHEMA:
        raise ValueError("pipeline checkpoint state schema is unsupported")
    if metadata["mechanism_status"] != "development_mechanism_only":
        raise ValueError("pipeline checkpoint mechanism status is invalid")
    if metadata["scientific_promotion_allowed"] is not False:
        raise ValueError("pipeline checkpoint promotion flag is invalid")
    config_payload = metadata["pipeline_config"]
    if type(config_payload) is not dict:
        raise ValueError("pipeline checkpoint config is invalid")
    pipeline = AlbertaPipeline(
        AlbertaPipelineConfig.from_dict(cast(dict[str, object], config_payload)),
        cumulant_fn=cumulant_fn,
    )
    initial_observation = jnp.zeros(
        (pipeline._observation_dim(),),  # noqa: SLF001
        dtype=(
            jnp.int32
            if pipeline.config.step2 == "associative"
            else jnp.float32
        ),
    )
    template = pipeline.init(jr.key(0), initial_observation)
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("pipeline checkpoint metadata changed between reads")
    state = cast(AlbertaPipelineState, restored)
    pipeline._validate_state_static_contract(state)  # noqa: SLF001
    if not bool(pipeline._state_valid(state)):  # noqa: SLF001
        raise ValueError("pipeline checkpoint state is invalid")
    resource_payload = metadata["resource_budget"]
    if type(resource_payload) is not dict:
        raise ValueError("pipeline checkpoint resource contract is invalid")
    if resource_payload != pipeline.resource_budget(state).to_dict():
        raise ValueError("pipeline checkpoint resource contract does not match")
    return pipeline, state


def make_alberta_pipeline(
    config: AlbertaPipelineConfig | None = None,
    *,
    cumulant_fn: CumulantFn | None = None,
) -> AlbertaPipeline:
    """Create an end-to-end Alberta production pipeline."""
    return AlbertaPipeline(config, cumulant_fn=cumulant_fn)


def run_pipeline_smoke(
    config: AlbertaPipelineConfig | None = None,
    *,
    steps: int = 24,
    seed: int = 0,
) -> AlbertaPipelineSmokeResult:
    """Run a deterministic Step 1-4 pipeline smoke probe."""
    if steps < 1:
        msg = f"steps must be positive, got {steps}"
        raise ValueError(msg)
    cfg = config or AlbertaPipelineConfig()
    pipeline = make_alberta_pipeline(cfg)

    observation_dim = pipeline._observation_dim()  # noqa: SLF001

    data_key, state_key = jr.split(jr.key(seed))
    if cfg.step2 == "associative" and cfg.associative is not None:
        observations = jr.randint(
            data_key,
            (steps + 1, observation_dim),
            minval=0,
            maxval=cfg.associative.vocab_size,
            dtype=jnp.int32,
        )
        rewards = jnp.tanh(observations[1:, 0].astype(jnp.float32))
        associative_labels = (
            observations[1:, -1] + 3 * observations[1:, -2] + observations[1:, 0]
        ) % cfg.associative.vocab_size
    else:
        observations = jr.normal(
            data_key,
            (steps + 1, observation_dim),
            dtype=jnp.float32,
        )
        rewards = jnp.tanh(observations[1:, 0])
        associative_labels = None
    terminated = jnp.zeros(steps, dtype=jnp.float32)
    cumulant_indices = jnp.arange(cfg.horde.n_demons) % observation_dim
    horde_cumulants = observations[1:, cumulant_indices].astype(jnp.float32)

    state = pipeline.init(state_key, observations[0])
    smoke_upgd_targets = (
        jnp.zeros((steps, cfg.upgd.n_heads), dtype=jnp.float32)
        if cfg.step2 == "upgd" and cfg.upgd is not None
        else None
    )
    result = pipeline.run_arrays(
        state,
        observations[1:],
        rewards,
        terminated,
        horde_cumulants,
        upgd_targets=smoke_upgd_targets,
        associative_labels=associative_labels,
    )
    result.q_values.block_until_ready()

    finite_actions = (
        jnp.all(result.actions >= 0)
        & jnp.all(result.actions < cfg.horde_ac.n_actions)
        if cfg.control_mode == "horde_ac" and cfg.horde_ac is not None
        else jnp.all(result.actions >= 0) & jnp.all(result.actions < cfg.control.n_actions)
    )
    finite = bool(
        jnp.all(jnp.isfinite(result.features))
        & jnp.all(jnp.isfinite(result.horde_predictions))
        & jnp.all(jnp.isfinite(result.horde_td_errors))
        & jnp.all(jnp.isfinite(result.q_values))
        & jnp.all(jnp.isfinite(result.control_td_errors))
        & jnp.all(result.update_applied)
        & finite_actions
    )
    return AlbertaPipelineSmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        feature_shape=tuple(int(dim) for dim in result.features.shape),
        horde_predictions_shape=tuple(
            int(dim) for dim in result.horde_predictions.shape
        ),
        q_values_shape=tuple(int(dim) for dim in result.q_values.shape),
        actions_shape=tuple(int(dim) for dim in result.actions.shape),
        finite=finite,
    )
