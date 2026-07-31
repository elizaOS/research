# mypy: disable-error-code="call-arg"
"""Two physically separate online learners for a binary signaling game.

The helper and beneficiary each own a zero-initialized float32 contextual
bandit table and an independent RNG key.  Policies are epsilon-greedy with
uniform random tie-breaking.  A transition update first forms *both*
candidate role states from the same old joint state, then atomically commits
each value-table candidate under its role-specific write mask.  Policy keys
always advance after decisions, including for value-frozen control roles; a
false mask preserves that role's float32 value table bit for bit without
turning stochastic exploration into a repeated deterministic draw.

The module intentionally has no target-label argument.  Both roles learn only
from their own ordinary inputs, their selected binary action, and the common
scalar reward supplied after both decisions have been made.

Public contexts index disjoint table rows.  This makes inactive-context value
retention structurally interference-free; it is useful for isolating genuine
co-learning and causal message use, but it is not evidence that a general
learner resists catastrophic forgetting in shared representations.
"""

from __future__ import annotations

import dataclasses
import math
from numbers import Real
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

N_CONTEXTS = 2
N_PRIVATE_INPUTS = 2
N_ACTIONS = 2
ROLE_VALUE_SHAPE = (N_CONTEXTS, N_PRIVATE_INPUTS, N_ACTIONS)

_HELPER_RNG_TAG = 0x484C50  # ASCII "HLP"
_BENEFICIARY_RNG_TAG = 0x424E46  # ASCII "BNF"


@dataclasses.dataclass(frozen=True)
class SignalingBanditConfig:
    """Learning and exploration rates shared by both role learners."""

    learning_rate: float = 0.1
    epsilon: float = 0.1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.learning_rate, Real)
            or isinstance(self.learning_rate, bool)
            or not math.isfinite(float(self.learning_rate))
            or not 0.0 < float(self.learning_rate) <= 1.0
        ):
            raise ValueError("learning_rate must lie in (0, 1]")
        if (
            not isinstance(self.epsilon, Real)
            or isinstance(self.epsilon, bool)
            or not math.isfinite(float(self.epsilon))
            or not 0.0 <= float(self.epsilon) <= 1.0
        ):
            raise ValueError("epsilon must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "learning_rate": float(self.learning_rate),
            "epsilon": float(self.epsilon),
        }


@chex.dataclass(frozen=True)
class SignalingBanditKeys:
    """Named independent policy streams for the two learning roles."""

    helper: PRNGKeyArray
    beneficiary: PRNGKeyArray


def signaling_bandit_keys(root_key: Array) -> SignalingBanditKeys:
    """Derive stable role keys without positional split coupling."""

    return SignalingBanditKeys(
        helper=jr.fold_in(root_key, _HELPER_RNG_TAG),
        beneficiary=jr.fold_in(root_key, _BENEFICIARY_RNG_TAG),
    )


@chex.dataclass(frozen=True)
class RoleBanditState:
    """One role's physically separate table and policy key."""

    values: Float[Array, "2 2 2"]
    key: PRNGKeyArray


@chex.dataclass(frozen=True)
class SignalingBanditState:
    """Joint container with no shared trainable parameters or RNG state."""

    helper: RoleBanditState
    beneficiary: RoleBanditState


@chex.dataclass(frozen=True)
class RoleDecision:
    """Pre-reward decision record produced from one old role state."""

    context: Int[Array, ""]
    private_input: Int[Array, ""]
    action: Int[Array, ""]
    selected_value: Float[Array, ""]
    next_key: PRNGKeyArray


@chex.dataclass(frozen=True)
class SignalingBanditUpdate:
    """Atomic update result with pre/post selected-cell diagnostics."""

    state: SignalingBanditState
    helper_value_pre: Float[Array, ""]
    helper_value_post: Float[Array, ""]
    beneficiary_value_pre: Float[Array, ""]
    beneficiary_value_post: Float[Array, ""]


@dataclasses.dataclass(frozen=True)
class RoleResourceBudget:
    """Exact persistent resource budget for one role state."""

    value_scalars: int
    key_scalars: int
    state_scalars: int
    state_bytes: int


@dataclasses.dataclass(frozen=True)
class SignalingBanditResourceBudget:
    """Exact persistent resource budget for a matched two-role learner."""

    helper: RoleResourceBudget
    beneficiary: RoleResourceBudget
    state_scalars: int
    state_bytes: int


def _zero_role(key: Array) -> RoleBanditState:
    return RoleBanditState(
        values=jnp.zeros(ROLE_VALUE_SHAPE, dtype=jnp.float32),
        key=key,
    )


def _select_role(
    state: RoleBanditState,
    context: Array,
    private_input: Array,
    epsilon: float,
) -> RoleDecision:
    """Select from one table, consuming only that role's policy key."""

    context_i = jnp.asarray(context, dtype=jnp.int32)
    input_i = jnp.asarray(private_input, dtype=jnp.int32)
    values = state.values[context_i, input_i]
    next_key, explore_key, random_action_key, tie_key = jr.split(state.key, 4)
    random_action = jr.randint(random_action_key, (), 0, N_ACTIONS, dtype=jnp.int32)
    tie_action = jr.randint(tie_key, (), 0, N_ACTIONS, dtype=jnp.int32)
    greedy_action = jnp.where(
        values[0] == values[1],
        tie_action,
        jnp.argmax(values).astype(jnp.int32),
    )
    explore = jr.uniform(explore_key, (), dtype=jnp.float32) < jnp.float32(epsilon)
    action = jnp.where(explore, random_action, greedy_action).astype(jnp.int32)
    return RoleDecision(
        context=context_i,
        private_input=input_i,
        action=action,
        selected_value=values[action],
        next_key=next_key,
    )


def decision_with_action(decision: RoleDecision, action: Array) -> RoleDecision:
    """Return a decision carrying an externally specified actual action.

    This is used only by explicit evaluator controls such as an oracle helper.
    It does not reveal an oracle target to the learner update; a frozen write
    mask discards the candidate role state in that control.
    """

    return RoleDecision(
        context=decision.context,
        private_input=decision.private_input,
        action=jnp.asarray(action, dtype=jnp.int32),
        selected_value=decision.selected_value,
        next_key=decision.next_key,
    )


def greedy_role_action(values: Array, context: Array, private_input: Array) -> Array:
    """Read-only deterministic greedy action for evaluator probes."""

    row = values[
        jnp.asarray(context, dtype=jnp.int32),
        jnp.asarray(private_input, dtype=jnp.int32),
    ]
    return jnp.argmax(row).astype(jnp.int32)


def _candidate_role(
    old_state: RoleBanditState,
    decision: RoleDecision,
    reward: Array,
    learning_rate: float,
) -> RoleBanditState:
    old_value = old_state.values[
        decision.context,
        decision.private_input,
        decision.action,
    ]
    candidate_value = old_value + jnp.float32(learning_rate) * (
        jnp.asarray(reward, dtype=jnp.float32) - old_value
    )
    candidate_values = old_state.values.at[
        decision.context,
        decision.private_input,
        decision.action,
    ].set(candidate_value)
    return RoleBanditState(values=candidate_values, key=decision.next_key)


def _commit_role(
    old_state: RoleBanditState,
    candidate_state: RoleBanditState,
    write_mask: Array | bool,
) -> RoleBanditState:
    """Gate value writes while always committing the consumed policy key."""

    values = jax.lax.cond(
        jnp.asarray(write_mask, dtype=jnp.bool_),
        lambda _: candidate_state.values,
        lambda _: old_state.values,
        operand=None,
    )
    return RoleBanditState(values=cast(Array, values), key=candidate_state.key)


class SignalingBanditAgent:
    """Independent contextual-bandit learners for helper and beneficiary."""

    def __init__(self, config: SignalingBanditConfig | None = None) -> None:
        self._config = config or SignalingBanditConfig()

    @property
    def config(self) -> SignalingBanditConfig:
        """Static learner configuration."""

        return self._config

    def init(self, keys: SignalingBanditKeys) -> SignalingBanditState:
        """Initialize both physically separate role states at exact zero."""

        return SignalingBanditState(
            helper=_zero_role(keys.helper),
            beneficiary=_zero_role(keys.beneficiary),
        )

    def init_beneficiary_only(self, key: Array) -> RoleBanditState:
        """Initialize the explicitly resource-unmatched beneficiary baseline."""

        return _zero_role(key)

    def select_helper(
        self,
        state: RoleBanditState,
        public_context: Array,
        private_cue: Array,
    ) -> RoleDecision:
        """Select a message without beneficiary or oracle information."""

        return _select_role(
            state,
            public_context,
            private_cue,
            self._config.epsilon,
        )

    def select_beneficiary(
        self,
        state: RoleBanditState,
        public_context: Array,
        delivered_message: Array,
    ) -> RoleDecision:
        """Select an action without the helper's private cue or oracle target."""

        return _select_role(
            state,
            public_context,
            delivered_message,
            self._config.epsilon,
        )

    def select_beneficiary_only(
        self,
        state: RoleBanditState,
        public_context: Array,
        delivered_message: Array,
    ) -> RoleDecision:
        """Select for the explicitly resource-unmatched one-role baseline."""

        return _select_role(
            state,
            public_context,
            delivered_message,
            self._config.epsilon,
        )

    def update(
        self,
        old_state: SignalingBanditState,
        helper_decision: RoleDecision,
        beneficiary_decision: RoleDecision,
        reward: Array,
        *,
        helper_write: Array | bool = True,
        beneficiary_write: Array | bool = True,
    ) -> SignalingBanditUpdate:
        """Form both candidates from ``old_state`` and commit atomically."""

        # These two candidates deliberately have no dependency on one another.
        helper_candidate = _candidate_role(
            old_state.helper,
            helper_decision,
            reward,
            self._config.learning_rate,
        )
        beneficiary_candidate = _candidate_role(
            old_state.beneficiary,
            beneficiary_decision,
            reward,
            self._config.learning_rate,
        )
        helper = _commit_role(old_state.helper, helper_candidate, helper_write)
        beneficiary = _commit_role(
            old_state.beneficiary,
            beneficiary_candidate,
            beneficiary_write,
        )
        helper_post = helper.values[
            helper_decision.context,
            helper_decision.private_input,
            helper_decision.action,
        ]
        beneficiary_post = beneficiary.values[
            beneficiary_decision.context,
            beneficiary_decision.private_input,
            beneficiary_decision.action,
        ]
        return SignalingBanditUpdate(
            state=SignalingBanditState(helper=helper, beneficiary=beneficiary),
            helper_value_pre=old_state.helper.values[
                helper_decision.context,
                helper_decision.private_input,
                helper_decision.action,
            ],
            helper_value_post=helper_post,
            beneficiary_value_pre=old_state.beneficiary.values[
                beneficiary_decision.context,
                beneficiary_decision.private_input,
                beneficiary_decision.action,
            ],
            beneficiary_value_post=beneficiary_post,
        )

    def update_beneficiary_only(
        self,
        old_state: RoleBanditState,
        decision: RoleDecision,
        reward: Array,
    ) -> RoleBanditState:
        """Update the one-role baseline from its old state."""

        return _candidate_role(
            old_state,
            decision,
            reward,
            self._config.learning_rate,
        )


def role_resource_budget(state: RoleBanditState) -> RoleResourceBudget:
    """Measure exact persistent table/key scalars and bytes."""

    values = np.asarray(state.values)
    key_data = np.asarray(jr.key_data(state.key))
    value_scalars = int(values.size)
    key_scalars = int(key_data.size)
    return RoleResourceBudget(
        value_scalars=value_scalars,
        key_scalars=key_scalars,
        state_scalars=value_scalars + key_scalars,
        state_bytes=int(values.nbytes + key_data.nbytes),
    )


def signaling_bandit_resource_budget(
    state: SignalingBanditState,
) -> SignalingBanditResourceBudget:
    """Measure exact persistent resources for both matched role states."""

    helper = role_resource_budget(state.helper)
    beneficiary = role_resource_budget(state.beneficiary)
    return SignalingBanditResourceBudget(
        helper=helper,
        beneficiary=beneficiary,
        state_scalars=helper.state_scalars + beneficiary.state_scalars,
        state_bytes=helper.state_bytes + beneficiary.state_bytes,
    )
