# mypy: disable-error-code="attr-defined,call-arg,index,no-untyped-def,untyped-decorator"
"""Exact-horizon and atomicity contracts for Step 2 prototype memory."""

from __future__ import annotations

import dataclasses

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import alberta_framework
import alberta_framework.core as core
from alberta_framework.core.prototype_memory import (
    PROTOTYPE_MEMORY_CONFIG_SCHEMA,
    PROTOTYPE_MEMORY_STATE_SCHEMA,
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
    load_prototype_memory_checkpoint,
    migrate_legacy_prototype_memory_state,
    prototype_memory_exact_clock_delta_nbytes,
    prototype_memory_state_delta_nbytes,
    save_prototype_memory_checkpoint,
)
from alberta_framework.core.upgd_memory import (
    UPGD_MEMORY_CONFIG_SCHEMA,
    UPGD_MEMORY_OUTER_CLOCK_DELTA_NBYTES,
    UPGD_MEMORY_STATE_SCHEMA,
    UPGDMemoryConfig,
    UPGDMemoryLearner,
    load_upgd_memory_checkpoint,
    migrate_legacy_upgd_memory_state,
    save_upgd_memory_checkpoint,
)

INT32_MAX = 2**31 - 1
UINT32_MAX = 2**32 - 1
pytestmark = [pytest.mark.unit, pytest.mark.slow]


def _prototype() -> PrototypeMemoryLearner:
    return PrototypeMemoryLearner(
        PrototypeMemoryConfig(
            feature_dim=2,
            n_classes=2,
            slots_per_class=2,
            novelty_threshold=0.01,
        )
    )


def _hybrid() -> UPGDMemoryLearner:
    return UPGDMemoryLearner(
        UPGDMemoryConfig(
            feature_dim=2,
            n_heads=2,
            hidden_sizes=(4,),
            target_trace_blend_scale=0.0,
        )
    )


def _set_prototype_clock(state, words: jax.Array):
    telemetry = jnp.asarray(
        INT32_MAX if int(words[0]) > 0 or int(words[1]) >= INT32_MAX else int(words[1]),
        dtype=jnp.int32,
    )
    return state.replace(step_count=telemetry, step_words=words)


def _set_hybrid_clock(learner: UPGDMemoryLearner, words: jax.Array):
    state = learner.init(jr.key(7))
    telemetry = jnp.asarray(INT32_MAX, dtype=jnp.int32)
    phase = jnp.asarray(int(words[1]) % 16, dtype=jnp.int32)
    upgd = state.upgd_state.replace(
        step_count=telemetry,
        step_words=words,
        perturbation_phase=phase,
    )
    memory = state.memory_state.replace(step_count=telemetry, step_words=words)
    return state.replace(
        upgd_state=upgd,
        memory_state=memory,
        step_count=telemetry,
        step_words=words,
    )


def test_prototype_exact_clock_carries_and_saturates_telemetry() -> None:
    """The exact authority must carry while int32 remains telemetry only."""
    learner = _prototype()
    words = jnp.asarray((1, UINT32_MAX), dtype=jnp.uint32)
    state = _set_prototype_clock(learner.init(), words)

    result = learner.update(
        state,
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray((2, 0), dtype=jnp.uint32),
    )
    assert int(result.state.step_count) == INT32_MAX
    chex.assert_trees_all_equal(result.state.last_update_words[0, 0], result.state.step_words)
    chex.assert_trees_all_equal(result.state.insertion_words[0, 0], result.state.step_words)


def test_prototype_visits_do_not_alias_at_float32_boundary() -> None:
    """Exact visits continue after float telemetry reaches its safe ceiling."""
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(
            feature_dim=1,
            n_classes=2,
            slots_per_class=1,
            novelty_threshold=2.0,
        )
    )
    state = learner.init()
    step = 2**24 + 10
    step_words = jnp.asarray((0, step), dtype=jnp.uint32)
    state = state.replace(
        means=state.means.at[0, 0, 0].set(0.0),
        counts=state.counts.at[0, 0].set(float(2**24 - 1)),
        visit_words=state.visit_words.at[0, 0].set(
            jnp.asarray((0, 2**24 - 1), dtype=jnp.uint32)
        ),
        last_update=state.last_update.at[0, 0].set(step),
        last_update_words=state.last_update_words.at[0, 0].set(step_words),
        insertion_step=state.insertion_step.at[0, 0].set(1),
        insertion_words=state.insertion_words.at[0, 0].set(
            jnp.asarray((0, 1), dtype=jnp.uint32)
        ),
        step_count=jnp.asarray(step, dtype=jnp.int32),
        step_words=step_words,
    )
    observation = jnp.asarray((0.0,), dtype=jnp.float32)
    target = jnp.asarray((1.0, 0.0), dtype=jnp.float32)

    first = learner.update(state, observation, target)
    second = learner.update(first.state, observation, target)

    chex.assert_trees_all_equal(
        first.state.visit_words[0, 0],
        jnp.asarray((0, 2**24), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        second.state.visit_words[0, 0],
        jnp.asarray((0, 2**24 + 1), dtype=jnp.uint32),
    )
    assert float(first.state.counts[0, 0]) == float(2**24)
    assert float(second.state.counts[0, 0]) == float(2**24)
    assert bool(learner.state_is_valid(second.state))


def test_prototype_eviction_uses_exact_visits_then_exact_recency() -> None:
    """LRU tie-breaking must remain deterministic beyond signed telemetry."""
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(
            feature_dim=1,
            n_classes=2,
            slots_per_class=3,
            novelty_threshold=0.01,
        )
    )
    state = learner.init()
    outer = jnp.asarray((1, 100), dtype=jnp.uint32)
    visits = jnp.asarray(((1, 7), (1, 7), (1, 8)), dtype=jnp.uint32)
    uses = jnp.asarray(((1, 50), (1, 60), (1, 70)), dtype=jnp.uint32)
    insertion = jnp.asarray(((0, 1), (0, 2), (0, 3)), dtype=jnp.uint32)
    state = state.replace(
        means=state.means.at[0, :, 0].set(jnp.asarray((0.0, 1.0, 2.0))),
        counts=state.counts.at[0].set(float(2**24)),
        visit_words=state.visit_words.at[0].set(visits),
        last_update=state.last_update.at[0].set(INT32_MAX),
        last_update_words=state.last_update_words.at[0].set(uses),
        insertion_step=state.insertion_step.at[0].set(jnp.asarray((1, 2, 3))),
        insertion_words=state.insertion_words.at[0].set(insertion),
        step_count=jnp.asarray(INT32_MAX, dtype=jnp.int32),
        step_words=outer,
    )
    assert bool(learner.state_is_valid(state))

    result = learner.update(
        state,
        jnp.asarray((100.0,), dtype=jnp.float32),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert float(result.state.means[0, 0, 0]) == 100.0
    chex.assert_trees_all_equal(
        result.state.visit_words[0, 0],
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(result.state.means[0, 1:, 0], jnp.asarray((1.0, 2.0)))


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_prototype_nonfinite_input_is_bit_exact_rollback(bad_value: float) -> None:
    """Dynamic non-finite values must never partially mutate memory clocks."""
    learner = _prototype()
    state = learner.init()
    result = learner.update(
        state,
        jnp.asarray((bad_value, 0.0), dtype=jnp.float32),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
    )
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_prototype_terminal_identity_refuses_without_wrap_under_jit() -> None:
    """All-ones is an explicit terminal resource boundary, not a wrap."""
    learner = _prototype()
    maximum = jnp.full((2,), UINT32_MAX, dtype=jnp.uint32)
    state = _set_prototype_clock(learner.init(), maximum)

    result = jax.jit(learner.update)(
        state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_hybrid_exact_children_and_outer_carry_together() -> None:
    """Composite, UPGD, and prototype event identities must stay identical."""
    learner = _hybrid()
    state = _set_hybrid_clock(learner, jnp.asarray((1, 5), dtype=jnp.uint32))
    result = learner.update(
        state,
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    expected = jnp.asarray((1, 6), dtype=jnp.uint32)
    chex.assert_trees_all_equal(result.state.step_words, expected)
    chex.assert_trees_all_equal(result.state.upgd_state.step_words, expected)
    chex.assert_trees_all_equal(result.state.memory_state.step_words, expected)
    assert int(result.state.step_count) == INT32_MAX


def test_hybrid_child_refusal_rolls_back_memory_rng_optimizer_and_blend() -> None:
    """A rejected UPGD proposal must undo an otherwise accepted memory child."""
    learner = _hybrid()
    state = learner.init(jr.key(13))
    result = learner.update(
        state,
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray((1e20, 0.0), dtype=jnp.float32),
    )
    assert not bool(result.upgd_update_applied)
    assert bool(result.memory_update_applied)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_hybrid_nonfinite_and_terminal_transactions_are_atomic_under_scan() -> None:
    """Rejected scan elements retain the entire incoming carry bit-for-bit."""
    learner = _hybrid()
    maximum = jnp.full((2,), UINT32_MAX, dtype=jnp.uint32)
    terminal = _set_hybrid_clock(learner, maximum)
    observations = jnp.asarray(((jnp.nan, 0.0), (1.0, 0.0)), dtype=jnp.float32)
    targets = jnp.asarray(((1.0, 0.0), (1.0, 0.0)), dtype=jnp.float32)

    def body(carry, batch):
        result = learner.update(carry, batch[0], batch[1])
        return result.state, result.update_applied

    final, applied = jax.jit(lambda s: jax.lax.scan(body, s, (observations, targets)))(
        terminal
    )
    chex.assert_trees_all_equal(applied, jnp.asarray((False, False)))
    chex.assert_trees_all_equal(final, terminal)


def test_legacy_migrations_require_unsaturated_unaliased_telemetry() -> None:
    """Migration must never invent wrapped or float-aliased history."""
    prototype = _prototype()
    short = prototype.update(
        prototype.init(),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
    ).state
    legacy_memory = {
        "means": short.means,
        "counts": short.counts,
        "last_update": short.last_update,
        "step_count": short.step_count,
    }
    migrated = migrate_legacy_prototype_memory_state(
        legacy_memory,
        config=prototype.config,
    )
    chex.assert_trees_all_equal(migrated.step_words, jnp.asarray((0, 1), dtype=jnp.uint32))
    with pytest.raises(ValueError, match="saturated"):
        migrate_legacy_prototype_memory_state(
            {**legacy_memory, "step_count": jnp.asarray(INT32_MAX, dtype=jnp.int32)},
            config=prototype.config,
        )
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_prototype_memory_state(
            {
                **legacy_memory,
                "counts": legacy_memory["counts"].at[0, 0].set(float(2**24)),
            },
            config=prototype.config,
        )

    hybrid = _hybrid()
    hybrid_state = hybrid.update(
        hybrid.init(jr.key(4)),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
    ).state
    legacy_hybrid = {
        field.name: getattr(hybrid_state, field.name)
        for field in dataclasses.fields(hybrid_state)
        if field.name != "step_words"
    }
    migrated_hybrid = migrate_legacy_upgd_memory_state(
        legacy_hybrid,
        config=hybrid.config,
    )
    assert bool(hybrid.state_is_valid(migrated_hybrid))
    with pytest.raises(ValueError, match="saturated"):
        migrate_legacy_upgd_memory_state(
            {
                **legacy_hybrid,
                "step_count": jnp.asarray(INT32_MAX, dtype=jnp.int32),
            },
            config=hybrid.config,
        )


def test_v2_schema_resources_and_checkpoint_roundtrip(tmp_path) -> None:
    """Schemas are strict and checkpoints bind exact resource declarations."""
    prototype = _prototype()
    assert prototype.to_config()["config"]["state_schema"] == PROTOTYPE_MEMORY_STATE_SCHEMA
    assert prototype.to_config()["config"]["config_schema"] == PROTOTYPE_MEMORY_CONFIG_SCHEMA
    bad_prototype_config = dict(prototype.config.to_config())
    bad_prototype_config["extra"] = True
    with pytest.raises(ValueError, match="manifest"):
        PrototypeMemoryConfig.from_config(bad_prototype_config)
    budget = prototype.resource_budget()
    assert budget.exact_clock_delta_nbytes == prototype_memory_exact_clock_delta_nbytes(
        prototype.config
    )
    assert budget.state_delta_nbytes == prototype_memory_state_delta_nbytes(prototype.config)
    prototype_path = tmp_path / "prototype"
    save_prototype_memory_checkpoint(prototype, prototype.init(), prototype_path)
    restored_prototype, restored_memory = load_prototype_memory_checkpoint(prototype_path)
    assert restored_prototype.config == prototype.config
    chex.assert_trees_all_equal(restored_memory, prototype.init())

    hybrid = _hybrid()
    assert hybrid.to_config()["config"]["state_schema"] == UPGD_MEMORY_STATE_SCHEMA
    assert hybrid.to_config()["config"]["config_schema"] == UPGD_MEMORY_CONFIG_SCHEMA
    bad_hybrid_config = dict(hybrid.config.to_config())
    bad_hybrid_config.pop("state_schema")
    with pytest.raises(ValueError, match="manifest"):
        UPGDMemoryConfig.from_config(bad_hybrid_config)
    hybrid_state = hybrid.init(jr.key(23))
    hybrid_budget = hybrid.resource_budget(hybrid_state)
    assert hybrid_budget.outer_clock_delta_nbytes == UPGD_MEMORY_OUTER_CLOCK_DELTA_NBYTES
    hybrid_path = tmp_path / "hybrid"
    save_upgd_memory_checkpoint(hybrid, hybrid_state, hybrid_path)
    restored_hybrid, restored_state = load_upgd_memory_checkpoint(hybrid_path)
    assert restored_hybrid.config == hybrid.config
    chex.assert_trees_all_equal(restored_state, hybrid_state)


def test_exact_memory_public_exports_are_available() -> None:
    """Both supported package surfaces expose horizon and persistence APIs."""
    names = (
        "PROTOTYPE_MEMORY_STATE_SCHEMA",
        "PROTOTYPE_MEMORY_CONFIG_SCHEMA",
        "PrototypeMemoryResourceBudget",
        "migrate_legacy_prototype_memory_state",
        "UPGD_MEMORY_STATE_SCHEMA",
        "UPGD_MEMORY_CONFIG_SCHEMA",
        "UPGDMemoryResourceBudget",
        "migrate_legacy_upgd_memory_state",
        "save_upgd_memory_checkpoint",
    )
    for name in names:
        assert hasattr(alberta_framework, name)
        assert hasattr(core, name)


def test_memory_configs_reject_nonfinite_hyperparameters() -> None:
    """NaN/inf hyperparameters must be rejected before tracing a learner."""
    with pytest.raises(ValueError, match="finite"):
        PrototypeMemoryLearner(
            PrototypeMemoryConfig(
                feature_dim=2,
                n_classes=2,
                novelty_threshold=float("nan"),
            )
        )
    with pytest.raises(ValueError, match="finite"):
        UPGDMemoryLearner(
            UPGDMemoryConfig(
                feature_dim=2,
                n_heads=2,
                confidence_logit_scale=float("inf"),
            )
        )
