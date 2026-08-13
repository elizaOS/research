"""Focused causal contracts for compositional-feature curation internals."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core import compositional_features as cf


def test_operation_logits_preserve_exact_zero_support() -> None:
    learner = cf.CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        operation_prior=(0.0, 0.5, 0.0, 0.5, 0.0),
    )

    logits = learner._op_logits()  # noqa: SLF001 - exact generator contract
    probabilities = jax.nn.softmax(logits)

    np.testing.assert_array_equal(
        np.isneginf(np.asarray(logits)),
        np.asarray((True, False, True, False, True)),
    )
    np.testing.assert_array_equal(
        np.asarray(probabilities)[[cf.OP_RAW, cf.OP_SUM, cf.OP_GATED]],
        np.zeros((3,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(probabilities)[[cf.OP_PRODUCT, cf.OP_TANH]],
        np.asarray((0.5, 0.5), dtype=np.float32),
    )

    forced = jax.nn.softmax(
        learner._op_logits(jnp.asarray(cf.OP_SUM, dtype=jnp.int32))  # noqa: SLF001
    )
    np.testing.assert_array_equal(
        np.asarray(forced),
        np.eye(cf.NUM_OPS, dtype=np.float32)[cf.OP_SUM],
    )


@pytest.mark.parametrize(
    ("operation_prior", "message"),
    (
        ((0.1, 0.9, 0.0, 0.0, 0.0), "cannot assign mass to OP_RAW"),
        ((0.0, float("nan"), 1.0, 0.0, 0.0), "finite real numbers"),
        ((0.0, float("inf"), 1.0, 0.0, 0.0), "finite real numbers"),
        ((0.0, True, 0.0, 0.0, 0.0), "finite real numbers"),
        ((0.0, 0.0, 0.0, 0.0, 0.0), "positive finite composing mass"),
    ),
)
def test_operation_prior_rejects_invalid_generated_op_support(
    operation_prior: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cf.CompositionalFeatureLearner(
            n_features=4,
            n_tasks=1,
            operation_prior=operation_prior,
        )


class _FailIfGeneratorPolicyIsSelected:
    def select(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("disabled generator resources sampled a meta-policy")


def test_disabled_generator_resources_never_sample_or_write_random_policy() -> None:
    learner = cf.CompositionalFeatureLearner(
        n_features=3,
        n_tasks=1,
        candidate_count=1,
        replacement_interval=1,
        min_feature_age=100,
        candidate_min_age=0,
        promotion_margin=1_000.0,
        learn_generator_resources=False,
        use_obgd=False,
    )
    state = learner.init(feature_dim=2, key=jr.key(71)).replace(  # type: ignore[attr-defined]
        candidate_ages=jnp.asarray((1,), dtype=jnp.int32),
        candidate_generator_policy=jnp.asarray((3,), dtype=jnp.int32),
    )
    learner._generator_resource_manager = (  # noqa: SLF001 - prove no selection call
        _FailIfGeneratorPolicyIsSelected()  # type: ignore[assignment]
    )

    result = learner.update(
        state,
        jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        jnp.asarray((0.75,), dtype=jnp.float32),
    )

    assert int(result.promoted_candidate) == -1
    assert int(result.state.candidate_ages[0]) == 0
    assert (
        int(result.state.candidate_generator_policy[0])
        == cf.FIXED_GENERATOR_POLICY_PLACEHOLDER
        == 0
    )


def test_named_curation_key_domains_are_pinned_and_disjoint() -> None:
    assert cf.COMPOSITIONAL_CURATION_PROPOSAL_CHANNEL == 0x50524F50
    assert cf.COMPOSITIONAL_CURATION_CASCADE_CHANNEL == 0x43415343
    assert cf.COMPOSITIONAL_CURATION_OVERDEPTH_REGENERATION_CHANNEL == 0x4F445247
    root = jr.wrap_key_data(
        jnp.asarray((0x13579BDF, 0x2468ACE0), dtype=jnp.uint32),
        impl="threefry2x32",
    )

    proposal_key, cascade_key = cf.compositional_curation_keys(root)
    overdepth_regeneration_key = jr.fold_in(
        root,
        jnp.uint32(cf.COMPOSITIONAL_CURATION_OVERDEPTH_REGENERATION_CHANNEL),
    )

    assert str(jr.key_impl(proposal_key)) == "threefry2x32"
    assert str(jr.key_impl(cascade_key)) == "threefry2x32"
    assert str(jr.key_impl(overdepth_regeneration_key)) == "threefry2x32"
    assert tuple(int(value) for value in jr.key_data(proposal_key)) == (
        552_745_620,
        1_216_138_420,
    )
    assert tuple(int(value) for value in jr.key_data(cascade_key)) == (
        2_761_025_115,
        2_096_087_055,
    )
    assert tuple(int(value) for value in jr.key_data(overdepth_regeneration_key)) == (
        3_550_815_958,
        3_549_559_133,
    )
    key_words = {
        tuple(int(value) for value in jr.key_data(key))
        for key in (proposal_key, cascade_key, overdepth_regeneration_key)
    }
    assert len(key_words) == 3
