"""Equation and transaction tests for the clean-room C-CHAIN L0 comparator."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework as package_root
import alberta_framework.core as core_root
import alberta_framework.core.cchain as cchain_module
from alberta_framework.core.cchain import (
    CCHAIN_EXTERNAL_OPTIMIZER_APPLICATION_AUTHENTICATED,
    CCHAIN_MECHANISM_STATUS,
    CCHAIN_SCIENTIFIC_PROMOTION_ALLOWED,
    CChain,
    CChainConfig,
    empirical_ntk_diagnostics,
    load_cchain_checkpoint,
    save_cchain_checkpoint,
    squared_output_churn,
)
from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata,
    save_checkpoint,
)

pytestmark = pytest.mark.unit


def test_public_roots_export_the_exact_cchain_surface_once() -> None:
    for root in (package_root, core_root):
        for name in cchain_module.__all__:
            assert getattr(root, name) is getattr(cchain_module, name)
            assert root.__all__.count(name) == 1


def _model(params: Any, batch: jax.Array) -> jax.Array:
    return batch @ params["w"] + params["b"]


def _flat_model(params: Any, batch: jax.Array) -> jax.Array:
    return jnp.squeeze(_model(params, batch), axis=-1)


def _deep_singleton_model(params: Any, batch: jax.Array) -> jax.Array:
    return _model(params, batch)[:, :, None]


def _vector_model(params: Any, batch: jax.Array) -> jax.Array:
    scalar = _model(params, batch)
    return jnp.concatenate((scalar, -scalar), axis=1)


def _loss(params: Any, batch: tuple[jax.Array, jax.Array]) -> jax.Array:
    observations, targets = batch
    error = _model(params, observations) - targets
    return 0.5 * jnp.mean(jnp.square(error))


def _params() -> dict[str, jax.Array]:
    return {
        "w": jnp.asarray([[0.25], [-0.5]], dtype=jnp.float32),
        "b": jnp.asarray([0.1], dtype=jnp.float32),
    }


def _ids(start: int, count: int) -> jax.Array:
    return jnp.stack(
        (
            jnp.zeros((count,), dtype=jnp.uint32),
            jnp.arange(start, start + count, dtype=jnp.uint32),
        ),
        axis=1,
    )


def _mechanism(**overrides: object) -> CChain:
    fields: dict[str, object] = {
        "model_binding_words": (0x4D4F444C, 1),
        "loss_binding_words": (0x4C4F5353, 1),
        "target_relative_scale": 0.2,
        "initial_coefficient": 0.75,
        "auto_scale_warmup_commits": 3,
        "auto_scale_window": 4,
        "minimum_coefficient": 0.0,
        "maximum_coefficient": 100.0,
        "max_commits": 100,
    }
    fields.update(overrides)
    return CChain(_model, _loss, CChainConfig(**fields))  # type: ignore[arg-type]


def _batches() -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    train_x = jnp.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=jnp.float32)
    train_y = jnp.asarray([[0.2], [-0.4]], dtype=jnp.float32)
    reference = jnp.asarray([[2.0, -1.0], [0.5, 0.25]], dtype=jnp.float32)
    return (train_x, train_y), reference


def _propose(mechanism: CChain, state: Any, params: Any) -> Any:
    train, reference = _batches()
    return mechanism.propose(
        state,
        params,
        train,
        reference,
        train_sample_ids=_ids(1, 2),
        reference_sample_ids=_ids(11, 2),
    )


def _sgd(params: Any, gradients: Any, step_size: float = 0.1) -> Any:
    return jax.tree.map(
        lambda parameter, gradient: parameter - step_size * gradient,
        params,
        gradients,
    )


def _tree_exact(left: Any, right: Any) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_config_roundtrip_and_machine_readable_l0_boundaries() -> None:
    mechanism = _mechanism()
    payload = mechanism.to_config()
    restored = CChain.from_config(payload, model_fn=_model, base_loss_fn=_loss)

    assert restored.to_config() == payload
    assert restored.config == mechanism.config
    assert payload["mechanism_status"] == "l0-development-only-not-assessed"
    assert payload["evidence_level"] == "L0"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["dispatch_authority"] is False
    assert payload["output_authority"] is False
    assert payload["model_binding_authenticated"] is False
    assert payload["loss_binding_authenticated"] is False
    assert payload["external_optimizer_application_authenticated"] is False
    assert payload["equation_8_objective_exact"] is True
    assert payload["comparator_scope"] == ("isolated-one-step-lag-combined-gradient-comparator")
    assert payload["full_sequential_algorithm_reproduced"] is False
    assert payload["efficacy_assessed"] is False
    assert payload["default_agent_integration"] is False
    paper_source = payload["paper_source"]
    assert isinstance(paper_source, dict)
    assert paper_source["implementation_origin"] == "clean-room-paper-equation"
    assert paper_source["public_repository_source_code_used"] is False
    assert payload["equation_8_objective_profile"] == (
        "tang-2025-equation-8-exact-squared-output-churn"
    )
    assert payload["autoscale_control_profile"] == (
        "appendix-absolute-loss-ratio-with-explicit-alberta-window-warmup-epsilon-bounds"
    )
    assert payload["autoscale_controls_are_equation_8"] is False
    assert payload["unrelated_selection_semantics"] is False
    assert CCHAIN_MECHANISM_STATUS == "l0-development-only-not-assessed"
    assert CCHAIN_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert CCHAIN_EXTERNAL_OPTIMIZER_APPLICATION_AUTHENTICATED is False

    missing = dict(payload)
    missing.pop("evidence_level")
    with pytest.raises(ValueError, match="fields"):
        CChain.from_config(missing, model_fn=_model, base_loss_fn=_loss)
    changed = dict(payload)
    changed["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="must remain false"):
        CChain.from_config(changed, model_fn=_model, base_loss_fn=_loss)


def test_equation_8_scalar_output_shape_is_exact_and_vector_extension_rejected() -> None:
    current = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    reference = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    scalar = squared_output_churn(current, reference)
    singleton = squared_output_churn(current[:, None], reference[:, None])
    deep_singleton = squared_output_churn(
        current[:, None, None],
        reference[:, None, None],
    )
    assert float(scalar) == pytest.approx(float(singleton))
    assert float(scalar) == pytest.approx(float(deep_singleton))

    with pytest.raises(ValueError, match="exactly one scalar per reference sample"):
        squared_output_churn(
            jnp.stack((current, -current), axis=1),
            jnp.stack((reference, -reference), axis=1),
        )

    config = CChainConfig(
        model_binding_words=(0x4D4F444C, 2),
        loss_binding_words=(0x4C4F5353, 2),
    )
    params = _params()
    for scalar_model in (_flat_model, _model, _deep_singleton_model):
        mechanism = CChain(scalar_model, _loss, config)
        assert bool(_propose(mechanism, mechanism.init(params), params).proposal.valid)

    mechanism = CChain(_vector_model, _loss, config)
    with pytest.raises(ValueError, match="exactly one scalar output per reference sample"):
        _propose(mechanism, mechanism.init(params), params)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_binding_words", (0, 0)),
        ("loss_binding_words", (1,)),
        ("target_relative_scale", -0.1),
        ("initial_coefficient", float("nan")),
        ("auto_scale_epsilon", 0.0),
        ("auto_scale_warmup_commits", -1),
        ("auto_scale_window", 0),
        ("auto_scale_window", 4_097),
        ("minimum_coefficient", -1.0),
        ("maximum_coefficient", 0.0),
        ("max_commits", 0),
        ("max_commits", 2**64),
    ],
)
def test_config_rejects_invalid_values(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "model_binding_words": (1, 2),
        "loss_binding_words": (3, 4),
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        CChainConfig(**kwargs)  # type: ignore[arg-type]


def test_squared_output_churn_matches_equation_eight_exactly() -> None:
    current = jnp.asarray([2.0, -1.0, 0.0, 3.0], dtype=jnp.float32)
    reference = jnp.asarray([1.0, 1.0, 2.0, 2.0], dtype=jnp.float32)
    expected = 0.5 * np.mean(np.square(np.asarray(current - reference)))

    actual = squared_output_churn(current, reference)

    assert float(actual) == pytest.approx(float(expected))
    gradient = jax.grad(lambda values: squared_output_churn(values, reference))(current)
    np.testing.assert_allclose(
        np.asarray(gradient),
        np.asarray(current - reference) / current.size,
    )
    stopped_reference_gradient = jax.grad(lambda values: squared_output_churn(current, values))(
        reference
    )
    np.testing.assert_array_equal(
        np.asarray(stopped_reference_gradient),
        np.zeros_like(np.asarray(reference)),
    )


def test_first_step_zero_churn_then_exact_one_update_lag() -> None:
    mechanism = _mechanism(auto_scale_warmup_commits=10)
    params = _params()
    state = mechanism.init(params)

    first = _propose(mechanism, state, params)
    assert bool(first.proposal.valid)
    assert float(first.proposal.churn_loss) == 0.0
    assert int(first.diagnostics.autodiff_pass_count) == 1

    applied = _sgd(params, first.proposal.gradients)
    committed = mechanism.commit(state, first.proposal, applied)
    assert bool(committed.diagnostics.applied)
    _tree_exact(committed.state.reference_params, params)
    _tree_exact(committed.state.expected_current_params, applied)

    second = _propose(mechanism, committed.state, applied)
    _, reference_batch = _batches()
    expected = 0.5 * jnp.mean(
        jnp.square(_model(applied, reference_batch) - _model(params, reference_batch))
    )
    assert float(second.proposal.churn_loss) == pytest.approx(float(expected))
    reference_gradient = jax.grad(
        lambda old: squared_output_churn(
            _model(applied, reference_batch),
            _model(old, reference_batch),
        )
    )(params)
    chex.assert_trees_all_equal(reference_gradient, jax.tree.map(jnp.zeros_like, params))


def test_combined_loss_and_gradient_match_manual_equation() -> None:
    mechanism = _mechanism(initial_coefficient=2.0, auto_scale_warmup_commits=10)
    params = _params()
    state = mechanism.init(params)
    first = _propose(mechanism, state, params)
    shifted = jax.tree.map(lambda value: value + 0.2, params)
    state = mechanism.commit(state, first.proposal, shifted).state

    result = _propose(mechanism, state, shifted)
    train, reference = _batches()

    def manual_loss(candidate: Any) -> jax.Array:
        base = _loss(candidate, train)
        churn = 0.5 * jnp.mean(jnp.square(_model(candidate, reference) - _model(params, reference)))
        return base + 2.0 * churn

    expected_loss, expected_gradient = jax.value_and_grad(manual_loss)(shifted)
    assert float(result.proposal.combined_loss) == pytest.approx(float(expected_loss))
    chex.assert_trees_all_close(result.proposal.gradients, expected_gradient)
    assert float(result.proposal.combined_loss) == pytest.approx(
        float(result.proposal.base_loss + 2.0 * result.proposal.churn_loss)
    )


def test_external_parameter_substitution_replay_and_tamper_fail_closed() -> None:
    mechanism = _mechanism()
    params = _params()
    state = mechanism.init(params)
    proposal = _propose(mechanism, state, params).proposal
    applied = _sgd(params, proposal.gradients)
    committed = mechanism.commit(state, proposal, applied)
    assert bool(committed.diagnostics.applied)

    substituted = jax.tree.map(lambda value: value + 7.0, applied)
    rejected = _propose(mechanism, committed.state, substituted)
    assert not bool(rejected.diagnostics.current_params_match)
    assert int(rejected.diagnostics.autodiff_pass_count) == 0
    assert not bool(rejected.proposal.valid)

    replay = mechanism.commit(committed.state, proposal, applied)
    assert not bool(replay.diagnostics.applied)
    assert not bool(replay.diagnostics.source_fresh)
    assert int(replay.diagnostics.autodiff_pass_count) == 0
    _tree_exact(replay.state, committed.state)

    tampered = proposal.replace(base_loss=proposal.base_loss + 1.0)
    tamper_result = mechanism.commit(state, tampered, applied)
    assert not bool(tamper_result.diagnostics.proposal_integrity_valid)
    assert not bool(tamper_result.diagnostics.applied)
    _tree_exact(tamper_result.state, state)

    binding_tamper = proposal.replace(
        model_binding_words=jnp.asarray((0xDEAD, 0xBEEF), dtype=jnp.uint32)
    )
    binding_result = mechanism.commit(state, binding_tamper, applied)
    assert not bool(binding_result.diagnostics.binding_words_match)
    assert not bool(binding_result.diagnostics.applied)
    _tree_exact(binding_result.state, state)


@pytest.mark.parametrize(
    "failure",
    ["train_zero", "train_duplicate", "reference_duplicate", "overlap"],
)
def test_invalid_sample_identity_preflight_performs_zero_autodiff(failure: str) -> None:
    mechanism = _mechanism()
    params = _params()
    state = mechanism.init(params)
    train, reference = _batches()
    train_ids = _ids(1, 2)
    reference_ids = _ids(11, 2)
    if failure == "train_zero":
        train_ids = train_ids.at[0].set(jnp.zeros((2,), dtype=jnp.uint32))
    elif failure == "train_duplicate":
        train_ids = train_ids.at[1].set(train_ids[0])
    elif failure == "reference_duplicate":
        reference_ids = reference_ids.at[1].set(reference_ids[0])
    else:
        reference_ids = reference_ids.at[0].set(train_ids[1])

    result = mechanism.propose(
        state,
        params,
        train,
        reference,
        train_sample_ids=train_ids,
        reference_sample_ids=reference_ids,
    )

    assert not bool(result.diagnostics.sample_identity_preflight_valid)
    assert not bool(result.proposal.valid)
    assert int(result.diagnostics.autodiff_pass_count) == 0


def test_invalid_preflight_never_invokes_model_or_loss_callables() -> None:
    def forbidden_model(params: Any, batch: Any) -> jax.Array:
        del params, batch
        raise AssertionError("invalid preflight invoked model_fn")

    def forbidden_loss(params: Any, batch: Any) -> jax.Array:
        del params, batch
        raise AssertionError("invalid preflight invoked base_loss_fn")

    mechanism = CChain(
        forbidden_model,
        forbidden_loss,
        CChainConfig(model_binding_words=(1, 2), loss_binding_words=(3, 4)),
    )
    params = _params()
    state = mechanism.init(params)
    train, reference = _batches()
    result = mechanism.propose(
        state,
        params,
        train,
        reference,
        train_sample_ids=_ids(1, 2),
        reference_sample_ids=_ids(1, 2),
    )

    assert not bool(result.diagnostics.preflight_valid)
    assert int(result.diagnostics.autodiff_pass_count) == 0


def test_valid_proposal_performs_one_autodiff_and_commit_performs_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_value_and_grad = cchain_module.jax.value_and_grad
    autodiff_transform_calls = 0

    def counted_value_and_grad(*args: Any, **kwargs: Any) -> Any:
        nonlocal autodiff_transform_calls
        autodiff_transform_calls += 1
        return original_value_and_grad(*args, **kwargs)

    monkeypatch.setattr(cchain_module.jax, "value_and_grad", counted_value_and_grad)
    mechanism = _mechanism()
    params = _params()
    state = mechanism.init(params)
    proposed = _propose(mechanism, state, params)
    assert bool(proposed.diagnostics.preflight_valid)
    assert int(proposed.diagnostics.autodiff_pass_count) == 1
    assert autodiff_transform_calls == 1

    applied = _sgd(params, proposed.proposal.gradients)
    committed = mechanism.commit(state, proposed.proposal, applied)
    assert int(committed.diagnostics.autodiff_pass_count) == 0
    assert bool(committed.diagnostics.applied)
    assert autodiff_transform_calls == 1


def test_numeric_invalid_proposal_and_applied_params_are_atomic() -> None:
    def invalid_loss(params: Any, batch: tuple[jax.Array, jax.Array]) -> jax.Array:
        del batch
        return jnp.asarray(jnp.nan, dtype=jnp.float32) * jnp.sum(params["w"])

    config = CChainConfig(model_binding_words=(1, 1), loss_binding_words=(2, 2))
    mechanism = CChain(_model, invalid_loss, config)
    params = _params()
    state = mechanism.init(params)
    train, reference = _batches()
    proposed = mechanism.propose(
        state,
        params,
        train,
        reference,
        train_sample_ids=_ids(1, 2),
        reference_sample_ids=_ids(11, 2),
    )
    assert int(proposed.diagnostics.autodiff_pass_count) == 1
    assert not bool(proposed.proposal.valid)
    rejected = mechanism.commit(state, proposed.proposal, params)
    assert not bool(rejected.diagnostics.applied)
    _tree_exact(rejected.state, state)

    good_mechanism = _mechanism()
    good_state = good_mechanism.init(params)
    good = _propose(good_mechanism, good_state, params).proposal
    nonfinite = dict(params)
    nonfinite["w"] = params["w"].at[0, 0].set(jnp.nan)
    rejected_applied = good_mechanism.commit(good_state, good, nonfinite)
    assert not bool(rejected_applied.diagnostics.applied_params_finite)
    assert not bool(rejected_applied.diagnostics.applied)
    _tree_exact(rejected_applied.state, good_state)


def test_auto_scale_uses_committed_trailing_window_and_warmup() -> None:
    def scalar_model(params: Any, batch: jax.Array) -> jax.Array:
        return jnp.ones_like(batch) * params["w"]

    def supplied_loss(params: Any, batch: jax.Array) -> jax.Array:
        return jnp.mean(batch) + jnp.asarray(0.0, dtype=jnp.float32) * params["w"]

    mechanism = CChain(
        scalar_model,
        supplied_loss,
        CChainConfig(
            model_binding_words=(10, 1),
            loss_binding_words=(11, 1),
            target_relative_scale=0.5,
            initial_coefficient=1.0,
            auto_scale_epsilon=0.1,
            auto_scale_warmup_commits=2,
            auto_scale_window=2,
            minimum_coefficient=0.1,
            maximum_coefficient=10.0,
            max_commits=10,
        ),
    )
    params = {"w": jnp.asarray(0.0, dtype=jnp.float32)}
    state = mechanism.init(params)
    reference = jnp.ones((2,), dtype=jnp.float32)

    def transition(
        source_state: Any,
        source_params: Any,
        base_loss: float,
        applied_value: float,
    ) -> Any:
        proposed = mechanism.propose(
            source_state,
            source_params,
            jnp.full((2,), base_loss, dtype=jnp.float32),
            reference,
            train_sample_ids=_ids(1, 2),
            reference_sample_ids=_ids(11, 2),
        )
        assert bool(proposed.proposal.valid)
        return proposed, mechanism.commit(
            source_state,
            proposed.proposal,
            {"w": jnp.asarray(applied_value, dtype=jnp.float32)},
        )

    first, committed = transition(state, params, 2.0, 2.0)
    assert float(first.proposal.churn_loss) == 0.0
    assert float(committed.state.coefficient) == pytest.approx(1.0)

    second, committed = transition(committed.state, {"w": jnp.float32(2.0)}, 4.0, 3.0)
    assert float(second.proposal.churn_loss) == pytest.approx(2.0)
    # beta * mean(|base|) / max(mean(churn), epsilon) = .5 * 3 / 1
    assert float(committed.state.coefficient) == pytest.approx(1.5)

    third, committed = transition(committed.state, {"w": jnp.float32(3.0)}, 8.0, 4.0)
    assert float(third.proposal.coefficient_used) == pytest.approx(1.5)
    assert float(third.proposal.churn_loss) == pytest.approx(0.5)
    # Trailing two commits only: .5 * mean(4, 8) / mean(2, .5) = 2.4.
    assert float(committed.state.coefficient) == pytest.approx(2.4)
    np.testing.assert_allclose(
        np.sort(np.asarray(committed.state.base_loss_window)),
        np.asarray([4.0, 8.0]),
    )


def test_target_zero_disables_regularization_and_coefficient_updates() -> None:
    with pytest.raises(ValueError, match="initial_coefficient must be zero"):
        CChainConfig(
            model_binding_words=(1, 2),
            loss_binding_words=(3, 4),
            target_relative_scale=0.0,
            initial_coefficient=1.0,
        )

    mechanism = _mechanism(target_relative_scale=0.0, initial_coefficient=0.0)
    params = _params()
    state = mechanism.init(params)
    proposed = _propose(mechanism, state, params)
    assert float(proposed.proposal.coefficient_used) == 0.0
    assert float(proposed.proposal.combined_loss) == pytest.approx(
        float(proposed.proposal.base_loss)
    )
    committed = mechanism.commit(state, proposed.proposal, params)
    assert float(committed.state.coefficient) == 0.0


def test_parameter_tree_and_sample_id_static_contracts() -> None:
    mechanism = _mechanism()
    with pytest.raises(ValueError, match="at least one"):
        mechanism.init({})
    with pytest.raises(TypeError, match="float32"):
        mechanism.init({"w": jnp.ones((2,), dtype=jnp.int32)})

    params = _params()
    state = mechanism.init(params)
    train, reference = _batches()
    with pytest.raises(TypeError, match="parameter PyTree"):
        mechanism.propose(
            state,
            {"different": jnp.ones((2,), dtype=jnp.float32)},
            train,
            reference,
            train_sample_ids=_ids(1, 2),
            reference_sample_ids=_ids(11, 2),
        )
    with pytest.raises(TypeError, match="uint32"):
        mechanism.propose(
            state,
            params,
            train,
            reference,
            train_sample_ids=_ids(1, 2).astype(jnp.int32),
            reference_sample_ids=_ids(11, 2),
        )
    with pytest.raises(TypeError, match="shape"):
        mechanism.propose(
            state,
            params,
            train,
            reference,
            train_sample_ids=jnp.ones((2, 3), dtype=jnp.uint32),
            reference_sample_ids=_ids(11, 2),
        )
    with pytest.raises(ValueError, match="sample-ID batch dimension"):
        mechanism.propose(
            state,
            params,
            train,
            reference[:1],
            train_sample_ids=_ids(1, 2),
            reference_sample_ids=_ids(11, 2),
        )


def test_eager_and_jit_proposal_commit_parity() -> None:
    mechanism = _mechanism()
    params = _params()
    state = mechanism.init(params)
    train, reference = _batches()
    arguments = (
        state,
        params,
        train,
        reference,
        _ids(1, 2),
        _ids(11, 2),
    )

    def propose(
        source_state: Any,
        source_params: Any,
        train_batch: Any,
        reference_batch: Any,
        train_ids: jax.Array,
        reference_ids: jax.Array,
    ) -> Any:
        return mechanism.propose(
            source_state,
            source_params,
            train_batch,
            reference_batch,
            train_sample_ids=train_ids,
            reference_sample_ids=reference_ids,
        )

    eager = propose(*arguments)
    compiled = jax.jit(propose)(*arguments)
    chex.assert_trees_all_close(eager, compiled)
    applied = _sgd(params, eager.proposal.gradients)
    eager_commit = mechanism.commit(state, eager.proposal, applied)
    compiled_commit = jax.jit(mechanism.commit)(state, eager.proposal, applied)
    chex.assert_trees_all_close(eager_commit, compiled_commit)


def test_max_commit_words_are_exact_and_capacity_fails_atomically() -> None:
    mechanism = _mechanism(max_commits=1)
    params = _params()
    state = mechanism.init(params)
    first = _propose(mechanism, state, params)
    committed = mechanism.commit(state, first.proposal, params)
    np.testing.assert_array_equal(
        np.asarray(committed.state.commit_count_words),
        np.asarray([0, 1], dtype=np.uint32),
    )
    blocked = _propose(mechanism, committed.state, params)
    assert not bool(blocked.diagnostics.commit_capacity_available)
    assert int(blocked.diagnostics.autodiff_pass_count) == 0
    assert not bool(blocked.proposal.valid)


def test_high_word_modulus_matches_python_bigint_for_non_power_of_two_window() -> None:
    words = jnp.asarray((0xFEDCBA98, 0x76543210), dtype=jnp.uint32)
    modulus = 1_000_003
    expected = ((0xFEDCBA98 << 32) | 0x76543210) % modulus

    actual = cchain_module._words_mod(words, modulus)  # noqa: SLF001

    assert int(actual) == expected


def test_state_cursor_remains_valid_across_uint32_low_word_rollover() -> None:
    window = 7
    mechanism = _mechanism(
        target_relative_scale=0.0,
        initial_coefficient=0.0,
        auto_scale_window=window,
        max_commits=(1 << 32) + 1,
    )
    params = _params()
    state = mechanism.init(params)
    pre_count = (1 << 32) - 1
    synthetic = state.replace(
        base_loss_window=jnp.zeros((window,), dtype=jnp.float32),
        churn_loss_window=jnp.zeros((window,), dtype=jnp.float32),
        loss_window_count=jnp.asarray(window, dtype=jnp.int32),
        loss_window_cursor=jnp.asarray(pre_count % window, dtype=jnp.int32),
        commit_count_words=jnp.asarray((0, 0xFFFFFFFF), dtype=jnp.uint32),
        state_integrity_tag=jnp.zeros((2,), dtype=jnp.uint32),
    )
    synthetic = mechanism._seal_state(synthetic)  # noqa: SLF001
    assert bool(mechanism.state_valid(synthetic))

    proposal = _propose(mechanism, synthetic, params).proposal
    np.testing.assert_array_equal(
        np.asarray(proposal.destination_commit_count_words),
        np.asarray((1, 0), dtype=np.uint32),
    )
    committed = mechanism.commit(synthetic, proposal, params)

    assert bool(committed.diagnostics.applied)
    np.testing.assert_array_equal(
        np.asarray(committed.state.commit_count_words),
        np.asarray((1, 0), dtype=np.uint32),
    )
    assert int(committed.state.loss_window_cursor) == (1 << 32) % window
    assert bool(mechanism.state_valid(committed.state))


def test_resource_declaration_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    mechanism = _mechanism()
    params = _params()
    state = mechanism.init(params)
    proposal = _propose(mechanism, state, params).proposal
    state = mechanism.commit(state, proposal, _sgd(params, proposal.gradients)).state
    resources = mechanism.resource_budget(state)
    assert resources.valid_proposal_autodiff_passes == 1
    assert resources.rejected_preflight_autodiff_passes == 0
    assert resources.commit_autodiff_passes == 0
    assert resources.dispatch_authority == 0
    assert resources.output_authority == 0
    assert resources.scientific_promotion_allowed is False
    assert resources.full_sequential_algorithm_reproduced is False
    assert resources.efficacy_assessed is False
    assert resources.default_agent_integration is False
    assert resources.max_commits == 100
    assert resources.reference_parameter_copies == 1
    assert resources.expected_current_parameter_copies == 1

    path = tmp_path / "cchain"
    save_cchain_checkpoint(mechanism, state, path)
    restored_mechanism, restored_state = load_cchain_checkpoint(
        path,
        params_template=params,
        model_fn=_model,
        base_loss_fn=_loss,
    )
    assert restored_mechanism.to_config() == mechanism.to_config()
    _tree_exact(restored_state, state)
    assert bool(restored_mechanism.state_valid(restored_state))

    wrong_template = {"w": jnp.ones((9,), dtype=jnp.float32)}
    with pytest.raises((TypeError, ValueError)):
        load_cchain_checkpoint(
            path,
            params_template=wrong_template,
            model_fn=_model,
            base_loss_fn=_loss,
        )


def test_checkpoint_rejects_tampered_source_and_config_metadata(tmp_path: Path) -> None:
    mechanism = _mechanism()
    params = _params()
    state = mechanism.init(params)
    original_path = tmp_path / "original"
    save_cchain_checkpoint(mechanism, state, original_path)
    metadata = load_checkpoint_metadata(original_path)

    source_tamper = copy.deepcopy(metadata)
    paper_source = source_tamper["paper_source"]
    assert isinstance(paper_source, dict)
    paper_source["url"] = "https://invalid.example"
    source_path = tmp_path / "source_tamper"
    save_checkpoint(state, source_path, metadata=source_tamper)
    with pytest.raises(ValueError, match="paper source metadata"):
        load_cchain_checkpoint(
            source_path,
            params_template=params,
            model_fn=_model,
            base_loss_fn=_loss,
        )

    digest_tamper = copy.deepcopy(metadata)
    digest_tamper["config_sha256"] = "0" * 64
    digest_path = tmp_path / "digest_tamper"
    save_checkpoint(state, digest_path, metadata=digest_tamper)
    with pytest.raises(ValueError, match="config digest"):
        load_cchain_checkpoint(
            digest_path,
            params_template=params,
            model_fn=_model,
            base_loss_fn=_loss,
        )


def test_empirical_ntk_exact_gram_rank_and_metrics() -> None:
    gradients = jnp.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    result = empirical_ntk_diagnostics(gradients, delta=0.2)

    np.testing.assert_allclose(
        np.asarray(result.gram_matrix),
        np.asarray([[1.0, 1.0], [1.0, 2.0]], dtype=np.float32),
    )
    expected_singular = np.asarray([(3.0 + np.sqrt(5.0)) / 2.0, (3.0 - np.sqrt(5.0)) / 2.0])
    np.testing.assert_allclose(np.asarray(result.singular_values), expected_singular, rtol=1e-5)
    assert int(result.approximate_rank) == 1
    assert float(result.off_diagonal_absolute_sum) == pytest.approx(2.0)
    assert float(result.off_diagonal_absolute_mean) == pytest.approx(1.0)
    assert float(result.off_diagonal_rms) == pytest.approx(1.0)
    assert float(result.diagonal_sum) == pytest.approx(3.0)
    assert float(result.diagonal_mean) == pytest.approx(1.5)
    assert float(result.diagonal_rms) == pytest.approx(np.sqrt(2.5))
    assert bool(result.input_finite)
    assert bool(result.derived_finite)
    assert not bool(result.zero_gradient)
    assert bool(result.valid)


def test_empirical_ntk_accepts_gradient_pytree_and_handles_zero_nonfinite() -> None:
    tree = {
        "w": jnp.zeros((3, 2, 2), dtype=jnp.float32),
        "b": jnp.zeros((3, 2), dtype=jnp.float32),
    }
    zero = empirical_ntk_diagnostics(tree)
    assert zero.gradient_matrix.shape == (3, 6)
    assert int(zero.approximate_rank) == 0
    assert bool(zero.zero_gradient)
    assert bool(zero.valid)
    np.testing.assert_array_equal(np.asarray(zero.gram_matrix), np.zeros((3, 3)))

    nonfinite_matrix = jnp.asarray([[1.0, jnp.nan], [0.0, 1.0]], dtype=jnp.float32)
    nonfinite = empirical_ntk_diagnostics(nonfinite_matrix)
    assert not bool(nonfinite.input_finite)
    assert bool(nonfinite.derived_finite)
    assert not bool(nonfinite.valid)
    assert not bool(nonfinite.zero_gradient)
    assert int(nonfinite.approximate_rank) == 0
    np.testing.assert_array_equal(np.asarray(nonfinite.gram_matrix), np.zeros((2, 2)))

    overflowing = empirical_ntk_diagnostics(
        jnp.full((2, 2), jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    )
    assert bool(overflowing.input_finite)
    assert not bool(overflowing.derived_finite)
    assert not bool(overflowing.valid)
    assert int(overflowing.approximate_rank) == 0
    np.testing.assert_array_equal(np.asarray(overflowing.gram_matrix), np.zeros((2, 2)))

    with pytest.raises(TypeError, match="float32"):
        empirical_ntk_diagnostics(jnp.ones((2, 2), dtype=jnp.int32))
    with pytest.raises(ValueError, match="leading sample"):
        empirical_ntk_diagnostics(
            {
                "a": jnp.ones((2, 1), dtype=jnp.float32),
                "b": jnp.ones((3, 1), dtype=jnp.float32),
            }
        )
    with pytest.raises(ValueError, match="delta"):
        empirical_ntk_diagnostics(jnp.ones((2, 2), dtype=jnp.float32), delta=1.0)


def test_state_and_proposal_content_tags_detect_direct_mutation() -> None:
    mechanism = _mechanism()
    params = _params()
    state = mechanism.init(params)
    assert bool(mechanism.state_valid(state))
    tampered_state = state.replace(coefficient=state.coefficient + 0.25)
    assert not bool(mechanism.state_valid(tampered_state))

    proposed = _propose(mechanism, state, params).proposal
    assert bool(mechanism.proposal_valid(state, proposed))
    tampered_proposal = dataclasses.replace(
        proposed,
        combined_loss=proposed.combined_loss + 0.25,
    )
    assert not bool(mechanism.proposal_valid(state, tampered_proposal))
