"""Cheap structural tests for the development-only v6 fixed scan runner."""

from __future__ import annotations

import ast
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner as runner_module
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    FORBIDDEN_SEED_NAMESPACES,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    build_v6_primary_controls,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    CANDIDATE_PAIR_SLOTS,
    MAX_CADENCE_LEDGER_ENTRIES,
    MAX_SCAN_STEPS,
    V6_CONTRACT_AUDIT_ORDER,
    V6_SOURCE_CLOSURE_PATHS,
    HiddenPartnerLifecycleWorldV6Runner,
    V6CadenceObservation,
    V6LifecycleObservation,
    compute_v6_source_closure_hashes,
    empty_v6_audit_totals,
    empty_v6_cadence_ledger,
    empty_v6_lifecycle_chain_state,
    empty_v6_row_head_totals,
    empty_v6_scan_carry,
    expected_v6_carry_survivors,
    expected_v6_focal_action,
    pack_v6_stream_code,
    reconstruct_v6_stream_code,
    record_v6_cadence_observation,
    require_v6_development_seed_namespace,
    update_v6_lifecycle_chain,
    update_v6_row_head_totals,
    v6_window_membership,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runtime import (
    capture_v6_runtime_record,
    validate_v6_runtime_record,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_scan_plan import (
    build_hidden_partner_lifecycle_world_v6_scan_plan_from_state,
    validate_hidden_partner_lifecycle_world_v6_scan_plan,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    HiddenPartnerWorldFeedbackWorld,
)

pytestmark = pytest.mark.unit


def _descriptors(*pairs: tuple[int, int]) -> jnp.ndarray:
    values = np.full((12, 2), -1, dtype=np.int32)
    for index, pair in enumerate(pairs):
        values[index] = pair
    return jnp.asarray(values)


def _cadence_observation(step: int = 64) -> V6CadenceObservation:
    descriptors = _descriptors((0, 4), (1, 4))
    false_candidates = jnp.zeros((2, 66), dtype=jnp.bool_)
    return V6CadenceObservation(
        transition_step=jnp.asarray(step, dtype=jnp.int32),
        regime_id=jnp.asarray(0, dtype=jnp.int32),
        pre_descriptors=descriptors,
        proposal_descriptors=descriptors,
        applied_descriptors=descriptors,
        proposal_event=jnp.full((6,), -1, dtype=jnp.int32),
        applied_event=jnp.full((6,), -1, dtype=jnp.int32),
        critical_slot=jnp.full((3, 2), -1, dtype=jnp.int32),
        critical_candidate_streak=jnp.zeros((2, 2), dtype=jnp.int32),
        critical_candidate_flags=jnp.zeros((2, 6), dtype=jnp.bool_),
        candidate_reset_mask=false_candidates,
        random_curation_flags=jnp.zeros((3,), dtype=jnp.bool_),
        random_curation_selected=jnp.full((3,), -1, dtype=jnp.int32),
        random_active_priorities=jnp.zeros((12,), dtype=jnp.float32),
        random_candidate_priorities=jnp.zeros((66,), dtype=jnp.float32),
        consumer_masks=jnp.zeros((9, 12), dtype=jnp.bool_),
        router_source_slots=jnp.arange(12, dtype=jnp.int32),
        router_masks=jnp.zeros((3, 12), dtype=jnp.bool_),
        router_flags=jnp.asarray((True, True, True, False), dtype=jnp.bool_),
        router_counts=jnp.asarray((63, 64, 0, 0), dtype=jnp.int32),
        transaction_exact=jnp.asarray(True, dtype=jnp.bool_),
        identity_carry_exact=jnp.asarray(True, dtype=jnp.bool_),
        retired_identity_reset_exact=jnp.asarray(True, dtype=jnp.bool_),
    )


def _consumer_gate_mechanism() -> SimpleNamespace:
    values: dict[str, object] = {
        "router_valid": jnp.asarray(True),
        "router_carry_survivors": jnp.asarray(True),
        "router_descriptors_changed": jnp.asarray(False),
        "router_source_slots": jnp.arange(12, dtype=jnp.int32),
        "router_survivor_mask": jnp.ones((12,), dtype=jnp.bool_),
        "router_new_mask": jnp.zeros((12,), dtype=jnp.bool_),
        "lifecycle_applied_retired_left": jnp.asarray(-1, dtype=jnp.int32),
        "lifecycle_applied_retired_right": jnp.asarray(-1, dtype=jnp.int32),
        "lifecycle_applied_candidate_reset_mask": jnp.zeros(
            (CANDIDATE_PAIR_SLOTS,), dtype=jnp.bool_
        ),
        "lifecycle_candidate_promotion_evidence_streak_post": jnp.zeros(
            (CANDIDATE_PAIR_SLOTS,), dtype=jnp.int32
        ),
        "lifecycle_candidate_reacquisition_required_post": jnp.zeros(
            (CANDIDATE_PAIR_SLOTS,), dtype=jnp.bool_
        ),
        "consumer_lifecycle_destination_reset_exact": jnp.asarray(True),
    }
    for field in (
        "consumer_route_source_slots_exact",
        "consumer_route_identity_masks_exact",
        "consumer_route_stable_prefix_exact",
        "consumer_route_survivor_values_exact",
        "consumer_route_reset_values_exact",
        "consumer_route_no_carry_reset_exact",
        "consumer_route_behavior_values_exact",
        "consumer_route_q_values_exact",
        "consumer_route_trace_values_exact",
        "consumer_route_last_observation_exact",
        "consumer_route_grounded_values_exact",
        "consumer_route_values_exact",
    ):
        values[field] = jnp.asarray(True)
    return SimpleNamespace(**values)


def test_window_membership_has_exact_entry_tail_and_final_overlap() -> None:
    starts = jnp.arange(18, dtype=jnp.int32) * 512
    ends = starts + 512
    run_steps = ends[-1]

    entry = v6_window_membership(jnp.asarray(0, dtype=jnp.int32), starts, ends, run_steps)
    assert entry.shape == (37,)
    assert bool(entry[0])
    assert int(jnp.sum(entry)) == 1

    final_step = run_steps - 1
    overlap = v6_window_membership(final_step, starts, ends, run_steps)
    assert bool(overlap[35])
    assert bool(overlap[36])
    assert int(jnp.sum(overlap)) == 2

    boundary = v6_window_membership(ends[0], starts, ends, run_steps)
    assert not bool(boundary[1])
    assert bool(boundary[2])


def test_stream_reconstruction_matches_exact_action_independent_world_draws() -> None:
    config = build_v6_primary_controls()[0].world_config
    world = HiddenPartnerWorldFeedbackWorld(config)
    initial = world.init(jr.key(807))
    expected = reconstruct_v6_stream_code(initial, config, 16)
    state = initial
    observed: list[jax.Array] = []
    for step in range(16):
        transition, state = world.step(state, jnp.asarray(step % 2, dtype=jnp.int32))
        observed.append(
            pack_v6_stream_code(
                next_signals=state.current_signals,
                partner_flipped=transition.oracle.partner_flipped,
                world_flipped=transition.oracle.world_flipped,
                next_cue_flipped=transition.oracle.next_world_cue_flipped,
                outcome_flipped=transition.oracle.outcome_flipped,
            )
        )

    np.testing.assert_array_equal(expected[:16], jnp.stack(observed))
    assert bool(jnp.all(expected[16:] == jnp.asarray(0, dtype=jnp.uint8)))


def test_cadence_ledger_has_canonical_sentinels_and_fails_closed_at_capacity() -> None:
    ledger = empty_v6_cadence_ledger()
    assert ledger.occupied.shape == (MAX_CADENCE_LEDGER_ENTRIES,)
    assert not bool(jnp.any(ledger.occupied))
    assert bool(jnp.all(ledger.transition_step == -1))
    assert bool(jnp.all(ledger.pre_descriptors == -1))
    assert bool(jnp.all(ledger.critical_candidate_streak == -1))

    ledger, count, overflow = record_v6_cadence_observation(
        ledger,
        jnp.asarray(0, dtype=jnp.int32),
        _cadence_observation(),
    )
    assert not bool(overflow)
    assert int(count) == 1
    assert bool(ledger.occupied[0])
    assert int(ledger.transition_step[0]) == 64

    unchanged, saturated_count, overflow = record_v6_cadence_observation(
        ledger,
        jnp.asarray(MAX_CADENCE_LEDGER_ENTRIES, dtype=jnp.int32),
        _cadence_observation(step=MAX_SCAN_STEPS - 1),
    )
    assert bool(overflow)
    assert int(saturated_count) == MAX_CADENCE_LEDGER_ENTRIES
    assert bool(jnp.array_equal(unchanged.transition_step, ledger.transition_step))


def test_intervention_accumulators_saturate_without_int32_wraparound() -> None:
    maximum = np.iinfo(np.int32).max
    near_failure = jnp.full((18,), maximum - 1, dtype=jnp.int32).at[0].set(maximum)
    near_witness = jnp.full((16,), maximum - 1, dtype=jnp.int32).at[0].set(maximum)
    totals = empty_v6_audit_totals().replace(
        intervention_failure_counts=near_failure,
        intervention_witness_counts=near_witness,
    )
    audit = runner_module.V6InterventionStepAudit(
        checks=jnp.zeros((18,), dtype=jnp.bool_),
        witnesses=jnp.ones((16,), dtype=jnp.bool_),
    )

    saturated = jax.jit(runner_module._accumulate_v6_intervention_audit)(  # noqa: SLF001
        totals,
        audit,
    )

    np.testing.assert_array_equal(
        saturated.intervention_failure_counts,
        np.full((18,), maximum, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        saturated.intervention_witness_counts,
        np.full((16,), maximum, dtype=np.int32),
    )
    assert not bool(jnp.any(saturated.intervention_failure_counts < 0))
    assert not bool(jnp.any(saturated.intervention_witness_counts < 0))


def test_row_head_accumulator_reconstructs_algebra_and_rejects_wrong_row_mask() -> None:
    targets = jnp.arange(10, dtype=jnp.float32) / 10.0
    feature = jnp.arange(10, dtype=jnp.float32) / 20.0
    bias = jnp.full((10,), 0.25, dtype=jnp.float32)
    raw = feature + bias
    errors = raw - targets
    fit = 0.5 * jnp.square(errors) / 10.0
    weights = jnp.asarray(
        (0.0, 10.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 10.0 / 3.0, 10.0 / 3.0, 0.0, 0.0), dtype=jnp.float32
    )
    by_head = jnp.zeros((10, 24), dtype=jnp.float32)
    by_head = by_head.at[:, 0].set(jnp.arange(10, dtype=jnp.float32) / 100.0)
    total_gradient = jnp.sum(by_head, axis=0)
    weight_mask = jnp.asarray((False, False, True, False))
    bias_mask = jnp.asarray((False, False, True, False))

    totals, valid = update_v6_row_head_totals(
        empty_v6_row_head_totals(),
        accepted=jnp.asarray(True),
        grounded_valid=jnp.asarray(True),
        executed_row=jnp.asarray(2, dtype=jnp.int32),
        feature_contribution=feature,
        row_bias=bias,
        raw_predictions=raw,
        targets=targets,
        errors=errors,
        fit_loss_by_head=fit,
        representation_loss_by_head=fit * weights,
        representation_gradient=total_gradient,
        representation_gradient_by_head=by_head,
        representation_gradient_norm_by_head=jnp.linalg.norm(by_head, axis=1),
        proposed_weight_change_mask=weight_mask,
        proposed_bias_change_mask=bias_mask,
        executed_weight_delta_norm_by_head=jnp.ones((10,), dtype=jnp.float32),
        executed_bias_delta_by_head=jnp.ones((10,), dtype=jnp.float32),
        row_update_isolated=jnp.asarray(True),
        target_weights=weights,
    )
    assert bool(valid)
    assert bool(jnp.all(totals.support[2] == 1))
    assert int(jnp.sum(totals.support)) == 10

    _, invalid = update_v6_row_head_totals(
        empty_v6_row_head_totals(),
        accepted=jnp.asarray(True),
        grounded_valid=jnp.asarray(True),
        executed_row=jnp.asarray(2, dtype=jnp.int32),
        feature_contribution=feature,
        row_bias=bias,
        raw_predictions=raw,
        targets=targets,
        errors=errors,
        fit_loss_by_head=fit,
        representation_loss_by_head=fit * weights,
        representation_gradient=total_gradient,
        representation_gradient_by_head=by_head,
        representation_gradient_norm_by_head=jnp.linalg.norm(by_head, axis=1),
        proposed_weight_change_mask=jnp.asarray((True, False, True, False)),
        proposed_bias_change_mask=bias_mask,
        executed_weight_delta_norm_by_head=jnp.ones((10,), dtype=jnp.float32),
        executed_bias_delta_by_head=jnp.ones((10,), dtype=jnp.float32),
        row_update_isolated=jnp.asarray(True),
        target_weights=weights,
    )
    assert not bool(invalid)


def test_balanced_action_and_stream_packing_are_exact() -> None:
    actions = [int(expected_v6_focal_action(jnp.asarray(i, dtype=jnp.int32), 0)) for i in range(6)]
    assert actions == [0, 1, 0, 1, 0, 1]
    reverse = [int(expected_v6_focal_action(jnp.asarray(i, dtype=jnp.int32), 1)) for i in range(5)]
    assert reverse == [1, 0, 1, 0, 1]

    code = pack_v6_stream_code(
        next_signals=jnp.asarray((1.0, -1.0, 1.0), dtype=jnp.float32),
        partner_flipped=jnp.asarray(True),
        world_flipped=jnp.asarray(False),
        next_cue_flipped=jnp.asarray((True, False)),
        outcome_flipped=jnp.asarray(True),
    )
    assert code.dtype == jnp.uint8
    assert int(code) == 0b10101101


def test_no_identity_carry_is_vacuously_true_until_descriptors_change() -> None:
    assert bool(expected_v6_carry_survivors(False, jnp.asarray(False)))
    assert not bool(expected_v6_carry_survivors(False, jnp.asarray(True)))
    assert bool(expected_v6_carry_survivors(True, jnp.asarray(False)))
    assert bool(expected_v6_carry_survivors(True, jnp.asarray(True)))


def test_runner_identity_and_retirement_gates_require_consumer_value_verdicts() -> None:
    mechanism = _consumer_gate_mechanism()
    assert bool(runner_module._identity_carry_exact(mechanism, carry_survivors=True))  # noqa: SLF001
    for field in (
        "consumer_route_source_slots_exact",
        "consumer_route_identity_masks_exact",
        "consumer_route_stable_prefix_exact",
        "consumer_route_survivor_values_exact",
        "consumer_route_reset_values_exact",
        "consumer_route_no_carry_reset_exact",
        "consumer_route_behavior_values_exact",
        "consumer_route_q_values_exact",
        "consumer_route_trace_values_exact",
        "consumer_route_last_observation_exact",
        "consumer_route_grounded_values_exact",
        "consumer_route_values_exact",
    ):
        corrupted = SimpleNamespace(**{**vars(mechanism), field: jnp.asarray(False)})
        assert not bool(
            runner_module._identity_carry_exact(  # noqa: SLF001
                corrupted,
                carry_survivors=True,
            )
        )

    retired_index = 38
    retired = SimpleNamespace(
        **{
            **vars(mechanism),
            "lifecycle_applied_retired_left": jnp.asarray(4, dtype=jnp.int32),
            "lifecycle_applied_retired_right": jnp.asarray(5, dtype=jnp.int32),
            "lifecycle_applied_candidate_reset_mask": jax.nn.one_hot(
                retired_index,
                CANDIDATE_PAIR_SLOTS,
                dtype=jnp.bool_,
            ),
            "lifecycle_candidate_reacquisition_required_post": jnp.zeros(
                (CANDIDATE_PAIR_SLOTS,), dtype=jnp.bool_
            )
            .at[retired_index]
            .set(True),
        }
    )
    assert bool(runner_module._retired_identity_reset_exact(retired))  # noqa: SLF001
    corrupted_retirement = SimpleNamespace(
        **{
            **vars(retired),
            "consumer_lifecycle_destination_reset_exact": jnp.asarray(False),
        }
    )
    assert not bool(
        runner_module._retired_identity_reset_exact(corrupted_retirement)  # noqa: SLF001
    )


def test_lifecycle_chain_order_is_an_outcome_not_structural_validity() -> None:
    empty = _descriptors((0, 4))
    with_d = _descriptors((0, 4), (4, 5))
    state = empty_v6_lifecycle_chain_state()

    state = update_v6_lifecycle_chain(
        state,
        V6LifecycleObservation(
            transition_step=jnp.asarray(100, dtype=jnp.int32),
            occurrence_index=jnp.asarray(3, dtype=jnp.int32),
            regime_id=jnp.asarray(3, dtype=jnp.int32),
            pre_descriptors=empty,
            applied_descriptors=with_d,
            promoted_candidate=jnp.asarray(38, dtype=jnp.int32),
            retired_pair=jnp.asarray((-1, -1), dtype=jnp.int32),
            d_reacquisition_required_pre=jnp.asarray(False),
            d_reacquisition_confirmed=jnp.asarray(False),
            d_reset_exact=jnp.asarray(True),
            structural_valid=jnp.asarray(True),
        ),
    )
    acquired_state = state
    replacement_disappearance = update_v6_lifecycle_chain(
        acquired_state,
        V6LifecycleObservation(
            transition_step=jnp.asarray(150, dtype=jnp.int32),
            occurrence_index=jnp.asarray(6, dtype=jnp.int32),
            regime_id=jnp.asarray(0, dtype=jnp.int32),
            pre_descriptors=with_d,
            applied_descriptors=empty,
            promoted_candidate=jnp.asarray(7, dtype=jnp.int32),
            retired_pair=jnp.asarray((-1, -1), dtype=jnp.int32),
            d_reacquisition_required_pre=jnp.asarray(False),
            d_reacquisition_confirmed=jnp.asarray(False),
            # This used to advance phase through a vacuous reset implication.
            d_reset_exact=jnp.asarray(True),
            structural_valid=jnp.asarray(True),
        ),
    )
    assert bool(replacement_disappearance.structural_valid)
    assert int(replacement_disappearance.d_phase) == 1
    assert int(replacement_disappearance.d_retirement_step) == -1
    assert not bool(replacement_disappearance.d_retirement_reset_exact)
    assert int(replacement_disappearance.out_of_order_event_count) == 1

    state = update_v6_lifecycle_chain(
        state,
        V6LifecycleObservation(
            transition_step=jnp.asarray(200, dtype=jnp.int32),
            occurrence_index=jnp.asarray(6, dtype=jnp.int32),
            regime_id=jnp.asarray(0, dtype=jnp.int32),
            pre_descriptors=with_d,
            applied_descriptors=empty,
            promoted_candidate=jnp.asarray(-1, dtype=jnp.int32),
            retired_pair=jnp.asarray((4, 5), dtype=jnp.int32),
            d_reacquisition_required_pre=jnp.asarray(False),
            d_reacquisition_confirmed=jnp.asarray(False),
            d_reset_exact=jnp.asarray(True),
            structural_valid=jnp.asarray(True),
        ),
    )
    state = update_v6_lifecycle_chain(
        state,
        V6LifecycleObservation(
            transition_step=jnp.asarray(300, dtype=jnp.int32),
            occurrence_index=jnp.asarray(12, dtype=jnp.int32),
            regime_id=jnp.asarray(3, dtype=jnp.int32),
            pre_descriptors=empty,
            applied_descriptors=with_d,
            promoted_candidate=jnp.asarray(38, dtype=jnp.int32),
            retired_pair=jnp.asarray((-1, -1), dtype=jnp.int32),
            d_reacquisition_required_pre=jnp.asarray(True),
            d_reacquisition_confirmed=jnp.asarray(True),
            d_reset_exact=jnp.asarray(True),
            structural_valid=jnp.asarray(True),
        ),
    )
    assert bool(state.structural_valid)
    assert int(state.d_phase) == 3
    assert bool(state.d_ordered_outcome)

    wrong_order = update_v6_lifecycle_chain(
        empty_v6_lifecycle_chain_state(),
        V6LifecycleObservation(
            transition_step=jnp.asarray(5, dtype=jnp.int32),
            occurrence_index=jnp.asarray(12, dtype=jnp.int32),
            regime_id=jnp.asarray(3, dtype=jnp.int32),
            pre_descriptors=empty,
            applied_descriptors=with_d,
            promoted_candidate=jnp.asarray(38, dtype=jnp.int32),
            retired_pair=jnp.asarray((-1, -1), dtype=jnp.int32),
            d_reacquisition_required_pre=jnp.asarray(False),
            d_reacquisition_confirmed=jnp.asarray(False),
            d_reset_exact=jnp.asarray(True),
            structural_valid=jnp.asarray(True),
        ),
    )
    assert bool(wrong_order.structural_valid)
    assert not bool(wrong_order.d_ordered_outcome)


def test_runtime_dataclasses_are_frozen() -> None:
    totals = empty_v6_audit_totals()
    assert totals.contract_failure_counts.shape == (27,)
    assert V6_CONTRACT_AUDIT_ORDER[10] == "next_selection_diagnostics"
    assert "padding_carry" not in V6_CONTRACT_AUDIT_ORDER
    assert "padding_sentinel" not in V6_CONTRACT_AUDIT_ORDER
    with pytest.raises(dataclasses.FrozenInstanceError):
        totals.active_steps = jnp.asarray(1, dtype=jnp.int32)  # type: ignore[misc]


def test_runner_refuses_every_reserved_evidence_namespace_before_construction() -> None:
    assert len(FORBIDDEN_SEED_NAMESPACES) >= 2
    control = build_v6_primary_controls()[0]
    assert require_v6_development_seed_namespace(None) is None
    for reserved in FORBIDDEN_SEED_NAMESPACES:
        with pytest.raises(PermissionError, match="reserved evidence namespace"):
            require_v6_development_seed_namespace(reserved)
        with pytest.raises(PermissionError, match="reserved evidence namespace"):
            HiddenPartnerLifecycleWorldV6Runner(control, seed_namespace=reserved)
    with pytest.raises(PermissionError, match="no seed-namespace authority"):
        HiddenPartnerLifecycleWorldV6Runner(
            control,
            seed_namespace="hidden-partner-v6-any-future-namespace",
        )
    with pytest.raises(TypeError, match="None or an exact string"):
        require_v6_development_seed_namespace(0)


def test_source_closure_rejects_files_changed_after_runner_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
        compute_v6_source_closure_hashes,
    )

    original = Path.read_bytes

    def tampered_read(path: Path) -> bytes:
        payload = original(path)
        return payload + b"tampered" if path.name == "uv.lock" else payload

    monkeypatch.setattr(Path, "read_bytes", tampered_read)
    with pytest.raises(RuntimeError, match="changed after runner import"):
        compute_v6_source_closure_hashes()


def test_behavioral_source_closure_contains_its_transitive_local_imports() -> None:
    repository = Path(__file__).resolve().parents[1]
    expected_inventory = (
        "pyproject.toml",
        "uv.lock",
        *tuple(
            sorted(
                path.relative_to(repository).as_posix()
                for path in (repository / "alberta_framework").rglob("*.py")
                if path.is_file()
            )
        ),
    )
    assert V6_SOURCE_CLOSURE_PATHS == expected_inventory
    closure = set(V6_SOURCE_CLOSURE_PATHS)
    for relative_path in V6_SOURCE_CLOSURE_PATHS:
        if not relative_path.endswith(".py"):
            continue
        tree = ast.parse((repository / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            elif isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            for module in modules:
                if not module.startswith("alberta_framework."):
                    continue
                candidate = f"{module.replace('.', '/')}.py"
                package = f"{module.replace('.', '/')}/__init__.py"
                if (repository / candidate).is_file():
                    assert candidate in closure, (relative_path, candidate)
                elif (repository / package).is_file():
                    assert package in closure, (relative_path, package)


@pytest.mark.parametrize(
    "changed_inventory",
    (
        (*V6_SOURCE_CLOSURE_PATHS, "alberta_framework/new_source_after_import.py"),
        V6_SOURCE_CLOSURE_PATHS[:-1],
    ),
)
def test_source_closure_rejects_added_or_deleted_package_sources(
    monkeypatch: pytest.MonkeyPatch,
    changed_inventory: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        runner_module,
        "_discover_v6_source_closure_paths",
        lambda: changed_inventory,
    )

    with pytest.raises(RuntimeError, match="inventory changed after runner import"):
        compute_v6_source_closure_hashes()


def test_v6_initialize_canonicalizes_only_birth_clocks_and_is_bit_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = build_v6_primary_controls()[0]
    runner = HiddenPartnerLifecycleWorldV6Runner(control)
    world_key = jr.key(43_001)
    agent_key = jr.key(43_002)

    real_initialize = runner.bridge.initialize
    bridge_initial = real_initialize(world_key, agent_key)
    call_count = 0

    def initialize_with_distinct_valid_births(*args: object) -> object:
        nonlocal call_count
        raw = real_initialize(*args)
        interaction_birth = jnp.asarray(128.0 + call_count, dtype=jnp.float32)
        control_birth = jnp.asarray(256.0 + call_count, dtype=jnp.float32)
        call_count += 1
        agent = raw.agent.replace(
            interaction=raw.agent.interaction.replace(
                birth_timestamp=interaction_birth
            ),
            control=raw.agent.control.replace(birth_timestamp=control_birth),
        )
        return raw.replace(agent=agent)

    monkeypatch.setattr(
        runner.bridge,
        "initialize",
        initialize_with_distinct_valid_births,
    )
    first = runner.initialize(world_key, agent_key)
    second = runner.initialize(world_key, agent_key)

    assert call_count == 2
    assert float(bridge_initial.agent.interaction.birth_timestamp) > 0.0
    assert float(bridge_initial.agent.control.birth_timestamp) > 0.0
    assert int(
        jax.lax.bitcast_convert_type(first.agent.interaction.birth_timestamp, jnp.uint32)
    ) == 0
    assert int(jax.lax.bitcast_convert_type(first.agent.control.birth_timestamp, jnp.uint32)) == 0
    assert int(jax.lax.bitcast_convert_type(first.agent.interaction.uptime_s, jnp.uint32)) == 0
    assert int(jax.lax.bitcast_convert_type(first.agent.control.uptime_s, jnp.uint32)) == 0
    np.testing.assert_array_equal(
        first.agent.interaction.uptime_s,
        bridge_initial.agent.interaction.uptime_s,
    )
    np.testing.assert_array_equal(
        first.agent.control.uptime_s,
        bridge_initial.agent.control.uptime_s,
    )

    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    assert first_tree == second_tree
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        if jnp.issubdtype(first_leaf.dtype, jax.dtypes.prng_key):
            assert str(jr.key_impl(first_leaf)) == str(jr.key_impl(second_leaf))
            first_array = np.asarray(jax.device_get(jr.key_data(first_leaf)))
            second_array = np.asarray(jax.device_get(jr.key_data(second_leaf)))
        else:
            first_array = np.asarray(jax.device_get(first_leaf))
            second_array = np.asarray(jax.device_get(second_leaf))
        assert first_array.dtype == second_array.dtype
        assert first_array.shape == second_array.shape
        assert first_array.tobytes(order="C") == second_array.tobytes(order="C")


@pytest.mark.parametrize(
    ("component", "malformed_birth"),
    (
        ("interaction", jnp.asarray(jnp.nan, dtype=jnp.float32)),
        ("control", jnp.asarray((1.0,), dtype=jnp.float32)),
    ),
)
def test_v6_initialize_rejects_malformed_bridge_birth_metadata(
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    malformed_birth: jax.Array,
) -> None:
    runner = HiddenPartnerLifecycleWorldV6Runner(build_v6_primary_controls()[0])
    world_key = jr.key(43_101)
    agent_key = jr.key(43_102)
    raw = runner.bridge.initialize(world_key, agent_key)
    if component == "interaction":
        corrupt_agent = raw.agent.replace(
            interaction=raw.agent.interaction.replace(birth_timestamp=malformed_birth)
        )
    else:
        corrupt_agent = raw.agent.replace(
            control=raw.agent.control.replace(birth_timestamp=malformed_birth)
        )
    corrupt = raw.replace(agent=corrupt_agent)
    monkeypatch.setattr(runner.bridge, "initialize", lambda *_args: corrupt)

    with pytest.raises(RuntimeError, match="birth_timestamp"):
        runner.initialize(world_key, agent_key)


def test_run_brackets_fake_scan_with_exact_runtime_and_source_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise run bracketing without executing the 30,318-step learner scan."""

    control = build_v6_primary_controls()[0]
    runner = HiddenPartnerLifecycleWorldV6Runner(control)
    runtime_record = capture_v6_runtime_record()
    # Bracketing needs one consistent snapshot object, not a live-disk re-read:
    # the on-disk guard is covered by the dedicated source-closure tests, and a
    # re-read here would fail whenever unrelated package sources change between
    # module import and this test in a long-lived development session.
    source_closure = runner_module._V6_IMPORT_SOURCE_CLOSURE_HASHES  # noqa: SLF001
    validation_calls: list[bool] = []
    capture_calls: list[int] = []
    source_calls: list[int] = []
    real_validate = validate_v6_runtime_record

    def validate_runtime(record: object, *, require_live_match: bool = True) -> object:
        validation_calls.append(require_live_match)
        return real_validate(record, require_live_match=require_live_match)

    def capture_runtime() -> object:
        capture_calls.append(len(capture_calls))
        return runtime_record

    def source_snapshot() -> object:
        source_calls.append(len(source_calls))
        return source_closure

    def fake_scan(carry: object, *_: object) -> tuple[object, jax.Array]:
        return carry, jnp.zeros((MAX_SCAN_STEPS,), dtype=jnp.uint8)

    monkeypatch.setattr(runner_module, "validate_v6_runtime_record", validate_runtime)
    monkeypatch.setattr(runner_module, "capture_v6_runtime_record", capture_runtime)
    monkeypatch.setattr(runner_module, "compute_v6_source_closure_hashes", source_snapshot)
    monkeypatch.setattr(runner, "_fixed_scan", fake_scan)

    result = runner.run(jr.key(501), jr.key(502))

    assert result.runtime == runtime_record
    assert result.source_closure_hashes == source_closure
    assert capture_calls == [0]
    assert source_calls == [0, 1]
    assert validation_calls == [True, False, True]
    config = runner.to_config()
    assert capture_calls == [0, 1]
    assert source_calls == [0, 1, 2]
    assert validation_calls == [True, False, True, True, False]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        runner.to_config(runtime=runtime_record)  # type: ignore[call-arg]
    assert config["runtime"] == dataclasses.asdict(runtime_record)
    assert config["source_closure"] == [dataclasses.asdict(record) for record in source_closure]
    assert config["initial_state_canonicalization"] == {
        "scope": "v6_runner_only",
        "canonicalized_leaves": [
            {
                "path": "agent.interaction.birth_timestamp",
                "required_precondition": "scalar finite positive float32",
                "canonical_value": "+0.0",
                "canonical_uint32_bits": 0,
            },
            {
                "path": "agent.control.birth_timestamp",
                "required_precondition": "scalar finite positive float32",
                "canonical_value": "+0.0",
                "canonical_uint32_bits": 0,
            },
        ],
        "required_preserved_leaves": [
            {
                "path": "agent.interaction.uptime_s",
                "dtype": "float32",
                "required_uint32_bits": 0,
            },
            {
                "path": "agent.control.uptime_s",
                "dtype": "float32",
                "required_uint32_bits": 0,
            },
        ],
    }
    fixed_scan = config["fixed_scan"]
    assert isinstance(fixed_scan, dict)
    assert fixed_scan["intervention_audit_order"] == list(
        runner_module.V6_INTERVENTION_AUDIT_ORDER
    )
    assert fixed_scan["intervention_witness_order"] == list(
        runner_module.V6_INTERVENTION_WITNESS_ORDER
    )
    assert fixed_scan["float32_bounded_replay"] == {
        "rtol": 2.0**-20,
        "atol": 2.0**-22,
        "disabled_persistence_bit_exact": True,
    }


@pytest.mark.development
@pytest.mark.skipif(
    os.environ.get("ALBERTA_RUN_V6_LARGE_MEMORY_TESTS") != "1",
    reason=("set ALBERTA_RUN_V6_LARGE_MEMORY_TESTS=1 in the larger-memory development lane"),
)
def test_live_runner_advance_active_executes_one_production_transition() -> None:
    assert MAX_SCAN_STEPS == runner_module._MAX_SCAN_STEPS == 30_318  # noqa: SLF001
    control = next(control for control in build_v6_primary_controls() if control.name == "full")
    runner = HiddenPartnerLifecycleWorldV6Runner(control)
    initial = runner.initialize(jr.key(91_001), jr.key(91_002))
    plan = validate_hidden_partner_lifecycle_world_v6_scan_plan(
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(initial.world)
    )
    starts = jnp.asarray(
        tuple(occurrence.start for occurrence in plan.segment_occurrences),
        dtype=jnp.int32,
    )
    ends = jnp.asarray(
        tuple(occurrence.end_exclusive for occurrence in plan.segment_occurrences),
        dtype=jnp.int32,
    )
    regimes = jnp.asarray(
        tuple(occurrence.regime_id for occurrence in plan.segment_occurrences),
        dtype=jnp.int32,
    )
    carry = empty_v6_scan_carry(initial)
    run_steps = jnp.asarray(1, dtype=jnp.int32)
    cycle_length = jnp.asarray(plan.cycle_length, dtype=jnp.int32)
    direct_carry, direct_code = runner._advance_active(  # noqa: SLF001
        carry,
        jnp.asarray(0, dtype=jnp.int32),
        run_steps=run_steps,
        cycle_length=cycle_length,
        occurrence_starts=starts,
        occurrence_ends=ends,
        occurrence_regimes=regimes,
    )
    direct_carry, direct_code = jax.block_until_ready((direct_carry, direct_code))
    assert int(direct_carry.bridge_state.step_count) == 1
    assert int(direct_carry.audits.active_steps) == 1
    assert int(direct_carry.audits.accepted_steps) == 1
    np.testing.assert_array_equal(
        direct_carry.audits.contract_failure_counts,
        np.zeros((len(V6_CONTRACT_AUDIT_ORDER),), dtype=np.int32),
    )
    assert direct_code.shape == ()
    assert direct_code.dtype == jnp.uint8


def test_fixed_scan_real_control_flow_runs_one_active_then_three_padding_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MAX_SCAN_STEPS == runner_module._MAX_SCAN_STEPS == 30_318  # noqa: SLF001
    monkeypatch.setattr(runner_module, "MAX_SCAN_STEPS", 4)
    runner = object.__new__(HiddenPartnerLifecycleWorldV6Runner)

    def advance_active(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        step: jax.Array,
        **_: object,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        scalar, failures, witnesses = carry
        return (
            (scalar + 1, failures + 1, witnesses + 1),
            jnp.asarray(0xA5 + step, dtype=jnp.uint8),
        )

    monkeypatch.setattr(runner, "_advance_active", advance_active)
    dummy_occurrences = jnp.zeros((18,), dtype=jnp.int32)
    with jax.disable_jit():
        final, stream = runner._fixed_scan(  # noqa: SLF001
            (
                jnp.asarray(0, dtype=jnp.int32),
                jnp.zeros((18,), dtype=jnp.int32),
                jnp.zeros((16,), dtype=jnp.int32),
            ),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            dummy_occurrences,
            dummy_occurrences,
            dummy_occurrences,
        )

    assert int(final[0]) == 1
    np.testing.assert_array_equal(final[1], np.ones((18,), dtype=np.int32))
    np.testing.assert_array_equal(final[2], np.ones((16,), dtype=np.int32))
    np.testing.assert_array_equal(
        stream,
        np.asarray((0xA5, 0, 0, 0), dtype=np.uint8),
    )
