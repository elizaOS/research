"""Synthetic host-validation tests for the authority-free v6 validator."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner as runner_module
import alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_validator as validator_module
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    build_v6_primary_controls,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_intervention_audit import (
    V6_INTERVENTION_WITNESS_ORDER,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    CRITICAL_PAIRS,
    CURATION_INTERVAL,
    MAX_CADENCE_LEDGER_ENTRIES,
    MAX_SCAN_STEPS,
    TARGET_HEADS,
    HiddenPartnerLifecycleWorldV6Runner,
    V6DevelopmentRun,
    V6ResourceRecord,
    V6RngRecord,
    empty_v6_action_totals,
    empty_v6_audit_totals,
    empty_v6_cadence_ledger,
    empty_v6_filter_totals,
    empty_v6_lifecycle_chain_state,
    empty_v6_row_head_totals,
    empty_v6_window_totals,
    reconstruct_v6_stream_code,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runtime import (
    capture_v6_runtime_record,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_scan_plan import (
    ENTRY_WINDOW_STEPS,
    FINAL_WINDOW_STEPS,
    build_hidden_partner_lifecycle_world_v6_scan_plan_from_state,
    require_v6_control_suite_ready,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_validator import (
    _AUDIT_CONTRACT,
    _LEDGER_CONTRACT,
    _ROW_HEAD_CONTRACT,
    DEVELOPMENT_ONLY,
    EVIDENCE_AUTHORIZED,
    EXECUTION_AUTHORIZED,
    REPLAY_VERIFIED,
    SCIENTIFIC_PROMOTION_ALLOWED,
    STRUCTURAL_ONLY,
    STRUCTURALLY_INVALID_DEVELOPMENT_RUN,
    STRUCTURALLY_VALID_DEVELOPMENT_RUN,
    _array,
    _recompute_lifecycle,
    _record_arrays,
    _validate_audits_and_counters,
    _validate_row_heads,
    _ValidationContext,
    validate_hidden_partner_lifecycle_world_v6_development_run,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _advance_roots(keys: jax.Array, steps: int, *, width: int) -> jax.Array:
    def body(_: int, current: jax.Array) -> jax.Array:
        return jax.vmap(lambda key: jr.split(key, width)[0])(current)

    return jax.lax.fori_loop(0, steps, body, keys)


def _tree_signature(tree: object) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in jax.tree_util.tree_leaves(tree))


def _critical_slots(bank: np.ndarray) -> np.ndarray:
    values: list[int] = []
    for pair in CRITICAL_PAIRS:
        matches = np.flatnonzero(np.all(bank == np.asarray(pair, dtype=np.int32), axis=1))
        values.append(int(matches[0]) if len(matches) == 1 else -1)
    return np.asarray(values, dtype=np.int32)


def _initial_stream_bits(state: object) -> jax.Array:
    world = state.world
    bits = jnp.concatenate(
        (
            world.current_signals > 0.0,
            jnp.reshape(world.world_sign > 0.0, (1,)),
            world.current_cues > 0.0,
            jnp.reshape(world.previous_outcome > 0.0, (1,)),
            jnp.reshape(world.has_partner_history, (1,)),
        )
    ).astype(jnp.uint8)
    return jnp.sum(
        jnp.left_shift(bits, jnp.arange(8, dtype=jnp.uint8)),
        dtype=jnp.uint8,
    )


@pytest.fixture(autouse=True, scope="module")
def _quiescent_source_closure() -> Iterator[None]:
    """Pin the validator's live closure read to the runner's import snapshot.

    A concurrent development session may edit unrelated package sources while
    this file runs, and the production guard correctly fail-closes on any such
    drift.  These tests exercise validator semantics against the import-time
    snapshot — exactly what the live read returns on a quiescent tree — while
    the on-disk guard keeps its own dedicated tests in the runner test file.
    """

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        validator_module,
        "compute_v6_source_closure_hashes",
        lambda: runner_module._V6_IMPORT_SOURCE_CLOSURE_HASHES,  # noqa: SLF001
    )
    yield
    patcher.undo()


@pytest.fixture(scope="module")
def synthetic_run() -> V6DevelopmentRun:
    """Build one internally coherent full-length record without executing its scan."""

    control = build_v6_primary_controls()[0]
    runner = HiddenPartnerLifecycleWorldV6Runner(control)
    world_key = jr.key(913)
    agent_key = jr.key(271)
    initial = runner.initialize(world_key, agent_key)
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(initial.world)
    run_steps = plan.run_steps

    world_roots = jnp.stack(
        (
            initial.world.signal_key,
            initial.world.partner_key,
            initial.world.world_key,
            initial.world.cue_key,
            initial.world.outcome_key,
        )
    )
    final_world_roots = _advance_roots(world_roots, run_steps, width=2)
    final_interaction_key = _advance_roots(
        jnp.reshape(initial.agent.interaction.key, (1,)),
        run_steps,
        width=2,
    )[0]
    policy_root = jnp.reshape(initial.agent.current_selection.rng_key_after, (1,))
    final_policy_before = _advance_roots(policy_root, run_steps - 1, width=4)[0]
    final_policy_after = _advance_roots(policy_root, run_steps, width=4)[0]
    final_selection = initial.agent.current_selection.replace(
        rng_key_before=final_policy_before,
        rng_key_after=final_policy_after,
    )
    grounded = initial.agent.grounded_world
    assert grounded is not None
    final_agent = initial.agent.replace(
        state_builder=initial.agent.state_builder.replace(
            step_count=initial.agent.state_builder.step_count + run_steps,
            update_count=initial.agent.state_builder.update_count + run_steps,
        ),
        behavior=initial.agent.behavior.replace(
            step_count=initial.agent.behavior.step_count + run_steps,
        ),
        interaction=initial.agent.interaction.replace(
            key=final_interaction_key,
            step_count=initial.agent.interaction.step_count + run_steps,
        ),
        joint_world=initial.agent.joint_world.replace(
            step_count=initial.agent.joint_world.step_count + run_steps,
        ),
        grounded_world=grounded.replace(update_count=grounded.update_count + run_steps),
        control=initial.agent.control.replace(
            rng_key=final_policy_after,
            step_count=initial.agent.control.step_count + run_steps,
        ),
        router=dataclasses.replace(
            initial.agent.router,
            route_count=initial.agent.router.route_count + run_steps,
        ),
        current_selection=final_selection,
        step_count=initial.agent.step_count + run_steps,
    )
    final_world = initial.world.replace(
        signal_key=final_world_roots[0],
        partner_key=final_world_roots[1],
        world_key=final_world_roots[2],
        cue_key=final_world_roots[3],
        outcome_key=final_world_roots[4],
        step_count=initial.world.step_count + run_steps,
    )
    final = initial.replace(
        world=final_world,
        agent=final_agent,
        world_filter=initial.world_filter.replace(
            step_count=initial.world_filter.step_count + run_steps,
        ),
        action=final_selection.action,
        step_count=initial.step_count + run_steps,
    )

    expected_windows = jnp.asarray(
        (ENTRY_WINDOW_STEPS,) * 36 + (FINAL_WINDOW_STEPS,),
        dtype=jnp.int32,
    )
    initial_bank = np.asarray(jax.device_get(initial.agent.router.descriptors))
    critical_present = np.asarray(
        [np.any(np.all(initial_bank == np.asarray(pair), axis=1)) for pair in CRITICAL_PAIRS],
        dtype=np.int32,
    )
    windows = empty_v6_window_totals().replace(
        scheduled_support=expected_windows,
        accepted_support=expected_windows,
        behavior_correct_count=expected_windows,
        grounded_support=expected_windows,
        critical_present_count=expected_windows[:, None] * critical_present[None, :],
    )

    joint = np.full((4,), run_steps // 4, dtype=np.int32)
    joint[: run_steps % 4] += 1
    focal = np.asarray((joint[0] + joint[1], joint[2] + joint[3]), dtype=np.int32)
    partner = np.asarray((joint[0] + joint[2], joint[1] + joint[3]), dtype=np.int32)
    action = empty_v6_action_totals().replace(
        focal_action_support=jnp.asarray(focal),
        partner_action_support=jnp.asarray(partner),
        joint_row_support=jnp.asarray(joint),
        ordinary_policy_action_support=jnp.asarray(focal),
        decision_count=jnp.asarray(run_steps + 1, dtype=jnp.int32),
    )
    row_support = np.broadcast_to(joint[:, None], (4, TARGET_HEADS)).copy()
    rows = empty_v6_row_head_totals().replace(support=jnp.asarray(row_support, dtype=jnp.int32))
    cue_support = np.full((4,), run_steps // 4, dtype=np.int32)
    cue_support[: run_steps % 4] += 1
    filters = empty_v6_filter_totals().replace(
        support=jnp.asarray(run_steps, dtype=jnp.int32),
        tied_support=jnp.asarray(run_steps, dtype=jnp.int32),
        tied_focal_action_support=jnp.asarray(focal),
        cue_pattern_support=jnp.asarray(cue_support),
        cue_flip_support=jnp.full((2,), run_steps, dtype=jnp.int32),
    )

    ledger_count = run_steps // CURATION_INTERVAL
    ledger = empty_v6_cadence_ledger()
    occupied = np.arange(MAX_CADENCE_LEDGER_ENTRIES) < ledger_count
    transition_steps = np.full((MAX_CADENCE_LEDGER_ENTRIES,), -1, dtype=np.int32)
    transition_steps[:ledger_count] = CURATION_INTERVAL * np.arange(
        1,
        ledger_count + 1,
        dtype=np.int32,
    )
    regimes = np.full((MAX_CADENCE_LEDGER_ENTRIES,), -1, dtype=np.int32)
    for index, transition_step in enumerate(transition_steps[:ledger_count]):
        scan_step = int(transition_step) - 1
        occurrence = next(
            item
            for item in plan.segment_occurrences
            if item.start <= scan_step < item.end_exclusive
        )
        regimes[index] = occurrence.regime_id
    descriptors = np.full(
        (MAX_CADENCE_LEDGER_ENTRIES, initial_bank.shape[0], 2),
        -1,
        dtype=np.int32,
    )
    descriptors[:ledger_count] = initial_bank
    critical = np.full((MAX_CADENCE_LEDGER_ENTRIES, 3, 2), -1, dtype=np.int32)
    critical[:ledger_count] = _critical_slots(initial_bank)[None, None, :]
    streak = np.full((MAX_CADENCE_LEDGER_ENTRIES, 2, 2), -1, dtype=np.int32)
    streak[:ledger_count] = 0
    random_flags = np.zeros((MAX_CADENCE_LEDGER_ENTRIES, 3), dtype=np.bool_)
    random_flags[:ledger_count, 1] = True
    random_selected = np.full((MAX_CADENCE_LEDGER_ENTRIES, 3), -1, dtype=np.int32)
    random_selected[:ledger_count, 2] = 0
    active_priorities = np.zeros(
        (MAX_CADENCE_LEDGER_ENTRIES, initial_bank.shape[0]),
        dtype=np.float32,
    )
    active_priorities[:ledger_count] = np.arange(initial_bank.shape[0], dtype=np.float32)
    candidate_priorities = np.zeros((MAX_CADENCE_LEDGER_ENTRIES, 66), dtype=np.float32)
    candidate_priorities[:ledger_count] = np.arange(66, dtype=np.float32)
    live = np.all(initial_bank >= 0, axis=1)
    sources = np.full((MAX_CADENCE_LEDGER_ENTRIES, initial_bank.shape[0]), -1, dtype=np.int32)
    sources[:ledger_count, live] = np.flatnonzero(live)
    router_masks = np.zeros(
        (MAX_CADENCE_LEDGER_ENTRIES, 3, initial_bank.shape[0]),
        dtype=np.bool_,
    )
    router_masks[:ledger_count, 0] = live
    router_flags = np.zeros((MAX_CADENCE_LEDGER_ENTRIES, 4), dtype=np.bool_)
    router_flags[:ledger_count, :3] = True
    router_counts = np.full((MAX_CADENCE_LEDGER_ENTRIES, 4), -1, dtype=np.int32)
    initial_route = int(initial.agent.router.route_count)
    initial_generation = int(initial.agent.router.generation_count)
    router_counts[:ledger_count, 0] = initial_route + transition_steps[:ledger_count] - 1
    router_counts[:ledger_count, 1] = initial_route + transition_steps[:ledger_count]
    router_counts[:ledger_count, 2:] = initial_generation
    exact_rows = np.zeros((MAX_CADENCE_LEDGER_ENTRIES,), dtype=np.bool_)
    exact_rows[:ledger_count] = True
    ledger = ledger.replace(
        occupied=jnp.asarray(occupied),
        transition_step=jnp.asarray(transition_steps),
        regime_id=jnp.asarray(regimes),
        pre_descriptors=jnp.asarray(descriptors),
        proposal_descriptors=jnp.asarray(descriptors),
        applied_descriptors=jnp.asarray(descriptors),
        critical_slot=jnp.asarray(critical),
        critical_candidate_streak=jnp.asarray(streak),
        random_curation_flags=jnp.asarray(random_flags),
        random_curation_selected=jnp.asarray(random_selected),
        random_active_priorities=jnp.asarray(active_priorities),
        random_candidate_priorities=jnp.asarray(candidate_priorities),
        router_source_slots=jnp.asarray(sources),
        router_masks=jnp.asarray(router_masks),
        router_flags=jnp.asarray(router_flags),
        router_counts=jnp.asarray(router_counts),
        transaction_exact=jnp.asarray(exact_rows),
        identity_carry_exact=jnp.asarray(exact_rows),
        retired_identity_reset_exact=jnp.asarray(exact_rows),
    )

    component_sums = jnp.asarray(
        (
            run_steps,
            run_steps,
            run_steps,
            run_steps,
            run_steps,
            run_steps,
            run_steps,
            run_steps,
            0,
            run_steps,
        ),
        dtype=jnp.int32,
    )
    audits = empty_v6_audit_totals().replace(
        component_delta_sums=component_sums,
        active_steps=jnp.asarray(run_steps, dtype=jnp.int32),
        accepted_steps=jnp.asarray(run_steps, dtype=jnp.int32),
        learner_valid_steps=jnp.asarray(run_steps, dtype=jnp.int32),
        filter_valid_steps=jnp.asarray(run_steps, dtype=jnp.int32),
        oracle_valid_steps=jnp.asarray(run_steps, dtype=jnp.int32),
        mechanism_valid_steps=jnp.asarray(run_steps, dtype=jnp.int32),
        all_finite_steps=jnp.asarray(run_steps, dtype=jnp.int32),
        curation_attempt_count=jnp.asarray(ledger_count, dtype=jnp.int32),
        ledger_count=jnp.asarray(ledger_count, dtype=jnp.int32),
    )

    initial_budget = runner.bridge.resource_budget(initial)
    final_budget = runner.bridge.resource_budget(final)
    resources = V6ResourceRecord(
        initial=initial_budget,
        final=final_budget,
        peak_total_state_nbytes=max(
            initial_budget.total_state_nbytes,
            final_budget.total_state_nbytes,
        ),
        static_total_state_nbytes=True,
        zero_replay=True,
        initial_tree_signature=_tree_signature(initial),
        final_tree_signature=_tree_signature(final),
        tree_structure_equal=True,
        tree_signature_equal=True,
    )
    rng = V6RngRecord(
        supplied_key_data=jnp.stack((jr.key_data(world_key), jr.key_data(agent_key))).astype(
            jnp.uint32
        ),
        initial_world_key_data=jax.vmap(jr.key_data)(world_roots).astype(jnp.uint32),
        final_world_key_data=jax.vmap(jr.key_data)(final_world_roots).astype(jnp.uint32),
        initial_policy_key_data=jnp.stack(
            (
                jr.key_data(initial.agent.current_selection.rng_key_before),
                jr.key_data(initial.agent.current_selection.rng_key_after),
            )
        ).astype(jnp.uint32),
        final_policy_key_data=jnp.stack(
            (jr.key_data(final_policy_before), jr.key_data(final_policy_after))
        ).astype(jnp.uint32),
        initial_interaction_key_data=jr.key_data(initial.agent.interaction.key).astype(jnp.uint32),
        final_interaction_key_data=jr.key_data(final_interaction_key).astype(jnp.uint32),
        initial_stream_bits=_initial_stream_bits(initial),
        world_draw_counts=jnp.full((5,), run_steps, dtype=jnp.int32),
        interaction_key_advance_count=jnp.asarray(run_steps, dtype=jnp.int32),
        policy_decision_count=jnp.asarray(run_steps + 1, dtype=jnp.int32),
    )

    readiness = require_v6_control_suite_ready()
    binding = next(
        item
        for item in readiness.bindings
        if item.family == "primary" and item.name == control.name
    )
    # The import-time snapshot is the exact value the live closure read yields
    # on a quiescent tree; re-reading disk here would couple this fixture to
    # unrelated concurrent edits elsewhere in the package.
    source_closure = runner_module._V6_IMPORT_SOURCE_CLOSURE_HASHES  # noqa: SLF001
    runtime_record = capture_v6_runtime_record()
    runner_digest = hashlib.sha256(
        _canonical_json_bytes(
            runner._config_for_source_closure(  # noqa: SLF001
                source_closure,
                runtime_record,
            )
        )
    ).hexdigest()
    return V6DevelopmentRun(
        control_name=control.name,
        primary=True,
        plan=plan,
        control_config_sha256=binding.control_config_sha256,
        control_matrix_sha256=readiness.control_matrix_sha256,
        bridge_config_sha256=binding.bridge_config_sha256,
        runner_config_sha256=runner_digest,
        source_closure_hashes=source_closure,
        runtime=runtime_record,
        initial_state=initial,
        final_state=final,
        windows=windows,
        row_heads=rows,
        filter_totals=filters,
        action_totals=action,
        audits=audits,
        ledger=ledger,
        lifecycle=empty_v6_lifecycle_chain_state(),
        rng=rng,
        resources=resources,
        stream_code=reconstruct_v6_stream_code(
            initial.world,
            control.world_config,
            plan.run_steps,
        ),
    )


def _codes(result: object) -> set[str]:
    return {error.code for error in result.errors}


def test_synthetic_negative_lifecycle_is_structurally_valid(
    synthetic_run: V6DevelopmentRun,
) -> None:
    result = validate_hidden_partner_lifecycle_world_v6_development_run(synthetic_run)

    assert result.status == STRUCTURALLY_VALID_DEVELOPMENT_RUN, result.errors
    assert result.errors == ()
    assert result.lifecycle.structural_chain_consistent
    assert not result.lifecycle.c_ever_acquired
    assert not result.lifecycle.d_ordered_outcome
    assert not result.quality.c_identity_outcome
    assert not result.quality.d_identity_outcome
    assert result.coverage.complete_window_support
    assert result.coverage.complete_row_head_support
    assert result.coverage.complete_filter_cue_support


def test_validator_has_no_authority_and_only_two_statuses(
    synthetic_run: V6DevelopmentRun,
) -> None:
    valid = validate_hidden_partner_lifecycle_world_v6_development_run(synthetic_run)
    invalid = validate_hidden_partner_lifecycle_world_v6_development_run(object())

    assert DEVELOPMENT_ONLY is True
    assert STRUCTURAL_ONLY is True
    assert REPLAY_VERIFIED is False
    assert EXECUTION_AUTHORIZED is False
    assert EVIDENCE_AUTHORIZED is False
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert {valid.status, invalid.status} == {
        STRUCTURALLY_VALID_DEVELOPMENT_RUN,
        STRUCTURALLY_INVALID_DEVELOPMENT_RUN,
    }
    assert not valid.execution_authorized
    assert valid.structural_only
    assert not valid.replay_verified
    assert not valid.evidence_authorized
    assert not valid.scientific_promotion_allowed


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    (
        (
            lambda run: dataclasses.replace(
                run,
                stream_code=np.zeros((MAX_SCAN_STEPS,), dtype=np.uint8),
            ),
            "ARRAY_CLASS",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                stream_code=jnp.zeros((MAX_SCAN_STEPS,), dtype=jnp.int32),
            ),
            "DTYPE",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                stream_code=jnp.zeros((MAX_SCAN_STEPS - 1,), dtype=jnp.uint8),
            ),
            "SHAPE",
        ),
        (
            lambda run: dataclasses.replace(run, runner_config_sha256="0" * 64),
            "LIVE_DIGEST",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                runtime=dataclasses.replace(
                    run.runtime,
                    python_version=run.runtime.python_version + ".stale",
                ),
            ),
            "RUNTIME_PROVENANCE",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                audits=run.audits.replace(accepted_steps=run.audits.accepted_steps - jnp.int32(1)),
            ),
            "ACTIVE_REJECTION",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                audits=run.audits.replace(
                    intervention_failure_counts=(
                        run.audits.intervention_failure_counts.at[0].set(1)
                    )
                ),
            ),
            "INTERVENTION_AUDIT",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                audits=run.audits.replace(
                    intervention_witness_counts=(
                        run.audits.intervention_witness_counts.at[0].set(
                            run.audits.accepted_steps + jnp.int32(1)
                        )
                    )
                ),
            ),
            "INTERVENTION_COUNT",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                audits=run.audits.replace(
                    intervention_witness_counts=(
                        run.audits.intervention_witness_counts.at[0].set(-1)
                    )
                ),
            ),
            "DOMAIN",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                audits=run.audits.replace(
                    intervention_failure_counts=jnp.zeros((17,), dtype=jnp.int32)
                ),
            ),
            "SHAPE",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                lifecycle=run.lifecycle.replace(c_ever_acquired=jnp.asarray(True)),
            ),
            "LIFECYCLE_RECOMPUTATION",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                resources=dataclasses.replace(run.resources, peak_total_state_nbytes=0),
            ),
            "RESOURCE_ALGEBRA",
        ),
        (
            lambda run: dataclasses.replace(
                run,
                rng=run.rng.replace(world_draw_counts=run.rng.world_draw_counts.at[0].add(1)),
            ),
            "RNG_COUNT",
        ),
    ),
)
def test_structural_tampering_fails_closed(
    synthetic_run: V6DevelopmentRun,
    mutate: object,
    expected_code: str,
) -> None:
    tampered = mutate(synthetic_run)
    result = validate_hidden_partner_lifecycle_world_v6_development_run(tampered)

    assert result.status == STRUCTURALLY_INVALID_DEVELOPMENT_RUN
    assert expected_code in _codes(result)


def test_nonfinite_and_negative_zero_are_rejected(synthetic_run: V6DevelopmentRun) -> None:
    nonfinite = dataclasses.replace(
        synthetic_run,
        windows=synthetic_run.windows.replace(
            reward_sum=synthetic_run.windows.reward_sum.at[0].set(jnp.float32(jnp.nan))
        ),
    )
    count = synthetic_run.plan.run_steps // CURATION_INTERVAL
    negative_zero_priorities = synthetic_run.ledger.random_active_priorities.at[count, 0].set(
        jnp.asarray(-0.0, dtype=jnp.float32)
    )
    negative_zero = dataclasses.replace(
        synthetic_run,
        ledger=synthetic_run.ledger.replace(random_active_priorities=negative_zero_priorities),
    )

    assert "NONFINITE" in _codes(
        validate_hidden_partner_lifecycle_world_v6_development_run(nonfinite)
    )
    assert "PADDING_SENTINEL" in _codes(
        validate_hidden_partner_lifecycle_world_v6_development_run(negative_zero)
    )


def test_plan_source_closure_and_final_state_are_independently_bound(
    synthetic_run: V6DevelopmentRun,
) -> None:
    bad_plan = dataclasses.replace(
        synthetic_run,
        plan=dataclasses.replace(
            synthetic_run.plan,
            run_steps=synthetic_run.plan.run_steps - 1,
        ),
    )
    first_source = dataclasses.replace(
        synthetic_run.source_closure_hashes[0],
        sha256="0" * 64,
    )
    bad_source = dataclasses.replace(
        synthetic_run,
        source_closure_hashes=(first_source, *synthetic_run.source_closure_hashes[1:]),
    )
    bad_final_world = synthetic_run.final_state.world.replace(
        current_signals=synthetic_run.final_state.world.current_signals.at[0].set(jnp.nan)
    )
    bad_final = dataclasses.replace(
        synthetic_run,
        final_state=synthetic_run.final_state.replace(world=bad_final_world),
    )

    assert "PLAN" in _codes(validate_hidden_partner_lifecycle_world_v6_development_run(bad_plan))
    assert "SOURCE_CLOSURE" in _codes(
        validate_hidden_partner_lifecycle_world_v6_development_run(bad_source)
    )
    assert "NONFINITE" in _codes(
        validate_hidden_partner_lifecycle_world_v6_development_run(bad_final)
    )


def test_router_consumer_event_random_and_stream_tampers_are_rejected(
    synthetic_run: V6DevelopmentRun,
) -> None:
    cases = (
        (
            synthetic_run.ledger.replace(
                router_counts=synthetic_run.ledger.router_counts.at[0, 1].add(1)
            ),
            "ROUTER_ALGEBRA",
        ),
        (
            synthetic_run.ledger.replace(
                consumer_masks=synthetic_run.ledger.consumer_masks.at[0, 6, 0].set(True)
            ),
            "CONSUMER_ALGEBRA",
        ),
        (
            synthetic_run.ledger.replace(
                proposal_event=synthetic_run.ledger.proposal_event.at[0, 0].set(0)
            ),
            "EVENT_ALGEBRA",
        ),
        (
            synthetic_run.ledger.replace(
                random_active_priorities=(
                    synthetic_run.ledger.random_active_priorities.at[0, 0].set(1.0)
                )
            ),
            "RANDOM_CURATION_ALGEBRA",
        ),
    )
    for ledger, expected_code in cases:
        result = validate_hidden_partner_lifecycle_world_v6_development_run(
            dataclasses.replace(synthetic_run, ledger=ledger)
        )
        assert result.status == STRUCTURALLY_INVALID_DEVELOPMENT_RUN
        assert expected_code in _codes(result)

    stream = synthetic_run.stream_code.at[synthetic_run.plan.run_steps].set(1)
    stream_result = validate_hidden_partner_lifecycle_world_v6_development_run(
        dataclasses.replace(synthetic_run, stream_code=stream)
    )
    assert "STREAM_PADDING" in _codes(stream_result)


def test_active_stream_byte_tamper_is_rejected(
    synthetic_run: V6DevelopmentRun,
) -> None:
    active_stream = synthetic_run.stream_code.at[0].set(
        jnp.bitwise_xor(synthetic_run.stream_code[0], jnp.asarray(1, dtype=jnp.uint8))
    )
    active_result = validate_hidden_partner_lifecycle_world_v6_development_run(
        dataclasses.replace(synthetic_run, stream_code=active_stream)
    )
    assert "STREAM_RECONSTRUCTION" in _codes(active_result)


def test_array_guard_rejects_tracer_without_materializing_it() -> None:
    captured: list[object] = []

    def trace(value: jax.Array) -> jax.Array:
        ctx = _ValidationContext()
        assert (
            _array(
                ctx,
                value,
                path="traced",
                shape=(2,),
                dtype=np.float32,
            )
            is None
        )
        captured.extend(ctx.errors)
        return jnp.asarray(0, dtype=jnp.int32)

    jax.make_jaxpr(trace)(jnp.zeros((2,), dtype=jnp.float32))

    assert any(error.code == "TRACER" for error in captured)


def test_frozen_grounded_lane_validates_proposed_compute_and_discard() -> None:
    frozen = build_v6_primary_controls()[1]
    assert frozen.name == "grounded_model_frozen"
    assert frozen.agent_config is not None
    assert not frozen.agent_config.grounded_world_learning_enabled
    joint = np.ones((4,), dtype=np.int32)
    support = np.ones((4, TARGET_HEADS), dtype=np.int32)
    totals = empty_v6_row_head_totals().replace(
        support=jnp.asarray(support),
        executed_weight_delta_norm_sum=jnp.ones((4, TARGET_HEADS), dtype=jnp.float32),
        executed_bias_delta_abs_sum=jnp.ones((4, TARGET_HEADS), dtype=jnp.float32),
        proposed_weight_change_count=jnp.ones((4,), dtype=jnp.int32),
        proposed_bias_change_count=jnp.ones((4,), dtype=jnp.int32),
    )
    ctx = _ValidationContext()
    arrays = _record_arrays(
        ctx,
        totals,
        type(totals),
        _ROW_HEAD_CONTRACT,
        path="rows",
    )

    assert _validate_row_heads(
        ctx,
        arrays,
        {"joint_row_support": joint},
        frozen,
    )
    assert ctx.errors == []

    bad = dict(arrays)
    bad["proposed_weight_change_count"] = np.asarray((2, 1, 1, 1), dtype=np.int32)
    bad_ctx = _ValidationContext()
    _validate_row_heads(
        bad_ctx,
        bad,
        {"joint_row_support": joint},
        frozen,
    )
    assert any(error.code == "ROW_HEAD_ALGEBRA" for error in bad_ctx.errors)


def test_frozen_grounded_lane_requires_zero_persistent_update_delta(
    synthetic_run: V6DevelopmentRun,
) -> None:
    frozen = build_v6_primary_controls()[1]
    initial_grounded = synthetic_run.initial_state.agent.grounded_world
    final_grounded = synthetic_run.final_state.agent.grounded_world
    assert initial_grounded is not None
    assert final_grounded is not None
    frozen_final_agent = synthetic_run.final_state.agent.replace(
        grounded_world=final_grounded.replace(
            update_count=initial_grounded.update_count,
        )
    )
    frozen_final = synthetic_run.final_state.replace(agent=frozen_final_agent)
    frozen_sums = synthetic_run.audits.component_delta_sums.at[5].set(0)
    grounded_witness_index = V6_INTERVENTION_WITNESS_ORDER.index(
        "grounded_parameter_proposal_nonzero"
    )
    frozen_witnesses = synthetic_run.audits.intervention_witness_counts.at[
        grounded_witness_index
    ].set(1)
    frozen_audits = synthetic_run.audits.replace(
        component_delta_sums=frozen_sums,
        intervention_witness_counts=frozen_witnesses,
    )
    frozen_run = dataclasses.replace(
        synthetic_run,
        final_state=frozen_final,
        audits=frozen_audits,
    )
    ctx = _ValidationContext()
    arrays = _record_arrays(
        ctx,
        frozen_audits,
        type(frozen_audits),
        _AUDIT_CONTRACT,
        path="audits",
    )

    _validate_audits_and_counters(
        ctx,
        arrays,
        frozen_run,
        synthetic_run.plan,
        frozen,
    )

    assert ctx.errors == []

    bad_ctx = _ValidationContext()
    bad_arrays = dict(arrays)
    bad_sums = arrays["component_delta_sums"].copy()
    bad_sums[5] = 1
    bad_arrays["component_delta_sums"] = bad_sums
    _validate_audits_and_counters(
        bad_ctx,
        bad_arrays,
        frozen_run,
        synthetic_run.plan,
        frozen,
    )
    assert any(error.code == "COMPONENT_ALGEBRA" for error in bad_ctx.errors)


def test_selected_control_requires_positive_intervention_witness_support(
    synthetic_run: V6DevelopmentRun,
) -> None:
    frozen = build_v6_primary_controls()[1]
    assert frozen.name == "grounded_model_frozen"
    ctx = _ValidationContext()
    arrays = _record_arrays(
        ctx,
        synthetic_run.audits,
        type(synthetic_run.audits),
        _AUDIT_CONTRACT,
        path="audits",
    )

    _validate_audits_and_counters(
        ctx,
        arrays,
        synthetic_run,
        synthetic_run.plan,
        frozen,
    )

    assert any(error.code == "INTERVENTION_WITNESS" for error in ctx.errors)


def test_lifecycle_steps_use_one_based_completed_transition_positions(
    synthetic_run: V6DevelopmentRun,
) -> None:
    ledger = {
        field.name: np.asarray(jax.device_get(getattr(synthetic_run.ledger, field.name))).copy()
        for field in dataclasses.fields(synthetic_run.ledger)
    }
    assert len(ledger) == len(_LEDGER_CONTRACT)
    count = synthetic_run.plan.run_steps // CURATION_INTERVAL
    acquired_bank = ledger["pre_descriptors"][0].copy()
    acquired_bank[0] = np.asarray(CRITICAL_PAIRS[0], dtype=np.int32)
    ledger["applied_descriptors"][0] = acquired_bank
    ledger["applied_event"][0, :3] = np.asarray((0, 1, -1), dtype=np.int32)
    ledger["pre_descriptors"][1:count] = acquired_bank
    ledger["applied_descriptors"][1:count] = acquired_bank

    outcome = _recompute_lifecycle(ledger, synthetic_run.plan)

    assert outcome.c_ever_acquired
    assert outcome.c_first_acquisition_step == CURATION_INTERVAL


def test_lifecycle_recomputation_does_not_treat_replacement_as_retirement(
    synthetic_run: V6DevelopmentRun,
) -> None:
    ledger = {
        field.name: np.asarray(jax.device_get(getattr(synthetic_run.ledger, field.name))).copy()
        for field in dataclasses.fields(synthetic_run.ledger)
    }
    count = synthetic_run.plan.run_steps // CURATION_INTERVAL
    empty = np.full_like(ledger["pre_descriptors"][0], -1)
    with_d = empty.copy()
    with_d[0] = np.asarray(CRITICAL_PAIRS[1], dtype=np.int32)
    ledger["pre_descriptors"][:count] = empty
    ledger["applied_descriptors"][:count] = empty
    ledger["pre_descriptors"][0] = empty
    ledger["applied_descriptors"][0] = with_d
    ledger["applied_event"][0, 1] = 38
    ledger["pre_descriptors"][1] = with_d
    ledger["applied_descriptors"][1] = empty
    # Descriptor D disappeared through replacement; no explicit applied
    # retirement pair exists even though the generic reset audit is true.
    ledger["applied_event"][1] = np.asarray((-1, 7, -1, -1, -1, -1), dtype=np.int32)
    ledger["retired_identity_reset_exact"][1] = True

    outcome = _recompute_lifecycle(ledger, synthetic_run.plan)

    assert outcome.structural_chain_consistent
    assert outcome.d_phase == 1
    assert outcome.d_retirement_step == -1
    assert not outcome.d_retirement_reset_exact
    assert outcome.out_of_order_event_count == 1
