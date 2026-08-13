"""Compiled, scan, and checkpoint integration for one-step Dyna."""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.one_step_dyna import (
    ONE_STEP_DYNA_CHECKPOINT_SCHEMA,
    OneStepDynaAuthority,
    OneStepDynaConfig,
    RealStateDynaAnchor,
    load_one_step_dyna_checkpoint,
    save_one_step_dyna_checkpoint,
)
from tests.test_one_step_dyna import (
    ANCHOR,
    REPRESENTATION_REVISION,
    _authority,
    _record,
    _system,
)

pytestmark = pytest.mark.integration


def _materialize_keys(tree: object) -> object:
    def materialize(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(materialize, tree)


def _assert_tree_equal(left: object, right: object) -> None:
    chex.assert_trees_all_equal(_materialize_keys(left), _materialize_keys(right))


def test_record_and_plan_have_eager_and_explicit_jit_parity() -> None:
    planner, model_state, control_state = _system()
    initial = planner.init(
        jr.key(101, impl="threefry2x32"),
        REPRESENTATION_REVISION,
        model_state,
        control_state,
    )
    authority = _authority(model_state, control_state)
    anchor = RealStateDynaAnchor(
        observation=ANCHOR,
        primitive_action=jnp.asarray(0, dtype=jnp.int32),
        decision_id_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        authority=authority,
    )
    eager_record = planner.record_real_anchor(
        initial,
        model_state,
        control_state,
        anchor,
    )
    compiled_record = jax.jit(planner.record_real_anchor)(
        initial,
        model_state,
        control_state,
        anchor,
    )
    _assert_tree_equal(eager_record, compiled_record)

    eager = planner.plan(
        eager_record.state,
        model_state,
        control_state,
        authority,
    )
    compiled = jax.jit(planner.plan)(
        eager_record.state,
        model_state,
        control_state,
        authority,
    )
    _assert_tree_equal(eager, compiled)
    _assert_tree_equal(model_state, planner.ensemble.init(jr.key(1)).replace(
        member_states=model_state.member_states,
    ))


def test_scan_matches_python_loop_with_dynamic_control_revision_authority() -> None:
    config = OneStepDynaConfig(
        anchor_capacity=2,
        backup_budget=1,
        min_action_support=1,
        max_epistemic_disagreement=100.0,
        max_residual_variance=100.0,
        require_residual_proxy_ready=False,
        max_anchor_records=4,
        max_planning_calls=4,
        max_planned_backups=4,
    )
    planner, model_state, control_state = _system(config=config)
    recorded = _record(planner, model_state, control_state)

    def body(carry: tuple[object, object], _: jax.Array):
        planner_state, current_control = carry
        authority = OneStepDynaAuthority(
            representation_revision_words=REPRESENTATION_REVISION,
            model_revision_words=model_state.event_count_words,
            control_revision_words=current_control.step_words,  # type: ignore[attr-defined]
        )
        result = planner.plan(
            planner_state,  # type: ignore[arg-type]
            model_state,
            current_control,  # type: ignore[arg-type]
            authority,
        )
        return (result.state, result.control_state), result.diagnostics.control_targets

    (scan_state, scan_control), scan_targets = jax.lax.scan(
        body,
        (recorded.state, control_state),
        jnp.arange(3, dtype=jnp.int32),
    )
    loop_state = recorded.state
    loop_control = control_state
    loop_targets = []
    for _ in range(3):
        authority = _authority(model_state, loop_control)
        result = planner.plan(loop_state, model_state, loop_control, authority)
        loop_state = result.state
        loop_control = result.control_state
        loop_targets.append(result.diagnostics.control_targets)
    _assert_tree_equal(scan_state, loop_state)
    _assert_tree_equal(scan_control, loop_control)
    chex.assert_trees_all_close(scan_targets, jnp.stack(loop_targets))
    assert int(scan_state.planned_backup_count_words[1]) == 3


def test_checkpoint_roundtrip_restores_only_planner_and_continues_exactly(
    tmp_path: Path,
) -> None:
    planner, model_state, control_state = _system()
    recorded = _record(planner, model_state, control_state)
    first = planner.plan(
        recorded.state,
        model_state,
        control_state,
        _authority(model_state, control_state),
    )
    path = tmp_path / "one-step-dyna"
    save_one_step_dyna_checkpoint(planner, first.state, path)
    restored_planner, restored_state = load_one_step_dyna_checkpoint(path)
    assert restored_planner.to_config() == planner.to_config()
    assert restored_planner.resource_budget == planner.resource_budget
    _assert_tree_equal(restored_state, first.state)

    authority = _authority(model_state, first.control_state)
    expected = planner.plan(
        first.state,
        model_state,
        first.control_state,
        authority,
    )
    resumed = restored_planner.plan(
        restored_state,
        model_state,
        first.control_state,
        authority,
    )
    _assert_tree_equal(resumed, expected)

    metadata_path = path / "metadata"
    assert metadata_path.exists()
    # Orbax metadata is intentionally opaque on disk; the public loader above
    # validates the exact v1 schema and both child-state exclusion flags.
    assert ONE_STEP_DYNA_CHECKPOINT_SCHEMA.endswith(".v1")


def test_non_threefry_planner_key_is_rejected_before_any_state_exists() -> None:
    planner, model_state, control_state = _system()
    with pytest.raises(TypeError, match="threefry2x32"):
        planner.init(
            jr.key(4, impl="rbg"),
            REPRESENTATION_REVISION,
            model_state,
            control_state,
        )
    assert str(jr.key_impl(jr.key(4, impl="threefry2x32"))) == "threefry2x32"
    assert np.asarray(jr.key_data(jr.key(4, impl="threefry2x32"))).shape == (2,)
