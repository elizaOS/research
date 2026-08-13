"""Adversarial contracts for the public compositional arm-analysis boundary."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Iterator
from typing import Any, cast

import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import CompositionalFeatureLearner
from alberta_framework.evaluation import (
    _compositional_future_utility_calibration_engine as engine,
)
from alberta_framework.evaluation import compositional_control_life_development as control

pytestmark = pytest.mark.integration

_CURATION_GEOMETRY_ARM = "dovetail_coverage_ancestor_headroom_leftpack"
_HARDENED_ANALYSIS_PARAMETERS = {
    "curation_geometry_arm_name",
    "pinned_curation_due_mask",
}


@pytest.fixture(scope="module")
def future_utility_execution() -> Iterator[
    tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ]
]:
    """Return one real production execution for coherently rehashed attacks."""

    protocol = control.build_short_test_protocol()
    source = control.build_bound_compositional_control_life_source(
        protocol,
        observation_key=jr.key(1_901),
        exploration_key=jr.key(1_902),
        random_action_key=jr.key(1_903),
        learner_key=jr.key(1_904),
    )
    arm = engine.FutureUtilityArmSpec(
        name="future_mix1_decay095_none",
        role="public arm-analysis adversarial fixture",
        mix=1.0,
        trace_decay=0.95,
        normalization="none",
    )
    learner = CompositionalFeatureLearner.from_config(
        engine.build_future_utility_learner_config(
            control.learner_config_for_arm(_CURATION_GEOMETRY_ARM),
            arm,
        )
    )
    execution = control.execute_compositional_control_life_arm(
        protocol,
        learner,
        source.learner_key,
        source.observations,
        source.phase_indices,
        source.exploration_mask,
        source.random_actions,
        composed_readout_enabled=True,
    )
    yield protocol, source, execution


def _analyze(
    protocol: control.CompositionalControlLifeProtocol,
    source: control.BoundCompositionalControlLifeSource,
    execution: control.CompositionalControlLifeArmExecution,
) -> control.CompositionalControlLifeArmAnalysisReceipt:
    """Call only the explicit geometry-and-cadence-bound public API."""

    parameters = inspect.signature(
        control.analyze_compositional_control_life_arm_execution
    ).parameters
    assert _HARDENED_ANALYSIS_PARAMETERS <= set(parameters), (
        "arm analysis must explicitly bind curation geometry and the pinned due mask"
    )
    return control.analyze_compositional_control_life_arm_execution(
        protocol,
        execution,
        curation_geometry_arm_name=_CURATION_GEOMETRY_ARM,
        pinned_curation_due_mask=source.curation_due_mask,
    )


def _rehash_execution(
    execution: control.CompositionalControlLifeArmExecution,
    events: Any,
) -> control.CompositionalControlLifeArmExecution:
    """Replace the event tree and coherently close its public digest."""

    return cast(
        control.CompositionalControlLifeArmExecution,
        dataclasses.replace(
            cast(Any, execution),
            events=events,
            trace_sha256=control._array_tree_sha256(events),
        ),
    )


def _replace_curation_trace(events: Any, **changes: object) -> Any:
    trace = dataclasses.replace(cast(Any, events.curation_trace), **changes)
    return events._replace(curation_trace=trace)


def _constructor_kwargs_without_validation_token(instance: object) -> dict[str, object]:
    """Project every init field except an intentionally private factory token."""

    return {
        field.name: getattr(instance, field.name)
        for field in dataclasses.fields(cast(Any, instance))
        if field.init and "validation_token" not in field.name
    }


def test_analysis_names_curation_geometry_without_claiming_source_identity(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution

    analysis = _analyze(protocol, source, execution)
    payload = analysis.to_config()

    assert analysis.curation_geometry_arm_name == _CURATION_GEOMETRY_ARM
    assert payload["curation_geometry_arm_name"] == _CURATION_GEOMETRY_ARM
    assert "source_arm_name" not in payload


def test_analysis_rejects_rehashed_experience_trailing_shape_forgery(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution
    executed_reward = np.asarray(execution.events.executed_reward)
    events = execution.events._replace(
        executed_reward=cast(Any, executed_reward[:, None])
    )
    broken = _rehash_execution(execution, events)

    with pytest.raises(
        (TypeError, ValueError, RuntimeError),
        match="executed_reward|experience|event tree|shape",
    ):
        _analyze(protocol, source, broken)


def test_analysis_rejects_rehashed_off_cadence_promotion(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution
    due_mask = np.asarray(source.curation_due_mask, dtype=np.bool_)
    off_cadence_index = int(np.flatnonzero(~due_mask)[0])
    promotion = np.array(
        execution.events.curation_trace.promotion_applied,
        dtype=np.bool_,
        copy=True,
    )
    assert not promotion[off_cadence_index]
    promotion[off_cadence_index] = True
    events = _replace_curation_trace(
        execution.events,
        promotion_applied=promotion,
    )
    broken = _rehash_execution(execution, events)

    with pytest.raises(
        (TypeError, ValueError, RuntimeError),
        match="promotion|cadence|curation",
    ):
        _analyze(protocol, source, broken)


def test_analysis_rejects_rehashed_false_due_step_should_try_replace(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution
    due_mask = np.asarray(source.curation_due_mask, dtype=np.bool_)
    due_index = int(np.flatnonzero(due_mask)[0])
    should_try_replace = np.array(
        execution.events.curation_trace.should_try_replace,
        dtype=np.bool_,
        copy=True,
    )
    assert should_try_replace[due_index]
    should_try_replace[due_index] = False
    events = _replace_curation_trace(
        execution.events,
        should_try_replace=should_try_replace,
    )
    broken = _rehash_execution(execution, events)

    with pytest.raises(
        (TypeError, ValueError, RuntimeError),
        match="should_try_replace|cadence|curation|due",
    ):
        _analyze(protocol, source, broken)


def test_analysis_rejects_rehashed_curation_count_forgery(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution
    counts = np.array(execution.events.curation_counts, dtype=np.int32, copy=True)
    proposal_index = control.CURATION_COUNT_NAMES.index("proposal")
    counts[0, proposal_index] += np.int32(1)
    events = execution.events._replace(curation_counts=cast(Any, counts))
    broken = _rehash_execution(execution, events)

    with pytest.raises(
        (TypeError, ValueError, RuntimeError),
        match="curation|count|mutation",
    ):
        _analyze(protocol, source, broken)


def test_analysis_rejects_rehashed_pre_post_temporal_discontinuity(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution
    pre_active = np.array(
        execution.events.pre_active_signature_slots,
        dtype=np.bool_,
        copy=True,
    )
    post_active = np.asarray(
        execution.events.post_active_signature_slots,
        dtype=np.bool_,
    )
    row = 1
    slot = control.RAW_DIM
    signature = 0
    assert pre_active[row, slot, signature] == post_active[
        row - 1,
        slot,
        signature,
    ]
    pre_active[row, slot, signature] = ~post_active[
        row - 1,
        slot,
        signature,
    ]
    events = execution.events._replace(
        pre_active_signature_slots=cast(Any, pre_active)
    )
    broken = _rehash_execution(execution, events)

    with pytest.raises(
        (TypeError, ValueError, RuntimeError),
        match="pre.*post|post.*pre|temporal|continuity|signature",
    ):
        _analyze(protocol, source, broken)


def test_execution_receipt_rejects_direct_reconstruction_without_private_token(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution
    receipt = control.validate_compositional_control_life_arm_execution(
        protocol,
        execution,
        pinned_curation_due_mask=source.curation_due_mask,
    )

    with pytest.raises(
        (TypeError, ValueError),
        match="private validation token|validation_token|produced by the validator",
    ):
        cast(Any, control.CompositionalControlLifeArmExecutionReceipt)(
            **_constructor_kwargs_without_validation_token(receipt),
        )


def test_analysis_receipt_rejects_direct_reconstruction_without_private_token(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution
    analysis = _analyze(protocol, source, execution)

    with pytest.raises(
        (TypeError, ValueError),
        match="private validation token|validation_token|produced by the analyzer",
    ):
        cast(Any, control.CompositionalControlLifeArmAnalysisReceipt)(
            **_constructor_kwargs_without_validation_token(analysis),
        )
