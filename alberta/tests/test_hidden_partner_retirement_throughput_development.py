"""Slow development falsification for bounded stale-retirement throughput."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from alberta_framework.evaluation.hidden_partner_retirement_throughput_development import (
    ARM_ORDER,
    CYCLE_STEPS,
    EARLIEST_D_STALE_STEP,
    FINAL_ABSENCE_START_STEP,
    LEGACY_FULL_BANK_QUEUE_BOUND_STEPS,
    ORDINARY_REPLACEMENT_INTERVAL,
    PROMPT_FULL_BANK_QUEUE_BOUND_STEPS,
    PROMPT_RETIREMENT_INTERVAL,
    RETIREMENT_SLACK_STEPS,
    RETIREMENT_THROUGHPUT_ARMS,
    RETIREMENT_THROUGHPUT_SEED,
    RetirementThroughputPanelResult,
    run_retirement_throughput_development,
    validate_retirement_throughput_static_contract,
)

pytestmark = [pytest.mark.development, pytest.mark.slow]


@functools.lru_cache(maxsize=1)
def _panel() -> RetirementThroughputPanelResult:
    return run_retirement_throughput_development()


def test_retirement_throughput_contract_is_fixed_bounded_and_nonpromoting() -> None:
    assert validate_retirement_throughput_static_contract() == ()
    assert CYCLE_STEPS == 2_304
    assert RETIREMENT_THROUGHPUT_SEED.namespace.endswith(
        "retirement-throughput-falsification-v1"
    )
    assert tuple(arm.name for arm in RETIREMENT_THROUGHPUT_ARMS) == ARM_ORDER
    prompt, legacy = RETIREMENT_THROUGHPUT_ARMS
    assert prompt.stale_retirement_interval == PROMPT_RETIREMENT_INTERVAL == 31
    assert legacy.stale_retirement_interval is None
    assert ORDINARY_REPLACEMENT_INTERVAL == 64
    assert EARLIEST_D_STALE_STEP == 1_665
    assert FINAL_ABSENCE_START_STEP == 2_047
    assert RETIREMENT_SLACK_STEPS == 382
    assert PROMPT_FULL_BANK_QUEUE_BOUND_STEPS == 372
    assert LEGACY_FULL_BANK_QUEUE_BOUND_STEPS == 768
    assert PROMPT_FULL_BANK_QUEUE_BOUND_STEPS <= RETIREMENT_SLACK_STEPS
    assert LEGACY_FULL_BANK_QUEUE_BOUND_STEPS > RETIREMENT_SLACK_STEPS


def test_prompt_retirement_records_valid_unchanged_gate_rejection() -> None:
    panel = _panel()
    assert panel.development_only
    assert not panel.scientific_promotion_allowed
    assert not panel.output_writes_allowed
    assert tuple(result.arm.name for result in panel.arms) == ARM_ORDER
    assert all(result.validation.valid for result in panel.arms), panel.to_report()

    prompt = panel.arm_result("prompt_retirement")
    legacy = panel.arm_result("legacy_throughput")
    assert prompt.interaction_config["stale_retirement_interval"] == 31
    assert legacy.interaction_config["stale_retirement_interval"] is None
    assert prompt.interaction_config["replacement_interval"] == 64
    assert legacy.interaction_config["replacement_interval"] == 64
    for result in panel.arms:
        live = np.asarray(result.run.trace.interaction_live_feature_count)
        vacancies = np.asarray(result.run.trace.interaction_vacancy_count)
        assert int(np.min(live)) >= 11, panel.to_report()
        assert int(np.max(vacancies)) <= 1, panel.to_report()

    lifecycle = prompt.lifecycle
    assert not lifecycle.d_task_learned, panel.to_report()
    assert lifecycle.d_deployed_through_exit, panel.to_report()
    assert lifecycle.d_retirement_event_steps == (1_673,), panel.to_report()
    assert legacy.lifecycle.d_retirement_event_steps == (1_791,), panel.to_report()
    assert lifecycle.d_retirement_event_aligned, panel.to_report()
    assert legacy.lifecycle.d_retirement_event_aligned, panel.to_report()
    assert lifecycle.d_linked_matching_candidate_reset_count == 1, panel.to_report()
    assert lifecycle.d_linked_candidate_utility_post == 0.0, panel.to_report()
    assert lifecycle.d_linked_candidate_head_linf_post == 0.0, panel.to_report()
    assert lifecycle.d_linked_candidate_age_post == 0, panel.to_report()
    assert lifecycle.d_repromotions_after_retirement == 0, panel.to_report()
    assert lifecycle.d_absent_entire_final_window, panel.to_report()
    assert legacy.lifecycle.d_absent_entire_final_window, panel.to_report()
    assert not lifecycle.d_learned_then_stably_retired, panel.to_report()
    assert not legacy.lifecycle.d_learned_then_stably_retired, panel.to_report()
    assert panel.d_requirement_failures == (
        "prompt_d_task_learned",
        "prompt_d_learned_then_stably_retired",
    ), panel.to_report()
    assert panel.prompt_retirement_precedes_legacy, panel.to_report()
    assert not panel.final_absence_isolated_to_prompt, panel.to_report()
    assert panel.cadence_alone_falsified_on_fixed_seed, panel.to_report()
    assert panel.status == "valid_development_rejection", panel.to_report()

    # C is deliberately outside this intervention's judged requirements.
    assert all(not name.startswith("c_") for name in panel.d_requirement_failures)
