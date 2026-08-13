from __future__ import annotations

import copy

import pytest

from alberta_framework.evaluation import partner_policy_fusion_stress_development as stress

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.parametrize("split", [0, 1, 47, 48, 95, 96])
def test_every_boundary_class_resumes_to_the_exact_full_report(split: int) -> None:
    expected = stress.run_partner_policy_fusion_stress_development()
    checkpoint = stress.make_partner_policy_fusion_stress_checkpoint(split)
    resumed = stress.resume_partner_policy_fusion_stress_checkpoint(checkpoint)
    assert resumed.payload() == expected.payload()


def test_core_checkpoint_tamper_is_rejected_after_outer_reseal() -> None:
    checkpoint = stress.make_partner_policy_fusion_stress_checkpoint(12)
    forged = copy.deepcopy(checkpoint)
    conditions = forged["conditions"]
    assert isinstance(conditions, dict)
    learned = conditions[stress.LEARNED_FUSION]
    assert isinstance(learned, dict)
    state_checkpoint = learned["state"]
    assert isinstance(state_checkpoint, dict)
    state = state_checkpoint["state"]
    assert isinstance(state, dict)
    weights = state["reliability_weights"]
    assert isinstance(weights, list)
    first_row = weights[0]
    assert isinstance(first_row, list)
    first_row[0] = float(first_row[0]) + 0.25

    # Resealing only the evaluator shell cannot repair the nested core digest.
    unsigned = dict(forged)
    unsigned.pop("checkpoint_digest")
    forged["checkpoint_digest"] = stress._digest(unsigned)
    with pytest.raises(ValueError, match="checkpoint"):
        stress.resume_partner_policy_fusion_stress_checkpoint(forged)


def test_repeated_runs_are_bit_exact_and_source_runtime_bound() -> None:
    first = stress.run_partner_policy_fusion_stress_development()
    second = stress.run_partner_policy_fusion_stress_development()
    assert first.payload() == second.payload()
    assert len(first.source_manifest) == 2
    device_count = first.runtime_manifest["device_count"]
    assert isinstance(device_count, int)
    assert device_count >= 1
    assert first.initial_snapshot_digest == second.initial_snapshot_digest
