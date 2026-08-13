"""Public surface for UPGD's exact-lifetime checkpoint contract."""

from __future__ import annotations

import alberta_framework as alberta
from alberta_framework.core import upgd


def test_upgd_exact_lifetime_helpers_are_public() -> None:
    names = (
        "UPGD_CHECKPOINT_SCHEMA",
        "UPGD_LIFETIME_COUNTER_DELTA_NBYTES",
        "UPGD_LIFETIME_COUNTER_NBYTES",
        "UPGD_STATE_SCHEMA",
        "UPGD_TRANSACTION_CLOCK_DELTA_NBYTES",
        "UPGD_TRANSACTION_CLOCK_NBYTES",
        "load_upgd_checkpoint",
        "measure_upgd_state_nbytes",
        "migrate_legacy_upgd_state",
        "save_upgd_checkpoint",
        "upgd_lifetime_counter_nbytes",
        "upgd_transaction_clock_nbytes",
    )
    for name in names:
        assert getattr(alberta, name) is getattr(upgd, name)
        assert name in alberta.__all__

    assert alberta.UPGD_STATE_SCHEMA == "alberta.upgd-state.v2"
    assert alberta.UPGD_CHECKPOINT_SCHEMA == "alberta.upgd-checkpoint.v2"
    assert alberta.upgd_lifetime_counter_nbytes() == 12
    assert alberta.upgd_transaction_clock_nbytes() == 16

