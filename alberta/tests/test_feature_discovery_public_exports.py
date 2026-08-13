"""Public exact-clock exports for fixed-budget feature discovery."""

from __future__ import annotations

import pytest

import alberta_framework as alberta
from alberta_framework.core import feature_discovery

pytestmark = pytest.mark.unit


def test_feature_discovery_exact_clock_surface_is_public_and_unique() -> None:
    names = (
        "FEATURE_DISCOVERY_CHECKPOINT_SCHEMA",
        "FEATURE_DISCOVERY_LIFETIME_COUNTER_DELTA_NBYTES",
        "FEATURE_DISCOVERY_LIFETIME_COUNTER_NBYTES",
        "FEATURE_DISCOVERY_STATE_SCHEMA",
        "FEATURE_DISCOVERY_TRANSACTION_CLOCK_DELTA_NBYTES",
        "FEATURE_DISCOVERY_TRANSACTION_CLOCK_NBYTES",
        "FeatureDiscoveryState",
        "FeatureDiscoveryUpdateResult",
        "feature_discovery_lifetime_counter_nbytes",
        "feature_discovery_transaction_clock_nbytes",
        "load_feature_discovery_checkpoint",
        "measure_feature_discovery_state_nbytes",
        "migrate_legacy_feature_discovery_state",
        "save_feature_discovery_checkpoint",
    )
    for name in names:
        assert alberta.__all__.count(name) == 1
        assert getattr(alberta, name) is getattr(feature_discovery, name)
