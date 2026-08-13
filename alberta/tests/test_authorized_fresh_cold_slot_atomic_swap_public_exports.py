"""Public-package identities for the v2 filtered atomic-swap surfaces."""

import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core import authorized_fresh_cold_slot_atomic_swap as outer
from alberta_framework.core import authorized_option_replacement as replacement
from alberta_framework.core import cumulant_option_scheduler as scheduler

pytestmark = pytest.mark.unit

_LOWER_REPLACEMENT_NAMES = (
    "AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_PREPARED_SCHEMA",
    "AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_RECEIPT_SCHEMA",
    "AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_RESULT_SCHEMA",
    "AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt",
    "AuthorizedOptionExternalCandidateAdoptionDiagnostics",
    "AuthorizedOptionExternalCandidateAdoptionPrepared",
    "AuthorizedOptionExternalCandidateAdoptionResult",
    "adopt_authorized_option_external_candidate",
    "authorized_option_external_candidate_adoption_authority_receipt",
    "prepare_authorized_option_external_candidate_adoption",
)
_LOWER_SCHEDULER_NAMES = (
    "CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_PREPARED_SCHEMA",
    "CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_RECEIPT_SCHEMA",
    "CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_RESULT_SCHEMA",
    "CumulantOptionExternalBundleAdoptionAuthorityReceipt",
    "CumulantOptionExternalBundleAdoptionDiagnostics",
    "CumulantOptionExternalBundleAdoptionPrepared",
    "CumulantOptionExternalBundleAdoptionResult",
    "adopt_cumulant_option_external_bundle",
    "cumulant_option_external_bundle_adoption_authority_receipt",
    "prepare_cumulant_option_external_bundle_adoption",
)


def test_outer_root_and_core_exports_have_exact_identity() -> None:
    assert outer.__all__
    assert len(outer.__all__) == len(set(outer.__all__))
    for name in outer.__all__:
        expected = getattr(outer, name)
        assert getattr(core, name) is expected
        assert getattr(alberta, name) is expected
        assert name in core.__all__
        assert name in alberta.__all__


def test_additive_lower_public_boundaries_have_exact_package_identity() -> None:
    for implementation, names in (
        (replacement, _LOWER_REPLACEMENT_NAMES),
        (scheduler, _LOWER_SCHEDULER_NAMES),
    ):
        for name in names:
            assert name in implementation.__all__
            expected = getattr(implementation, name)
            assert getattr(core, name) is expected
            assert getattr(alberta, name) is expected
            assert name in core.__all__
            assert name in alberta.__all__


def test_all_export_manifests_remain_duplicate_free() -> None:
    assert len(replacement.__all__) == len(set(replacement.__all__))
    assert len(scheduler.__all__) == len(set(scheduler.__all__))
    assert len(core.__all__) == len(set(core.__all__))
    assert len(alberta.__all__) == len(set(alberta.__all__))
