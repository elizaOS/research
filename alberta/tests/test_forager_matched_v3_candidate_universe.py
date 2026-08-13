"""Contracts for the unexecuted matched-Forager v3 candidate universe."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_candidate_universe as v3,
)

_PROTOCOL_SHA256 = "1" * 64
_RESULT_SHA256 = "2" * 64
_EXPECTED_UNIVERSE_SHA256 = (
    "a441b35eed4ec6327bf03463099a46e9c2596f2a169182fd317fe51c98b4c750"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _accepted_receipt(
    *,
    causal: str = "causal_e025_q050",
    horde: str = "alberta_horde_default",
    local_rtu: str = "alberta_rtu_h08_taylor",
    plasticity: str = "external_dqn_crelu",
    protocol_sha256: str = _PROTOCOL_SHA256,
    result_sha256: str = _RESULT_SHA256,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": v3.FORAGER_MATCHED_V3_SELECTION_RECEIPT_SCHEMA_VERSION,
        "status": "accepted",
        "classification": "development_selection_nonpromoting",
        "development_universe": {
            "schema_version": v3.FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION,
            "sha256": v3.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256,
        },
        "development_protocol": {
            "schema_version": v3.FORAGER_MATCHED_V3_DEVELOPMENT_PROTOCOL_SCHEMA_VERSION,
            "sha256": protocol_sha256,
        },
        "development_result": {
            "schema_version": v3.FORAGER_MATCHED_V3_DEVELOPMENT_RESULT_SCHEMA_VERSION,
            "sha256": result_sha256,
        },
        "selections": {
            "causal_candidate_id": causal,
            "horde_candidate_id": horde,
            "local_rtu_candidate_id": local_rtu,
            "dqn_plasticity_candidate_id": plasticity,
        },
        "selection_rule": "exact_group_winners_from_bound_development_result",
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }
    receipt["payload_sha256"] = _canonical_sha256(receipt)
    return receipt


def _rehash_receipt(receipt: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = _canonical_sha256(unsigned)


@pytest.mark.unit
def test_development_universe_counts_order_and_detachment() -> None:
    descriptor = v3.matched_v3_development_universe_descriptor()
    ids = tuple(item["candidate_id"] for item in descriptor["candidates"])

    assert descriptor["schema_version"] == (
        "alberta.forager_matched_v3_development_universe.v1"
    )
    assert ids == v3.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS
    assert ids[:9] == v3.MATCHED_V3_CAUSAL_SELECTION_CANDIDATE_IDS
    assert ids[9:13] == v3.MATCHED_V3_HORDE_SELECTION_CANDIDATE_IDS
    assert ids[13] == "alberta_rtu_h08_taylor"
    assert ids[14:25] == v3.MATCHED_V3_EXTERNAL_INFERENTIAL_CANDIDATE_IDS
    assert ids[25:] == v3.MATCHED_V3_DESCRIPTIVE_CANDIDATE_IDS
    assert len(ids) == 28
    assert len(set(ids)) == 28
    assert descriptor["scope"] == {
        "development_candidate_count": 28,
        "development_inferential_candidate_count": 25,
        "alberta_inferential_candidate_count": 14,
        "external_inferential_candidate_count": 11,
        "descriptive_candidate_count": 3,
        "confirmatory_inferential_candidate_count": 11,
        "confirmatory_descriptive_candidate_count": 3,
        "research_literature_exhaustive": False,
        "universal_sota_claim_allowed": False,
        "scientific_promotion_allowed": False,
    }

    raw = v3.canonical_matched_v3_development_universe_bytes()
    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_UNIVERSE_SHA256
    assert v3.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256 == _EXPECTED_UNIVERSE_SHA256
    assert v3.parse_matched_v3_development_universe_artifact(raw) == descriptor

    descriptor["candidates"][0]["candidate_id"] = "mutated"
    assert v3.matched_v3_development_universe_descriptor()["candidates"][0][
        "candidate_id"
    ] == "causal_e025_q050"


@pytest.mark.unit
def test_canonical_universe_reads_ignore_mutated_private_construction_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = v3.canonical_matched_v3_development_universe_bytes()
    first = v3._MATCHED_V3_DEVELOPMENT_UNIVERSE["candidates"][0]
    monkeypatch.setitem(first, "rationale", "mutated private construction graph")
    monkeypatch.setitem(
        v3._MATCHED_V3_DEVELOPMENT_UNIVERSE["source_pins"][0],
        "archive_size_bytes",
        1,
    )

    assert v3.canonical_matched_v3_development_universe_bytes() == raw
    assert v3.matched_v3_development_universe_descriptor()["candidates"][0][
        "rationale"
    ] != "mutated private construction graph"
    assert v3.parse_matched_v3_development_universe_artifact(raw)["source_pins"][0][
        "archive_size_bytes"
    ] == 314_961_920


@pytest.mark.unit
def test_candidate_dataclasses_are_frozen_and_returned_as_an_immutable_tuple() -> None:
    candidates = v3.matched_v3_development_candidates()
    assert isinstance(candidates, tuple)
    assert tuple(item.candidate_id for item in candidates) == (
        v3.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS
    )
    with pytest.raises(FrozenInstanceError):
        candidates[0].candidate_id = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_selection_groups_are_exact_and_disjoint() -> None:
    descriptor = v3.matched_v3_development_universe_descriptor()
    groups = descriptor["selection_groups"]

    assert groups == [
        {
            "group_id": "causal",
            "candidate_ids": list(v3.MATCHED_V3_CAUSAL_SELECTION_CANDIDATE_IDS),
            "confirmatory_selection_count": 1,
        },
        {
            "group_id": "horde",
            "candidate_ids": list(v3.MATCHED_V3_HORDE_SELECTION_CANDIDATE_IDS),
            "confirmatory_selection_count": 1,
        },
        {
            "group_id": "local_rtu",
            "candidate_ids": ["alberta_rtu_h08_taylor"],
            "confirmatory_selection_count": 1,
        },
        {
            "group_id": "dqn_plasticity",
            "candidate_ids": list(
                v3.MATCHED_V3_DQN_PLASTICITY_SELECTION_CANDIDATE_IDS
            ),
            "confirmatory_selection_count": 1,
        },
    ]
    selectable = [candidate_id for group in groups for candidate_id in group["candidate_ids"]]
    assert len(selectable) == len(set(selectable))
    assert set(v3.MATCHED_V3_FIXED_EXTERNAL_INFERENTIAL_CANDIDATE_IDS).isdisjoint(
        selectable
    )
    assert set(v3.MATCHED_V3_DESCRIPTIVE_CANDIDATE_IDS).isdisjoint(selectable)


@pytest.mark.unit
def test_source_pins_and_new_adapters_are_exact_and_not_execution_ready() -> None:
    descriptor = v3.matched_v3_development_universe_descriptor()
    pins = {item["repository_id"]: item for item in descriptor["source_pins"]}
    candidates = {item["candidate_id"]: item for item in descriptor["candidates"]}

    assert pins["foragax_agents"]["commit_git_sha1"] == (
        "9710f60fa30da5badc451ad7ce3ff296d5070830"
    )
    assert pins["foragax_agents"]["tree_git_sha1"] == (
        "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
    )
    assert pins["foragax_agents"]["archive_sha256"] == (
        "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
    )
    assert pins["foragax_agents"]["archive_size_bytes"] == 314_961_920
    assert pins["dopamine"]["commit_git_sha1"] == (
        "5873f5494ee0c2d7c016d0ab2ad530354fec59d0"
    )
    assert pins["dopamine"]["tree_git_sha1"] == (
        "578408662e298d00e4e855f13f67dc08bd784e7c"
    )
    assert pins["dopamine"]["archive_sha256"] == (
        "bea46f755c86725d7ca90c531a08aad86cab62201ac2b9224c82f66dfada7456"
    )
    assert pins["pobax"]["commit_git_sha1"] == (
        "a5e1d62d14e4efe783885b9d4f19cffa2a568eec"
    )
    assert pins["pobax"]["tree_git_sha1"] == (
        "d67cf5c209f2e7de9ce517d4bc72a2741ccaf6a6"
    )
    assert pins["pobax"]["archive_sha256"] == (
        "f354028549d79a1b3f1ee67deaa46454a0be60d9346764e5aed9e8ab93768ad9"
    )

    dopamine_files = {
        item["path"]: item["sha256"] for item in pins["dopamine"]["relevant_files"]
    }
    assert dopamine_files == {
        "LICENSE": "e47b2783cb7131207707c35d0aea22277aa1beded6bf9d7c2436cd7de9462323",
        "dopamine/jax/agents/full_rainbow/configs/full_rainbow.gin": (
            "f926614f7c99ec248f3bafdbb920a7d8497476c0a27d5aad9ca8c69ca9ebc130"
        ),
        "dopamine/jax/agents/full_rainbow/full_rainbow_agent.py": (
            "cc85222d9b60b6f05cbb8e6af170a57a3f74c20c9dd72067b70d8daf4cf50595"
        ),
        "dopamine/jax/losses.py": (
            "42c10699bebf5b41b7bcd5cbeb18693c0f606f3bc427b988426368741e3cbd39"
        ),
        "dopamine/jax/agents/dqn/dqn_agent.py": (
            "53a37912775c1fcce84f3c158c29fb9d63094ba8dc9f8a0c9c627e0f8c519dca"
        ),
        "dopamine/jax/networks.py": (
            "fac813138454e2c947aca78a284b0e79b8f021beaf27b5f99981177ec8ca3bb9"
        ),
        "dopamine/jax/agents/rainbow/rainbow_agent.py": (
            "02c90de41f68c18e66938bc9c5664a5e6154b8c67571114c8955d04a9e67cef8"
        ),
        "dopamine/jax/replay_memory/accumulator.py": (
            "cfe4c849b2121f259fce5cd23e0a349f6ffba45f3c5c167dd63f36da2fc9cd25"
        ),
        "dopamine/jax/replay_memory/samplers.py": (
            "de33adddd80fa4194e5eda14182f1eee50c65492c575e16e5c45630b9c75bb0b"
        ),
    }
    pobax_files = {
        item["path"]: item["sha256"] for item in pins["pobax"]["relevant_files"]
    }
    assert pobax_files == {
        "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "pobax/algos/ppo.py": (
            "0c82725027e6022d48847bca45a87e6f8d9b54d720bbb844f053d4b8448ce153"
        ),
        "pobax/models/network.py": (
            "b3ea151f6a7f9000dd1b529cbcc262c150b767c66664399008aa89283a2e520a"
        ),
        "pobax/config.py": (
            "38bb46c93734c8882ab7ad7bdfbee9d64bb21db04231ccd15b9ec2a6eb02034c"
        ),
        "pobax/models/actor_critic.py": (
            "bb707481b32eefc1219adbc38abd527c3c600cf8941ae963bf6b6540c9b2158f"
        ),
        "pobax/models/discrete.py": (
            "ad7ac11a03b49f7ea53fcf11b0b97cc7697f57447f4661a22fb235a6ab90885c"
        ),
        "pobax/models/__init__.py": (
            "c4434b0b1eba13c227cdf479380f5347aa57aba4d2f78a12112c056cdada323a"
        ),
        "pobax/models/value.py": (
            "e875e7ef951aba37ea4648328442aaece0fc3415de580c6b5115843eb32366bd"
        ),
        "pyproject.toml": (
            "4f02e96a5d8471f9637ec36dc9536398183f49fb28fa07c5b7f371ffcdbe81d5"
        ),
        "requirements.txt": (
            "8d8a36a4428d481b15c47b9ed1aec573c3dc2472af746be611e9a17dae40a17c"
        ),
    }

    for candidate_id, repository_id in (
        ("adapted_full_rainbow", "dopamine"),
        ("adapted_ppo_gru", "pobax"),
    ):
        item = candidates[candidate_id]
        assert item["source_repository_id"] == repository_id
        assert item["source_relationship"] == (
            "derived_adapter_not_exact_upstream_execution"
        )
        assert item["adapter_status"] == "existing_but_unqualified_for_v3"
        assert item["execution_ready"] is False
        assert "implemented" in item["rationale"].lower()
        assert "runner" in item["rationale"].lower()


@pytest.mark.unit
def test_claim_flags_are_hardcoded_false_for_universe_and_every_candidate() -> None:
    descriptor = v3.matched_v3_development_universe_descriptor()
    claims = descriptor["claim_boundaries"]

    assert claims == {
        "descriptor_supports_performance_claim": False,
        "development_results_support_confirmatory_performance_claim": False,
        "research_literature_exhaustive": False,
        "universal_sota_claim_allowed": False,
        "scientific_promotion_allowed": False,
        "execution_authorized": False,
        "confirmatory_panel_is_executed": False,
        "builder_infers_selection_from_scores": False,
    }
    for candidate in descriptor["candidates"]:
        assert candidate["execution_ready"] is False
        assert candidate["scientific_promotion_allowed"] is False
        assert candidate["universal_sota_claim_allowed"] is False


@pytest.mark.unit
def test_accepted_receipt_builds_exact_eleven_plus_three_panel_without_scores() -> None:
    receipt = _accepted_receipt(
        causal="causal_e050_q075",
        horde="alberta_horde_recurrent64",
        plasticity="external_dqn_redo",
    )
    panel = v3.build_matched_v3_confirmatory_panel(
        receipt,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )

    assert panel["inferential_candidate_ids"] == [
        "causal_e050_q075",
        "alberta_horde_recurrent64",
        "alberta_rtu_h08_taylor",
        "external_dqn_plain",
        "external_dqn_redo",
        "external_pt_dqn_xfinal",
        "external_drqn_xfinal",
        "isolated_ppo_generic",
        "isolated_rtu_paper_scale",
        "adapted_full_rainbow",
        "adapted_ppo_gru",
    ]
    assert panel["descriptive_candidate_ids"] == [
        "random_policy",
        "search_nearest",
        "search_oracle",
    ]
    assert panel["candidate_ids"] == (
        panel["inferential_candidate_ids"] + panel["descriptive_candidate_ids"]
    )
    assert len(panel["inferential_candidate_ids"]) == 11
    assert len(panel["descriptive_candidate_ids"]) == 3
    assert "scores" not in panel
    assert panel["claim_boundaries"]["builder_inferred_selection_from_scores"] is False
    assert panel["claim_boundaries"]["scientific_promotion_allowed"] is False
    assert panel["claim_boundaries"]["universal_sota_claim_allowed"] is False
    assert v3.validate_matched_v3_confirmatory_panel(
        panel,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    ) == panel

    panel["candidate_ids"].append("mutation")
    rebuilt = v3.build_matched_v3_confirmatory_panel(
        receipt,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )
    assert "mutation" not in rebuilt["candidate_ids"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("selection_key", "candidate_id"),
    [
        ("causal_candidate_id", "search_nearest"),
        ("horde_candidate_id", "random_policy"),
        ("local_rtu_candidate_id", "search_oracle"),
        ("dqn_plasticity_candidate_id", "search_oracle"),
    ],
)
def test_descriptive_candidates_cannot_leak_into_selection(
    selection_key: str, candidate_id: str
) -> None:
    receipt = _accepted_receipt()
    receipt["selections"][selection_key] = candidate_id
    _rehash_receipt(receipt)

    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="selection group"):
        v3.build_matched_v3_confirmatory_panel(
            receipt,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("selection_key", "candidate_id"),
    [
        ("causal_candidate_id", "alberta_horde_eps05"),
        ("horde_candidate_id", "causal_e050_q050"),
        ("local_rtu_candidate_id", "alberta_rtu_h08_taylor_wrong"),
        ("dqn_plasticity_candidate_id", "external_dqn_plain"),
    ],
)
def test_wrong_group_choices_are_rejected(selection_key: str, candidate_id: str) -> None:
    receipt = _accepted_receipt()
    receipt["selections"][selection_key] = candidate_id
    _rehash_receipt(receipt)

    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="selection group"):
        v3.build_matched_v3_confirmatory_panel(
            receipt,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
def test_duplicate_choices_are_rejected_before_panel_construction() -> None:
    receipt = _accepted_receipt()
    receipt["selections"]["horde_candidate_id"] = receipt["selections"][
        "causal_candidate_id"
    ]
    _rehash_receipt(receipt)

    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="duplicate"):
        v3.build_matched_v3_confirmatory_panel(
            receipt,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "binding", ["development_universe", "development_protocol", "development_result"]
)
def test_receipt_digest_mismatches_are_rejected(binding: str) -> None:
    receipt = _accepted_receipt()
    receipt[binding]["sha256"] = "3" * 64
    _rehash_receipt(receipt)

    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="digest"):
        v3.build_matched_v3_confirmatory_panel(
            receipt,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
def test_receipt_payload_digest_and_exact_keys_are_required() -> None:
    bad_digest = _accepted_receipt()
    bad_digest["payload_sha256"] = "0" * 64
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="payload digest"):
        v3.build_matched_v3_confirmatory_panel(
            bad_digest,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )

    extra = _accepted_receipt()
    extra["scores"] = {"causal_e025_q050": 123.0}
    _rehash_receipt(extra)
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="exact keys"):
        v3.build_matched_v3_confirmatory_panel(
            extra,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
def test_only_strict_accepted_receipts_are_admitted() -> None:
    for key, value in (
        ("status", "rejected"),
        ("scientific_promotion_allowed", True),
        ("universal_sota_claim_allowed", True),
    ):
        receipt = _accepted_receipt()
        receipt[key] = value
        _rehash_receipt(receipt)
        with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError):
            v3.build_matched_v3_confirmatory_panel(
                receipt,
                expected_development_protocol_sha256=_PROTOCOL_SHA256,
                expected_development_result_sha256=_RESULT_SHA256,
            )


@pytest.mark.unit
def test_cross_version_universe_receipt_and_panel_are_rejected() -> None:
    raw_descriptor = v3.matched_v3_development_universe_descriptor()
    raw_descriptor["schema_version"] = "alberta.forager_matched_candidate_universe.v2"
    raw = json.dumps(
        raw_descriptor,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError):
        v3.parse_matched_v3_development_universe_artifact(raw)

    receipt = _accepted_receipt()
    receipt["schema_version"] = "alberta.forager_matched_candidate_selection.v2"
    _rehash_receipt(receipt)
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="schema"):
        v3.build_matched_v3_confirmatory_panel(
            receipt,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )

    good_receipt = _accepted_receipt()
    panel = v3.build_matched_v3_confirmatory_panel(
        good_receipt,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )
    panel["schema_version"] = "alberta.forager_matched_confirmatory_panel.v2"
    unsigned = {key: value for key, value in panel.items() if key != "payload_sha256"}
    panel["payload_sha256"] = _canonical_sha256(unsigned)
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="schema"):
        v3.validate_matched_v3_confirmatory_panel(
            panel,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
def test_expected_digest_arguments_are_strict_sha256_values() -> None:
    receipt = _accepted_receipt()
    for protocol_digest, result_digest in (
        ("short", _RESULT_SHA256),
        (_PROTOCOL_SHA256, "G" * 64),
    ):
        with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="SHA-256"):
            v3.build_matched_v3_confirmatory_panel(
                copy.deepcopy(receipt),
                expected_development_protocol_sha256=protocol_digest,
                expected_development_result_sha256=result_digest,
            )


@pytest.mark.unit
def test_receipt_is_snapshotted_before_validation_and_never_uses_hostile_deepcopy() -> None:
    class HostileSelections(dict[str, Any]):
        def __deepcopy__(self, memo: dict[int, object]) -> dict[str, Any]:
            del memo
            mutated = dict(self)
            mutated["causal_candidate_id"] = "search_nearest"
            return mutated

    receipt = _accepted_receipt(causal="causal_e100_q090")
    receipt["selections"] = HostileSelections(receipt["selections"])
    _rehash_receipt(receipt)

    panel = v3.build_matched_v3_confirmatory_panel(
        receipt,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )

    assert panel["selection_receipt"]["selections"]["causal_candidate_id"] == (
        "causal_e100_q090"
    )
    assert v3.validate_matched_v3_confirmatory_panel(
        panel,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    ) == panel


@pytest.mark.unit
def test_oversized_receipt_unknown_field_is_rejected_before_json_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _accepted_receipt()
    receipt["unknown"] = "x" * (2 * 1024 * 1024)

    def forbidden_loads(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("oversized caller JSON reached the parser")

    monkeypatch.setattr(json, "loads", forbidden_loads)
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="too large"):
        v3.validate_matched_v3_development_selection_receipt(
            receipt,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
def test_oversized_nested_receipt_payload_is_rejected() -> None:
    receipt = _accepted_receipt()
    receipt["selections"]["causal_candidate_id"] = "x" * (2 * 1024 * 1024)

    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="too large"):
        v3.build_matched_v3_confirmatory_panel(
            receipt,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
def test_oversized_nested_panel_payload_is_rejected_before_exact_key_validation() -> None:
    panel = v3.build_matched_v3_confirmatory_panel(
        _accepted_receipt(),
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )
    panel["unknown"] = {"nested": {"payload": "x" * (2 * 1024 * 1024)}}

    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="too large"):
        v3.validate_matched_v3_confirmatory_panel(
            panel,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
@pytest.mark.parametrize("target", ["receipt", "panel"])
def test_invalid_unicode_is_normalized_to_the_public_error(target: str) -> None:
    receipt = _accepted_receipt()
    value: dict[str, Any]
    if target == "receipt":
        value = receipt
    else:
        value = v3.build_matched_v3_confirmatory_panel(
            receipt,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )
    value["unknown"] = "\ud800"

    with pytest.raises(
        v3.ForagerMatchedV3CandidateUniverseError, match="canonical JSON"
    ):
        if target == "receipt":
            v3.validate_matched_v3_development_selection_receipt(
                value,
                expected_development_protocol_sha256=_PROTOCOL_SHA256,
                expected_development_result_sha256=_RESULT_SHA256,
            )
        else:
            v3.validate_matched_v3_confirmatory_panel(
                value,
                expected_development_protocol_sha256=_PROTOCOL_SHA256,
                expected_development_result_sha256=_RESULT_SHA256,
            )


@pytest.mark.unit
def test_dynamic_artifact_bytes_round_trip_to_detached_validated_snapshots() -> None:
    receipt = _accepted_receipt(causal="causal_e100_q090", plasticity="external_dqn_redo")
    receipt_raw = v3.canonical_matched_v3_development_selection_receipt_bytes(
        receipt,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )
    parsed_receipt = v3.parse_matched_v3_development_selection_receipt_artifact(
        receipt_raw,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )

    assert receipt_raw == _canonical_bytes(receipt)
    assert parsed_receipt == receipt
    assert parsed_receipt is not receipt
    assert parsed_receipt["selections"] is not receipt["selections"]

    panel = v3.build_matched_v3_confirmatory_panel(
        receipt,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )
    panel_raw = v3.canonical_matched_v3_confirmatory_panel_bytes(
        panel,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )
    parsed_panel = v3.parse_matched_v3_confirmatory_panel_artifact(
        panel_raw,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )

    assert panel_raw == _canonical_bytes(panel)
    assert parsed_panel == panel
    assert parsed_panel is not panel
    assert parsed_panel["selection_receipt"] is not panel["selection_receipt"]


@pytest.mark.unit
@pytest.mark.parametrize("artifact", ["receipt", "panel"])
def test_dynamic_artifact_parsers_reject_nested_duplicate_keys(artifact: str) -> None:
    receipt = _accepted_receipt()
    if artifact == "receipt":
        raw = _canonical_bytes(receipt)
        marker = b'"selections":{'
        duplicate = b'"causal_candidate_id":"causal_e025_q050",'
        parser = v3.parse_matched_v3_development_selection_receipt_artifact
    else:
        panel = v3.build_matched_v3_confirmatory_panel(
            receipt,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )
        raw = _canonical_bytes(panel)
        marker = b'"counts":{'
        duplicate = b'"candidate_count":14,'
        parser = v3.parse_matched_v3_confirmatory_panel_artifact
    duplicate_raw = raw.replace(marker, marker + duplicate, 1)
    assert duplicate_raw != raw

    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="duplicate JSON"):
        parser(
            duplicate_raw,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
@pytest.mark.parametrize("artifact", ["receipt", "panel"])
def test_dynamic_artifact_parsers_reject_noncanonical_whitespace_and_order(
    artifact: str,
) -> None:
    receipt = _accepted_receipt()
    if artifact == "receipt":
        raw = _canonical_bytes(receipt)
        parser = v3.parse_matched_v3_development_selection_receipt_artifact
    else:
        panel = v3.build_matched_v3_confirmatory_panel(
            receipt,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )
        raw = _canonical_bytes(panel)
        parser = v3.parse_matched_v3_confirmatory_panel_artifact

    parsed = json.loads(raw)
    reordered = dict(reversed(tuple(parsed.items())))
    noncanonical_order = json.dumps(
        reordered,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")
    for invalid_raw in (b" " + raw, noncanonical_order):
        assert invalid_raw != raw
        with pytest.raises(
            v3.ForagerMatchedV3CandidateUniverseError, match="canonical encoding"
        ):
            parser(
                invalid_raw,
                expected_development_protocol_sha256=_PROTOCOL_SHA256,
                expected_development_result_sha256=_RESULT_SHA256,
            )


@pytest.mark.unit
@pytest.mark.parametrize("artifact", ["receipt", "panel"])
@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"value":"\xff"}', "UTF-8"),
        (b'{"value":"\\ud800"}', "invalid Unicode"),
        (b'{"value":NaN}', "non-finite JSON"),
        (b'{"value":1e999}', "non-finite JSON"),
        (b" " * (2 * 1024 * 1024 + 1), "too large"),
    ],
    ids=(
        "invalid_utf8",
        "invalid_unicode_scalar",
        "nan",
        "overflowing_float",
        "oversized",
    ),
)
def test_dynamic_artifact_parsers_reject_invalid_or_oversized_bytes(
    artifact: str, raw: bytes, message: str
) -> None:
    parser = (
        v3.parse_matched_v3_development_selection_receipt_artifact
        if artifact == "receipt"
        else v3.parse_matched_v3_confirmatory_panel_artifact
    )
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match=message):
        parser(
            raw,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
@pytest.mark.parametrize("artifact", ["receipt", "panel"])
def test_dynamic_artifact_parsers_require_the_exact_bytes_type(artifact: str) -> None:
    class BytesSubclass(bytes):
        pass

    parser = (
        v3.parse_matched_v3_development_selection_receipt_artifact
        if artifact == "receipt"
        else v3.parse_matched_v3_confirmatory_panel_artifact
    )
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="exact bytes"):
        parser(
            BytesSubclass(b"{}"),
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
def test_dynamic_canonicalizers_require_plain_unaliased_json_inputs() -> None:
    class DictSubclass(dict[str, Any]):
        pass

    class ListSubclass(list[str]):
        pass

    receipt = _accepted_receipt()
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="plain JSON"):
        v3.canonical_matched_v3_development_selection_receipt_bytes(
            DictSubclass(receipt),
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )

    aliased = _accepted_receipt()
    aliased["development_result"] = aliased["development_protocol"]
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="unaliased"):
        v3.canonical_matched_v3_development_selection_receipt_bytes(
            aliased,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )

    panel = v3.build_matched_v3_confirmatory_panel(
        receipt,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )
    panel["candidate_ids"] = ListSubclass(panel["candidate_ids"])
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="plain JSON"):
        v3.canonical_matched_v3_confirmatory_panel_bytes(
            panel,
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )


@pytest.mark.unit
def test_dynamic_artifact_snapshots_ignore_later_mutation_and_private_graph_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _accepted_receipt(causal="causal_e050_q075")
    raw = _canonical_bytes(receipt)
    original = json.loads(raw)
    first_candidate = v3._MATCHED_V3_DEVELOPMENT_UNIVERSE["candidates"][0]
    monkeypatch.setitem(first_candidate, "candidate_id", "private_graph_mutation")
    receipt["selections"]["causal_candidate_id"] = "search_nearest"

    parsed = v3.parse_matched_v3_development_selection_receipt_artifact(
        raw,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )
    assert parsed == original
    parsed["selections"]["causal_candidate_id"] = "mutated_return_value"
    assert v3.parse_matched_v3_development_selection_receipt_artifact(
        raw,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    ) == original


@pytest.mark.unit
def test_dynamic_artifact_parsers_keep_all_authority_denied() -> None:
    receipt = _accepted_receipt()
    receipt["scientific_promotion_allowed"] = True
    _rehash_receipt(receipt)
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="nonpromoting"):
        v3.parse_matched_v3_development_selection_receipt_artifact(
            _canonical_bytes(receipt),
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )

    accepted = _accepted_receipt()
    panel = v3.build_matched_v3_confirmatory_panel(
        accepted,
        expected_development_protocol_sha256=_PROTOCOL_SHA256,
        expected_development_result_sha256=_RESULT_SHA256,
    )
    panel["claim_boundaries"]["execution_authorized"] = True
    unsigned = {key: value for key, value in panel.items() if key != "payload_sha256"}
    panel["payload_sha256"] = _canonical_sha256(unsigned)
    with pytest.raises(v3.ForagerMatchedV3CandidateUniverseError, match="must be false"):
        v3.parse_matched_v3_confirmatory_panel_artifact(
            _canonical_bytes(panel),
            expected_development_protocol_sha256=_PROTOCOL_SHA256,
            expected_development_result_sha256=_RESULT_SHA256,
        )
