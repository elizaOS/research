"""Unexecuted, nonpromoting design contract for a future matched Forager v3 panel.

This module declares a broader *development* universe and the mechanical rule
that can turn an independently accepted development-selection receipt into a
confirmatory panel.  It deliberately contains no benchmark runner, score
reader, ranking implementation, or receipt issuer.  In particular, the panel
builder never infers a choice from scores: it accepts only four explicit,
group-valid choices in a content-addressed receipt bound to the exact universe,
development protocol, and development result.

The two new third-party adapters have implemented but unqualified exact-task
cores and full-horizon runner contracts.  They remain runtime-unqualified,
not execution-ready, and derived rather than exact executions of their pinned
upstream repositories.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_development_universe.v1"
)
FORAGER_MATCHED_V3_SELECTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_development_selection_receipt.v1"
)
FORAGER_MATCHED_V3_DEVELOPMENT_PROTOCOL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_development_protocol.v1"
)
FORAGER_MATCHED_V3_DEVELOPMENT_RESULT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_development_result.v1"
)
FORAGER_MATCHED_V3_CONFIRMATORY_PANEL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_confirmatory_panel.v1"
)

_MAX_JSON_BYTES: Final = 2 * 1024 * 1024
_MAX_DYNAMIC_JSON_DEPTH: Final = 64
_MAX_DYNAMIC_JSON_NODES: Final = 200_000
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}\Z")

AnalysisRole = Literal["inferential", "descriptive_only"]
DevelopmentSelectionGroup = Literal[
    "causal",
    "horde",
    "local_rtu",
    "dqn_plasticity",
    "fixed_external",
    "descriptive",
]
AdapterStatus = Literal[
    "existing_but_unqualified_for_v3",
    "existing_reference_unqualified_for_v3",
    "unimplemented",
]
# This is a universe-availability class, distinct from an adapter artifact's
# canonical implementation-status literal.


class ForagerMatchedV3CandidateUniverseError(ValueError):
    """The v3 design descriptor, selection receipt, or panel is invalid."""


@dataclass(frozen=True)
class RelevantFilePin:
    """Exact SHA-256 binding for one reviewed file in a source repository."""

    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class SourceRepositoryPin:
    """Immutable source identity for one reviewed upstream repository."""

    repository_id: str
    canonical_url: str
    commit_git_sha1: str
    tree_git_sha1: str
    archive_sha256: str | None
    archive_size_bytes: int | None
    relevant_files: tuple[RelevantFilePin, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "canonical_url": self.canonical_url,
            "commit_git_sha1": self.commit_git_sha1,
            "tree_git_sha1": self.tree_git_sha1,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "relevant_files": [item.to_dict() for item in self.relevant_files],
        }


@dataclass(frozen=True)
class DevelopmentCandidate:
    """One immutable candidate definition in the future development universe."""

    candidate_id: str
    analysis_role: AnalysisRole
    development_selection_group: DevelopmentSelectionGroup
    confirmatory_disposition: str
    source_repository_id: str
    source_relationship: str
    adapter_status: AdapterStatus
    observation_access: str
    pairing_eligible: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "analysis_role": self.analysis_role,
            "development_selection_group": self.development_selection_group,
            "confirmatory_disposition": self.confirmatory_disposition,
            "source_repository_id": self.source_repository_id,
            "source_relationship": self.source_relationship,
            "adapter_status": self.adapter_status,
            "observation_access": self.observation_access,
            "pairing_eligible": self.pairing_eligible,
            "execution_ready": False,
            "scientific_promotion_allowed": False,
            "universal_sota_claim_allowed": False,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ValidatedDevelopmentSelection:
    """Detached scalar choices extracted from one strictly validated receipt."""

    causal_candidate_id: str
    horde_candidate_id: str
    local_rtu_candidate_id: str
    dqn_plasticity_candidate_id: str

    def candidate_ids(self) -> tuple[str, str, str, str]:
        return (
            self.causal_candidate_id,
            self.horde_candidate_id,
            self.local_rtu_candidate_id,
            self.dqn_plasticity_candidate_id,
        )


def _causal_ids() -> tuple[str, ...]:
    return tuple(
        f"causal_e{exploration}_q{quantile}"
        for exploration in ("025", "050", "100")
        for quantile in ("050", "075", "090")
    )


MATCHED_V3_CAUSAL_SELECTION_CANDIDATE_IDS: Final = _causal_ids()
MATCHED_V3_HORDE_SELECTION_CANDIDATE_IDS: Final = (
    "alberta_horde_default",
    "alberta_horde_eps05",
    "alberta_horde_recurrent64",
    "alberta_horde_step3e3",
)
MATCHED_V3_LOCAL_RTU_CANDIDATE_ID: Final = "alberta_rtu_h08_taylor"
MATCHED_V3_DQN_PLASTICITY_SELECTION_CANDIDATE_IDS: Final = (
    "external_dqn_crelu",
    "external_dqn_redo",
    "external_dqn_reward_trace",
    "external_dqn_l2_init",
)
MATCHED_V3_FIXED_EXTERNAL_INFERENTIAL_CANDIDATE_IDS: Final = (
    "external_dqn_plain",
    "external_pt_dqn_xfinal",
    "external_drqn_xfinal",
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
    "adapted_full_rainbow",
    "adapted_ppo_gru",
)
MATCHED_V3_EXTERNAL_INFERENTIAL_CANDIDATE_IDS: Final = (
    "external_dqn_plain",
    "external_dqn_crelu",
    "external_dqn_redo",
    "external_dqn_reward_trace",
    "external_dqn_l2_init",
    "external_pt_dqn_xfinal",
    "external_drqn_xfinal",
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
    "adapted_full_rainbow",
    "adapted_ppo_gru",
)
MATCHED_V3_DESCRIPTIVE_CANDIDATE_IDS: Final = (
    "random_policy",
    "search_nearest",
    "search_oracle",
)
MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS: Final = (
    MATCHED_V3_CAUSAL_SELECTION_CANDIDATE_IDS
    + MATCHED_V3_HORDE_SELECTION_CANDIDATE_IDS
    + (MATCHED_V3_LOCAL_RTU_CANDIDATE_ID,)
    + MATCHED_V3_EXTERNAL_INFERENTIAL_CANDIDATE_IDS
)
MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS: Final = (
    MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
    + MATCHED_V3_DESCRIPTIVE_CANDIDATE_IDS
)

_SOURCE_PINS: Final = (
    SourceRepositoryPin(
        repository_id="foragax_agents",
        canonical_url="https://github.com/steventango/continual-foragax-agents",
        commit_git_sha1="9710f60fa30da5badc451ad7ce3ff296d5070830",
        tree_git_sha1="a5ad878ac4be0567c43dfd9177471c4b5a910bfa",
        archive_sha256=(
            "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
        ),
        archive_size_bytes=314_961_920,
        relevant_files=(),
    ),
    SourceRepositoryPin(
        repository_id="dopamine",
        canonical_url="https://github.com/google/dopamine",
        commit_git_sha1="5873f5494ee0c2d7c016d0ab2ad530354fec59d0",
        tree_git_sha1="578408662e298d00e4e855f13f67dc08bd784e7c",
        archive_sha256=(
            "bea46f755c86725d7ca90c531a08aad86cab62201ac2b9224c82f66dfada7456"
        ),
        archive_size_bytes=82_933_760,
        relevant_files=(
            RelevantFilePin(
                "LICENSE",
                "e47b2783cb7131207707c35d0aea22277aa1beded6bf9d7c2436cd7de9462323",
            ),
            RelevantFilePin(
                "dopamine/jax/agents/full_rainbow/configs/full_rainbow.gin",
                "f926614f7c99ec248f3bafdbb920a7d8497476c0a27d5aad9ca8c69ca9ebc130",
            ),
            RelevantFilePin(
                "dopamine/jax/agents/full_rainbow/full_rainbow_agent.py",
                "cc85222d9b60b6f05cbb8e6af170a57a3f74c20c9dd72067b70d8daf4cf50595",
            ),
            RelevantFilePin(
                "dopamine/jax/losses.py",
                "42c10699bebf5b41b7bcd5cbeb18693c0f606f3bc427b988426368741e3cbd39",
            ),
            RelevantFilePin(
                "dopamine/jax/agents/dqn/dqn_agent.py",
                "53a37912775c1fcce84f3c158c29fb9d63094ba8dc9f8a0c9c627e0f8c519dca",
            ),
            RelevantFilePin(
                "dopamine/jax/networks.py",
                "fac813138454e2c947aca78a284b0e79b8f021beaf27b5f99981177ec8ca3bb9",
            ),
            RelevantFilePin(
                "dopamine/jax/agents/rainbow/rainbow_agent.py",
                "02c90de41f68c18e66938bc9c5664a5e6154b8c67571114c8955d04a9e67cef8",
            ),
            RelevantFilePin(
                "dopamine/jax/replay_memory/accumulator.py",
                "cfe4c849b2121f259fce5cd23e0a349f6ffba45f3c5c167dd63f36da2fc9cd25",
            ),
            RelevantFilePin(
                "dopamine/jax/replay_memory/samplers.py",
                "de33adddd80fa4194e5eda14182f1eee50c65492c575e16e5c45630b9c75bb0b",
            ),
        ),
    ),
    SourceRepositoryPin(
        repository_id="pobax",
        canonical_url="https://github.com/taodav/pobax",
        commit_git_sha1="a5e1d62d14e4efe783885b9d4f19cffa2a568eec",
        tree_git_sha1="d67cf5c209f2e7de9ce517d4bc72a2741ccaf6a6",
        archive_sha256=(
            "f354028549d79a1b3f1ee67deaa46454a0be60d9346764e5aed9e8ab93768ad9"
        ),
        archive_size_bytes=1_699_840,
        relevant_files=(
            RelevantFilePin(
                "LICENSE",
                "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
            ),
            RelevantFilePin(
                "pobax/algos/ppo.py",
                "0c82725027e6022d48847bca45a87e6f8d9b54d720bbb844f053d4b8448ce153",
            ),
            RelevantFilePin(
                "pobax/models/network.py",
                "b3ea151f6a7f9000dd1b529cbcc262c150b767c66664399008aa89283a2e520a",
            ),
            RelevantFilePin(
                "pobax/config.py",
                "38bb46c93734c8882ab7ad7bdfbee9d64bb21db04231ccd15b9ec2a6eb02034c",
            ),
            RelevantFilePin(
                "pobax/models/actor_critic.py",
                "bb707481b32eefc1219adbc38abd527c3c600cf8941ae963bf6b6540c9b2158f",
            ),
            RelevantFilePin(
                "pobax/models/discrete.py",
                "ad7ac11a03b49f7ea53fcf11b0b97cc7697f57447f4661a22fb235a6ab90885c",
            ),
            RelevantFilePin(
                "pobax/models/__init__.py",
                "c4434b0b1eba13c227cdf479380f5347aa57aba4d2f78a12112c056cdada323a",
            ),
            RelevantFilePin(
                "pobax/models/value.py",
                "e875e7ef951aba37ea4648328442aaece0fc3415de580c6b5115843eb32366bd",
            ),
            RelevantFilePin(
                "pyproject.toml",
                "4f02e96a5d8471f9637ec36dc9536398183f49fb28fa07c5b7f371ffcdbe81d5",
            ),
            RelevantFilePin(
                "requirements.txt",
                "8d8a36a4428d481b15c47b9ed1aec573c3dc2472af746be611e9a17dae40a17c",
            ),
        ),
    ),
)


def _candidate(
    candidate_id: str,
    group: DevelopmentSelectionGroup,
    *,
    repository_id: str,
    relationship: str,
    status: AdapterStatus = "existing_but_unqualified_for_v3",
    role: AnalysisRole = "inferential",
    observation_access: str = "matched_partial_color_aperture_9",
    rationale: str,
) -> DevelopmentCandidate:
    if group in {"causal", "horde", "local_rtu", "dqn_plasticity"}:
        disposition = "selected_only_by_accepted_development_receipt"
    elif group == "fixed_external":
        disposition = "fixed_after_any_accepted_development_receipt"
    else:
        disposition = "fixed_descriptive_after_any_accepted_development_receipt"
    return DevelopmentCandidate(
        candidate_id=candidate_id,
        analysis_role=role,
        development_selection_group=group,
        confirmatory_disposition=disposition,
        source_repository_id=repository_id,
        source_relationship=relationship,
        adapter_status=status,
        observation_access=observation_access,
        pairing_eligible=role == "inferential",
        rationale=rationale,
    )


def _development_candidates() -> tuple[DevelopmentCandidate, ...]:
    candidates: list[DevelopmentCandidate] = []
    for candidate_id in MATCHED_V3_CAUSAL_SELECTION_CANDIDATE_IDS:
        candidates.append(
            _candidate(
                candidate_id,
                "causal",
                repository_id="local_alberta",
                relationship="existing_local_candidate_requiring_new_v3_development",
                rationale="One member of the nine-arm causal-map development selection group.",
            )
        )
    for candidate_id in MATCHED_V3_HORDE_SELECTION_CANDIDATE_IDS:
        candidates.append(
            _candidate(
                candidate_id,
                "horde",
                repository_id="local_alberta",
                relationship="existing_local_candidate_requiring_new_v3_development",
                rationale="One member of the four-arm Horde development selection group.",
            )
        )
    candidates.append(
        _candidate(
            MATCHED_V3_LOCAL_RTU_CANDIDATE_ID,
            "local_rtu",
            repository_id="local_alberta",
            relationship="existing_local_candidate_requiring_new_v3_development",
            rationale="The sole local RTU candidate; the accepted receipt must retain it.",
        )
    )

    for candidate_id in MATCHED_V3_EXTERNAL_INFERENTIAL_CANDIDATE_IDS:
        group: DevelopmentSelectionGroup = (
            "dqn_plasticity"
            if candidate_id in MATCHED_V3_DQN_PLASTICITY_SELECTION_CANDIDATE_IDS
            else "fixed_external"
        )
        repository_id = "foragax_agents"
        relationship = "derived_exact_task_configuration_not_exact_upstream_execution"
        status: AdapterStatus = "existing_but_unqualified_for_v3"
        rationale = "Transferred comparator requiring fresh v3 development and qualification."
        if candidate_id == "adapted_full_rainbow":
            repository_id = "dopamine"
            relationship = "derived_adapter_not_exact_upstream_execution"
            status = "existing_but_unqualified_for_v3"
            rationale = (
                "The Full Rainbow exact-task core and full-horizon runner are implemented "
                "but no Foragax runtime or execution qualification exists."
            )
        elif candidate_id == "adapted_ppo_gru":
            repository_id = "pobax"
            relationship = "derived_adapter_not_exact_upstream_execution"
            status = "existing_but_unqualified_for_v3"
            rationale = (
                "The recurrent PPO-GRU exact-task core and full-horizon runner are "
                "implemented but no Foragax runtime or execution qualification exists."
            )
        candidates.append(
            _candidate(
                candidate_id,
                group,
                repository_id=repository_id,
                relationship=relationship,
                status=status,
                rationale=rationale,
            )
        )

    for candidate_id in MATCHED_V3_DESCRIPTIVE_CANDIDATE_IDS:
        observation_access = (
            "observation_independent_random_action_reference"
            if candidate_id == "random_policy"
            else "privileged_global_object_access"
        )
        candidates.append(
            _candidate(
                candidate_id,
                "descriptive",
                repository_id="foragax_agents",
                relationship="derived_exact_task_reference_not_inferential_comparator",
                status="existing_reference_unqualified_for_v3",
                role="descriptive_only",
                observation_access=observation_access,
                rationale=(
                    "Descriptive context only; excluded from selection and every inferential "
                    "claim."
                ),
            )
        )
    return tuple(candidates)


_DEVELOPMENT_CANDIDATES: Final = _development_candidates()


def _development_universe_descriptor() -> dict[str, Any]:
    return {
        "schema_version": FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION,
        "status": "unexecuted_design",
        "classification": "development_only_nonpromoting_candidate_universe",
        "scope": {
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
        },
        "source_pins": [pin.to_dict() for pin in _SOURCE_PINS],
        "candidates": [candidate.to_dict() for candidate in _DEVELOPMENT_CANDIDATES],
        "selection_groups": [
            {
                "group_id": "causal",
                "candidate_ids": list(MATCHED_V3_CAUSAL_SELECTION_CANDIDATE_IDS),
                "confirmatory_selection_count": 1,
            },
            {
                "group_id": "horde",
                "candidate_ids": list(MATCHED_V3_HORDE_SELECTION_CANDIDATE_IDS),
                "confirmatory_selection_count": 1,
            },
            {
                "group_id": "local_rtu",
                "candidate_ids": [MATCHED_V3_LOCAL_RTU_CANDIDATE_ID],
                "confirmatory_selection_count": 1,
            },
            {
                "group_id": "dqn_plasticity",
                "candidate_ids": list(
                    MATCHED_V3_DQN_PLASTICITY_SELECTION_CANDIDATE_IDS
                ),
                "confirmatory_selection_count": 1,
            },
        ],
        "confirmatory_panel_rule": {
            "requires_strict_accepted_development_selection_receipt": True,
            "receipt_must_bind_exact_universe_protocol_and_result_digests": True,
            "builder_reads_or_infers_scores": False,
            "selected_group_ids": ["causal", "horde", "local_rtu", "dqn_plasticity"],
            "fixed_external_inferential_candidate_ids": list(
                MATCHED_V3_FIXED_EXTERNAL_INFERENTIAL_CANDIDATE_IDS
            ),
            "fixed_descriptive_candidate_ids": list(MATCHED_V3_DESCRIPTIVE_CANDIDATE_IDS),
        },
        "claim_boundaries": {
            "descriptor_supports_performance_claim": False,
            "development_results_support_confirmatory_performance_claim": False,
            "research_literature_exhaustive": False,
            "universal_sota_claim_allowed": False,
            "scientific_promotion_allowed": False,
            "execution_authorized": False,
            "confirmatory_panel_is_executed": False,
            "builder_infers_selection_from_scores": False,
        },
        "limitations": [
            "This is an unexecuted development-universe design, not an evaluation result.",
            "No v3 development protocol, result, or accepted selection receipt is supplied here.",
            "Every candidate requires fresh v3 qualification before execution.",
            (
                "The Rainbow and PPO-GRU exact-task cores and full-horizon runners are "
                "implemented but runtime and execution qualification are missing."
            ),
            "The audited source corpus is not an exhaustive research-literature search.",
            "A constructed panel cannot authorize execution, evidence promotion, or a SOTA claim.",
        ],
    }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except UnicodeError as exc:
        raise ForagerMatchedV3CandidateUniverseError(
            "value is not valid UTF-8 canonical JSON"
        ) from exc
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3CandidateUniverseError(
            "value is not finite canonical JSON"
        ) from exc


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_plain_unaliased_json_tree(value: Any, context: str) -> None:
    """Require exact built-in JSON types without cycles or shared containers."""

    pending: list[tuple[Any, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    node_count = 0
    while pending:
        item, depth = pending.pop()
        node_count += 1
        if node_count > _MAX_DYNAMIC_JSON_NODES:
            raise ForagerMatchedV3CandidateUniverseError(
                f"{context} contains too many JSON nodes"
            )
        if depth > _MAX_DYNAMIC_JSON_DEPTH:
            raise ForagerMatchedV3CandidateUniverseError(
                f"{context} exceeds the JSON nesting limit"
            )

        if type(item) is dict:
            identity = id(item)
            if identity in seen_containers:
                raise ForagerMatchedV3CandidateUniverseError(
                    f"{context} must be an unaliased acyclic JSON tree"
                )
            seen_containers.add(identity)
            for key, child in item.items():
                if type(key) is not str:
                    raise ForagerMatchedV3CandidateUniverseError(
                        f"{context} must be a plain JSON tree with string object keys"
                    )
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ForagerMatchedV3CandidateUniverseError(
                        f"{context} contains invalid Unicode"
                    ) from exc
                pending.append((child, depth + 1))
        elif type(item) is list:
            identity = id(item)
            if identity in seen_containers:
                raise ForagerMatchedV3CandidateUniverseError(
                    f"{context} must be an unaliased acyclic JSON tree"
                )
            seen_containers.add(identity)
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedV3CandidateUniverseError(
                    f"{context} contains invalid Unicode"
                ) from exc
        elif type(item) is float:
            if not math.isfinite(item):
                raise ForagerMatchedV3CandidateUniverseError(
                    f"{context} contains a non-finite JSON number"
                )
        elif item is None or type(item) in (bool, int):
            continue
        else:
            raise ForagerMatchedV3CandidateUniverseError(
                f"{context} must be a plain JSON tree; got {type(item).__name__}"
            )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ForagerMatchedV3CandidateUniverseError(
                f"duplicate JSON object key {key!r} is forbidden"
            )
        value[key] = item
    return value


def _reject_nonfinite_json_constant(token: str) -> Any:
    raise ForagerMatchedV3CandidateUniverseError(
        f"non-finite JSON number {token!r} is forbidden"
    )


def _parse_finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ForagerMatchedV3CandidateUniverseError(
            f"non-finite JSON number {token!r} is forbidden"
        )
    return value


def _parse_canonical_dynamic_json_object(raw: bytes, context: str) -> dict[str, Any]:
    """Decode one bounded, duplicate-free, exact-canonical JSON object."""

    if type(raw) is not bytes:
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} artifact must be exact bytes"
        )
    if len(raw) > _MAX_JSON_BYTES:
        raise ForagerMatchedV3CandidateUniverseError(f"{context} artifact is too large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} artifact is not strict UTF-8 JSON"
        ) from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except ForagerMatchedV3CandidateUniverseError:
        raise
    except (json.JSONDecodeError, OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} artifact is not strict UTF-8 JSON"
        ) from exc
    _validate_plain_unaliased_json_tree(decoded, f"decoded {context} artifact")
    if type(decoded) is not dict:
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} artifact must encode a plain JSON object"
        )
    snapshot = cast(dict[str, Any], decoded)
    if raw != _canonical_bytes(snapshot):
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} artifact is not the exact canonical encoding"
        )
    return snapshot


def _plain_json_object_snapshot(
    value: Mapping[str, Any], context: str
) -> dict[str, Any]:
    """Snapshot caller-owned mappings without invoking custom deepcopy hooks."""
    raw = _canonical_bytes(value)
    if len(raw) > _MAX_JSON_BYTES:
        raise ForagerMatchedV3CandidateUniverseError(f"{context} is too large")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} could not be snapshotted as canonical JSON"
        ) from exc
    if type(parsed) is not dict:
        raise ForagerMatchedV3CandidateUniverseError(f"{context} must be a JSON object")
    return cast(dict[str, Any], parsed)


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} must have exact keys; missing={missing!r}, extra={extra!r}"
        )


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForagerMatchedV3CandidateUniverseError(f"{context} must be an object")
    return cast(Mapping[str, Any], value)


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} must be a lowercase hexadecimal SHA-256"
        )
    return value


def _validate_internal_universe(descriptor: Mapping[str, Any]) -> None:
    expected_top_keys = frozenset(
        {
            "schema_version",
            "status",
            "classification",
            "scope",
            "source_pins",
            "candidates",
            "selection_groups",
            "confirmatory_panel_rule",
            "claim_boundaries",
            "limitations",
        }
    )
    _require_exact_keys(descriptor, expected_top_keys, "development universe")
    if descriptor["schema_version"] != FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION:
        raise AssertionError("development-universe schema drift")

    pins = cast(list[dict[str, Any]], descriptor["source_pins"])
    if [item["repository_id"] for item in pins] != [
        "foragax_agents",
        "dopamine",
        "pobax",
    ]:
        raise AssertionError("source-pin order or membership drift")
    for pin in pins:
        _require_exact_keys(
            pin,
            frozenset(
                {
                    "repository_id",
                    "canonical_url",
                    "commit_git_sha1",
                    "tree_git_sha1",
                    "archive_sha256",
                    "archive_size_bytes",
                    "relevant_files",
                }
            ),
            "source pin",
        )
        if _GIT_SHA1_RE.fullmatch(cast(str, pin["commit_git_sha1"])) is None:
            raise AssertionError("invalid pinned commit identity")
        if _GIT_SHA1_RE.fullmatch(cast(str, pin["tree_git_sha1"])) is None:
            raise AssertionError("invalid pinned tree identity")
        archive_sha256 = pin["archive_sha256"]
        archive_size = pin["archive_size_bytes"]
        if (archive_sha256 is None) != (archive_size is None):
            raise AssertionError("archive identity and size must be present together")
        if archive_sha256 is not None:
            _require_sha256(archive_sha256, "source archive digest")
            if type(archive_size) is not int or archive_size <= 0:
                raise AssertionError("archive size must be a positive exact integer")
        for artifact in cast(list[dict[str, Any]], pin["relevant_files"]):
            _require_exact_keys(artifact, frozenset({"path", "sha256"}), "file pin")
            _require_sha256(artifact["sha256"], "relevant-file digest")

    candidates = cast(list[dict[str, Any]], descriptor["candidates"])
    ids = tuple(cast(str, item["candidate_id"]) for item in candidates)
    if ids != MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS or len(ids) != len(set(ids)):
        raise AssertionError("development-candidate order or membership drift")
    candidate_keys = frozenset(
        {
            "candidate_id",
            "analysis_role",
            "development_selection_group",
            "confirmatory_disposition",
            "source_repository_id",
            "source_relationship",
            "adapter_status",
            "observation_access",
            "pairing_eligible",
            "execution_ready",
            "scientific_promotion_allowed",
            "universal_sota_claim_allowed",
            "rationale",
        }
    )
    for candidate in candidates:
        _require_exact_keys(candidate, candidate_keys, "candidate")
        role = candidate["analysis_role"]
        if candidate["pairing_eligible"] is not (role == "inferential"):
            raise AssertionError("candidate role/pairing mismatch")
        if any(
            candidate[key] is not False
            for key in (
                "execution_ready",
                "scientific_promotion_allowed",
                "universal_sota_claim_allowed",
            )
        ):
            raise AssertionError("every v3 design candidate must remain non-ready/nonpromoting")
    by_id = {cast(str, item["candidate_id"]): item for item in candidates}
    for candidate_id, repository_id in (
        ("adapted_full_rainbow", "dopamine"),
        ("adapted_ppo_gru", "pobax"),
    ):
        item = by_id[candidate_id]
        if (
            item["source_repository_id"] != repository_id
            or item["source_relationship"]
            != "derived_adapter_not_exact_upstream_execution"
            or item["adapter_status"] != "existing_but_unqualified_for_v3"
        ):
            raise AssertionError("new adapter provenance/readiness drift")

    claims = cast(Mapping[str, Any], descriptor["claim_boundaries"])
    if any(value is not False for value in claims.values()):
        raise AssertionError("all universe claim/authorization flags must be false")


_MATCHED_V3_DEVELOPMENT_UNIVERSE: Final = _development_universe_descriptor()
_validate_internal_universe(_MATCHED_V3_DEVELOPMENT_UNIVERSE)
_MATCHED_V3_DEVELOPMENT_UNIVERSE_BYTES: Final = _canonical_bytes(
    _MATCHED_V3_DEVELOPMENT_UNIVERSE
)
MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256: Final = (
    "a441b35eed4ec6327bf03463099a46e9c2596f2a169182fd317fe51c98b4c750"
)
if not hmac.compare_digest(
    hashlib.sha256(_MATCHED_V3_DEVELOPMENT_UNIVERSE_BYTES).hexdigest(),
    MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256,
):
    raise AssertionError("canonical matched-v3 development universe drifted")


def _frozen_development_universe_snapshot() -> dict[str, Any]:
    """Decode a fresh snapshot from the frozen canonical universe bytes."""

    try:
        value = json.loads(_MATCHED_V3_DEVELOPMENT_UNIVERSE_BYTES.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise ForagerMatchedV3CandidateUniverseError(
            "frozen development-universe bytes could not be decoded"
        ) from exc
    if type(value) is not dict:  # pragma: no cover
        raise ForagerMatchedV3CandidateUniverseError(
            "frozen development-universe bytes must encode a plain object"
        )
    return cast(dict[str, Any], value)


def matched_v3_development_candidates() -> tuple[DevelopmentCandidate, ...]:
    """Return the immutable candidate definitions in exact development order."""
    return _DEVELOPMENT_CANDIDATES


def matched_v3_development_universe_descriptor() -> dict[str, Any]:
    """Return a detached copy of the canonical, unexecuted universe descriptor."""
    return _frozen_development_universe_snapshot()


def canonical_matched_v3_development_universe_bytes() -> bytes:
    """Return the canonical content-addressed development-universe bytes."""
    return _MATCHED_V3_DEVELOPMENT_UNIVERSE_BYTES


def parse_matched_v3_development_universe_artifact(raw: bytes) -> dict[str, Any]:
    """Accept only the exact canonical v3 development-universe artifact."""
    if not isinstance(raw, bytes):
        raise ForagerMatchedV3CandidateUniverseError(
            "development-universe artifact must be bytes"
        )
    if len(raw) > _MAX_JSON_BYTES:
        raise ForagerMatchedV3CandidateUniverseError(
            "development-universe artifact is too large"
        )
    if not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256
    ):
        raise ForagerMatchedV3CandidateUniverseError(
            "development-universe artifact does not match the frozen digest"
        )
    canonical = canonical_matched_v3_development_universe_bytes()
    if raw != canonical:
        raise ForagerMatchedV3CandidateUniverseError(
            "development-universe artifact is not the exact canonical encoding"
        )
    return _frozen_development_universe_snapshot()


_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "classification",
        "development_universe",
        "development_protocol",
        "development_result",
        "selections",
        "selection_rule",
        "scientific_promotion_allowed",
        "universal_sota_claim_allowed",
        "payload_sha256",
    }
)
_BINDING_KEYS: Final = frozenset({"schema_version", "sha256"})
_SELECTION_KEYS: Final = frozenset(
    {
        "causal_candidate_id",
        "horde_candidate_id",
        "local_rtu_candidate_id",
        "dqn_plasticity_candidate_id",
    }
)


def _validate_receipt_payload_digest(receipt: Mapping[str, Any]) -> None:
    supplied = _require_sha256(receipt["payload_sha256"], "receipt payload digest")
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    calculated = _canonical_sha256(unsigned)
    if not hmac.compare_digest(supplied, calculated):
        raise ForagerMatchedV3CandidateUniverseError(
            "receipt payload digest does not match its canonical payload"
        )


def _validate_binding(
    value: Any,
    *,
    context: str,
    expected_schema: str,
    expected_sha256: str,
) -> None:
    binding = _require_mapping(value, context)
    _require_exact_keys(binding, _BINDING_KEYS, context)
    if binding["schema_version"] != expected_schema:
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} schema does not match the exact required schema"
        )
    supplied = _require_sha256(binding["sha256"], f"{context} digest")
    if not hmac.compare_digest(supplied, expected_sha256):
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} digest does not match the exact required digest"
        )


def _validated_selection(receipt: Mapping[str, Any]) -> ValidatedDevelopmentSelection:
    selections = _require_mapping(receipt["selections"], "receipt selections")
    _require_exact_keys(selections, _SELECTION_KEYS, "receipt selections")
    values: dict[str, str] = {}
    for key in _SELECTION_KEYS:
        value = selections[key]
        if not isinstance(value, str):
            raise ForagerMatchedV3CandidateUniverseError(
                f"receipt selection {key} must be a candidate ID string"
            )
        values[key] = value
    ordered_values = (
        values["causal_candidate_id"],
        values["horde_candidate_id"],
        values["local_rtu_candidate_id"],
        values["dqn_plasticity_candidate_id"],
    )
    if len(set(ordered_values)) != len(ordered_values):
        raise ForagerMatchedV3CandidateUniverseError(
            "receipt selections contain a duplicate candidate ID"
        )
    memberships = (
        ("causal", ordered_values[0], MATCHED_V3_CAUSAL_SELECTION_CANDIDATE_IDS),
        ("horde", ordered_values[1], MATCHED_V3_HORDE_SELECTION_CANDIDATE_IDS),
        ("local_rtu", ordered_values[2], (MATCHED_V3_LOCAL_RTU_CANDIDATE_ID,)),
        (
            "dqn_plasticity",
            ordered_values[3],
            MATCHED_V3_DQN_PLASTICITY_SELECTION_CANDIDATE_IDS,
        ),
    )
    for group_id, candidate_id, allowed in memberships:
        if candidate_id not in allowed:
            raise ForagerMatchedV3CandidateUniverseError(
                f"{candidate_id!r} is not in the exact {group_id!r} selection group"
            )
    return ValidatedDevelopmentSelection(*ordered_values)


def _validate_and_extract_receipt(
    receipt_value: Mapping[str, Any],
    *,
    expected_development_protocol_sha256: str,
    expected_development_result_sha256: str,
) -> ValidatedDevelopmentSelection:
    expected_protocol = _require_sha256(
        expected_development_protocol_sha256,
        "expected development protocol SHA-256",
    )
    expected_result = _require_sha256(
        expected_development_result_sha256,
        "expected development result SHA-256",
    )
    _require_exact_keys(receipt_value, _RECEIPT_KEYS, "development-selection receipt")
    if receipt_value["schema_version"] != FORAGER_MATCHED_V3_SELECTION_RECEIPT_SCHEMA_VERSION:
        raise ForagerMatchedV3CandidateUniverseError(
            "development-selection receipt schema is not the exact v3 schema"
        )
    if receipt_value["status"] != "accepted":
        raise ForagerMatchedV3CandidateUniverseError(
            "development-selection receipt status must be exactly accepted"
        )
    if receipt_value["classification"] != "development_selection_nonpromoting":
        raise ForagerMatchedV3CandidateUniverseError(
            "development-selection receipt classification is invalid"
        )
    if receipt_value["selection_rule"] != (
        "exact_group_winners_from_bound_development_result"
    ):
        raise ForagerMatchedV3CandidateUniverseError(
            "development-selection receipt selection rule is invalid"
        )
    if receipt_value["scientific_promotion_allowed"] is not False:
        raise ForagerMatchedV3CandidateUniverseError(
            "development-selection receipt must remain nonpromoting"
        )
    if receipt_value["universal_sota_claim_allowed"] is not False:
        raise ForagerMatchedV3CandidateUniverseError(
            "development-selection receipt cannot authorize a universal SOTA claim"
        )
    _validate_binding(
        receipt_value["development_universe"],
        context="development-universe binding",
        expected_schema=FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION,
        expected_sha256=MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256,
    )
    _validate_binding(
        receipt_value["development_protocol"],
        context="development-protocol binding",
        expected_schema=FORAGER_MATCHED_V3_DEVELOPMENT_PROTOCOL_SCHEMA_VERSION,
        expected_sha256=expected_protocol,
    )
    _validate_binding(
        receipt_value["development_result"],
        context="development-result binding",
        expected_schema=FORAGER_MATCHED_V3_DEVELOPMENT_RESULT_SCHEMA_VERSION,
        expected_sha256=expected_result,
    )
    selection = _validated_selection(receipt_value)
    _validate_receipt_payload_digest(receipt_value)
    return selection


def validate_matched_v3_development_selection_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_development_protocol_sha256: str,
    expected_development_result_sha256: str,
) -> dict[str, Any]:
    """Validate and detach one externally issued accepted selection receipt.

    This validator does not issue a receipt and does not read a development
    result.  The expected result digest must be supplied by the caller's
    separately authenticated artifact boundary.
    """
    if not isinstance(receipt, Mapping):
        raise ForagerMatchedV3CandidateUniverseError(
            "development-selection receipt must be an object"
        )
    snapshot = _plain_json_object_snapshot(receipt, "development-selection receipt")
    _validate_and_extract_receipt(
        snapshot,
        expected_development_protocol_sha256=expected_development_protocol_sha256,
        expected_development_result_sha256=expected_development_result_sha256,
    )
    return snapshot


def parse_matched_v3_development_selection_receipt_artifact(
    raw: bytes,
    *,
    expected_development_protocol_sha256: str,
    expected_development_result_sha256: str,
) -> dict[str, Any]:
    """Parse exact canonical receipt bytes, then validate every binding and denial."""

    snapshot = _parse_canonical_dynamic_json_object(
        raw, "development-selection receipt"
    )
    return validate_matched_v3_development_selection_receipt(
        snapshot,
        expected_development_protocol_sha256=expected_development_protocol_sha256,
        expected_development_result_sha256=expected_development_result_sha256,
    )


def canonical_matched_v3_development_selection_receipt_bytes(
    receipt: Mapping[str, Any],
    *,
    expected_development_protocol_sha256: str,
    expected_development_result_sha256: str,
) -> bytes:
    """Validate a plain unaliased receipt tree and return its canonical bytes."""

    _validate_plain_unaliased_json_tree(receipt, "development-selection receipt")
    if type(receipt) is not dict:
        raise ForagerMatchedV3CandidateUniverseError(
            "development-selection receipt must be a plain JSON object"
        )
    raw = _canonical_bytes(receipt)
    parse_matched_v3_development_selection_receipt_artifact(
        raw,
        expected_development_protocol_sha256=expected_development_protocol_sha256,
        expected_development_result_sha256=expected_development_result_sha256,
    )
    return raw


_PANEL_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "classification",
        "selection_receipt",
        "selection_receipt_sha256",
        "inferential_candidate_ids",
        "descriptive_candidate_ids",
        "candidate_ids",
        "counts",
        "claim_boundaries",
        "payload_sha256",
    }
)
_PANEL_COUNT_KEYS: Final = frozenset(
    {"inferential_candidate_count", "descriptive_candidate_count", "candidate_count"}
)
_PANEL_CLAIM_KEYS: Final = frozenset(
    {
        "builder_inferred_selection_from_scores",
        "research_literature_exhaustive",
        "universal_sota_claim_allowed",
        "scientific_promotion_allowed",
        "execution_authorized",
        "panel_is_executed",
        "panel_supports_performance_claim",
    }
)


def _panel_inferential_ids(selection: ValidatedDevelopmentSelection) -> tuple[str, ...]:
    selected = set(selection.candidate_ids()) | set(
        MATCHED_V3_FIXED_EXTERNAL_INFERENTIAL_CANDIDATE_IDS
    )
    result = tuple(
        candidate_id
        for candidate_id in MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
        if candidate_id in selected
    )
    if len(result) != 11 or len(set(result)) != 11:
        raise AssertionError("confirmatory inferential panel rule did not yield 11 unique arms")
    return result


def build_matched_v3_confirmatory_panel(
    selection_receipt: Mapping[str, Any],
    *,
    expected_development_protocol_sha256: str,
    expected_development_result_sha256: str,
) -> dict[str, Any]:
    """Build the exact 11-inferential plus 3-descriptive future panel.

    Choices are copied only from a strictly accepted, content-addressed
    receipt.  No score field is accepted and no ranking or tie break occurs in
    this function.  The returned descriptor remains unexecuted and
    nonpromoting.
    """
    if not isinstance(selection_receipt, Mapping):
        raise ForagerMatchedV3CandidateUniverseError(
            "development-selection receipt must be an object"
        )
    detached_receipt = _plain_json_object_snapshot(
        selection_receipt, "development-selection receipt"
    )
    selection = _validate_and_extract_receipt(
        detached_receipt,
        expected_development_protocol_sha256=expected_development_protocol_sha256,
        expected_development_result_sha256=expected_development_result_sha256,
    )
    inferential_ids = _panel_inferential_ids(selection)
    descriptive_ids = MATCHED_V3_DESCRIPTIVE_CANDIDATE_IDS
    panel: dict[str, Any] = {
        "schema_version": FORAGER_MATCHED_V3_CONFIRMATORY_PANEL_SCHEMA_VERSION,
        "status": "unexecuted_design",
        "classification": "confirmatory_panel_from_accepted_development_selection",
        "selection_receipt": detached_receipt,
        "selection_receipt_sha256": _canonical_sha256(detached_receipt),
        "inferential_candidate_ids": list(inferential_ids),
        "descriptive_candidate_ids": list(descriptive_ids),
        "candidate_ids": list(inferential_ids + descriptive_ids),
        "counts": {
            "inferential_candidate_count": 11,
            "descriptive_candidate_count": 3,
            "candidate_count": 14,
        },
        "claim_boundaries": {
            "builder_inferred_selection_from_scores": False,
            "research_literature_exhaustive": False,
            "universal_sota_claim_allowed": False,
            "scientific_promotion_allowed": False,
            "execution_authorized": False,
            "panel_is_executed": False,
            "panel_supports_performance_claim": False,
        },
    }
    panel["payload_sha256"] = _canonical_sha256(panel)
    return panel


def _require_string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ForagerMatchedV3CandidateUniverseError(
            f"{context} must be a JSON array of strings"
        )
    return cast(list[str], value)


def validate_matched_v3_confirmatory_panel(
    panel: Mapping[str, Any],
    *,
    expected_development_protocol_sha256: str,
    expected_development_result_sha256: str,
) -> dict[str, Any]:
    """Strictly validate and detach a panel built from an accepted receipt."""
    if not isinstance(panel, Mapping):
        raise ForagerMatchedV3CandidateUniverseError("confirmatory panel must be an object")
    snapshot = _plain_json_object_snapshot(panel, "confirmatory panel")
    _require_exact_keys(snapshot, _PANEL_KEYS, "confirmatory panel")
    if snapshot["schema_version"] != FORAGER_MATCHED_V3_CONFIRMATORY_PANEL_SCHEMA_VERSION:
        raise ForagerMatchedV3CandidateUniverseError(
            "confirmatory panel schema is not the exact v3 schema"
        )
    if snapshot["status"] != "unexecuted_design" or snapshot["classification"] != (
        "confirmatory_panel_from_accepted_development_selection"
    ):
        raise ForagerMatchedV3CandidateUniverseError(
            "confirmatory panel status/classification is invalid"
        )
    receipt = _require_mapping(snapshot["selection_receipt"], "panel selection receipt")
    selection = _validate_and_extract_receipt(
        receipt,
        expected_development_protocol_sha256=expected_development_protocol_sha256,
        expected_development_result_sha256=expected_development_result_sha256,
    )
    receipt_sha256 = _require_sha256(
        snapshot["selection_receipt_sha256"], "panel selection-receipt digest"
    )
    if not hmac.compare_digest(receipt_sha256, _canonical_sha256(receipt)):
        raise ForagerMatchedV3CandidateUniverseError(
            "panel selection-receipt digest does not match the embedded receipt"
        )

    expected_inferential = list(_panel_inferential_ids(selection))
    inferential = _require_string_list(
        snapshot["inferential_candidate_ids"], "panel inferential candidate IDs"
    )
    descriptive = _require_string_list(
        snapshot["descriptive_candidate_ids"], "panel descriptive candidate IDs"
    )
    all_ids = _require_string_list(snapshot["candidate_ids"], "panel candidate IDs")
    if inferential != expected_inferential:
        raise ForagerMatchedV3CandidateUniverseError(
            "panel inferential candidates do not follow the exact receipt-bound rule"
        )
    if descriptive != list(MATCHED_V3_DESCRIPTIVE_CANDIDATE_IDS):
        raise ForagerMatchedV3CandidateUniverseError(
            "panel descriptive candidates are not the exact fixed references"
        )
    if all_ids != inferential + descriptive or len(all_ids) != len(set(all_ids)):
        raise ForagerMatchedV3CandidateUniverseError(
            "panel candidate order/membership is invalid or duplicated"
        )

    counts = _require_mapping(snapshot["counts"], "panel counts")
    _require_exact_keys(counts, _PANEL_COUNT_KEYS, "panel counts")
    if any(
        (
            type(counts["inferential_candidate_count"]) is not int,
            type(counts["descriptive_candidate_count"]) is not int,
            type(counts["candidate_count"]) is not int,
            counts["inferential_candidate_count"] != 11,
            counts["descriptive_candidate_count"] != 3,
            counts["candidate_count"] != 14,
        )
    ):
        raise ForagerMatchedV3CandidateUniverseError("panel counts are not exact")
    claims = _require_mapping(snapshot["claim_boundaries"], "panel claim boundaries")
    _require_exact_keys(claims, _PANEL_CLAIM_KEYS, "panel claim boundaries")
    if any(value is not False for value in claims.values()):
        raise ForagerMatchedV3CandidateUniverseError(
            "all panel claim/authorization flags must be false"
        )
    supplied_payload = _require_sha256(snapshot["payload_sha256"], "panel payload digest")
    unsigned = {key: value for key, value in snapshot.items() if key != "payload_sha256"}
    if not hmac.compare_digest(supplied_payload, _canonical_sha256(unsigned)):
        raise ForagerMatchedV3CandidateUniverseError(
            "panel payload digest does not match its canonical payload"
        )
    return snapshot


def parse_matched_v3_confirmatory_panel_artifact(
    raw: bytes,
    *,
    expected_development_protocol_sha256: str,
    expected_development_result_sha256: str,
) -> dict[str, Any]:
    """Parse exact canonical panel bytes, then replay its receipt and denials."""

    snapshot = _parse_canonical_dynamic_json_object(raw, "confirmatory panel")
    return validate_matched_v3_confirmatory_panel(
        snapshot,
        expected_development_protocol_sha256=expected_development_protocol_sha256,
        expected_development_result_sha256=expected_development_result_sha256,
    )


def canonical_matched_v3_confirmatory_panel_bytes(
    panel: Mapping[str, Any],
    *,
    expected_development_protocol_sha256: str,
    expected_development_result_sha256: str,
) -> bytes:
    """Validate a plain unaliased panel tree and return its canonical bytes."""

    _validate_plain_unaliased_json_tree(panel, "confirmatory panel")
    if type(panel) is not dict:
        raise ForagerMatchedV3CandidateUniverseError(
            "confirmatory panel must be a plain JSON object"
        )
    raw = _canonical_bytes(panel)
    parse_matched_v3_confirmatory_panel_artifact(
        raw,
        expected_development_protocol_sha256=expected_development_protocol_sha256,
        expected_development_result_sha256=expected_development_result_sha256,
    )
    return raw
