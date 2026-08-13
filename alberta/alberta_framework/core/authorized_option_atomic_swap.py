# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Opt-in atomic retirement plus replacement over the v1 controller.

The v1 :mod:`authorized_option_replacement` transaction deliberately exposes a
real cold-slot phase between retirement and replacement.  That is useful for
auditing each authority boundary, but it cannot satisfy a caller that requires
the persistent cohort to remain fully installed.  This v2 host adapter keeps
the v1 controller unchanged and treats its retirement state as transient: the
original all-installed source is replaced only when a freshly derived candidate
can complete the exact retired slot in the same adoption.

``prepare`` is not authority.  It records the complete retirement and
replacement derivation and an unkeyed corruption checksum.  ``commit`` reruns
that complete derivation, bit-compares every leaf, revalidates the separately
supplied retirement and installation authority receipts through the v1
controller, and adopts only a full all-installed result.  Missing freshness,
decline, tampering, replay, or destination drift is an exact persistent no-op;
the transient cold state is never selected.

This is an L0 mechanism.  It writes no outputs and owns no retirement,
replacement, safety, go/no-go, evidence, promotion, dispatch, discovery, or
autonomous-curation authority.  Its receipts are integrity declarations, not
caller authentication.
"""

from __future__ import annotations

import dataclasses
from typing import cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.authorized_option_replacement import (
    AuthorizedOptionReplacementArm,
    AuthorizedOptionReplacementController,
    AuthorizedOptionReplacementPrepared,
    AuthorizedOptionReplacementResult,
    AuthorizedOptionReplacementRetirementResult,
    AuthorizedOptionReplacementState,
    OptionReplacementAuthorityReceipt,
    _checksum_arrays,
    _require_array,
    _tree_array_equal,
)
from alberta_framework.core.authorized_option_retirement import (
    OptionRetirementAuthorityReceipt,
)
from alberta_framework.core.cumulant_option_installation import CumulantOptionLiveInputs
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionInstallationAuthorityReceipt,
    CumulantOptionRetirementHandoff,
    CumulantOptionSchedulerArmInputs,
    CumulantOptionSchedulerObservation,
)

AUTHORIZED_OPTION_ATOMIC_SWAP_ASSESSMENT = "not_assessed"
AUTHORIZED_OPTION_ATOMIC_SWAP_OUTPUT_WRITES = False
AUTHORIZED_OPTION_ATOMIC_SWAP_EVIDENCE_AUTHORITY = False
AUTHORIZED_OPTION_ATOMIC_SWAP_PROMOTION_AUTHORITY = False
AUTHORIZED_OPTION_ATOMIC_SWAP_SAFETY_AUTHORITY = False
AUTHORIZED_OPTION_ATOMIC_SWAP_GO_NO_GO_AUTHORITY = False
AUTHORIZED_OPTION_ATOMIC_SWAP_RETIREMENT_AUTHORITY = False
AUTHORIZED_OPTION_ATOMIC_SWAP_REPLACEMENT_AUTHORITY = False
AUTHORIZED_OPTION_ATOMIC_SWAP_DISCOVERY_AUTHORITY = False
AUTHORIZED_OPTION_ATOMIC_SWAP_DISPATCH_AUTHORITY = False
AUTHORIZED_OPTION_ATOMIC_SWAP_AUTONOMOUS_CURATION_AUTHORITY = False
AUTHORIZED_OPTION_ATOMIC_SWAP_SCIENTIFIC_PROMOTION_ALLOWED = False


@chex.dataclass(frozen=True)
class AuthorizedOptionAtomicSwapPrepareDiagnostics:
    """Source, transient retirement, and fresh replacement feasibility."""

    source_state_valid: Bool[Array, ""]
    source_all_slots_installed: Bool[Array, ""]
    transient_retirement_applied: Bool[Array, ""]
    transient_retirement_state_valid: Bool[Array, ""]
    exact_one_transient_cold_slot: Bool[Array, ""]
    replacement_preparation_valid: Bool[Array, ""]
    fresh_candidate_available: Bool[Array, ""]
    exact_target_semantic_change: Bool[Array, ""]
    live_slots_semantically_preserved: Bool[Array, ""]
    atomic_swap_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AuthorizedOptionAtomicSwapPrepared:
    """Complete source-bound derivation; transient and never persistent."""

    source_state: AuthorizedOptionReplacementState
    retirement_handoff: CumulantOptionRetirementHandoff
    retirement_authority: OptionRetirementAuthorityReceipt
    phase_one_key: Array
    phase_two_key: Array
    arm_inputs: CumulantOptionSchedulerArmInputs
    observation: CumulantOptionSchedulerObservation
    live_inputs: CumulantOptionLiveInputs
    retirement_result: AuthorizedOptionReplacementRetirementResult
    replacement_arm: AuthorizedOptionReplacementArm
    replacement_prepared: AuthorizedOptionReplacementPrepared
    diagnostics: AuthorizedOptionAtomicSwapPrepareDiagnostics
    prepared_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class AuthorizedOptionAtomicSwapAuthorityReceipt:
    """Caller declaration binding one full swap and its nested v1 receipt."""

    replacement_authority: OptionReplacementAuthorityReceipt
    swap_authorized: Bool[Array, ""]
    controller_owner_digest: UInt[Array, " 8"]
    source_binding_checksum: UInt[Array, " 2"]
    source_controller_revision: Int[Array, ""]
    retired_binding_checksum: UInt[Array, " 2"]
    retired_controller_revision: Int[Array, ""]
    prepared_checksum: UInt[Array, " 2"]
    replacement_prepared_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class AuthorizedOptionAtomicSwapResult:
    """Atomic full-swap result or the exact unchanged destination state."""

    state: AuthorizedOptionReplacementState
    retirement_result: AuthorizedOptionReplacementRetirementResult
    replacement_result: AuthorizedOptionReplacementResult
    destination_state_valid: Bool[Array, ""]
    destination_matches_source: Bool[Array, ""]
    prepared_integrity_valid: Bool[Array, ""]
    preparation_derivation_valid: Bool[Array, ""]
    authority_valid: Bool[Array, ""]
    atomic_swap_ready: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    retirement_applied: Bool[Array, ""]
    replacement_applied: Bool[Array, ""]
    cold_state_persisted: Bool[Array, ""]
    reset_slots: Bool[Array, " option_budget"]
    preserved_slots: Bool[Array, " option_budget"]
    caller_authenticated: Bool[Array, ""]


class AuthorizedOptionAtomicSwapController:
    """Host-only atomic-swap adapter borrowing one exact v1 owner."""

    def __init__(self, replacement: AuthorizedOptionReplacementController) -> None:
        if type(replacement) is not AuthorizedOptionReplacementController:
            raise TypeError(
                "replacement must be an exact AuthorizedOptionReplacementController"
            )
        self._replacement = replacement

    @property
    def replacement(self) -> AuthorizedOptionReplacementController:
        """Return the sole borrowed v1 controller."""

        return self._replacement

    def _prepared_payload_arrays(
        self,
        prepared: AuthorizedOptionAtomicSwapPrepared,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                tuple(
                    getattr(prepared, field.name)
                    for field in dataclasses.fields(AuthorizedOptionAtomicSwapPrepared)
                    if field.name != "prepared_checksum"
                )
            )
        )

    def _with_prepared_checksum(
        self,
        prepared: AuthorizedOptionAtomicSwapPrepared,
    ) -> AuthorizedOptionAtomicSwapPrepared:
        return dataclasses.replace(
            prepared,
            prepared_checksum=_checksum_arrays(self._prepared_payload_arrays(prepared)),
        )

    def prepare(
        self,
        state: AuthorizedOptionReplacementState,
        retirement_handoff: CumulantOptionRetirementHandoff,
        retirement_authority: OptionRetirementAuthorityReceipt,
        phase_one_key: Array,
        phase_two_key: Array,
        arm_inputs: CumulantOptionSchedulerArmInputs,
        observation: CumulantOptionSchedulerObservation,
        live_inputs: CumulantOptionLiveInputs,
    ) -> AuthorizedOptionAtomicSwapPrepared:
        """Derive retirement and a fresh replacement without persisting either."""

        self._replacement._check_state_contract(state)
        retired = self._replacement.retire(
            state,
            retirement_handoff,
            retirement_authority,
            phase_one_key,
            phase_two_key,
        )
        arm = self._replacement.arm(retired.state, arm_inputs)
        replacement_prepared = self._replacement.prepare(
            retired.state,
            arm,
            observation,
            live_inputs,
        )

        source_valid = self._replacement.state_valid(state)
        source_all_installed = jnp.all(state.installed_slot_mask)
        transient_valid = self._replacement.state_valid(retired.state)
        exact_one_cold = jnp.sum(~retired.state.installed_slot_mask, dtype=jnp.int32) == 1
        replacement_valid = replacement_prepared.diagnostics.transaction_valid
        fresh_candidate = replacement_prepared.diagnostics.candidate_ready_for_authority
        exact_target_change = jnp.array_equal(
            replacement_prepared.changed_slots,
            replacement_prepared.target_mask,
        ) & (jnp.sum(replacement_prepared.target_mask, dtype=jnp.int32) == 1)
        live_preserved = ~jnp.any(
            replacement_prepared.changed_slots & retired.state.installed_slot_mask
        )
        ready = (
            source_valid
            & source_all_installed
            & retired.transaction_applied
            & transient_valid
            & exact_one_cold
            & replacement_valid
            & fresh_candidate
            & exact_target_change
            & live_preserved
            & _tree_array_equal(replacement_prepared.source_state, retired.state)
        )
        prepared = AuthorizedOptionAtomicSwapPrepared(
            source_state=state,
            retirement_handoff=retirement_handoff,
            retirement_authority=retirement_authority,
            phase_one_key=phase_one_key,
            phase_two_key=phase_two_key,
            arm_inputs=arm_inputs,
            observation=observation,
            live_inputs=live_inputs,
            retirement_result=retired,
            replacement_arm=arm,
            replacement_prepared=replacement_prepared,
            diagnostics=AuthorizedOptionAtomicSwapPrepareDiagnostics(
                source_state_valid=source_valid,
                source_all_slots_installed=source_all_installed,
                transient_retirement_applied=retired.transaction_applied,
                transient_retirement_state_valid=transient_valid,
                exact_one_transient_cold_slot=exact_one_cold,
                replacement_preparation_valid=replacement_valid,
                fresh_candidate_available=fresh_candidate,
                exact_target_semantic_change=exact_target_change,
                live_slots_semantically_preserved=live_preserved,
                atomic_swap_ready=ready,
            ),
            prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_prepared_checksum(prepared)

    def authority_receipt(
        self,
        prepared: AuthorizedOptionAtomicSwapPrepared,
        installation_authority: CumulantOptionInstallationAuthorityReceipt,
        *,
        swap_authorized: bool | Array,
    ) -> AuthorizedOptionAtomicSwapAuthorityReceipt:
        """Bind explicit swap authority to the original and transient sources."""

        if type(prepared) is not AuthorizedOptionAtomicSwapPrepared:
            raise TypeError("prepared must be an exact AuthorizedOptionAtomicSwapPrepared")
        authorized = jnp.asarray(swap_authorized, dtype=jnp.bool_)
        nested = self._replacement.authority_receipt(
            prepared.replacement_prepared,
            installation_authority,
            replacement_authorized=authorized,
        )
        source = prepared.source_state
        retired = prepared.retirement_result.state
        return AuthorizedOptionAtomicSwapAuthorityReceipt(
            replacement_authority=nested,
            swap_authorized=authorized,
            controller_owner_digest=source.controller_owner_digest,
            source_binding_checksum=source.binding_checksum,
            source_controller_revision=source.controller_revision,
            retired_binding_checksum=retired.binding_checksum,
            retired_controller_revision=retired.controller_revision,
            prepared_checksum=prepared.prepared_checksum,
            replacement_prepared_checksum=(prepared.replacement_prepared.prepared_checksum),
        )

    def _check_authority_contract(
        self,
        receipt: AuthorizedOptionAtomicSwapAuthorityReceipt,
    ) -> None:
        if type(receipt) is not AuthorizedOptionAtomicSwapAuthorityReceipt:
            raise TypeError(
                "authority_receipt must be an exact AuthorizedOptionAtomicSwapAuthorityReceipt"
            )
        self._replacement._check_receipt_contract(receipt.replacement_authority)
        contracts = (
            (receipt.swap_authorized, "swap_authorized", (), jnp.bool_),
            (receipt.controller_owner_digest, "controller_owner_digest", (8,), jnp.uint32),
            (receipt.source_binding_checksum, "source_binding_checksum", (2,), jnp.uint32),
            (receipt.source_controller_revision, "source_controller_revision", (), jnp.int32),
            (receipt.retired_binding_checksum, "retired_binding_checksum", (2,), jnp.uint32),
            (receipt.retired_controller_revision, "retired_controller_revision", (), jnp.int32),
            (receipt.prepared_checksum, "prepared_checksum", (2,), jnp.uint32),
            (
                receipt.replacement_prepared_checksum,
                "replacement_prepared_checksum",
                (2,),
                jnp.uint32,
            ),
        )
        for value, name, shape, dtype in contracts:
            _require_array(
                value,
                name=f"authority_receipt.{name}",
                shape=shape,
                dtype=dtype,
            )

    def _authority_valid(
        self,
        prepared: AuthorizedOptionAtomicSwapPrepared,
        receipt: AuthorizedOptionAtomicSwapAuthorityReceipt,
    ) -> Array:
        source = prepared.source_state
        retired = prepared.retirement_result.state
        nested = receipt.replacement_authority
        return (
            receipt.swap_authorized
            & nested.replacement_authorized
            & jnp.array_equal(receipt.controller_owner_digest, source.controller_owner_digest)
            & jnp.array_equal(receipt.source_binding_checksum, source.binding_checksum)
            & (receipt.source_controller_revision == source.controller_revision)
            & jnp.array_equal(receipt.retired_binding_checksum, retired.binding_checksum)
            & (receipt.retired_controller_revision == retired.controller_revision)
            & jnp.array_equal(receipt.prepared_checksum, prepared.prepared_checksum)
            & jnp.array_equal(
                receipt.replacement_prepared_checksum,
                prepared.replacement_prepared.prepared_checksum,
            )
            & self._replacement._authority_valid(
                retired,
                prepared.replacement_prepared,
                nested,
            )
        )

    def commit(
        self,
        state: AuthorizedOptionReplacementState,
        prepared: AuthorizedOptionAtomicSwapPrepared,
        authority_receipt: AuthorizedOptionAtomicSwapAuthorityReceipt,
    ) -> AuthorizedOptionAtomicSwapResult:
        """Rederive and atomically adopt only a complete all-installed swap."""

        self._replacement._check_state_contract(state)
        if type(prepared) is not AuthorizedOptionAtomicSwapPrepared:
            raise TypeError("prepared must be an exact AuthorizedOptionAtomicSwapPrepared")
        self._check_authority_contract(authority_receipt)

        integrity = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._prepared_payload_arrays(prepared)),
        )
        recomputed = self.prepare(
            prepared.source_state,
            prepared.retirement_handoff,
            prepared.retirement_authority,
            prepared.phase_one_key,
            prepared.phase_two_key,
            prepared.arm_inputs,
            prepared.observation,
            prepared.live_inputs,
        )
        derivation = _tree_array_equal(prepared, recomputed)
        destination_valid = self._replacement.state_valid(state)
        destination_matches = _tree_array_equal(state, recomputed.source_state)
        authority_valid = self._authority_valid(recomputed, authority_receipt)

        replacement_result = self._replacement.commit(
            recomputed.retirement_result.state,
            recomputed.replacement_prepared,
            authority_receipt.replacement_authority,
        )
        exact_reset = jnp.array_equal(
            replacement_result.reset_slots,
            recomputed.replacement_prepared.target_mask,
        )
        exact_preserve = jnp.array_equal(
            replacement_result.preserved_slots,
            ~recomputed.replacement_prepared.target_mask,
        )
        final_valid = self._replacement.state_valid(replacement_result.state)
        final_all_installed = jnp.all(replacement_result.state.installed_slot_mask)
        applied = (
            destination_valid
            & destination_matches
            & integrity
            & derivation
            & recomputed.diagnostics.atomic_swap_ready
            & authority_valid
            & replacement_result.diagnostics.prepared_integrity_valid
            & replacement_result.diagnostics.preparation_derivation_valid
            & replacement_result.diagnostics.authority_valid
            & replacement_result.diagnostics.replacement_applied
            & exact_reset
            & exact_preserve
            & final_valid
            & final_all_installed
        )
        next_state = cast(
            AuthorizedOptionReplacementState,
            jax.tree_util.tree_map(
                lambda candidate, destination: jnp.where(applied, candidate, destination),
                replacement_result.state,
                state,
            ),
        )
        cold_persisted = (
            jnp.all(recomputed.source_state.installed_slot_mask)
            & jnp.any(~next_state.installed_slot_mask)
        )
        return AuthorizedOptionAtomicSwapResult(
            state=next_state,
            retirement_result=recomputed.retirement_result,
            replacement_result=replacement_result,
            destination_state_valid=destination_valid,
            destination_matches_source=destination_matches,
            prepared_integrity_valid=integrity,
            preparation_derivation_valid=derivation,
            authority_valid=authority_valid,
            atomic_swap_ready=recomputed.diagnostics.atomic_swap_ready,
            transaction_applied=applied,
            retirement_applied=applied,
            replacement_applied=applied,
            cold_state_persisted=cold_persisted,
            reset_slots=applied & replacement_result.reset_slots,
            preserved_slots=applied & replacement_result.preserved_slots,
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )


__all__ = [
    "AUTHORIZED_OPTION_ATOMIC_SWAP_ASSESSMENT",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_AUTONOMOUS_CURATION_AUTHORITY",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_DISCOVERY_AUTHORITY",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_DISPATCH_AUTHORITY",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_EVIDENCE_AUTHORITY",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_GO_NO_GO_AUTHORITY",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_OUTPUT_WRITES",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_PROMOTION_AUTHORITY",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_REPLACEMENT_AUTHORITY",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_RETIREMENT_AUTHORITY",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_SAFETY_AUTHORITY",
    "AUTHORIZED_OPTION_ATOMIC_SWAP_SCIENTIFIC_PROMOTION_ALLOWED",
    "AuthorizedOptionAtomicSwapAuthorityReceipt",
    "AuthorizedOptionAtomicSwapController",
    "AuthorizedOptionAtomicSwapPrepareDiagnostics",
    "AuthorizedOptionAtomicSwapPrepared",
    "AuthorizedOptionAtomicSwapResult",
]
