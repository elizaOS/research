"""Focused tests for the unapplied generated-class reacquisition epoch."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import struct
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    OP_GATED,
    OP_PRODUCT,
    OP_RAW,
    OP_SUM,
    OP_TANH,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)
from alberta_framework.evaluation import (
    generated_reacquisition_epoch as reacquisition_epoch_module,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    ACTIVE_MASKED_LEAF_PATHS,
    CANDIDATE_MASKED_LEAF_PATHS,
    CROSS_MASKED_LEAF_PATHS,
    GeneratedClassScrubConfig,
    scrub_compositional_feature_state,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    GeneratedExpression,
    product_expression,
    raw_expression,
)
from alberta_framework.evaluation.generated_expression_lineage import (
    ExpandedExpressionLineageConfig,
    ExpandedExpressionLineagePlan,
    compile_expanded_expression_lineage_masks,
)
from alberta_framework.evaluation.generated_reacquisition_epoch import (
    GENERATED_REACQUISITION_EPOCH_SCHEMA,
    GENERATED_REACQUISITION_EPOCH_STATUS,
    GENERATED_REACQUISITION_KEY_NAMESPACE,
    GeneratedReacquisitionEpochConfig,
    GeneratedReacquisitionEpochConstructionError,
    GeneratedReacquisitionEpochExecutionUnauthorizedError,
    GeneratedReacquisitionEpochPlan,
    build_generated_reacquisition_epoch_plan,
    derive_generated_reacquisition_epoch_key,
    require_generated_reacquisition_epoch_executable,
    validate_generated_reacquisition_epoch_plan,
)

pytestmark = pytest.mark.unit


def _tree_bit_records(value: object) -> tuple[tuple[str, str, tuple[int, ...], bytes], ...]:
    records: list[tuple[str, str, tuple[int, ...], bytes]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        if isinstance(leaf, jax.Array) and jnp.issubdtype(
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            array = np.asarray(jr.key_data(leaf))
            records.append((str(path), str(leaf.dtype), array.shape, array.tobytes()))
        elif isinstance(leaf, jax.Array):
            array = np.asarray(leaf)
            records.append((str(path), array.dtype.str, array.shape, array.tobytes()))
        elif type(leaf) is float:
            records.append((str(path), "python-float", (), struct.pack(">d", leaf)))
        else:
            raise TypeError(type(leaf))
    return tuple(records)


def _lineage_config() -> ExpandedExpressionLineageConfig:
    return ExpandedExpressionLineageConfig(
        feature_dim=3,
        active_slots=8,
        candidate_slots=4,
        n_tasks=1,
        generator_contexts=1,
        generator_policy_count=4,
    )


def _scrub_config(
    config: ExpandedExpressionLineageConfig,
) -> GeneratedClassScrubConfig:
    return GeneratedClassScrubConfig(
        feature_dim=config.feature_dim,
        active_slots=config.active_slots,
        candidate_slots=config.candidate_slots,
        n_tasks=config.n_tasks,
        filler_op=config.filler_op,
        filler_parent_a=config.filler_parent_a,
        filler_parent_b=config.filler_parent_b,
    )


def _pre_state(*, key: jax.Array | None = None, step_count: int = 17) -> CompositionalFeatureState:
    learner = CompositionalFeatureLearner(
        n_features=8,
        n_tasks=1,
        candidate_count=4,
        replacement_interval=0,
        max_depth=3,
    )
    state = learner.init(3, jr.key(402))
    theta = (
        jnp.zeros((8, 2), dtype=jnp.float32)
        .at[6]
        .set(jnp.asarray((-1.0, 2.0), dtype=jnp.float32))
    )
    updated = state.replace(  # type: ignore[attr-defined]
        key=state.key if key is None else key,
        ops=jnp.asarray(
            (OP_RAW, OP_RAW, OP_RAW, OP_PRODUCT, OP_SUM, OP_GATED, OP_TANH, OP_PRODUCT),
            dtype=jnp.int32,
        ),
        parent_a=jnp.asarray((0, 1, 2, 0, 3, 1, 1, 4), dtype=jnp.int32),
        parent_b=jnp.asarray((-1, -1, -1, 1, 2, 2, 0, 5), dtype=jnp.int32),
        theta=theta,
        depth=jnp.asarray((0, 0, 0, 1, 2, 1, 1, 3), dtype=jnp.int32),
        candidate_ops=jnp.asarray((OP_GATED, OP_PRODUCT, OP_RAW, OP_SUM), dtype=jnp.int32),
        candidate_parent_a=jnp.asarray((4, 6, 0, 4), dtype=jnp.int32),
        candidate_parent_b=jnp.asarray((0, 2, -1, 5), dtype=jnp.int32),
        candidate_theta=jnp.zeros((4, 2), dtype=jnp.float32),
        candidate_depth=jnp.asarray((3, 2, 0, 3), dtype=jnp.int32),
        feature_generator_policy=jnp.asarray((0, 1, 2, 3, 0, 1, 2, 3), dtype=jnp.int32),
        candidate_generator_policy=jnp.asarray((3, 2, 1, 0), dtype=jnp.int32),
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
        step_words=jnp.asarray((0, step_count), dtype=jnp.uint32),
        birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
        uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
    )
    return cast(CompositionalFeatureState, updated)


def _bound_scrub(
    *,
    key: jax.Array | None = None,
    step_count: int = 17,
) -> tuple[
    CompositionalFeatureState,
    CompositionalFeatureState,
    GeneratedExpression,
    ExpandedExpressionLineagePlan,
    ExpandedExpressionLineageConfig,
    GeneratedClassScrubConfig,
]:
    pre = _pre_state(key=key, step_count=step_count)
    target = product_expression(raw_expression(0), raw_expression(1))
    lineage_config = _lineage_config()
    scrub_config = _scrub_config(lineage_config)
    lineage_plan = compile_expanded_expression_lineage_masks(
        pre,
        target,
        config=lineage_config,
    )
    result = scrub_compositional_feature_state(
        pre,
        lineage_plan.active_mask,
        lineage_plan.candidate_mask,
        jnp.asarray(True, dtype=jnp.bool_),
        config=scrub_config,
    )
    assert bool(result.diagnostics.committed)
    return pre, result.state, target, lineage_plan, lineage_config, scrub_config


def _build(
    inputs: tuple[
        CompositionalFeatureState,
        CompositionalFeatureState,
        GeneratedExpression,
        ExpandedExpressionLineagePlan,
        ExpandedExpressionLineageConfig,
        GeneratedClassScrubConfig,
    ],
    *,
    config: GeneratedReacquisitionEpochConfig | None = None,
) -> GeneratedReacquisitionEpochPlan:
    pre, post, target, lineage_plan, lineage_config, scrub_config = inputs
    return build_generated_reacquisition_epoch_plan(
        pre,
        post,
        target,
        lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        config=GeneratedReacquisitionEpochConfig() if config is None else config,
    )


def test_plan_binds_successful_scrub_freeze_and_unapplied_fresh_epoch() -> None:
    inputs = _bound_scrub()
    pre, post, target, lineage_plan, lineage_config, scrub_config = inputs
    before_pre = _tree_bit_records(pre)
    before_post = _tree_bit_records(post)
    plan = _build(inputs)
    replay = _build(inputs)

    assert _tree_bit_records(pre) == before_pre
    assert _tree_bit_records(post) == before_post
    assert plan.contract == replay.contract
    np.testing.assert_array_equal(
        jr.key_data(plan.fresh_learner_key),
        jr.key_data(replay.fresh_learner_key),
    )

    contract = plan.contract
    assert contract.schema == GENERATED_REACQUISITION_EPOCH_SCHEMA
    assert contract.schema == "alberta.generated-reacquisition-epoch.development.v1"
    assert contract.status == GENERATED_REACQUISITION_EPOCH_STATUS
    assert contract.structural_scrub_valid
    assert contract.scrub_preserved_learner_key_exactly
    assert contract.pre_scrub_state_bit_sha256 != contract.post_scrub_state_bit_sha256
    assert contract.key_namespace == GENERATED_REACQUISITION_KEY_NAMESPACE
    assert contract.previous_epoch_counter == 0
    assert contract.reacquisition_epoch_counter == 1
    assert contract.paired_life_seed == 101
    assert contract.paired_life_seed in contract.paired_life_seeds_uint32
    assert len(contract.paired_life_seeds_uint32) == 8
    assert contract.paired_life_seed_matched_across_arms_required
    assert contract.all_manifest_lives_required_for_campaign
    assert contract.distinct_epoch_keys_across_manifest_lives_observed
    assert contract.previous_epoch_key_data_uint32 != contract.fresh_learner_key_data_uint32
    assert contract.fresh_key_collision_with_bound_or_manifest_prior_count == 0
    assert contract.manifest_epoch_key_collision_count == 0
    assert not contract.fresh_key_matches_bound_pre_or_post_state_key
    assert contract.generation_write_freeze_start_state_step_count == 17
    assert contract.generation_write_freeze_end_state_step_count == 49
    assert contract.generation_write_freeze_updates == 32
    assert contract.scheduled_curation_decision_slots_in_freeze == 1
    source_path = inspect.getsourcefile(CompositionalFeatureLearner)
    assert source_path is not None
    assert contract.reviewed_compositional_features_module_byte_sha256 == (
        hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    )
    reviewed_source_hashes = {
        "__init__": contract.reviewed_learner_init_source_sha256,
        "_op_logits": contract.reviewed_learner_op_logits_source_sha256,
        "update": contract.reviewed_learner_update_source_sha256,
        "_generate_one": contract.reviewed_learner_generate_one_source_sha256,
        "_curation_stage_guidance": (
            contract.reviewed_learner_curation_stage_guidance_source_sha256
        ),
        "_cascade_replace": contract.reviewed_learner_cascade_replace_source_sha256,
        "_cascade_replace_with_mask": (
            contract.reviewed_learner_cascade_replace_with_mask_source_sha256
        ),
    }
    for method_name, reviewed_sha256 in reviewed_source_hashes.items():
        method = getattr(CompositionalFeatureLearner, method_name)
        assert reviewed_sha256 == hashlib.sha256(
            inspect.getsource(method).encode("utf-8")
        ).hexdigest()
    assert contract.public_curation_trace_available
    assert not contract.public_curation_trace_consumed
    assert not contract.public_curation_trace_bound
    assert contract.identical_curation_proposal_required
    assert contract.conditional_curation_write_components_permitted_during_freeze == 0
    assert contract.ordinary_noncuration_learning_writes_permitted
    assert set(contract.conditional_curation_write_leaf_paths) == (
        ACTIVE_MASKED_LEAF_PATHS
        | CANDIDATE_MASKED_LEAF_PATHS
        | CROSS_MASKED_LEAF_PATHS
        | {
            "generator_resource_state.action_counts",
            "generator_resource_state.log_weights",
            "generator_resource_state.reward_ema",
            "generator_resource_state.step_count",
            "replacement_accumulator",
        }
    )
    assert contract.fresh_key_apply_state_step_count == 49
    assert not contract.freeze_runner_implemented
    assert not contract.fresh_key_application_implemented
    assert contract.learner_observation_fields == ("raw_features",)
    assert not set(contract.learner_observation_fields) & set(contract.evaluator_only_fields)
    assert not contract.evaluator_metadata_in_learner_observations
    assert contract.evaluator_intervention
    assert not contract.generation_only_substream_claimed
    assert not contract.generator_policy_state_reset
    assert not contract.global_rng_use_ledger_implemented
    assert not contract.stochastic_independence_claimed
    assert not contract.acquisition_claimed
    assert not contract.adequate_acquisition_probability_claimed
    assert not contract.behavioral_memory_erasure_claimed
    assert contract.learner_updates_executed == 0
    assert contract.generated_life_steps_executed == 0
    assert contract.learner_rng_draws_executed == 0
    assert contract.artifact_bytes_written == 0
    assert len(contract.contract_sha256) == 64

    validation = validate_generated_reacquisition_epoch_plan(
        plan,
        pre,
        post,
        target,
        lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        config=GeneratedReacquisitionEpochConfig(),
    )
    assert validation.valid
    assert dataclasses.asdict(validation) == {
        name: True for name in dataclasses.asdict(validation)
    }


def test_module_digest_and_guidance_source_tampering_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _bound_scrub()
    with monkeypatch.context() as patch:
        patch.setattr(
            reacquisition_epoch_module,
            "_compositional_features_module_byte_sha256",
            lambda: "0" * 64,
        )
        with pytest.raises(
            GeneratedReacquisitionEpochConstructionError,
            match="module bytes changed",
        ):
            _build(inputs)

    original = CompositionalFeatureLearner._curation_stage_guidance

    def tampered_guidance(*args: object, **kwargs: object) -> object:
        return original(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as patch:
        patch.setattr(
            CompositionalFeatureLearner,
            "_curation_stage_guidance",
            tampered_guidance,
        )
        with pytest.raises(
            GeneratedReacquisitionEpochConstructionError,
            match="_curation_stage_guidance source changed",
        ):
            _build(inputs)


def test_key_derivation_is_pinned_typed_threefry_and_has_no_target_parameter() -> None:
    config = GeneratedReacquisitionEpochConfig()
    parameters = tuple(inspect.signature(derive_generated_reacquisition_epoch_key).parameters)
    assert parameters == ("epoch_counter", "config")
    baseline = derive_generated_reacquisition_epoch_key(0, config=config)
    fresh = derive_generated_reacquisition_epoch_key(1, config=config)

    assert str(jr.key_impl(baseline)) == "threefry2x32"
    assert str(jr.key_impl(fresh)) == "threefry2x32"
    assert baseline.shape == fresh.shape == ()
    assert tuple(int(word) for word in jr.key_data(baseline)) == (
        0x8402811F,
        0x8CDF6549,
    )
    assert tuple(int(word) for word in jr.key_data(fresh)) == (
        0x070E65CE,
        0x787A2732,
    )
    assert not np.array_equal(jr.key_data(baseline), jr.key_data(fresh))

    second_life_config = dataclasses.replace(config, paired_life_seed=211)
    second_life_fresh = derive_generated_reacquisition_epoch_key(
        1,
        config=second_life_config,
    )
    assert tuple(int(word) for word in jr.key_data(second_life_fresh)) == (
        0x092D9271,
        0x0678A1BB,
    )
    assert not np.array_equal(jr.key_data(fresh), jr.key_data(second_life_fresh))

    with pytest.raises(TypeError, match="exact Python integer"):
        derive_generated_reacquisition_epoch_key(True, config=config)
    with pytest.raises(ValueError, match="outside the preregistered"):
        derive_generated_reacquisition_epoch_key(2, config=config)


def test_stale_post_state_and_fresh_key_reuse_fail_closed() -> None:
    inputs = _bound_scrub()
    pre, post, target, lineage_plan, lineage_config, scrub_config = inputs
    stale_post = post.replace(  # type: ignore[attr-defined]
        output_bias=post.output_bias.at[0].set(jnp.float32(1.0))
    )
    with pytest.raises(GeneratedReacquisitionEpochConstructionError, match="did not accept"):
        build_generated_reacquisition_epoch_plan(
            pre,
            stale_post,
            target,
            lineage_plan,
            lineage_config=lineage_config,
            scrub_config=scrub_config,
            config=GeneratedReacquisitionEpochConfig(),
        )

    fresh_key = derive_generated_reacquisition_epoch_key(
        1,
        config=GeneratedReacquisitionEpochConfig(),
    )
    reused_inputs = _bound_scrub(key=fresh_key)
    with pytest.raises(GeneratedReacquisitionEpochConstructionError, match="reuses"):
        _build(reused_inputs)


def test_forged_target_leakage_metadata_and_plan_key_are_rejected() -> None:
    inputs = _bound_scrub()
    pre, post, target, lineage_plan, lineage_config, scrub_config = inputs
    plan = _build(inputs)
    forged_contract = dataclasses.replace(
        plan.contract,
        key_derivation_input_fields=(
            *plan.contract.key_derivation_input_fields,
            "target_digest",
        ),
    )
    forged_plan = dataclasses.replace(plan, contract=forged_contract)
    leakage = validate_generated_reacquisition_epoch_plan(
        forged_plan,
        pre,
        post,
        target,
        lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        config=GeneratedReacquisitionEpochConfig(),
    )
    assert not leakage.supplied_contract_self_hash_valid
    assert not leakage.contract_matches_canonical
    assert not leakage.target_or_outcome_derivation_fields_absent
    assert not leakage.valid

    reused_key_plan = dataclasses.replace(plan, fresh_learner_key=post.key)
    reuse = validate_generated_reacquisition_epoch_plan(
        reused_key_plan,
        pre,
        post,
        target,
        lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        config=GeneratedReacquisitionEpochConfig(),
    )
    assert not reuse.fresh_key_matches_supplied_contract
    assert not reuse.fresh_key_matches_canonical
    assert not reuse.no_detectable_key_reuse
    assert not reuse.valid


def test_config_and_plan_key_types_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="freeze length"):
        GeneratedReacquisitionEpochConfig(generation_write_freeze_updates=31)
    with pytest.raises(ValueError, match="next monotone epoch"):
        GeneratedReacquisitionEpochConfig(reacquisition_epoch_counter=0)
    with pytest.raises(ValueError, match="not canonical"):
        GeneratedReacquisitionEpochConfig(
            key_namespace=f"{GENERATED_REACQUISITION_KEY_NAMESPACE}/target-D"
        )
    with pytest.raises(ValueError, match="cannot grant"):
        GeneratedReacquisitionEpochConfig(execution_authorized=True)
    with pytest.raises(ValueError, match="expose the public trace"):
        GeneratedReacquisitionEpochConfig(public_curation_trace_available=False)
    with pytest.raises(ValueError, match="cannot grant"):
        GeneratedReacquisitionEpochConfig(public_curation_trace_consumed=True)
    with pytest.raises(ValueError, match="cannot grant"):
        GeneratedReacquisitionEpochConfig(public_curation_trace_bound=True)
    with pytest.raises(ValueError, match="outside the fixed development manifest"):
        GeneratedReacquisitionEpochConfig(paired_life_seed=102)

    inputs = _bound_scrub()
    pre, post, target, lineage_plan, lineage_config, scrub_config = inputs
    plan = _build(inputs)
    legacy = dataclasses.replace(plan, fresh_learner_key=jr.PRNGKey(0))
    with pytest.raises(TypeError, match="typed JAX PRNG dtype"):
        validate_generated_reacquisition_epoch_plan(
            legacy,
            pre,
            post,
            target,
            lineage_plan,
            lineage_config=lineage_config,
            scrub_config=scrub_config,
            config=GeneratedReacquisitionEpochConfig(),
        )
    rbg = dataclasses.replace(plan, fresh_learner_key=jr.key(0, impl="rbg"))
    with pytest.raises(ValueError, match="threefry2x32"):
        validate_generated_reacquisition_epoch_plan(
            rbg,
            pre,
            post,
            target,
            lineage_plan,
            lineage_config=lineage_config,
            scrub_config=scrub_config,
            config=GeneratedReacquisitionEpochConfig(),
        )


def test_step_capacity_and_execution_authority_remain_fail_closed() -> None:
    overflow_inputs = _bound_scrub(step_count=np.iinfo(np.int32).max - 16)
    with pytest.raises(GeneratedReacquisitionEpochConstructionError, match="step capacity"):
        _build(overflow_inputs)

    plan = _build(_bound_scrub())
    with pytest.raises(
        GeneratedReacquisitionEpochExecutionUnauthorizedError,
        match="not executable",
    ):
        require_generated_reacquisition_epoch_executable(plan)
    assert not plan.contract.execution_authorized
    assert not plan.contract.runner_authorized
    assert not plan.contract.campaign_authorized
    assert not plan.contract.artifact_writes_authorized
    assert not plan.contract.threshold_authorized
    assert not plan.contract.evidence_authorized
    assert not plan.contract.scientific_promotion_allowed
