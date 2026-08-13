"""Canonical API for the non-gradient prospective-exploration selector.

The implementation remains in the historical ``delightful_exploration``
module so v1 imports and strict config-payload migration can be maintained.
Legacy v1 checkpoints are deliberately unsupported and fail closed. New code
uses this module: its score is expected improvement times capped host-relative
surprisal, and it neither computes DG delight nor executes an actor backward.
"""

from alberta_framework.core.delightful_exploration import (
    PROSPECTIVE_EXPLORATION_ACTION_DISPATCH_AUTHORITY,
    PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA,
    PROSPECTIVE_EXPLORATION_CONFIG_SCHEMA,
    PROSPECTIVE_EXPLORATION_EVIDENCE_LEVEL,
    PROSPECTIVE_EXPLORATION_EXECUTES_ACTOR_BACKWARD,
    PROSPECTIVE_EXPLORATION_GRADIENT_DELIGHT_SEMANTICS,
    PROSPECTIVE_EXPLORATION_LIFETIME_SEMANTICS,
    PROSPECTIVE_EXPLORATION_MODES,
    PROSPECTIVE_EXPLORATION_NOISY_BANDIT_SEMANTICS,
    PROSPECTIVE_EXPLORATION_OUTCOME_STATUS,
    PROSPECTIVE_EXPLORATION_OUTPUT_WRITE_AUTHORITY,
    PROSPECTIVE_EXPLORATION_PHYSICAL_SAFETY_CLAIM,
    PROSPECTIVE_EXPLORATION_POLICY_OVERRIDE_AUTHORITY,
    PROSPECTIVE_EXPLORATION_RESOURCE_SCHEMA,
    PROSPECTIVE_EXPLORATION_REVEALED_VALUE_EQUIVALENCE,
    PROSPECTIVE_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED,
    PROSPECTIVE_EXPLORATION_SCORE_SEMANTICS,
    PROSPECTIVE_EXPLORATION_STATE_SCHEMA,
    ExplorationCandidateBatch,
    ExplorationMode,
    ProspectiveExploration,
    ProspectiveExplorationConfig,
    ProspectiveExplorationResourceBudget,
    ProspectiveExplorationResult,
    ProspectiveExplorationScanResult,
    ProspectiveExplorationState,
    measure_prospective_exploration_state_nbytes,
    run_prospective_exploration_from_batches,
)

__all__ = [
    "PROSPECTIVE_EXPLORATION_ACTION_DISPATCH_AUTHORITY",
    "PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA",
    "PROSPECTIVE_EXPLORATION_CONFIG_SCHEMA",
    "PROSPECTIVE_EXPLORATION_EVIDENCE_LEVEL",
    "PROSPECTIVE_EXPLORATION_EXECUTES_ACTOR_BACKWARD",
    "PROSPECTIVE_EXPLORATION_GRADIENT_DELIGHT_SEMANTICS",
    "PROSPECTIVE_EXPLORATION_LIFETIME_SEMANTICS",
    "PROSPECTIVE_EXPLORATION_MODES",
    "PROSPECTIVE_EXPLORATION_NOISY_BANDIT_SEMANTICS",
    "PROSPECTIVE_EXPLORATION_OUTCOME_STATUS",
    "PROSPECTIVE_EXPLORATION_OUTPUT_WRITE_AUTHORITY",
    "PROSPECTIVE_EXPLORATION_PHYSICAL_SAFETY_CLAIM",
    "PROSPECTIVE_EXPLORATION_POLICY_OVERRIDE_AUTHORITY",
    "PROSPECTIVE_EXPLORATION_RESOURCE_SCHEMA",
    "PROSPECTIVE_EXPLORATION_REVEALED_VALUE_EQUIVALENCE",
    "PROSPECTIVE_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROSPECTIVE_EXPLORATION_SCORE_SEMANTICS",
    "PROSPECTIVE_EXPLORATION_STATE_SCHEMA",
    "ExplorationCandidateBatch",
    "ExplorationMode",
    "ProspectiveExploration",
    "ProspectiveExplorationConfig",
    "ProspectiveExplorationResourceBudget",
    "ProspectiveExplorationResult",
    "ProspectiveExplorationScanResult",
    "ProspectiveExplorationState",
    "measure_prospective_exploration_state_nbytes",
    "run_prospective_exploration_from_batches",
]
