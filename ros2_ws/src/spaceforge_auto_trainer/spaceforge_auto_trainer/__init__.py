"""Curriculum management and optional Optuna search; no training on import."""
from .curriculum import (
    SCHEMA_VERSION as CURRICULUM_SCHEMA_VERSION,
    STAGES,
    STAGE_DESCRIPTIONS,
    WINDOW_SIZE,
    AdvancementThresholds,
    RollbackThresholds,
    CurriculumManager,
    EpisodeOutcome,
    StageTransition,
    WindowMetrics,
)
from .hpo import (
    SCHEMA_VERSION as HPO_SCHEMA_VERSION,
    ALGORITHMS,
    HPOConfig,
    ObjectiveWeights,
    SearchSpace,
    objective_value,
    run_study,
    suggest_params,
)

__all__ = [
    "CURRICULUM_SCHEMA_VERSION", "STAGES", "STAGE_DESCRIPTIONS", "WINDOW_SIZE",
    "AdvancementThresholds", "RollbackThresholds", "CurriculumManager",
    "EpisodeOutcome", "StageTransition", "WindowMetrics",
    "HPO_SCHEMA_VERSION", "ALGORITHMS", "HPOConfig", "ObjectiveWeights",
    "SearchSpace", "objective_value", "run_study", "suggest_params",
]
