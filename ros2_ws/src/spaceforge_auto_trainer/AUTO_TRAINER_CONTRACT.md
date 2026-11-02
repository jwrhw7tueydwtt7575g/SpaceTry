# SpaceForge auto_trainer contract

Five curriculum stages are explicit and stable: 1 stage1_1d approach; 2 stage2_3d_static rendezvous; 3 stage3_tumbling_target; 4 stage4_debris_avoidance; 5 stage5_full_inspection.

Advancement policy: retain the most recent 50 episode outcomes for the current stage. Advance only when the window is full and success_rate >= 0.90, average fuel_used <= 0.35, average safety_overrides <= 1.0 per episode, and average collision_risk <= 0.05. Threshold equality passes.

Hold policy: incomplete windows or metrics between advancement and rollback remain at the current stage. Rollback one stage when success_rate < 0.50, average safety_overrides > 5.0, or average collision_risk > 0.20. Rollback clears the current window. Stage 1 cannot roll back; stage 5 cannot advance.

State persistence stores schema version, base seed, deterministic episode index, current stage, rolling outcomes, and transition history. The next episode seed is base_seed + episode_index, so resume and replay are reproducible.

HPO uses Optuna only when explicitly requested. Studies use SQLite storage, load_if_exists, bounded n_trials and timeout, trial artifact metadata, and an objective containing success, fuel, safety overrides, and collision risk. No expensive study runs in the default test suite.

Two orchestration semantics are explicit (Optuna >= 4 changed both defaults):
n_trials bounds the whole study, not each call: resuming with the same study name and storage tops up to the budget instead of overshooting it. Failed train_fn calls mark that trial FAILED (catch=(Exception,)) and the study spends the rest of its bounded budget; only storage/sampler errors propagate. Every trial records algorithm, stage, suggested params, the four objective metrics, and the scalarized value as user attributes; hpo_report.json carries the schema version, budgets, per-trial artifacts, and the best trial.
