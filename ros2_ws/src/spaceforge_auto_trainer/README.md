# SpaceForge auto_trainer

This package manages the five-stage curriculum and optional PPO/SAC Optuna search. Curriculum decisions are deterministic and persisted. Advancement requires a full 50-episode window and the thresholds in AUTO_TRAINER_CONTRACT.md.

CurriculumManager keeps the most recent 50 episode outcomes per stage, advances on the documented band (equality passes), holds incomplete or in-between windows, and rolls back one stage on the documented failure band (clearing the window; stage 1 holds instead). Episode seeds are base_seed + episode_index and state (schema version, seeds, outcomes, transitions) persists to JSON, so resume and replay are reproducible.

HPO (`hpo.py`) needs `pip install optuna` (or the `hpo` extra); without it, curriculum and all default tests still work and study tests skip. Studies use SQLite storage with load_if_exists: `n_trials` bounds the whole study across resumes, failed trials are recorded (not aborting), and every trial stores algorithm/stage/params/metrics/objective metadata. The scalarized objective trades success against fuel, safety overrides, and collision risk with explicit weights. Default tests use stub training with n_trials <= 3 and need no SB3, Gazebo or network; expensive studies require explicit opt-in (see test_live_optuna.py).
