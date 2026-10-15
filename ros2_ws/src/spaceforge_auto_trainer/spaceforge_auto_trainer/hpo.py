"""Optuna SQLite-backed hyperparameter search for PPO/SAC (optional dependency).

Optuna is imported lazily: curriculum management and every default test work
without it. Request a study explicitly via run_study; studies persist in SQLite
with load_if_exists so bounded budgets are resumable.

Objective (maximize), with explicit weights and no hidden constants:
  value = success_rate
          - fuel_weight * avg_fuel_used
          - override_weight * avg_overrides_per_episode
          - risk_weight * avg_collision_risk
All four contract metrics (success, fuel, safety overrides, collision risk)
enter every trial's value and artifact metadata.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import json

#: Machine-readable HPO report schema version.
SCHEMA_VERSION = 1

#: Selectable algorithms; PPO is the default, SAC the explicit alternative.
ALGORITHMS: Tuple[str, ...] = ("PPO", "SAC")


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "optuna is required for hyperparameter search "
            "(pip install optuna, or the 'hpo' extra)"
        ) from exc
    return optuna


@dataclass(frozen=True)
class SearchSpace:
    """Explicit suggest ranges; every bound is config, never a literal elsewhere."""
    lr_low: float = 1e-5
    lr_high: float = 1e-3
    gamma_low: float = 0.90
    gamma_high: float = 0.999
    batch_choices: Tuple[int, ...] = (32, 64, 256)
    ppo_n_steps_choices: Tuple[int, ...] = (64, 256, 1024)
    sac_buffer_choices: Tuple[int, ...] = (10_000, 100_000)

    def __post_init__(self):
        if not 0.0 < self.lr_low < self.lr_high:
            raise ValueError("learning-rate bounds must satisfy 0 < low < high")
        if not 0.0 < self.gamma_low < self.gamma_high <= 1.0:
            raise ValueError("gamma bounds must satisfy 0 < low < high <= 1")
        for choices in (self.batch_choices, self.ppo_n_steps_choices, self.sac_buffer_choices):
            if not choices or any(int(v) < 1 for v in choices):
                raise ValueError("choice lists must be non-empty positive ints")


@dataclass(frozen=True)
class ObjectiveWeights:
    """Explicit trade-offs in the scalarized trial value (all non-negative)."""
    success: float = 1.0
    fuel: float = 1.0
    overrides: float = 0.05
    risk: float = 2.0

    def __post_init__(self):
        if min(self.success, self.fuel, self.overrides, self.risk) < 0.0:
            raise ValueError("objective weights must be non-negative")


@dataclass(frozen=True)
class HPOConfig:
    algorithm: str = "PPO"
    n_trials: int = 10
    timeout_s: float = 0.0  # 0 means no wall-clock limit
    storage_path: str = "runs/hpo/study.db"
    study_name: str = "spaceforge_hpo"
    seed: int = 0  # TPE sampler seed: identical seeds replay identical suggestions
    eval_episodes: int = 5
    stage: str = "stage1_1d"
    output_dir: str = "runs/hpo"
    search: SearchSpace = SearchSpace()
    weights: ObjectiveWeights = ObjectiveWeights()

    def __post_init__(self):
        algorithm = str(self.algorithm).upper()
        object.__setattr__(self, "algorithm", algorithm)
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        if int(self.n_trials) < 1:
            raise ValueError("n_trials is a bounded budget of at least one trial")
        if float(self.timeout_s) < 0.0:
            raise ValueError("timeout_s must be non-negative (0 disables it)")
        if int(self.eval_episodes) < 1:
            raise ValueError("eval_episodes must be at least one")
        if not str(self.study_name):
            raise ValueError("study_name must be non-empty")


def objective_value(metrics: Dict[str, Any], weights: ObjectiveWeights) -> float:
    """Scalarize trial metrics; raises ValueError on missing/non-finite inputs."""
    values = {}
    for key in ("success_rate", "avg_fuel_used", "avg_overrides_per_episode", "avg_collision_risk"):
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"trial metrics must contain finite {key!r}") from exc
        if value != value or abs(value) == float("inf"):
            raise ValueError(f"trial metrics must contain finite {key!r}")
        values[key] = value
    return (weights.success * values["success_rate"]
            - weights.fuel * values["avg_fuel_used"]
            - weights.overrides * values["avg_overrides_per_episode"]
            - weights.risk * values["avg_collision_risk"])


def suggest_params(trial, config: HPOConfig) -> Dict[str, Any]:
    """Suggest one hyperparameter set from the explicit search space."""
    space = config.search
    params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", space.lr_low, space.lr_high, log=True),
        "gamma": trial.suggest_float("gamma", space.gamma_low, space.gamma_high),
        "batch_size": trial.suggest_categorical("batch_size", list(space.batch_choices)),
    }
    if config.algorithm == "PPO":
        params["n_steps"] = trial.suggest_categorical("n_steps", list(space.ppo_n_steps_choices))
    else:
        params["buffer_size"] = trial.suggest_categorical("buffer_size", list(space.sac_buffer_choices))
    return params


def run_study(config: HPOConfig, train_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    """Run (or resume) a bounded SQLite study.

    train_fn maps suggested params to metrics with keys success_rate,
    avg_fuel_used, avg_overrides_per_episode, avg_collision_risk. Exceptions
    mark that trial FAILED; the study continues with the remaining budget.
    Writes hpo_report.json and returns it.
    """
    optuna = require_optuna()
    storage_path = Path(config.storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=config.study_name,
        storage=f"sqlite:///{storage_path.resolve()}",
        load_if_exists=True,  # resumability: same name+storage continues
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.NopPruner(),  # every trial runs to completion; no partial pruning
    )

    def _objective(trial) -> float:
        params = suggest_params(trial, config)
        trial.set_user_attr("algorithm", config.algorithm)
        trial.set_user_attr("stage", config.stage)
        trial.set_user_attr("params", params)
        try:
            metrics = train_fn(dict(params))
            value = objective_value(metrics, config.weights)
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise
        trial.set_user_attr("metrics", {k: float(metrics[k]) for k in
                                        ("success_rate", "avg_fuel_used", "avg_overrides_per_episode",
                                         "avg_collision_risk")})
        trial.set_user_attr("objective_value", float(value))
        return float(value)

    # Total-budget accounting across resumes: n_trials bounds the whole study,
    # not each optimize call (Optuna counts per call). Re-running with the same
    # name+storage tops up to the budget instead of overshooting it.
    # catch=(Exception,) is explicit because Optuna >= 4 defaults to catch=()
    # (first failure aborts). Here a failed train_fn marks that trial FAILED
    # and the study spends the rest of its bounded budget.
    remaining = max(0, int(config.n_trials) - len(study.trials))
    if remaining:
        study.optimize(_objective, n_trials=remaining,
                       timeout=config.timeout_s if config.timeout_s > 0 else None,
                       catch=(Exception,))

    trials = [{
        "number": t.number,
        "state": t.state.name,
        "value": t.value,
        "params": dict(t.params),
        "user_attrs": {k: v for k, v in t.user_attrs.items()},
    } for t in study.trials]
    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    best = study.best_trial if complete else None
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_name": config.study_name,
        "storage": str(storage_path),
        "direction": "maximize",
        "algorithm": config.algorithm,
        "stage": config.stage,
        "budgets": {"n_trials": int(config.n_trials), "timeout_s": float(config.timeout_s)},
        "n_complete": len(complete),
        "n_failed": sum(1 for t in study.trials if t.state.name == "FAIL"),
        "best": ({"number": best.number, "value": best.value, "params": dict(best.params),
                  "user_attrs": {k: v for k, v in best.user_attrs.items()}} if best is not None else None),
        "trials": trials,
        "config": {"algorithm": config.algorithm, "seed": int(config.seed),
                   "eval_episodes": int(config.eval_episodes), "stage": config.stage,
                   "search": asdict(config.search),
                   "weights": asdict(config.weights)},
    }
    (output_dir / "hpo_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
