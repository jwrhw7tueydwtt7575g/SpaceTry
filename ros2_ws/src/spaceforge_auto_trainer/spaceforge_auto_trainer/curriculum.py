"""Deterministic five-stage curriculum manager with persisted rolling windows.

Contract (AUTO_TRAINER_CONTRACT.md), encoded without hidden constants:
  stages: stage1_1d -> stage2_3d_static -> stage3_tumbling_target ->
    stage4_debris_avoidance -> stage5_full_inspection.
  window: most recent WINDOW_SIZE (50) episode outcomes for the current stage.
  advance: window full AND success_rate >= 0.90 AND avg fuel <= 0.35 AND
    avg overrides <= 1.0/ep AND avg collision_risk <= 0.05 (equality passes).
  hold: incomplete windows, or metrics between the advance and rollback bands.
  rollback: one stage back when success_rate < 0.50 OR avg overrides > 5.0 OR
    avg collision_risk > 0.20. Checked before advancement (safety first);
    clears the window. Stage 1 cannot roll back; stage 5 cannot advance.
  reproducibility: episode seeds are base_seed + episode_index, persisted, so
    resume and replay produce identical streams.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

#: Persisted-state schema version; bump on any format change.
SCHEMA_VERSION = 1

#: Rolling window size: advancement/rollback read exactly this many outcomes.
WINDOW_SIZE = 50

#: The five curriculum stages, in order. Stable: never reorder or rename.
STAGES: Tuple[str, ...] = (
    "stage1_1d",
    "stage2_3d_static",
    "stage3_tumbling_target",
    "stage4_debris_avoidance",
    "stage5_full_inspection",
)

STAGE_DESCRIPTIONS: Dict[str, str] = {
    "stage1_1d": "1-D approach along a single axis at rest",
    "stage2_3d_static": "3-D static rendezvous from an offset start",
    "stage3_tumbling_target": "rendezvous with a rotating target",
    "stage4_debris_avoidance": "approach with debris keep-out constraints",
    "stage5_full_inspection": "full inspection orbit and station-keeping",
}


@dataclass(frozen=True)
class AdvancementThresholds:
    min_success_rate: float = 0.90
    max_fuel_used: float = 0.35
    max_overrides_per_episode: float = 1.0
    max_collision_risk: float = 0.05


@dataclass(frozen=True)
class RollbackThresholds:
    min_success_rate: float = 0.50
    max_overrides_per_episode: float = 5.0
    max_collision_risk: float = 0.20


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    fuel_used: float
    safety_overrides: int
    collision_risk: float
    seed: int
    stage: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeOutcome":
        return cls(bool(data["success"]), float(data["fuel_used"]), int(data["safety_overrides"]),
                   float(data["collision_risk"]), int(data["seed"]), str(data["stage"]))


@dataclass(frozen=True)
class WindowMetrics:
    episodes: int
    success_rate: float
    avg_fuel_used: float
    avg_overrides_per_episode: float
    avg_collision_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageTransition:
    from_stage: str
    to_stage: str
    direction: str  # "advance" | "rollback"
    episode_index: int
    metrics: WindowMetrics

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageTransition":
        metrics = data["metrics"]
        return cls(str(data["from_stage"]), str(data["to_stage"]), str(data["direction"]),
                   int(data["episode_index"]), WindowMetrics(int(metrics["episodes"]), float(metrics["success_rate"]),
                   float(metrics["avg_fuel_used"]), float(metrics["avg_overrides_per_episode"]),
                   float(metrics["avg_collision_risk"])))


class CurriculumManager:
    """Rolling-window curriculum state machine; deterministic and persisted."""

    def __init__(
        self,
        base_seed: int = 0,
        stage: str = STAGES[0],
        advance: AdvancementThresholds = AdvancementThresholds(),
        rollback: RollbackThresholds = RollbackThresholds(),
        window_size: int = WINDOW_SIZE,
    ) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown curriculum stage: {stage!r}")
        if int(window_size) < 1:
            raise ValueError("window_size must be at least one")
        self.base_seed = int(base_seed)
        self._stage = stage
        self.advance_thresholds = advance
        self.rollback_thresholds = rollback
        self.window_size = int(window_size)
        self._outcomes: List[EpisodeOutcome] = []
        self._episode_index = 0
        self._transitions: List[StageTransition] = []

    # ------------------------------------------------------------ accessors
    @property
    def current_stage(self) -> str:
        return self._stage

    @property
    def stage_index(self) -> int:
        return STAGES.index(self._stage)

    @property
    def episode_index(self) -> int:
        return self._episode_index

    @property
    def next_seed(self) -> int:
        """Seed for the next episode: base_seed + episode_index."""
        return self.base_seed + self._episode_index

    @property
    def window(self) -> Tuple[EpisodeOutcome, ...]:
        return tuple(self._outcomes)

    @property
    def transitions(self) -> Tuple[StageTransition, ...]:
        return tuple(self._transitions)

    # ------------------------------------------------------------ recording
    def record_outcome(self, success: bool, fuel_used: float, safety_overrides: int,
                       collision_risk: float) -> Optional[StageTransition]:
        """Record one episode outcome; returns the transition if the window fired."""
        for name, value in (("fuel_used", fuel_used), ("collision_risk", collision_risk)):
            if not isinstance(value, (int, float)) or not (value == value and abs(value) != float("inf")) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if int(safety_overrides) < 0:
            raise ValueError("safety_overrides must be non-negative")
        outcome = EpisodeOutcome(bool(success), float(fuel_used), int(safety_overrides),
                                 float(collision_risk), self.next_seed, self._stage)
        self._outcomes.append(outcome)
        # Keep only the most recent window; older outcomes never influence decisions.
        if len(self._outcomes) > self.window_size:
            del self._outcomes[:-self.window_size]
        self._episode_index += 1
        return self._maybe_transition()

    def window_metrics(self) -> Optional[WindowMetrics]:
        if not self._outcomes:
            return None
        count = len(self._outcomes)
        return WindowMetrics(
            episodes=count,
            success_rate=sum(1 for o in self._outcomes if o.success) / count,
            avg_fuel_used=sum(o.fuel_used for o in self._outcomes) / count,
            avg_overrides_per_episode=sum(o.safety_overrides for o in self._outcomes) / count,
            avg_collision_risk=sum(o.collision_risk for o in self._outcomes) / count,
        )

    def evaluate(self) -> Tuple[str, Optional[WindowMetrics]]:
        """Decide advance|rollback|hold for the current window without mutating."""
        if len(self._outcomes) < self.window_size:
            return "hold", self.window_metrics()
        metrics = self.window_metrics()
        assert metrics is not None
        rollback = self.rollback_thresholds
        if (metrics.success_rate < rollback.min_success_rate
                or metrics.avg_overrides_per_episode > rollback.max_overrides_per_episode
                or metrics.avg_collision_risk > rollback.max_collision_risk):
            if self.stage_index > 0:
                return "rollback", metrics
            return "hold", metrics  # stage 1 cannot roll back: hold and re-fill
        advance = self.advance_thresholds
        if (self.stage_index < len(STAGES) - 1
                and metrics.success_rate >= advance.min_success_rate
                and metrics.avg_fuel_used <= advance.max_fuel_used
                and metrics.avg_overrides_per_episode <= advance.max_overrides_per_episode
                and metrics.avg_collision_risk <= advance.max_collision_risk):
            return "advance", metrics
        return "hold", metrics  # stage 5 at ceiling, or between the bands

    def _maybe_transition(self) -> Optional[StageTransition]:
        decision, metrics = self.evaluate()
        if decision == "hold" or metrics is None:
            return None
        from_stage = self._stage
        to_stage = STAGES[self.stage_index + 1] if decision == "advance" else STAGES[self.stage_index - 1]
        transition = StageTransition(from_stage, to_stage, decision, self._episode_index, metrics)
        self._stage = to_stage
        self._outcomes = []  # fresh per-stage window after any transition
        self._transitions.append(transition)
        return transition

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "base_seed": self.base_seed,
            "episode_index": self._episode_index,
            "current_stage": self._stage,
            "window_size": self.window_size,
            "advance_thresholds": asdict(self.advance_thresholds),
            "rollback_thresholds": asdict(self.rollback_thresholds),
            "outcomes": [o.to_dict() for o in self._outcomes],
            "transitions": [t.to_dict() for t in self._transitions],
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path) -> "CurriculumManager":
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported curriculum schema_version: {data.get('schema_version')!r}")
        manager = cls(base_seed=int(data["base_seed"]), stage=str(data["current_stage"]),
                      advance=AdvancementThresholds(**data.get("advance_thresholds", {})),
                      rollback=RollbackThresholds(**data.get("rollback_thresholds", {})),
                      window_size=int(data.get("window_size", WINDOW_SIZE)))
        manager._episode_index = int(data["episode_index"])
        manager._outcomes = [EpisodeOutcome.from_dict(o) for o in data.get("outcomes", [])]
        manager._transitions = [StageTransition.from_dict(t) for t in data.get("transitions", [])]
        return manager
