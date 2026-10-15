"""Replay mode: recorded JSON artifacts become versioned telemetry envelopes.

Sources (all optional, all offline, no ROS/Gazebo):
  scenario / telemetry traces: spaceforge_mvp/viz/scenarios/*.json and
    runs/*/telemetry.json — {summary, frames} with t/pos/vel/cmd/applied/
    override/dist/speed/fuel/dv per frame.
  training_report.json (rl_trainer): policy vs classical vs safety-filtered.
  hpo_report.json (auto_trainer): best-trial summary.
  DecisionTrace JSON (llm_planner): strategic trace.
  curriculum JSON (auto_trainer CurriculumManager.save): stage + window.

Mapping is documented per loader; unknown files raise ValueError instead of
guessing. Use merge_sorted for multi-source chronological playback.
"""
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from .schema import make_envelope

ZERO3 = (0.0, 0.0, 0.0)


def _read_json(path) -> Any:
    """Read JSON or raise ValueError (missing files are unknown files, not crashes)."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable replay file {path}: {exc}") from exc


def _frame_stamp_ns(frame: Dict[str, Any]) -> int:
    return int(float(frame.get("t", 0.0)) * 1_000_000_000)


def load_trace_frames(path, label: str = "", frame_id: str = "target_lvlh") -> List[Dict[str, Any]]:
    """Scenario/telemetry trace -> odometry/command/fuel/safety envelopes + summary.

    Works for both viz/scenarios/*.json and runs/*/telemetry.json: identical
    {summary, frames} shape from record_episode.
    """
    data = _read_json(path)
    if not isinstance(data, dict) or "frames" not in data or "summary" not in data:
        raise ValueError(f"not a trace file: {path}")
    summary = data["summary"]
    label = label or str(summary.get("label", Path(path).stem))
    events: List[Dict[str, Any]] = []
    for frame in data["frames"]:
        stamp = _frame_stamp_ns(frame)
        pos = tuple(float(v) for v in frame["pos"])
        vel = tuple(float(v) for v in frame["vel"])
        cmd = tuple(float(v) for v in frame.get("cmd", ZERO3))
        applied = tuple(float(v) for v in frame.get("applied", ZERO3))
        override = frame.get("override")
        fuel = float(frame.get("fuel", 1.0))
        events.append(make_envelope("odometry", {"position_m": pos, "velocity_mps": vel},
                                    stamp, frame_id))
        events.append(make_envelope("commanded", {"force_n": cmd, "torque_nm": ZERO3},
                                    stamp, frame_id))
        events.append(make_envelope("applied_command", {"force_n": applied, "torque_nm": ZERO3,
                                                        "override": override}, stamp, frame_id))
        events.append(make_envelope("fuel", {"fraction": fuel}, stamp))
        if override is not None:
            events.append(make_envelope("safety_event", {
                "event_id": f"{label}-{frame.get('t', 0)}",
                "goal_type": "APPROACH", "reason_code": str(override),
                "reason": f"interlock {override} at t={frame.get('t', 0)}s",
                "commanded_force_n": cmd, "applied_force_n": applied}, stamp))
    last_stamp = _frame_stamp_ns(data["frames"][-1]) if data["frames"] else 0
    events.append(make_envelope("episode_metrics", {
        "label": label, "seed": int(summary.get("seed", 0)), "steps": int(summary.get("steps", 0)),
        "start_dist_m": float(summary.get("start_dist", 0.0)),
        "end_dist_m": float(summary.get("end_dist", 0.0)),
        "end_speed_mps": float(summary.get("end_speed", 0.0)),
        "success": bool(summary.get("success", False)),
        "fuel_used": float(summary.get("fuel_used", 0.0)),
        "dv_mps": float(summary.get("dv", 0.0)),
        "safety_overrides": int(summary.get("overrides", 0)),
        "terminal_reason": str(summary.get("terminal_reason", ""))}, last_stamp))
    return events


def load_training_report(path) -> List[Dict[str, Any]]:
    """rl_trainer training_report.json -> per-controller metrics + tensorboard link."""
    data = _read_json(path)
    if "policy" not in data or "classical" not in data:
        raise ValueError(f"not a training report: {path}")
    events = []
    for key in ("policy", "classical", "safety_filtered"):
        metrics = data.get(key)
        if not isinstance(metrics, dict):
            continue
        events.append(make_envelope("episode_metrics", {
            "label": f"rl_trainer/{key}", "seed": int(metrics.get("seed", 0)),
            "steps": 0, "start_dist_m": 0.0, "end_dist_m": 0.0, "end_speed_mps": 0.0,
            "success": bool(metrics.get("success_rate", 0.0) >= 1.0),
            "success_rate": float(metrics.get("success_rate", 0.0)),
            "fuel_used": float(metrics.get("avg_fuel_used", 0.0)),
            "safety_overrides": int(metrics.get("safety_overrides", metrics.get("overrides", 0))),
            "terminal_reason": "", "episodes": int(metrics.get("episodes", 0))}, 0))
    checkpoint = str(data.get("checkpoint", ""))
    if checkpoint:
        events.append(make_envelope("tensorboard", {
            "url": "http://localhost:6006/", "run_dir": str(Path(checkpoint).parent)}, 0))
    return events


def load_hpo_report(path) -> List[Dict[str, Any]]:
    """auto_trainer hpo_report.json -> best-trial metrics envelope."""
    data = _read_json(path)
    best = data.get("best")
    if not isinstance(best, dict):
        raise ValueError(f"no best trial in HPO report: {path}")
    return [make_envelope("episode_metrics", {
        "label": f"hpo-best/{data.get('algorithm', '')}", "seed": 0, "steps": 0,
        "start_dist_m": 0.0, "end_dist_m": 0.0, "end_speed_mps": 0.0,
        "success": False, "fuel_used": 0.0, "safety_overrides": 0, "terminal_reason": "",
        "objective_value": float(best.get("value", 0.0) or 0.0),
        "best_params": dict(best.get("params", {}))}, 0)]


def load_decision_trace(path_or_dict) -> List[Dict[str, Any]]:
    """llm_planner DecisionTrace JSON -> redacted llm_trace envelope."""
    from .schema import redact
    data = path_or_dict if isinstance(path_or_dict, dict) else _read_json(path_or_dict)
    if "trace_id" not in data:
        raise ValueError("not a decision trace")
    payload = {"trace_id": str(data["trace_id"]),
               "prompt_version": str(data.get("prompt_version", "")),
               "schema_version": str(data.get("schema_version", "")),
               "provider": str(data.get("provider", "")),
               "metadata": redact(dict(data.get("metadata", {}))),
               "status": str(data.get("mission_status", data.get("parse_status", ""))),
               "goal_type": "", "reason": str(data.get("mission_reason", data.get("failure", ""))),
               "token_count": int(data.get("token_count", 0)),
               "cost_usd": float(data.get("cost_usd", 0.0)),
               "attempts": int(len(data.get("attempts", [])))}
    return [make_envelope("llm_trace", redact(payload), 0)]


def load_curriculum_state(path) -> List[Dict[str, Any]]:
    """auto_trainer curriculum JSON -> curriculum_state envelope."""
    data = _read_json(path)
    if data.get("schema_version") != 1 or "current_stage" not in data:
        raise ValueError(f"not a curriculum state file: {path}")
    outcomes = data.get("outcomes", [])
    count = len(outcomes)
    payload = {"current_stage": str(data["current_stage"]),
               "episode_index": int(data.get("episode_index", 0)),
               "window_episodes": count,
               "success_rate": (sum(1 for o in outcomes if o.get("success")) / count) if count else 0.0,
               "transitions": len(data.get("transitions", []))}
    return [make_envelope("curriculum_state", payload, 0)]


def merge_sorted(*event_lists: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chronological merge (stable) across sources for multi-pane playback."""
    merged = [e for events in event_lists for e in events]
    merged.sort(key=lambda e: (e["stamp_ns"], e["stream"]))
    return merged


class ReplayPlayer:
    """Timed iterator over envelopes for dashboard testing without ROS."""

    def __init__(self, events: List[Dict[str, Any]], speed: float = 1.0):
        if speed <= 0:
            raise ValueError("replay speed must be positive")
        self.events = list(events)
        self.speed = float(speed)
        self.position = 0

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterable[Tuple[float, Dict[str, Any]]]:
        """Yield (delay_seconds, envelope) honoring original timestamps / speed."""
        previous = self.events[0]["stamp_ns"] if self.events else 0
        for envelope in self.events:
            delay = max(0.0, (envelope["stamp_ns"] - previous) / 1e9 / self.speed)
            previous = envelope["stamp_ns"]
            yield delay, envelope

    def drain(self) -> List[Dict[str, Any]]:
        """All remaining envelopes without timing (tests, thumbnails)."""
        remaining = self.events[self.position:]
        self.position = len(self.events)
        return remaining
