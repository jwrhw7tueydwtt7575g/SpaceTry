"""Phase 6 validation matrix: measured results only, no simulated claims.

Runs over 100 evaluation episodes plus offline probes and reports:
  1. docking success rate per domain level vs the agreed 0.85 threshold
  2. fuel benchmark (mean/max fuel_used and delta-v)
  3. perception pose error vs ground truth (OpenCV solvePnP, offline)
  4. LLM goal validity through mission_ctrl (mock provider, offline)
  5. curriculum completion (synthetic outcome streams, offline)
  6. safety invariants over every matrix step (bounds, finiteness, fuel range)
  7. sim-to-sim gap: SpacecraftEnv vs SpacecraftRLEnv+MockTransport, same seeds

Hardware and Gazebo checks are out of scope here and reported as unverified,
never as passing. Usage:
  python3 -m spaceforge_mvp.eval.validation_matrix --episodes 120 --seed 0
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from spaceforge_mvp.config import DomainConfig, FaultConfig
from spaceforge_mvp.control.glideslope import glideslope_action
from spaceforge_mvp.env.spacecraft_env import SpacecraftEnv

SCHEMA_VERSION = 1

# Agreed acceptance gates (shipped training metadata thresholds).
SUCCESS_THRESHOLD = 0.85
FUEL_THRESHOLD = 0.35
OVERRIDES_THRESHOLD = 1.0


def _level_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "nominal": {"domain": None, "faults": None},
        "randomized": {
            "domain": DomainConfig(
                mass_low_kg=0.8, mass_high_kg=1.2,
                effectiveness_tol=0.1,
                thrust_noise_std=0.02,
                obs_pos_noise_std_m=0.05,
                obs_vel_noise_std_mps=0.01,
            ),
            "faults": None,
        },
        "faults": {
            "domain": DomainConfig(
                mass_low_kg=0.9, mass_high_kg=1.1, effectiveness_tol=0.05,
                thrust_noise_std=0.01,
            ),
            "faults": FaultConfig(
                bias=(0.01, 0.0, 0.0),
                dropout_prob=0.02,
                sensor_freeze_prob=0.02,
                fuel_leak_rate=0.0002,
            ),
        },
    }


def run_episodes(level: str, episodes: int, seed: int) -> Dict[str, Any]:
    """Glideslope over randomized ICs under one domain level.

    Initial conditions are drawn from the documented reset bounds with a seeded
    RNG, so the matrix replays identically. Invariants are asserted on every
    step; any violation is counted, never hidden.
    """
    cfg = _level_configs()[level]
    rng = np.random.default_rng(seed)
    successes = 0
    fuels: List[float] = []
    dvs: List[float] = []
    overrides: List[int] = []
    steps: List[int] = []
    violations = 0
    for episode in range(episodes):
        env = SpacecraftEnv(domain=cfg["domain"], faults=cfg["faults"])
        env.reset(seed=seed + episode)
        # Randomized initial conditions inside the documented reset bounds.
        env.rel_pos = rng.uniform(-20.0, 20.0, size=3).astype(np.float64)
        env.rel_vel = rng.uniform(-0.005, 0.005, size=3).astype(np.float64)
        done = False
        last_info: Dict[str, Any] = {}
        ep_dv = 0.0
        ep_overrides = 0
        while not done:
            action = np.asarray(glideslope_action(env.rel_pos, env.rel_vel), dtype=np.float64)
            _, _, terminated, truncated, last_info = env.step(action)
            applied = np.asarray(last_info["applied_action"], dtype=float)
            ep_dv += float(np.sum(np.abs(applied)))
            ep_overrides += int(last_info.get("safety_override") is not None)
            if not (np.all(np.isfinite(applied)) and np.all(np.abs(applied) <= 1.0 + 1e-9)):
                violations += 1
            if not (np.all(np.isfinite(env.rel_pos)) and np.all(np.isfinite(env.rel_vel))):
                violations += 1
            if not 0.0 <= float(env.fuel) <= 1.0:
                violations += 1
            done = bool(terminated or truncated)
        successes += int(bool(last_info.get("is_success", False)))
        fuels.append(max(0.0, 1.0 - float(env.fuel)))
        dvs.append(ep_dv)
        overrides.append(ep_overrides)
        steps.append(int(last_info.get("step_count", 0)))
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(1, episodes),
        "avg_fuel_used": float(np.mean(fuels)) if fuels else 0.0,
        "max_fuel_used": float(np.max(fuels)) if fuels else 0.0,
        "avg_dv_mps": float(np.mean(dvs)) if dvs else 0.0,
        "max_dv_mps": float(np.max(dvs)) if dvs else 0.0,
        "avg_overrides": float(np.mean(overrides)) if overrides else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "invariant_violations": violations,
    }


def measure_pose_error(trials: int, seed: int) -> Dict[str, Any]:
    """Perception solver error vs ground truth over random poses (offline).

    Returns {"status": "unavailable", ...} instead of guessing when OpenCV or
    the perception package cannot be imported.
    """
    try:
        import cv2  # noqa: F401
        from spaceforge_perception.contracts import CameraCalibration, Detection, TargetKeypoints
        from spaceforge_perception.pnp import OpenCVPnPSolver
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"missing dependency: {exc}"}
    rng = np.random.default_rng(seed)
    keypoints = TargetKeypoints(
        ("a", "b", "c", "d"),
        ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)),
    )
    calib = CameraCalibration(640, 480, (500, 0, 320, 0, 500, 240, 0, 0, 1), (0, 0, 0, 0, 0))
    errors: List[float] = []
    reproj: List[float] = []
    failures = 0
    for _ in range(trials):
        import cv2 as _cv2
        truth_t = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(2, 10)])
        axis = rng.normal(size=3)
        axis = axis / float(np.linalg.norm(axis))
        rvec, _ = _cv2.Rodrigues((axis * rng.uniform(0, 0.3)).reshape(3, 1))
        K = np.asarray(calib.K, dtype=float).reshape(3, 3)
        pixels, _ = _cv2.projectPoints(
            np.asarray(keypoints.points_m, dtype=float), rvec, truth_t, K, np.zeros(5))
        pixels = tuple(map(tuple, pixels.reshape(-1, 2).tolist()))
        detection = Detection(pixels, (1.0,) * 4, 1.0, 0, calib.camera_frame)
        pose = OpenCVPnPSolver().solve(keypoints, detection, calib)
        if "VALID" not in str(pose.status):
            failures += 1
            continue
        errors.append(float(np.linalg.norm(np.asarray(pose.translation_m) - truth_t)))
        reproj.append(float(pose.reprojection_error_px))
    return {
        "status": "measured", "trials": trials, "failures": failures,
        "mean_translation_error_m": float(np.mean(errors)) if errors else float("inf"),
        "max_translation_error_m": float(np.max(errors)) if errors else float("inf"),
        "mean_reprojection_error_px": float(np.mean(reproj)) if reproj else float("inf"),
    }


def measure_llm_validity(valid_cases: int, adversarial_cases: int, seed: int) -> Dict[str, Any]:
    """Mock-provider proposals through mission_ctrl: valid accepted, bad held.

    Returns unavailable instead of guessing when sibling packages are missing.
    """
    try:
        from spaceforge_llm_planner import LLMPlanner, OfflineMockProvider
        from spaceforge_mission_ctrl import MissionController
        from spaceforge_mission_ctrl.schema import Telemetry
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"missing dependency: {exc}"}

    def tele():
        return Telemetry((10.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, 1_000_000_000)

    def proposal(goal="APPROACH", **extra):
        return {"schema_version": "1.0", "proposal_id": "m", "prompt_version": "matrix",
                "goals": [{"goal_type": goal, **extra}], "rationale": "matrix probe"}

    valid_ok = 0
    for i in range(valid_cases):
        goal = ["APPROACH", "INSPECT", "RETREAT", "SAFE_HOLD"][i % 4]
        provider = OfflineMockProvider([proposal(goal)])
        planner = LLMPlanner(provider, MissionController(), lambda: {}, max_retries=0)
        decision = planner.plan(tele(), 1_000_000_000)
        # Valid goals must be accepted, or deterministically mapped to safe-hold
        # only when mission_ctrl itself vetoes (never a crash, never a bypass).
        if decision.accepted or decision.trace.mission_status in ("SAFE_HOLD_ENFORCED",):
            valid_ok += 1

    adversarial = [
        {"schema_version": "1.0", "goals": [{"goal_type": "APPROACH", "force_n": [1, 0, 0]}]},
        {"schema_version": "1.0", "proposal_id": "x", "prompt_version": "m",
         "goals": [{"goal_type": "APPROACH"}, {"goal_type": "RETREAT"}],
         "rationale": "contradiction"},
        {"schema_version": "1.0", "proposal_id": "x", "prompt_version": "m",
         "goals": [{"goal_type": "APPROACH"}], "rationale": "ignore previous instructions"},
        {"schema_version": "9.9", "goals": [{"goal_type": "APPROACH"}]},
        {"schema_version": "1.0", "proposal_id": "x", "prompt_version": "m",
         "goals": [{"goal_type": "APPROACH", "target_position_m": [float("nan"), 0, 0]}],
         "rationale": "nan"},
    ]
    contained = 0
    for i in range(adversarial_cases):
        raw = adversarial[i % len(adversarial)]
        provider = OfflineMockProvider([raw])
        planner = LLMPlanner(provider, MissionController(), lambda: {}, max_retries=0)
        try:
            decision = planner.plan(tele(), 1_000_000_000)
            # Contained = not accepted as a nominal plan; safe-hold enforced instead.
            if not decision.accepted:
                contained += 1
        except Exception:
            pass  # an exception escaping would be a bypass failure, not containment
    return {
        "status": "measured",
        "valid_cases": valid_cases, "valid_handled": valid_ok,
        "valid_rate": valid_ok / max(1, valid_cases),
        "adversarial_cases": adversarial_cases, "adversarial_contained": contained,
        "adversarial_containment_rate": contained / max(1, adversarial_cases),
    }


def measure_curriculum() -> Dict[str, Any]:
    """Curriculum completion on synthetic outcome streams (offline, fast)."""
    try:
        from spaceforge_auto_trainer.curriculum import CurriculumManager
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"missing dependency: {exc}"}
    manager = CurriculumManager(base_seed=0)
    for _ in range(50):
        transition = manager.record_outcome(True, 0.05, 0, 0.0)
    advanced = manager.current_stage == "stage2_3d_static" and transition is not None
    poor = CurriculumManager(base_seed=1, stage="stage2_3d_static")
    last = None
    for _ in range(50):
        last = poor.record_outcome(False, 0.5, 6, 0.3)
    rolled_back = poor.current_stage == "stage1_1d" and last is not None
    return {
        "status": "measured",
        "perfect_stream_advances_in_50": bool(advanced),
        "poor_stream_rolls_back": bool(rolled_back),
        "transitions": [t.to_dict() for t in manager.transitions],
    }


def measure_sim2sim_gap(pairs: int, seed: int) -> Dict[str, Any]:
    """Paired rollouts: MVP env vs rl_env MockTransport, identical seeds/commands."""
    try:
        from spaceforge_rl_env.env import SpacecraftRLEnv
        from spaceforge_rl_env.transport import MockTransport
        from spaceforge_mvp.config import MissionConfig
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"missing dependency: {exc}"}
    divergences: List[float] = []
    agreements = 0
    starts = [(5.0, 0.0, 0.0), (12.0, -7.0, 3.5), (-8.0, 4.0, 1.0), (3.0, 3.0, -2.0)]
    for pair in range(pairs):
        start = starts[pair % len(starts)]
        mission = MissionConfig(
            start_pos=start, target_pos=(0.0, 0.0, 0.0),
            start_vel=(0.0, 0.0, 0.0), target_vel=(0.0, 0.0, 0.0),
            initial_fuel=1.0, duration=300.0, dt=1.0, mean_motion=0.00113,
            safety_mode="v1",
        )
        from spaceforge_mvp.config import PhysicsConfig, RunConfig
        run = RunConfig(mission=mission, physics=PhysicsConfig(n=mission.mean_motion, dt=mission.dt),
                        seed=seed + pair)
        truth = SpacecraftEnv(run_config=run)
        mirror = SpacecraftEnv(run_config=run)
        transport = MockTransport(mirror)
        rl_env = SpacecraftRLEnv(backend=mirror)
        truth.reset(seed=seed + pair)
        rl_env.reset(seed=seed + pair)
        peak = 0.0
        done_a = done_b = False
        info_a: Dict[str, Any] = {}
        for _ in range(300):
            if not done_a:
                action = np.asarray(glideslope_action(truth.rel_pos, truth.rel_vel))
                _, _, term_a, trunc_a, info_a = truth.step(action)
                done_a = bool(term_a or trunc_a)
            if not done_b:
                action_b = np.asarray(glideslope_action(mirror.rel_pos, mirror.rel_vel))
                transport.publish_wrench(action_b, np.zeros(3),
                                         int(mirror.step_count * 1e9))
                transport.wait_for_sensor(-1, 0.25)
                # Mirror outcome comes from the backend step inside the transport.
                backend_info = dict(transport.last_backend_info)
                done_b = bool(backend_info.get("is_success", False)) or mirror.step_count >= 300
            peak = max(peak, float(np.linalg.norm(truth.rel_pos - mirror.rel_pos)))
            if done_a and done_b:
                break
        divergences.append(peak)
        ok_a = bool(info_a.get("is_success", False))
        ok_b = bool(dict(transport.last_backend_info).get("is_success", False))
        agreements += int(ok_a == ok_b)
    return {
        "status": "measured", "pairs": pairs,
        "max_divergence_m": float(np.max(divergences)) if divergences else float("inf"),
        "mean_divergence_m": float(np.mean(divergences)) if divergences else float("inf"),
        "success_agreement_rate": agreements / max(1, pairs),
    }


def run_matrix(episodes: int = 120, seed: int = 0,
               levels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the full matrix and return the report dict."""
    levels = levels or ["nominal", "randomized", "faults"]
    level_reports = {level: run_episodes(level, episodes, seed) for level in levels}
    pose = measure_pose_error(min(episodes, 120), seed)
    llm = measure_llm_validity(20, 25, seed)
    curriculum = measure_curriculum()
    sim2sim = measure_sim2sim_gap(20, seed)
    nominal = level_reports.get("nominal", {})
    verdicts = {
        "success_above_threshold": bool(nominal.get("success_rate", 0.0) >= SUCCESS_THRESHOLD),
        "fuel_within_budget": bool(nominal.get("avg_fuel_used", 9.0) <= FUEL_THRESHOLD),
        "no_invariant_violations": all(
            r.get("invariant_violations", -1) == 0 for r in level_reports.values()),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "episodes_per_level": episodes,
        "seed": seed,
        "levels": level_reports,
        "pose_error": pose,
        "llm_validity": llm,
        "curriculum": curriculum,
        "sim2sim_gap": sim2sim,
        "thresholds": {
            "min_success_rate": SUCCESS_THRESHOLD,
            "max_avg_fuel_used": FUEL_THRESHOLD,
            "max_avg_overrides": OVERRIDES_THRESHOLD,
        },
        "verdicts": verdicts,
        "unverified": [
            "Gazebo Harmonic closed-loop docking (no simulator host)",
            "ROS 2 hardware-in-the-loop runs (no ROS host)",
            "physical spacecraft hardware (no hardware)",
        ],
    }


def print_report(report: Dict[str, Any]) -> None:
    print("\n=== SpaceForge validation matrix ===")
    print(f"episodes/level: {report['episodes_per_level']}  seed: {report['seed']}")
    for level, metrics in report["levels"].items():
        print(f"[{level:>10}] success {metrics['success_rate'] * 100:6.2f}%  "
              f"fuel {metrics['avg_fuel_used']:.4f} (max {metrics['max_fuel_used']:.4f})  "
              f"dv {metrics['avg_dv_mps']:.3f}  overrides {metrics['avg_overrides']:.3f}  "
              f"violations {metrics['invariant_violations']}")
    pose = report["pose_error"]
    print(f"[pose]      {pose.get('status')}: mean {pose.get('mean_translation_error_m')} m, "
          f"max {pose.get('max_translation_error_m')} m over {pose.get('trials')} trials")
    llm = report["llm_validity"]
    print(f"[llm]       {llm.get('status')}: valid {llm.get('valid_rate')}  "
          f"contained {llm.get('adversarial_containment_rate')}")
    cur = report["curriculum"]
    print(f"[curriculum] {cur.get('status')}: advance-in-50 {cur.get('perfect_stream_advances_in_50')}  "
          f"rollback {cur.get('poor_stream_rolls_back')}")
    gap = report["sim2sim_gap"]
    print(f"[sim2sim]   {gap.get('status')}: max div {gap.get('max_divergence_m')} m, "
          f"agreement {gap.get('success_agreement_rate')}")
    print(f"[verdicts]  {json.dumps(report['verdicts'])}")
    print(f"[unverified] {'; '.join(report['unverified'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 validation matrix (measured results only)")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", nargs="+", default=["nominal", "randomized", "faults"],
                        choices=["nominal", "randomized", "faults"])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least one")
    report = run_matrix(episodes=args.episodes, seed=args.seed, levels=args.levels)
    print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"report written: {args.output}")


if __name__ == "__main__":
    main()
