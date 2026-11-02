"""Build the recorded scenarios that drive the videos and the dashboard.

Run:  python -m spaceforge_mvp.viz.scenarios
Writes JSON traces to spaceforge_mvp/viz/scenarios/ and prints the measured
outcome of every scenario, so nothing shown in a video is hand-drawn.
"""

from typing import Any, Dict

from spaceforge_mvp.control.glideslope import glideslope_action, hot_approach_action
from spaceforge_mvp.viz.record import (
    SCENARIO_DIR,
    record_episode,
    rl_controller,
    write_episode,
)

CHECKPOINT = "checkpoints/ppo_spacecraft_final.zip"

# A deliberately awkward start: 3-D offset, at rest, ~14 m out.
DOCK_START = ([12.0, -7.0, 3.5], [0.0, 0.0, 0.0])
# The reckless case: 6 m out, at rest, full thrust straight at the target.
HOT_START = ([6.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def build(write: bool = True) -> Dict[str, Dict[str, Any]]:
    traces: Dict[str, Dict[str, Any]] = {}

    # 1. The certified glideslope docking — the hero run.
    traces["docking"] = record_episode(
        glideslope_action, seed=42, label="Certified glideslope",
        initial=DOCK_START, safety="v1",
    )

    # 2. The shipped RL checkpoint, either under its own steam or with the
    #    certified controller holding the stick.
    traces["rl_policy"] = record_episode(
        rl_controller(CHECKPOINT), seed=42, label="Shipped PPO checkpoint",
        safety="v1", annotate_continuous=False,
    )

    # 3. Reckless slam, no runtime assurance at all: powered penetration.
    traces["hot_open"] = record_episode(
        hot_approach_action, seed=7, label="Reckless slam, no interlock",
        initial=HOT_START, safety="off", max_steps=120, annotate_continuous=True,
    )

    # 4. Same slam under the MVP interlock: veto = zero thrust.
    traces["hot_v1"] = record_episode(
        hot_approach_action, seed=7, label="Reckless slam, interlock v1 (zero thrust)",
        initial=HOT_START, safety="v1", max_steps=120, annotate_continuous=True,
    )

    # 5. Same slam under runtime assurance v2: predictive envelope filter with
    #    the certified glideslope as backup controller.
    traces["hot_v2"] = record_episode(
        hot_approach_action, seed=7, label="Reckless slam, RTA v2 (predictive filter + certified backup)",
        initial=HOT_START, safety="v2", max_steps=120, annotate_continuous=True,
    )

    if write:
        SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
        for key, trace in traces.items():
            write_episode(trace, SCENARIO_DIR / f"{key}.json")

    return traces


def _report(traces: Dict[str, Dict[str, Any]]) -> None:
    print(f"{'scenario':<12} {'start':>8} {'sampled':>9} {'continuous':>11} {'v@pass':>8} "
          f"{'steps':>6} {'dv':>7} {'fires':>6}  outcome")
    print("-" * 112)
    for key, trace in traces.items():
        s = trace["summary"]
        cont = s.get("continuous_min_dist")
        cont_s = f"{cont * 100:>8.2f}cm" if cont is not None else "         -"
        vpass = s.get("continuous_min_speed")
        vpass_s = f"{vpass:>6.2f} m/s" if vpass is not None else "         -"
        outcome = s["terminal_reason"]
        if s.get("penetrated"):
            outcome += "  (fly-through)"
        print(f"{key:<12} {s['start_dist']:>7.2f}m {s['min_dist']:>8.3f}m {cont_s:>11} {vpass_s:>10} "
              f"{s['steps']:>6} {s['dv']:>7.3f} {s['overrides']:>6}  {outcome}")


if __name__ == "__main__":
    traces = build()
    _report(traces)
    print(f"\ntraces written to {SCENARIO_DIR}")
