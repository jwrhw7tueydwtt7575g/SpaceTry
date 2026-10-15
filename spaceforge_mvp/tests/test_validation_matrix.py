"""Validation matrix smoke test: tiny run proving the report contract (offline)."""
from spaceforge_mvp.eval.validation_matrix import (
    OVERRIDES_THRESHOLD,
    SUCCESS_THRESHOLD,
    run_episodes,
    run_matrix,
)


def test_matrix_report_contract_on_two_episodes():
    report = run_matrix(episodes=2, seed=0, levels=["nominal"])
    assert report["schema_version"] == 1
    assert report["episodes_per_level"] == 2
    assert set(report["levels"]) == {"nominal"}
    nominal = report["levels"]["nominal"]
    assert nominal["episodes"] == 2
    assert 0.0 <= nominal["success_rate"] <= 1.0
    assert nominal["invariant_violations"] == 0
    for key in ("pose_error", "llm_validity", "curriculum", "sim2sim_gap"):
        assert report[key]["status"] in ("measured", "unavailable")
    assert report["thresholds"]["min_success_rate"] == SUCCESS_THRESHOLD
    assert report["thresholds"]["max_avg_overrides"] == OVERRIDES_THRESHOLD
    assert set(report["verdicts"]) == {
        "success_above_threshold", "fuel_within_budget", "no_invariant_violations"}
    assert any("Gazebo" in item for item in report["unverified"])


def test_episode_runner_counts_overrides_and_dv():
    metrics = run_episodes("nominal", episodes=3, seed=1)
    assert metrics["episodes"] == 3
    assert metrics["avg_dv_mps"] >= 0.0
    assert metrics["avg_overrides"] >= 0.0
    assert metrics["successes"] <= 3
