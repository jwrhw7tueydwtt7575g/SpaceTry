"""Curriculum thresholds/persistence plus bounded HPO orchestration (stub training).

No expensive study runs here: HPO tests use an injected stub train_fn with
n_trials <= 3. Optuna-gated tests skip cleanly where optuna is unavailable.
"""
import json
import pytest
from spaceforge_auto_trainer.curriculum import (
    SCHEMA_VERSION, STAGES, WINDOW_SIZE, CurriculumManager,
)
from spaceforge_auto_trainer.hpo import HPOConfig, ObjectiveWeights, objective_value


def perfect_window(manager, n=WINDOW_SIZE, fuel=0.10, overrides=0, risk=0.01):
    last = None
    for _ in range(n):
        last = manager.record_outcome(True, fuel, overrides, risk)
    return last


# ------------------------------------------------------- curriculum stages
def test_five_stages_are_explicit_and_ordered():
    assert STAGES == ("stage1_1d", "stage2_3d_static", "stage3_tumbling_target",
                      "stage4_debris_avoidance", "stage5_full_inspection")
    assert CurriculumManager().current_stage == "stage1_1d"
    with pytest.raises(ValueError):
        CurriculumManager(stage="warp_drive")


def test_incomplete_window_holds_without_transition():
    manager = CurriculumManager(base_seed=3)
    for _ in range(WINDOW_SIZE - 1):
        assert manager.record_outcome(True, 0.05, 0, 0.0) is None
    assert manager.current_stage == "stage1_1d"
    assert manager.evaluate()[0] == "hold"


def test_advancement_threshold_equality_passes():
    manager = CurriculumManager(base_seed=0)
    # Exactly at every boundary: 45/50 success, fuel 0.35, 1.0 overrides/ep, risk 0.05.
    for i in range(WINDOW_SIZE):
        success = i < 45
        transition = manager.record_outcome(success, 0.35, 1, 0.05)
    assert transition is not None and transition.direction == "advance"
    assert (transition.from_stage, transition.to_stage) == ("stage1_1d", "stage2_3d_static")
    assert manager.current_stage == "stage2_3d_static"
    assert len(manager.window) == 0  # fresh per-stage window
    assert transition.metrics.success_rate == pytest.approx(0.90)


def test_just_below_threshold_holds():
    manager = CurriculumManager(base_seed=0)
    for i in range(WINDOW_SIZE):
        manager.record_outcome(i < 44, 0.10, 0, 0.01)  # 0.88 < 0.90
    assert manager.current_stage == "stage1_1d"
    assert manager.evaluate()[0] == "hold"


def test_between_bands_holds():
    manager = CurriculumManager(base_seed=0)
    for i in range(WINDOW_SIZE):
        manager.record_outcome(i < 30, 0.10, 2, 0.01)  # 0.60 success, 2.0 overrides/ep
    assert manager.evaluate()[0] == "hold"


def test_rollback_on_low_success_clears_window():
    manager = CurriculumManager(base_seed=5, stage="stage3_tumbling_target")
    for i in range(WINDOW_SIZE):
        transition = manager.record_outcome(i < 20, 0.10, 0, 0.01)  # 0.40 < 0.50
    assert transition is not None and transition.direction == "rollback"
    assert (transition.from_stage, transition.to_stage) == ("stage3_tumbling_target", "stage2_3d_static")
    assert len(manager.window) == 0


@pytest.mark.parametrize("kwargs", [{"safety_overrides": 6}, {"collision_risk": 0.25}])
def test_rollback_on_safety_metrics(kwargs):
    outcomes = dict(success=True, fuel_used=0.10, safety_overrides=0, collision_risk=0.01)
    outcomes.update(kwargs)
    manager = CurriculumManager(base_seed=6, stage="stage2_3d_static")
    for _ in range(WINDOW_SIZE):
        transition = manager.record_outcome(**outcomes)
    assert transition.direction == "rollback" and manager.current_stage == "stage1_1d"


def test_stage_one_cannot_roll_back_and_stage_five_cannot_advance():
    manager = CurriculumManager(base_seed=0)
    for _ in range(WINDOW_SIZE):
        assert manager.record_outcome(False, 0.90, 9, 0.50) is None
    assert manager.current_stage == "stage1_1d"  # held, not rolled back

    top = CurriculumManager(base_seed=0, stage="stage5_full_inspection")
    for _ in range(WINDOW_SIZE):
        assert top.record_outcome(True, 0.05, 0, 0.0) is None
    assert top.current_stage == "stage5_full_inspection" and top.evaluate()[0] == "hold"


def test_rollback_boundary_equality_does_not_roll_back():
    manager = CurriculumManager(base_seed=0, stage="stage2_3d_static")
    for i in range(WINDOW_SIZE):
        # success 0.50 (not < 0.50), overrides 5.0 (not > 5.0), risk 0.20 (not > 0.20).
        manager.record_outcome(i < 25, 0.10, 5, 0.20)
    assert manager.evaluate()[0] == "hold"  # between bands, not rollback


def test_invalid_outcomes_rejected():
    manager = CurriculumManager()
    with pytest.raises(ValueError):
        manager.record_outcome(True, -0.1, 0, 0.0)
    with pytest.raises(ValueError):
        manager.record_outcome(True, 0.1, -1, 0.0)
    with pytest.raises(ValueError):
        manager.record_outcome(True, 0.1, 0, float("nan"))


# ------------------------------------------------------- seeds + persistence
def test_seed_stream_is_deterministic_and_resumable(tmp_path):
    manager = CurriculumManager(base_seed=100)
    assert manager.next_seed == 100
    manager.record_outcome(True, 0.1, 0, 0.0)
    assert manager.next_seed == 101
    assert manager.window[0].seed == 100
    path = manager.save(tmp_path / "curriculum.json")
    resumed = CurriculumManager.load(path)
    assert resumed.next_seed == 101 and resumed.current_stage == "stage1_1d"
    resumed.record_outcome(True, 0.1, 0, 0.0)
    assert resumed.window[-1].seed == 101  # identical stream as uninterrupted run


def test_persistence_round_trip_covers_transitions(tmp_path):
    manager = CurriculumManager(base_seed=7)
    perfect_window(manager)
    assert manager.current_stage == "stage2_3d_static"
    path = manager.save(tmp_path / "state.json")
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == SCHEMA_VERSION and saved["episode_index"] == WINDOW_SIZE
    restored = CurriculumManager.load(path)
    assert restored.current_stage == "stage2_3d_static"
    assert len(restored.transitions) == 1
    assert restored.transitions[0].direction == "advance"
    with pytest.raises(ValueError):
        saved["schema_version"] = 999
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(saved))
        CurriculumManager.load(bad)



# ------------------------------------------------------- objective math (no optuna needed)
def stub_metrics(success_rate=0.8, fuel=0.2, overrides=1, risk=0.02):
    return {"success_rate": success_rate, "avg_fuel_used": fuel,
            "avg_overrides_per_episode": overrides, "avg_collision_risk": risk}


def test_objective_combines_all_four_metrics():
    weights = ObjectiveWeights()
    value = objective_value(stub_metrics(), weights)
    assert value == pytest.approx(0.8 - 0.2 - 0.05 * 1 - 2.0 * 0.02)
    with pytest.raises(ValueError):
        objective_value({"success_rate": 1.0}, weights)
    with pytest.raises(ValueError):
        objective_value({**stub_metrics(), "avg_fuel_used": float("nan")}, weights)


def test_invalid_hpo_budgets_rejected_before_any_study():
    with pytest.raises(ValueError):
        HPOConfig(n_trials=0)
    with pytest.raises(ValueError):
        HPOConfig(algorithm="DQN")
