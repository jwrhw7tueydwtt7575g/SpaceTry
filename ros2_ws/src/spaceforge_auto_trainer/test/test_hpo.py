"""Bounded Optuna orchestration with stub training (no expensive studies).

Every test uses n_trials <= 3 and a pure-Python stub train_fn: no SB3, no
Gazebo, no network. Skipped cleanly where optuna is unavailable.
"""
import json
import pytest

optuna = pytest.importorskip("optuna", reason="optuna unavailable; HPO orchestration not claimed")

from spaceforge_auto_trainer.hpo import HPOConfig, ObjectiveWeights, objective_value, run_study


def stub_metrics(success_rate=0.8, fuel=0.2, overrides=1, risk=0.02):
    return {"success_rate": success_rate, "avg_fuel_used": fuel,
            "avg_overrides_per_episode": overrides, "avg_collision_risk": risk}


def test_bounded_budget_and_artifact_metadata(tmp_path):
    config = HPOConfig(n_trials=2, storage_path=str(tmp_path / "study.db"),
                       output_dir=str(tmp_path / "hpo"), seed=0)
    report = run_study(config, lambda params: stub_metrics())
    assert report["n_complete"] == 2 and report["n_failed"] == 0
    assert len(report["trials"]) == 2
    assert report["best"]["value"] == pytest.approx(objective_value(stub_metrics(), ObjectiveWeights()))
    for trial in report["trials"]:
        assert trial["user_attrs"]["algorithm"] == "PPO" and trial["user_attrs"]["stage"] == "stage1_1d"
        assert set(trial["user_attrs"]["metrics"]) == {"success_rate", "avg_fuel_used",
                                                       "avg_overrides_per_episode", "avg_collision_risk"}
        assert set(trial["params"]) >= {"learning_rate", "gamma", "batch_size", "n_steps"}
    saved = json.loads((tmp_path / "hpo" / "hpo_report.json").read_text())
    assert saved["schema_version"] == 1 and saved["best"]["value"] == report["best"]["value"]
    assert saved["budgets"] == {"n_trials": 2, "timeout_s": 0.0}


def test_sac_search_space_is_explicitly_selectable(tmp_path):
    config = HPOConfig(algorithm="SAC", n_trials=1, storage_path=str(tmp_path / "study.db"),
                       output_dir=str(tmp_path / "hpo"), seed=0)
    report = run_study(config, lambda params: stub_metrics())
    assert report["algorithm"] == "SAC" and report["n_complete"] == 1
    assert "buffer_size" in report["trials"][0]["params"]
    assert "n_steps" not in report["trials"][0]["params"]


def test_failed_trials_do_not_abort_bounded_study(tmp_path):
    calls = {"n": 0}

    def flaky(params):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated training crash")
        return stub_metrics()

    config = HPOConfig(n_trials=3, storage_path=str(tmp_path / "study.db"),
                       output_dir=str(tmp_path / "hpo"), seed=1)
    report = run_study(config, flaky)
    assert report["n_complete"] == 2 and report["n_failed"] == 1
    assert report["best"] is not None
    failed = [t for t in report["trials"] if t["state"] == "FAIL"]
    assert len(failed) == 1 and "simulated training crash" in failed[0]["user_attrs"]["error"]


def test_resume_continues_study_without_repeating_trials(tmp_path):
    storage = str(tmp_path / "study.db")
    first = HPOConfig(n_trials=1, storage_path=storage, output_dir=str(tmp_path / "hpo"),
                      study_name="resume_me", seed=2)
    run_study(first, lambda params: stub_metrics())
    second = HPOConfig(n_trials=2, storage_path=storage, output_dir=str(tmp_path / "hpo"),
                       study_name="resume_me", seed=2)
    report = run_study(second, lambda params: stub_metrics())
    assert len(report["trials"]) == 2  # one reused, one new
    assert report["n_complete"] == 2
    assert report["best"] is not None
