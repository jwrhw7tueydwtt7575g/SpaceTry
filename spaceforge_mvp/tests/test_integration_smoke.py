"""End-to-end integration smoke test for full pipeline."""

from pathlib import Path
from spaceforge_mvp.eval.plot_trajectory import plot_trajectory
from spaceforge_mvp.eval.run_eval import run_eval
from spaceforge_mvp.training.train import train


def test_full_pipeline_integration_smoke(tmp_path):
    """
    End-to-end integration test:
      1. Train PPO policy for 1000 timesteps into tmp_path
      2. Run run_eval against the output checkpoint for 5 episodes
      3. Run plot_trajectory to generate trajectory plot PNG
      4. Assert all metrics present and non-zero-byte PNG produced
    """
    save_dir = str(tmp_path / "checkpoints")
    plot_output = str(tmp_path / "trajectory.png")

    # Step 1: Train for 1000 timesteps
    model, tracker = train(
        total_timesteps=1000,
        save_freq=1000,
        save_dir=save_dir,
        verbose=0,
    )
    assert model is not None, "Training model object should not be None"

    ckpt_path = Path(save_dir) / "ppo_spacecraft_final.zip"
    if not ckpt_path.exists():
        ckpt_path = Path(save_dir) / "ppo_spacecraft_1000_steps.zip"

    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} was not created."

    # Step 2: Run evaluation over 5 episodes
    report = run_eval(checkpoint_path=str(ckpt_path), num_episodes=5)

    assert "success_rate" in report
    assert "avg_fuel_used" in report
    assert "avg_episode_length" in report
    assert "safety_overrides_count" in report

    # Step 3: Run plot_trajectory once
    plot_trajectory(checkpoint_path=str(ckpt_path), output_path=plot_output)

    # Step 4: Assert non-zero-byte trajectory plot PNG produced
    png_path = Path(plot_output)
    assert png_path.exists(), f"Trajectory plot {png_path} was not generated."
    assert png_path.stat().st_size > 0, f"Trajectory plot {png_path} is a 0-byte file."
