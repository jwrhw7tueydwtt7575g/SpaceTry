import json
from spaceforge_rl_trainer.config import TrainerConfig
from spaceforge_rl_trainer.trainer import _algorithm, build_vec_env, train

def test_ppo_smoke_writes_machine_report(tmp_path):
    config = TrainerConfig(total_timesteps=64, ppo_n_steps=32, ppo_batch_size=16, eval_freq=0, checkpoint_freq=64, eval_episodes=1, max_steps=100, output_dir=str(tmp_path / "ppo"), tensorboard=False)
    model, report = train(config)
    assert model is not None
    assert (tmp_path / "ppo" / "policy_final.zip").is_file()
    saved = json.loads((tmp_path / "ppo" / "training_report.json").read_text())
    assert {"policy", "classical", "safety_filtered", "stage1_proof"} <= set(saved)
    assert saved["stage1_proof"]["success"]

def test_sac_is_explicitly_selectable(tmp_path):
    config = TrainerConfig(algorithm="SAC", total_timesteps=2, output_dir=str(tmp_path / "sac"), tensorboard=False)
    env = build_vec_env(config)
    model = _algorithm(config, env)
    assert model.__class__.__name__ == "SAC"
    env.close()
