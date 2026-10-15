import numpy as np
from spaceforge_rl_env.classical import prove_stage1_approach
from spaceforge_rl_env.contract import ACTION_DIM, OBSERVATION_DIM
from spaceforge_rl_env.env import SpacecraftRLEnv

def test_contract_is_frozen_and_normalized():
    env = SpacecraftRLEnv()
    assert env.observation_space.shape == (18,)
    assert env.action_space.shape == (6,)
    obs, info = env.reset(seed=4)
    assert obs.shape == (OBSERVATION_DIM,)
    assert np.all(np.isfinite(obs))
    assert ACTION_DIM == 6
    assert env.observation_contract()[0].startswith("relative_position")
    assert info["sim_time_ns"] == 0

def test_mock_transport_command_flow_and_timestamps():
    env = SpacecraftRLEnv()
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32))
    assert info["sim_time_ns"] > 0
    assert not info["stale_sensor"]
    assert env.transport.published_commands[-1][1][0] == 1.0
    assert obs.shape == (18,)
    assert not terminated and not truncated

def test_action_limits_and_timeout():
    env = SpacecraftRLEnv(max_steps=1)
    env.reset(seed=2)
    _, _, _, truncated, info = env.step(np.full(6, 3.0, dtype=np.float32))
    assert truncated and info["action_clipped"]

def test_stage1_classical_controller_completes():
    result = prove_stage1_approach()
    assert result.success
    assert result.distance_m < 0.5
    assert result.speed_mps < 0.1
