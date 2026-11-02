"""The browser simulator is only trustworthy if its fixtures match Python.

``lab/sim.test.js`` checks the JavaScript port against ``lab/fixtures.json``;
these tests check that ``fixtures.json`` and ``policy.js`` are what the current
Python code and checkpoint produce, so neither side can drift silently.
Regenerate with ``python -m spaceforge_mvp.viz.lab_export``.
"""

import json
import shutil
import subprocess

import numpy as np
import pytest

from spaceforge_mvp.viz import lab_export


def _close(a, b, path="fixtures"):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            _close(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, (int, float)):
        assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _committed():
    return json.loads((lab_export.LAB_DIR / "fixtures.json").read_text())


def test_fixtures_match_the_python_reference():
    committed = _committed()
    fresh = json.loads(lab_export.fixtures_text(lab_export.build_fixtures(include_policy=False)))
    committed["episodes"] = [ep for ep in committed["episodes"] if ep["controller"] != "ppo"]
    _close(fresh, committed)


def test_policy_episodes_replay_exactly_and_match_the_checkpoint():
    """PPO episodes: physics must replay exactly; network outputs only to float32 tolerance.

    Torch's float32 matmul rounds differently across CPUs and BLAS builds (~1e-8 per
    action), so regenerating these episodes is not bit-reproducible. Instead, drive the
    Python env with the committed actions (pos/vel must match to 1e-9) and check the
    shipped checkpoint produces those actions from the same observations.
    """
    from stable_baselines3 import PPO

    model = PPO.load(str(lab_export.CHECKPOINT), device="cpu")
    episodes = [ep for ep in _committed()["episodes"] if ep["controller"] == "ppo"]
    assert len(episodes) == 2
    for ep in episodes:
        actions = iter(step["action"] for step in ep["steps"])

        def replay(obs, model=model, actions=actions):
            action = np.asarray(next(actions), dtype=np.float64)
            live = model.predict(obs, deterministic=True)[0]
            np.testing.assert_allclose(live, action, atol=2e-5)
            return action

        fresh = lab_export._episode(ep["name"], "ppo", ep["start_pos"], ep["start_vel"], ep["safety"],
                                    max_steps=ep["max_steps"], fuel=ep["fuel"], policy=replay)
        _close(json.loads(json.dumps(fresh)), ep, ep["name"])


def test_policy_weights_match_the_shipped_checkpoint():
    committed = (lab_export.LAB_DIR / "policy.js").read_text()
    assert committed == lab_export.policy_js(lab_export.export_policy())


def test_exported_network_reproduces_sb3_actions():
    """Decode the exported weights in numpy and compare with SB3's own predict()."""
    import base64

    from stable_baselines3 import PPO

    spec = lab_export.export_policy()
    model = PPO.load(str(lab_export.CHECKPOINT), device="cpu")

    def layers(key):
        out = []
        for layer in spec[key]:
            w = np.frombuffer(base64.b64decode(layer["w"]), dtype="<f4").reshape(layer["out"], layer["in"])
            b = np.frombuffer(base64.b64decode(layer["b"]), dtype="<f4")
            out.append((w.astype(np.float64), b.astype(np.float64)))
        return out

    actor = layers("actor")
    rng = np.random.default_rng(3)
    for _ in range(50):
        obs = rng.normal(0, 0.6, 10).astype(np.float32)
        h = obs.astype(np.float64)
        for i, (w, b) in enumerate(actor):
            h = w @ h + b
            if i < len(actor) - 1:
                h = np.tanh(h)
        expected = model.predict(obs, deterministic=True)[0]
        np.testing.assert_allclose(np.clip(h, -1, 1), expected, atol=2e-5)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_javascript_port_passes_parity():
    result = subprocess.run(["node", str(lab_export.LAB_DIR / "sim.test.js")], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
