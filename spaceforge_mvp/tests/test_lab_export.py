"""The browser Mission Lab is only trustworthy if its fixtures match Python.

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


def test_fixtures_match_the_python_reference():
    committed = json.loads((lab_export.LAB_DIR / "fixtures.json").read_text())
    fresh = json.loads(lab_export.fixtures_text(lab_export.build_fixtures()))
    _close(fresh, committed)


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
