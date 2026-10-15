"""Observability tests: schema, replay, bridge robustness. No ROS/Gazebo needed."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from spaceforge_observability.bridge import BridgeConfig, TelemetryBridge
from spaceforge_observability.replay import (
    ReplayPlayer, load_curriculum_state, load_decision_trace, load_hpo_report,
    load_trace_frames, load_training_report, merge_sorted,
)
from spaceforge_observability.schema import (
    SCHEMA_VERSION, STREAMS, is_stale, make_envelope, redact, validate_envelope,
)

ROOT = Path(__file__).resolve().parents[4]
SCENARIOS = ROOT / "spaceforge_mvp" / "viz" / "scenarios"


def envelope(stream="odometry", **kwargs):
    payloads = {
        "odometry": {"position_m": (1.0, 2.0, 3.0), "velocity_mps": (0.1, 0.0, 0.0)},
        "target_pose": {"translation_m": (0, 0, 5), "quaternion_xyzw": (0, 0, 0, 1), "confidence": 0.9},
        "commanded": {"force_n": (0.1, 0, 0), "torque_nm": (0, 0, 0)},
        "applied_command": {"force_n": (0.1, 0, 0), "torque_nm": (0, 0, 0), "override": None},
        "fuel": {"fraction": 0.8},
        "safety_event": {"event_id": "OVR-1", "goal_type": "APPROACH", "reason_code": "REJECT",
                         "reason": "too fast", "commanded_force_n": (1, 0, 0), "applied_force_n": (0, 0, 0)},
        "episode_metrics": {"label": "demo", "seed": 1, "steps": 10, "start_dist_m": 5.0,
                            "end_dist_m": 0.4, "end_speed_mps": 0.05, "success": True,
                            "fuel_used": 0.01, "dv_mps": 0.5, "safety_overrides": 0,
                            "terminal_reason": "docked"},
        "curriculum_state": {"current_stage": "stage1_1d", "episode_index": 50,
                             "window_episodes": 50, "success_rate": 0.95, "transitions": 1},
        "tensorboard": {"url": "http://localhost:6006/", "run_dir": "runs/ppo"},
        "llm_trace": {"trace_id": "t1", "status": "ACCEPTED", "provider": "offline_mock",
                      "token_count": 10, "cost_usd": 0.001},
    }
    payload = dict(payloads[stream])
    payload.update(kwargs.pop("payload", {}))
    return make_envelope(stream, payload, kwargs.pop("stamp_ns", 1_000_000_000),
                         kwargs.pop("frame_id", "target_lvlh" if stream in
                                    ("odometry", "target_pose", "commanded", "applied_command") else ""))


# ------------------------------------------------------- schema
def test_all_streams_validate_and_version_is_pinned():
    assert SCHEMA_VERSION == 1 and len(STREAMS) == 10
    for stream in STREAMS:
        assert validate_envelope(envelope(stream))[0]


@pytest.mark.parametrize("message", [
    {"stream": "odometry"},  # no version
    {"schema_version": 2, "stream": "odometry", "stamp_ns": 1, "frame_id": "f",
     "payload": {"position_m": [0, 0, 0], "velocity_mps": [0, 0, 0]}},
    {"schema_version": 1, "stream": "nope", "stamp_ns": 1, "frame_id": "",
     "payload": {}},
    {"schema_version": 1, "stream": "fuel", "stamp_ns": -1, "frame_id": "",
     "payload": {"fraction": 0.5}},
    {"schema_version": 1, "stream": "odometry", "stamp_ns": 1, "frame_id": "",
     "payload": {"position_m": [0, 0, 0], "velocity_mps": [0, 0, 0]}},  # spatial w/o frame
    {"schema_version": 1, "stream": "fuel", "stamp_ns": 1, "frame_id": "",
     "payload": {"fraction": 1.5}},
    {"schema_version": 1, "stream": "target_pose", "stamp_ns": 1, "frame_id": "c",
     "payload": {"translation_m": [0, 0, 5], "quaternion_xyzw": [0, 0, 0, 0.1], "confidence": 1}},
    "just a string",
])
def test_malformed_envelopes_rejected_with_reason(message):
    ok, reason = validate_envelope(message)
    assert not ok and reason


def test_secrets_redacted_before_egress():
    bridge = TelemetryBridge()
    client = bridge.register()
    payload = {"trace_id": "t", "status": "ok", "provider": "openai",
               "api_key": "sk-live-123", "nested": {"Authorization": "Bearer abc"}}
    ok, _ = bridge.ingest({"schema_version": 1, "stream": "llm_trace", "stamp_ns": 5,
                           "frame_id": "", "payload": payload})
    assert ok
    stored = bridge.drain(client.client_id)[0]["payload"]
    assert stored["api_key"] == "***REDACTED***" and stored["nested"]["Authorization"] == "***REDACTED***"
    assert stored["trace_id"] == "t"  # non-secrets untouched
    assert "sk-live-123" not in json.dumps(bridge.history["llm_trace"][0])


def test_redact_handles_nesting_and_lists():
    assert redact({"a": [{"bearer": "x", "ok": 1}]}) == {"a": [{"bearer": "***REDACTED***", "ok": 1}]}


# ------------------------------------------------------- replay
def test_scenario_replay_covers_all_runtime_streams():
    events = load_trace_frames(SCENARIOS / "docking.json")
    streams = {e["stream"] for e in events}
    assert {"odometry", "commanded", "applied_command", "fuel", "episode_metrics"} <= streams
    assert all(validate_envelope(e)[0] for e in events)
    metrics = [e for e in events if e["stream"] == "episode_metrics"][0]
    assert metrics["payload"]["success"] is True


def test_hot_scenarios_surface_safety_events():
    v1 = load_trace_frames(SCENARIOS / "hot_v1.json")
    assert any(e["stream"] == "safety_event" for e in v1)
    merged = merge_sorted(v1, load_trace_frames(SCENARIOS / "hot_v2.json"))
    assert [e["stamp_ns"] for e in merged] == sorted(e["stamp_ns"] for e in merged)


def test_replay_player_rejects_bad_speed_and_unknown_files(tmp_path):
    with pytest.raises(ValueError):
        ReplayPlayer([], speed=0)
    with pytest.raises(ValueError):
        load_trace_frames(tmp_path / "missing.json")
    from spaceforge_observability.cli import load_replay_file
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(ValueError):
        load_replay_file(str(empty))


def test_report_and_state_loaders(tmp_path):
    report = {"policy": {"success_rate": 0.5, "avg_fuel_used": 0.1, "safety_overrides": 2,
                         "episodes": 4, "seed": 3},
              "classical": {"success_rate": 1.0, "avg_fuel_used": 0.05, "safety_overrides": 0,
                            "episodes": 4, "seed": 3},
              "safety_filtered": {"success_rate": 1.0, "avg_fuel_used": 0.2, "safety_overrides": 1,
                                  "episodes": 4, "seed": 3},
              "checkpoint": "runs/x/policy_final.zip"}
    path = tmp_path / "training_report.json"
    path.write_text(json.dumps(report))
    events = load_training_report(path)
    assert {e["stream"] for e in events} == {"episode_metrics", "tensorboard"}

    hpo = tmp_path / "hpo_report.json"
    hpo.write_text(json.dumps({"algorithm": "PPO", "best": {"value": 0.7, "params": {"lr": 0.001}}}))
    assert load_hpo_report(hpo)[0]["stream"] == "episode_metrics"

    trace = {"trace_id": "t", "mission_status": "ACCEPTED", "token_count": 5, "cost_usd": 0.01,
             "attempts": [], "metadata": {"api_key": "sk-x", "note": "kept"}}
    loaded_trace = load_decision_trace(trace)[0]
    assert loaded_trace["payload"]["metadata"] == {"api_key": "***REDACTED***", "note": "kept"}

    curriculum = tmp_path / "curriculum.json"
    curriculum.write_text(json.dumps({"schema_version": 1, "current_stage": "stage2_3d_static",
                                      "episode_index": 60, "outcomes": [{"success": True}] * 10,
                                      "transitions": []}))
    loaded = load_curriculum_state(curriculum)[0]
    assert loaded["payload"]["success_rate"] == pytest.approx(1.0)


# ------------------------------------------------------- bridge robustness
def test_backpressure_drops_oldest_and_counts():
    bridge = TelemetryBridge(BridgeConfig(max_queue=4))
    client = bridge.register()
    for i in range(10):
        assert bridge.ingest(envelope(stamp_ns=i))[0]
    assert client.dropped == 6
    buffered = bridge.drain(client.client_id)
    assert [e["stamp_ns"] for e in buffered] == [6, 7, 8, 9]  # newest retained
    assert bridge.drain(client.client_id) == []


def test_slow_client_does_not_block_fast_client():
    bridge = TelemetryBridge(BridgeConfig(max_queue=2))
    slow = bridge.register()
    fast = bridge.register()
    for i in range(5):
        bridge.ingest(envelope(stamp_ns=i))
        bridge.drain(fast.client_id)  # fast keeps up
    assert slow.dropped == 3 and fast.dropped == 0


def test_connection_loss_frees_state_and_keeps_bridge():
    bridge = TelemetryBridge()
    client = bridge.register()
    bridge.ingest(envelope())
    bridge.unregister(client.client_id)
    assert client.client_id not in bridge.clients and not client.connected
    assert bridge.ingest(envelope())[0]  # bridge survives
    assert bridge.status_snapshot()["clients"] == 0


def test_malformed_ingress_never_raises_and_stays_connected():
    bridge = TelemetryBridge()
    client = bridge.register()
    for bad in ["{oops", {"type": "bogus"}, {"stream": "fuel"}, None, 42]:
        reply = bridge.handle_message(client, bad)
        assert reply is None or reply.get("type") == "error"
    assert client.connected and bridge.malformed >= 4
    assert bridge.ingest(envelope())[0]


def test_subscriptions_filter_fanout():
    bridge = TelemetryBridge()
    fuel_only = bridge.register(subscriptions=["fuel"])
    bridge.ingest(envelope("odometry"))
    bridge.ingest(envelope("fuel"))
    assert [e["stream"] for e in bridge.drain(fuel_only.client_id)] == ["fuel"]
    with pytest.raises(ValueError):
        bridge.register(subscriptions=["warp"])
    reply = bridge.handle_message(fuel_only, {"type": "subscribe", "streams": ["odometry"]})
    assert reply["type"] == "ok"
    reply = bridge.handle_message(fuel_only, {"type": "ping"})
    assert reply == {"type": "pong"}


def test_stale_streams_flagged_in_status():
    bridge = TelemetryBridge()
    bridge.ingest(envelope("odometry", stamp_ns=0))
    snapshot = bridge.status_snapshot(now_ns=10_000_000_000)
    assert snapshot["stale"]["odometry"] is True
    assert snapshot["published"] == 1 and "dropped" in snapshot
    assert is_stale(None, 0, 1) and not is_stale(10, 11, 5)


# ------------------------------------------------------- dashboard + standalone
def test_dashboard_logic_in_node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable; dashboard JS logic not claimed")
    test_js = ROOT / "ros2_ws" / "src" / "spaceforge_observability" / "dashboard" / "src" / "state.test.js"
    result = subprocess.run([node, str(test_js)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "all assertions passed" in result.stdout


def test_dashboard_files_present_and_referenced():
    dashboard = ROOT / "ros2_ws" / "src" / "spaceforge_observability" / "dashboard"
    index = (dashboard / "index.html").read_text()
    for token in ("react", "three", "src/state.js", "src/app.js", "replay"):
        assert token in index.lower()
    app = (dashboard / "src" / "app.js").read_text()
    assert "unsubscribe" in app and "safety_event" in app


def test_dashboard_browser_smoke():
    """Real React + Three boot in jsdom: views, ingest, filters, 2D fallback.

    Needs smoke/vendor bundles + jsdom (dashboard/smoke/install.sh); skips
    cleanly without them. Runs with WebGL unavailable, so the 2D fallback path
    is what gets exercised alongside everything else.
    """
    import os
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable; browser smoke not claimed")
    smoke = ROOT / "ros2_ws" / "src" / "spaceforge_observability" / "dashboard" / "smoke"
    harness = smoke / "harness.js"
    node_modules = vendor = None
    for modules_dir, vendor_dir in (
            (smoke / "node_modules", smoke / "vendor"),
            (Path("/tmp/dashtest/node_modules"), Path("/tmp/dashtest/vendor"))):
        if (modules_dir / "jsdom").is_dir() and vendor_dir.is_dir():
            node_modules, vendor = modules_dir, vendor_dir
            break
    if not harness.is_file() or node_modules is None:
        pytest.skip("browser smoke needs smoke/vendor bundles + jsdom (see dashboard/smoke/install.sh)")
    dashboard = smoke.parent
    env = dict(os.environ, NODE_PATH=str(node_modules))
    result = subprocess.run(
        [node, str(harness), str(dashboard), str(vendor)],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(smoke))
    assert result.returncode == 0, result.stderr[-2000:]
    assert "browser smoke: boot, views, ingest, filters all OK" in result.stdout


def test_standalone_dashboard_still_builds():
    from spaceforge_mvp.viz import dashboard as standalone
    original = standalone.OUT.read_bytes()  # build() rewrites in place; restore after
    try:
        out = standalone.build()
        assert out.is_file() and out.stat().st_size > 100_000
        head = out.read_text()[:2000].lower()
        assert "spaceforge" in head
    finally:
        standalone.OUT.write_bytes(original)
