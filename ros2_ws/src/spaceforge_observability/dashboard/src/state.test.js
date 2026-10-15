/* Dashboard logic tests: node state.test.js (no browser, no network). */
"use strict";
const assert = require("assert");
const S = require("./state.js");

// store + data ingest
{
  const store = S.createStore();
  assert.strictEqual(store.connection, "disconnected");
  const ok = S.applyData(store, [
    { stream: "odometry", stamp_ns: 1e9, frame_id: "target_lvlh",
      payload: { position_m: [10, 0, 0], velocity_mps: [0, 0, 0] } },
    { stream: "fuel", stamp_ns: 1e9, frame_id: "", payload: { fraction: 0.8 } },
  ]);
  assert.strictEqual(ok, 2);
  assert.deepStrictEqual(store.latest.odometry.payload.position_m, [10, 0, 0]);
}

// malformed never throws, only counts
{
  const store = S.createStore();
  const ok = S.applyData(store, [null, 42, { stream: "nope", stamp_ns: 1 }, { stream: "fuel" }]);
  assert.strictEqual(ok, 0);
  assert.strictEqual(store.malformed, 4);
}

// history capped, latest wins
{
  const store = S.createStore();
  const frames = [];
  for (let i = 0; i < 600; i++) {
    frames.push({ stream: "odometry", stamp_ns: i, frame_id: "f",
      payload: { position_m: [i, 0, 0], velocity_mps: [0, 0, 0] } });
  }
  S.applyData(store, frames);
  assert.strictEqual(store.history.odometry.length, 500);
  assert.strictEqual(store.latest.odometry.stamp_ns, 599);
}

// status + stale
{
  const store = S.createStore();
  assert.strictEqual(S.applyStatus(store, { type: "nope" }), false);
  assert.strictEqual(S.applyStatus(store, { type: "status", stale: { odometry: true, fuel: false }, dropped: {} }), true);
  assert.deepStrictEqual(S.staleStreams(store), ["odometry"]);
}

// connection backoff: 500, 1000, 2000 ... capped at 15000, reset on connect
{
  const store = S.createStore();
  assert.strictEqual(S.onDisconnected(store), 500);
  assert.strictEqual(S.onDisconnected(store), 1000);
  assert.strictEqual(S.onDisconnected(store), 2000);
  for (let i = 0; i < 10; i++) S.onDisconnected(store);
  assert.strictEqual(store.reconnectDelayMs, 15000);
  S.onConnected(store);
  assert.strictEqual(store.connection, "live");
  assert.strictEqual(store.reconnectDelayMs, 500);
}

// formatters
{
  assert.strictEqual(S.fmtMeters(1500), "1.50 km");
  assert.strictEqual(S.fmtMeters(0.5), "50.0 cm");
  assert.strictEqual(S.fmtMeters(NaN), "—");
  assert.strictEqual(S.fmtNs(95e9), "1:35");
  assert.strictEqual(S.fmtClock(3661e9), "1:01:01");
  assert.strictEqual(S.fmtClock(95e9), "01:35");
}

// trace converter mirrors the bridge replay mapping
{
  const trace = {
    summary: { label: "demo", seed: 7, steps: 2, start_dist: 6, end_dist: 0.4,
      end_speed: 0.02, success: true, fuel_used: 0.01, dv: 1.2, overrides: 1,
      terminal_reason: "docked" },
    frames: [
      { t: 0, pos: [6, 0, 0], vel: [0, 0, 0], cmd: [0, 0, 0], applied: [0, 0, 0],
        override: null, fuel: 1 },
      { t: 1, pos: [5, 0, 0], vel: [-1, 0, 0], cmd: [-1, 0, 0], applied: [0, 0, 0],
        override: "rejected", fuel: 0.99 },
      { t: 2, pos: null, vel: [0, 0, 0] }, // ragged frame skipped, never throws
    ],
  };
  const { envelopes, skipped } = S.traceToEnvelopes(trace);
  assert.strictEqual(skipped, 1);
  const streams = envelopes.map((e) => e.stream);
  assert.deepStrictEqual(streams, ["odometry", "commanded", "applied_command", "fuel",
    "odometry", "commanded", "applied_command", "fuel", "safety_event", "episode_metrics"]);
  const metrics = envelopes[envelopes.length - 1].payload;
  assert.strictEqual(metrics.success, true);
  assert.strictEqual(metrics.seed, 7);
  assert.strictEqual(envelopes[8].payload.reason_code, "rejected");
}

// deterministic player: seek, tick, step, speed
{
  const { envelopes } = S.traceToEnvelopes({
    summary: { label: "p" },
    frames: [
      { t: 0, pos: [3, 0, 0], vel: [0, 0, 0], fuel: 1 },
      { t: 10, pos: [2, 0, 0], vel: [0, 0, 0], fuel: 0.9 },
      { t: 20, pos: [1, 0, 0], vel: [0, 0, 0], fuel: 0.8 },
    ],
  });
  const store = S.createStore();
  const player = S.createPlayer(envelopes);
  assert.strictEqual(player.startNs, 0);
  assert.strictEqual(player.endNs, 20e9);
  S.playerSeek(player, store, 10e9);
  assert.strictEqual(store.latest.fuel.payload.fraction, 0.9);
  player.playing = true;
  assert.strictEqual(S.playerSetSpeed(player, 2), true);
  assert.strictEqual(S.playerSetSpeed(player, -1), false);
  S.playerTick(player, store, 5e9); // +10s at 2x -> reaches the end
  assert.strictEqual(store.latest.fuel.payload.fraction, 0.8);
  assert.strictEqual(player.done, true);
  assert.strictEqual(player.playing, false);
  S.playerSeek(player, store, 0);
  assert.strictEqual(player.done, false);
  assert.strictEqual(store.latest.fuel.payload.fraction, 1);
  S.playerStep(player, store, 1);
  assert.strictEqual(store.latest.fuel.payload.fraction, 0.9);
  S.playerStep(player, store, -1);
  assert.strictEqual(store.latest.fuel.payload.fraction, 1);
  assert.strictEqual(S.playerToggle(player), true);
  assert.strictEqual(S.playerToggle(player), false);
  const empty = S.createPlayer([]);
  assert.strictEqual(empty.done, true);
}

console.log("dashboard state logic: all assertions passed");
