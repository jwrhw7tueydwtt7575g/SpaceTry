/* Pure dashboard state: no DOM, no network. UMD so node can test it. */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory();
  else root.ObsState = factory();
}(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var STREAMS = ["odometry", "target_pose", "commanded", "applied_command", "fuel",
    "safety_event", "episode_metrics", "curriculum_state", "tensorboard", "llm_trace"];
  var MAX_HISTORY = 500;
  var BACKOFF_BASE_MS = 500;
  var BACKOFF_CAP_MS = 15000;
  var ZERO3 = [0, 0, 0];

  function createStore() {
    return {
      latest: {}, history: {}, status: null, malformed: 0,
      connection: "disconnected", // disconnected | connecting | live
      reconnectAttempt: 0, reconnectDelayMs: BACKOFF_BASE_MS,
      dropped: {}, errors: [],
    };
  }

  function clearData(store) {
    store.latest = {};
    store.history = {};
  }

  function isEnvelope(message) {
    return !!message && typeof message === "object" && typeof message.stream === "string" &&
      typeof message.stamp_ns === "number" && message.payload !== undefined;
  }

  function applyData(store, envelopes) {
    var accepted = 0;
    (envelopes || []).forEach(function (envelope) {
      if (!isEnvelope(envelope) || STREAMS.indexOf(envelope.stream) < 0) {
        store.malformed += 1;
        return;
      }
      accepted += 1;
      store.latest[envelope.stream] = envelope;
      var history = store.history[envelope.stream] || (store.history[envelope.stream] = []);
      history.push(envelope);
      if (history.length > MAX_HISTORY) history.splice(0, history.length - MAX_HISTORY);
    });
    return accepted;
  }

  function applyStatus(store, status) {
    if (!status || status.type !== "status") { store.malformed += 1; return false; }
    store.status = status;
    store.dropped = status.dropped || {};
    return true;
  }

  function applyError(store, message) {
    store.errors.push(String((message && message.reason) || "unknown error"));
    if (store.errors.length > 20) store.errors.splice(0, store.errors.length - 20);
  }

  function onConnected(store) {
    store.connection = "live";
    store.reconnectAttempt = 0;
    store.reconnectDelayMs = BACKOFF_BASE_MS;
  }

  function onDisconnected(store) {
    store.connection = "disconnected";
    store.reconnectAttempt += 1;
    store.reconnectDelayMs = Math.min(BACKOFF_CAP_MS,
      BACKOFF_BASE_MS * Math.pow(2, store.reconnectAttempt - 1));
    return store.reconnectDelayMs;
  }

  function onConnecting(store) { store.connection = "connecting"; }

  function staleStreams(store) {
    if (!store.status || !store.status.stale) return [];
    return Object.keys(store.status.stale).filter(function (s) { return store.status.stale[s]; });
  }

  function fmtMeters(value) {
    if (typeof value !== "number" || !isFinite(value)) return "—";
    if (value >= 1000) return (value / 1000).toFixed(2) + " km";
    if (value >= 1) return value.toFixed(2) + " m";
    return (value * 100).toFixed(1) + " cm";
  }

  function fmtNs(ns) {
    if (typeof ns !== "number" || !isFinite(ns)) return "—";
    var s = ns / 1e9;
    return Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");
  }

  function fmtClock(ns) {
    if (typeof ns !== "number" || !isFinite(ns) || ns < 0) return "—";
    var s = Math.floor(ns / 1e9);
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var sec = s % 60;
    function pad(n) { return String(n).padStart(2, "0"); }
    return (h > 0 ? h + ":" : "") + pad(m) + ":" + pad(sec);
  }

  /* Convert a record_episode trace {summary, frames} into telemetry envelopes.
   * Mirrors the bridge replay mapping so raw trace files play without a server.
   * Returns {envelopes, skipped}. Never throws on ragged input. */
  function traceToEnvelopes(trace) {
    var envelopes = [];
    var skipped = 0;
    function num(value, fallback) {
      return (typeof value === "number" && isFinite(value)) ? value : fallback;
    }
    function vec3(value) {
      if (!value || typeof value.length !== "number" || value.length < 3) return null;
      var out = [num(value[0], NaN), num(value[1], NaN), num(value[2], NaN)];
      return (isFinite(out[0]) && isFinite(out[1]) && isFinite(out[2])) ? out : null;
    }
    var frames = (trace && trace.frames) || [];
    var summary = (trace && trace.summary) || {};
    var label = String(summary.label || "replay");
    frames.forEach(function (frame, i) {
      var pos = vec3(frame.pos);
      var vel = vec3(frame.vel);
      if (!pos || !vel) { skipped += 1; return; }
      var stamp = Math.round(num(frame.t, i) * 1e9);
      var cmd = vec3(frame.cmd) || ZERO3.slice();
      var applied = vec3(frame.applied) || ZERO3.slice();
      var override = (frame.override === undefined || frame.override === null)
        ? null : String(frame.override);
      envelopes.push({ stream: "odometry", stamp_ns: stamp, frame_id: "target_lvlh",
        payload: { position_m: pos, velocity_mps: vel } });
      envelopes.push({ stream: "commanded", stamp_ns: stamp, frame_id: "target_lvlh",
        payload: { force_n: cmd, torque_nm: ZERO3.slice() } });
      envelopes.push({ stream: "applied_command", stamp_ns: stamp, frame_id: "target_lvlh",
        payload: { force_n: applied, torque_nm: ZERO3.slice(), override: override } });
      envelopes.push({ stream: "fuel", stamp_ns: stamp, frame_id: "",
        payload: { fraction: Math.min(1, Math.max(0, num(frame.fuel, 1))) } });
      if (override !== null) {
        envelopes.push({ stream: "safety_event", stamp_ns: stamp, frame_id: "",
          payload: { event_id: label + "-" + i, goal_type: "APPROACH",
            reason_code: override, reason: "interlock " + override + " at t=" + frame.t + "s",
            commanded_force_n: cmd, applied_force_n: applied } });
      }
    });
    var lastStamp = envelopes.length ? envelopes[envelopes.length - 1].stamp_ns : 0;
    envelopes.push({ stream: "episode_metrics", stamp_ns: lastStamp, frame_id: "",
      payload: { label: label, seed: Math.round(num(summary.seed, 0)),
        steps: Math.round(num(summary.steps, frames.length)),
        start_dist_m: num(summary.start_dist, 0), end_dist_m: num(summary.end_dist, 0),
        end_speed_mps: num(summary.end_speed, 0), success: summary.success === true,
        fuel_used: num(summary.fuel_used, 0), dv_mps: num(summary.dv, 0),
        safety_overrides: Math.round(num(summary.overrides, 0)),
        terminal_reason: String(summary.terminal_reason || "") } });
    return { envelopes: envelopes, skipped: skipped };
  }

  /* Deterministic playback cursor over sorted envelopes. Virtual clock only —
   * the browser drives tick() from requestAnimationFrame; tests drive it by hand. */
  function createPlayer(events) {
    var sorted = (events || []).slice().sort(function (a, b) { return a.stamp_ns - b.stamp_ns; });
    return {
      events: sorted, index: 0, playing: false, speed: 1,
      clockNs: sorted.length ? sorted[0].stamp_ns : 0,
      done: sorted.length === 0,
      startNs: sorted.length ? sorted[0].stamp_ns : 0,
      endNs: sorted.length ? sorted[sorted.length - 1].stamp_ns : 0,
    };
  }

  function playerApplyUntil(player, store) {
    var applied = 0;
    while (player.index < player.events.length &&
           player.events[player.index].stamp_ns <= player.clockNs) {
      applyData(store, [player.events[player.index]]);
      player.index += 1;
      applied += 1;
    }
    if (player.index >= player.events.length) { player.done = true; player.playing = false; }
    return applied;
  }

  function playerSeek(player, store, stampNs) {
    clearData(store);
    player.index = 0;
    player.done = false;
    player.clockNs = Math.min(Math.max(stampNs, player.startNs), player.endNs);
    return playerApplyUntil(player, store);
  }

  function playerTick(player, store, dtNs) {
    if (!player.playing || player.done) return 0;
    player.clockNs = Math.min(player.clockNs + dtNs * player.speed, player.endNs);
    return playerApplyUntil(player, store);
  }

  function playerToggle(player) {
    if (player.done) return false;
    player.playing = !player.playing;
    return player.playing;
  }

  function playerSetSpeed(player, speed) {
    if (typeof speed !== "number" || !isFinite(speed) || speed <= 0) return false;
    player.speed = Math.min(speed, 64);
    return true;
  }

  function playerStep(player, store, direction) {
    var stamps = [];
    player.events.forEach(function (e) {
      if (stamps.indexOf(e.stamp_ns) < 0) stamps.push(e.stamp_ns);
    });
    var target;
    if (direction > 0) {
      target = player.endNs;
      for (var i = 0; i < stamps.length; i++) {
        if (stamps[i] > player.clockNs) { target = stamps[i]; break; }
      }
    } else {
      target = player.startNs;
      for (var j = stamps.length - 1; j >= 0; j--) {
        if (stamps[j] < player.clockNs) { target = stamps[j]; break; }
      }
    }
    return playerSeek(player, store, target);
  }

  return {
    STREAMS: STREAMS, BACKOFF_BASE_MS: BACKOFF_BASE_MS, BACKOFF_CAP_MS: BACKOFF_CAP_MS,
    createStore: createStore, clearData: clearData,
    isEnvelope: isEnvelope, applyData: applyData,
    applyStatus: applyStatus, applyError: applyError, onConnected: onConnected,
    onDisconnected: onDisconnected, onConnecting: onConnecting,
    staleStreams: staleStreams, fmtMeters: fmtMeters, fmtNs: fmtNs, fmtClock: fmtClock,
    traceToEnvelopes: traceToEnvelopes, createPlayer: createPlayer,
    playerSeek: playerSeek, playerTick: playerTick, playerToggle: playerToggle,
    playerSetSpeed: playerSetSpeed, playerStep: playerStep,
  };
}));
