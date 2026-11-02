/* SpaceForge mission console: live bridge + replay, React panels, Three.js stage.
 * Needs CDN React/ReactDOM/Three (see index.html); degrades to panels + replay
 * files without them. Playback and state math live in state.js (node-tested).
 * Keyboard: 1-4 views, space play/pause, arrows seek, F follow camera.
 */
(function () {
  "use strict";

  var store = ObsState.createStore();
  var ws = null;
  var wsUrl = "ws://127.0.0.1:8765";
  var reconnectTimer = null;
  var unsubscribed = {};
  var player = null; // ObsState player when a replay is loaded, else live mode
  var view = "mission";
  var followCam = true;
  var camPreset = "chase";
  var chartHover = -1;
  var lastRenderAt = 0;
  try {
    var savedUrl = window.localStorage.getItem("spaceforge.bridgeUrl");
    if (savedUrl) wsUrl = savedUrl;
  } catch (e) { /* storage unavailable: keep default */ }

  /* ---------------- connection ---------------- */
  function setConnection(state) {
    if (state === "live") ObsState.onConnected(store);
    else if (state === "connecting") ObsState.onConnecting(store);
    requestRender();
  }

  function scheduleReconnect() {
    var delay = ObsState.onDisconnected(store);
    requestRender();
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, delay);
  }

  function connect() {
    if (typeof WebSocket === "undefined") return;
    if (ws) { try { ws.close(); } catch (e) { /* replaced below */ } }
    setConnection("connecting");
    try { ws = new WebSocket(wsUrl); } catch (e) { scheduleReconnect(); return; }
    ws.onopen = function () {
      setConnection("live");
      var streams = ObsState.STREAMS.filter(function (s) { return !unsubscribed[s]; });
      ws.send(JSON.stringify({ type: "subscribe", streams: streams }));
    };
    ws.onmessage = function (event) {
      var message;
      try { message = JSON.parse(event.data); } catch (e) { store.malformed += 1; requestRender(); return; }
      if (!message || typeof message !== "object") { store.malformed += 1; requestRender(); return; }
      if (message.type === "hello") {
        Object.keys(message.history || {}).forEach(function (s) {
          ObsState.applyData(store, message.history[s]);
        });
      } else if (message.type === "data") {
        ObsState.applyData(store, message.envelopes);
      } else if (message.type === "status") {
        ObsState.applyStatus(store, message);
      } else if (message.type === "error") {
        ObsState.applyError(store, message);
      }
      requestRender();
    };
    ws.onclose = scheduleReconnect;
    ws.onerror = function () { try { ws.close(); } catch (e) { /* handled by onclose */ } };
  }

  function toggleStream(stream) {
    if (unsubscribed[stream]) {
      delete unsubscribed[stream];
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "subscribe", streams: [stream] }));
    } else {
      unsubscribed[stream] = true;
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "unsubscribe", streams: [stream] }));
      delete store.latest[stream];
      delete store.history[stream];
    }
    requestRender();
  }

  /* ---------------- replay ---------------- */
  var SCENARIOS = [
    { key: "docking", name: "Certified glideslope", hint: "Clean docking, no interlock fires" },
    { key: "rl_policy", name: "Shipped PPO policy", hint: "Flies away, 0% success" },
    { key: "hot_open", name: "Reckless slam, no interlock", hint: "Collision at 6.5 m/s" },
    { key: "hot_v1", name: "Reckless slam, veto interlock", hint: "Still collides at 2.0 m/s" },
    { key: "hot_v2", name: "Reckless slam, braking filter", hint: "Soft capture in 8 s" },
  ];

  function loadEnvelopes(envelopes, sourceName) {
    player = ObsState.createPlayer(envelopes);
    ObsState.playerSeek(player, store, player.startNs);
    player.playing = true;
    ObsState.onConnected(store);
    store.errors.push("replay: " + envelopes.length + " envelopes from " + sourceName);
    setView("mission");
    requestRender();
  }

  function loadTraceObject(trace, sourceName) {
    var converted = ObsState.traceToEnvelopes(trace);
    if (!converted.envelopes.length) {
      store.errors.push("replay file has no usable frames: " + sourceName);
      requestRender();
      return;
    }
    if (converted.skipped) {
      store.errors.push(converted.skipped + " ragged frame(s) skipped in " + sourceName);
    }
    loadEnvelopes(converted.envelopes, sourceName);
  }

  function loadReplayText(text, sourceName) {
    var parsed = JSON.parse(text);
    if (parsed && parsed.summary && parsed.frames) loadTraceObject(parsed, sourceName);
    else if (Array.isArray(parsed)) loadEnvelopes(parsed, sourceName);
    else if (parsed && Array.isArray(parsed.envelopes)) loadEnvelopes(parsed.envelopes, sourceName);
    else throw new Error("unrecognized replay JSON: need a trace, envelope array, or {envelopes}");
  }

  function loadReplayFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      try { loadReplayText(reader.result, file.name); }
      catch (e) { store.errors.push("replay file rejected: " + e.message); requestRender(); }
    };
    reader.onerror = function () {
      store.errors.push("could not read file: " + file.name);
      requestRender();
    };
    reader.readAsText(file);
  }

  function loadScenario(key) {
    var candidates = [
      "replay/" + key + ".json",
      "../spaceforge_mvp/viz/scenarios/" + key + ".json",
      "/spaceforge_mvp/viz/scenarios/" + key + ".json",
    ];
    store.errors.push("fetching scenario " + key + "…");
    requestRender();
    (function next(i) {
      if (i >= candidates.length) {
        store.errors.push("scenario not reachable: serve the repo root over HTTP, or drop the JSON file here.");
        requestRender();
        return;
      }
      fetch(candidates[i]).then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.text();
      }).then(function (text) { loadReplayText(text, key + ".json"); })
        .catch(function () { next(i + 1); });
    })(0);
  }

  /* ---------------- Three.js mission stage ---------------- */
  var three = null;
  var glFailed = false; // sticky: a dead GPU context falls back to 2D, never throws per frame

  function makeLabelSprite(text, color) {
    var canvas = document.createElement("canvas");
    canvas.width = 256; canvas.height = 64;
    var ctx = canvas.getContext("2d");
    ctx.font = "600 28px 'JetBrains Mono', ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillStyle = color || "#8FA1B5";
    ctx.fillText(text, 128, 42);
    var texture = new THREE.CanvasTexture(canvas);
    var sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
    sprite.scale.set(4, 1, 1);
    return sprite;
  }

  function makeGlowTexture() {
    var canvas = document.createElement("canvas");
    canvas.width = 64; canvas.height = 64;
    var ctx = canvas.getContext("2d");
    var gradient = ctx.createRadialGradient(32, 32, 2, 32, 32, 30);
    gradient.addColorStop(0, "rgba(70,216,255,1)");
    gradient.addColorStop(0.4, "rgba(70,216,255,.45)");
    gradient.addColorStop(1, "rgba(70,216,255,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(canvas);
  }

  function initThree(canvas) {
    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    } catch (e) { return false; }
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0x04070c);
    var camera = new THREE.PerspectiveCamera(50, canvas.width / canvas.height, 0.1, 20000);
    camera.position.set(20, -24, 15);

    var starGeo = new THREE.BufferGeometry();
    var starPos = new Float32Array(900 * 3);
    for (var i = 0; i < 900; i++) {
      var r = 1500 + Math.random() * 2500;
      var theta = Math.random() * Math.PI * 2;
      var phi = Math.acos(2 * Math.random() - 1);
      starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      starPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      starPos[i * 3 + 2] = r * Math.cos(phi);
    }
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    scene.add(new THREE.Points(starGeo,
      new THREE.PointsMaterial({ color: 0x8fa1b5, size: 2.2, sizeAttenuation: false })));

    var grid = new THREE.GridHelper(120, 24, 0x1b2635, 0x111a26);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    // Target station: bus + twin wings with technical edge lines.
    var station = new THREE.Group();
    var edgeMat = new THREE.LineBasicMaterial({ color: 0x5cd4ff });
    var bus = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1.4, 1.4),
      new THREE.MeshBasicMaterial({ color: 0x22303f }));
    station.add(bus);
    station.add(new THREE.LineSegments(new THREE.EdgesGeometry(bus.geometry), edgeMat));
    [-1, 1].forEach(function (side) {
      var wing = new THREE.Mesh(new THREE.BoxGeometry(0.08, 4.4, 1.8),
        new THREE.MeshBasicMaterial({ color: 0x14324a }));
      wing.position.x = side * 2.7;
      station.add(wing);
      station.add(new THREE.LineSegments(new THREE.EdgesGeometry(wing.geometry), edgeMat));
    });
    scene.add(station);

    var keepOut = new THREE.Mesh(
      new THREE.SphereGeometry(5, 32, 24),
      new THREE.MeshBasicMaterial({ color: 0xffb547, wireframe: true, transparent: true, opacity: 0.16 }));
    scene.add(keepOut);
    var capture = new THREE.Mesh(
      new THREE.SphereGeometry(0.5, 24, 18),
      new THREE.MeshBasicMaterial({ color: 0x3fdd9b, wireframe: true, transparent: true, opacity: 0.5 }));
    scene.add(capture);
    var keepLabel = makeLabelSprite("keep-out 5 m", "#FFB454");
    keepLabel.position.set(0, 0, 5.4);
    scene.add(keepLabel);

    // Chaser: cone aligned to velocity + glow sprite.
    var chaser = new THREE.Group();
    var cone = new THREE.Mesh(new THREE.ConeGeometry(0.45, 1.4, 20),
      new THREE.MeshBasicMaterial({ color: 0x5cd4ff }));
    cone.rotation.x = Math.PI / 2;
    chaser.add(cone);
    var glow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: makeGlowTexture(), transparent: true, depthWrite: false, opacity: 0.9 }));
    glow.scale.set(2.4, 2.4, 1);
    chaser.add(glow);
    scene.add(chaser);

    // Gradient trail: newest cyan, oldest transparent.
    var TRAIL = 600;
    var trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
    trailGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
    var trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.95 }));
    trail.frustumCulled = false;
    scene.add(trail);

    three = {
      renderer: renderer, scene: scene, camera: camera, canvas: canvas,
      chaser: chaser, trailGeo: trailGeo, trailMax: TRAIL,
      orbit: { theta: -0.7, phi: 1.05, radius: 34, target: new THREE.Vector3() },
      dragging: false, lastX: 0, lastY: 0,
    };
    attachOrbit(canvas);
    applyPreset();
    return true;
  }

  function attachOrbit(canvas) {
    canvas.addEventListener("pointerdown", function (event) {
      three.dragging = true;
      three.lastX = event.clientX; three.lastY = event.clientY;
      try { canvas.setPointerCapture(event.pointerId); } catch (e) { /* mouse input */ }
    });
    canvas.addEventListener("pointermove", function (event) {
      if (!three.dragging) return;
      var dx = event.clientX - three.lastX;
      var dy = event.clientY - three.lastY;
      three.lastX = event.clientX; three.lastY = event.clientY;
      three.orbit.theta -= dx * 0.005;
      three.orbit.phi = Math.min(1.55, Math.max(0.05, three.orbit.phi - dy * 0.005));
    });
    function endDrag() { three.dragging = false; }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("wheel", function (event) {
      event.preventDefault();
      three.orbit.radius = Math.min(400, Math.max(3, three.orbit.radius * (1 + event.deltaY * 0.001)));
    }, { passive: false });
    canvas.addEventListener("dblclick", function () { applyPreset(); });
  }

  function applyPreset() {
    if (!three) return;
    if (camPreset === "top") {
      three.orbit.theta = 0; three.orbit.phi = 0.06; three.orbit.radius = 40;
    } else if (camPreset === "side") {
      three.orbit.theta = Math.PI / 2; three.orbit.phi = 1.5; three.orbit.radius = 30;
    } else { // chase
      three.orbit.theta = -0.7; three.orbit.phi = 1.05; three.orbit.radius = 34;
    }
  }

  function chaserPoint() {
    var history = store.history.odometry || [];
    if (!history.length) return null;
    // Hill frame (x radial, y in-track, z cross) -> scene (y right, x up, z depth).
    var p = history[history.length - 1].payload.position_m;
    return { x: p[1], y: p[0], z: p[2] };
  }

  function updateThree() {
    if (!three || three.broken) return;
    var history = store.history.odometry || [];
    var positions = three.trailGeo.attributes.position.array;
    var colors = three.trailGeo.attributes.color.array;
    var count = Math.min(history.length, three.trailMax);
    var start = history.length - count;
    for (var i = 0; i < count; i++) {
      var p = history[start + i].payload.position_m;
      positions[i * 3] = p[1]; positions[i * 3 + 1] = p[0]; positions[i * 3 + 2] = p[2];
      var fade = count <= 1 ? 1 : i / (count - 1);
      colors[i * 3] = 0.27 * fade + 0.05;
      colors[i * 3 + 1] = 0.85 * fade + 0.05;
      colors[i * 3 + 2] = 1.0 * fade + 0.05;
    }
    three.trailGeo.setDrawRange(0, count);
    three.trailGeo.attributes.position.needsUpdate = true;
    three.trailGeo.attributes.color.needsUpdate = true;

    var point = chaserPoint();
    if (point) {
      three.chaser.visible = true;
      three.chaser.position.set(point.x, point.y, point.z);
      if (history.length > 1) {
        var prev = history[history.length - 2].payload.position_m;
        var dx = point.x - prev[1];
        var dy = point.y - prev[0];
        var dz = point.z - prev[2];
        if (dx * dx + dy * dy + dz * dz > 1e-12) {
          three.chaser.lookAt(point.x + dx, point.y + dy, point.z + dz);
        }
      }
      if (followCam) three.orbit.target.set(point.x, point.y, point.z);
      else three.orbit.target.set(0, 0, 0);
    } else {
      three.chaser.visible = false;
    }
    var o = three.orbit;
    three.camera.position.set(
      o.target.x + o.radius * Math.sin(o.phi) * Math.cos(o.theta),
      o.target.y + o.radius * Math.sin(o.phi) * Math.sin(o.theta),
      o.target.z + o.radius * Math.cos(o.phi));
    three.camera.lookAt(o.target);
    var w = three.canvas.clientWidth || three.canvas.width;
    var h = three.canvas.clientHeight || three.canvas.height;
    if (three.canvas.width !== w || three.canvas.height !== h) {
      three.renderer.setSize(w, h, false);
      three.camera.aspect = w / h;
      three.camera.updateProjectionMatrix();
    }
    try {
      three.renderer.render(three.scene, three.camera);
    } catch (e) {
      // GPU context died mid-session: park 3D permanently, show the 2D plot.
      three.broken = true;
      glFailed = true;
      three = null;
      requestRender();
    }
  }

  function drawFallback2D(canvas) {
    var ctx = canvas.getContext("2d");
    var history = store.history.odometry || [];
    var w = canvas.width;
    var h = canvas.height;
    ctx.fillStyle = "#04070c";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#1b2635";
    var g;
    for (g = 0; g <= 10; g++) {
      ctx.beginPath(); ctx.moveTo((w / 10) * g, 0); ctx.lineTo((w / 10) * g, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, (h / 10) * g); ctx.lineTo(w, (h / 10) * g); ctx.stroke();
    }
    if (!history.length) return;
    var maxR = 1;
    history.forEach(function (e) {
      var p = e.payload.position_m;
      maxR = Math.max(maxR, Math.abs(p[0]), Math.abs(p[1]));
    });
    var scale = (Math.min(w, h) / 2 - 20) / maxR;
    var cx = w / 2;
    var cy = h / 2;
    ctx.strokeStyle = "#FFB454";
    ctx.beginPath(); ctx.arc(cx, cy, 5 * scale, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = "#5cd4ff";
    ctx.beginPath();
    history.forEach(function (e, i) {
      var p = e.payload.position_m;
      var x = cx + p[1] * scale;
      var y = cy - p[0] * scale;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  /* ---------------- charts ---------------- */
  function drawChart(canvas, config) {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth || canvas.width;
    var h = canvas.clientHeight || canvas.height;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    var padL = 56;
    var padB = 20;
    var padT = 8;
    var plotW = Math.max(10, w - padL - 8);
    var plotH = Math.max(10, h - padT - padB);
    var points = config.points;
    if (!points.length) {
      ctx.fillStyle = "#54657a";
      ctx.font = "12px 'JetBrains Mono', ui-monospace, monospace";
      ctx.fillText("awaiting " + config.stream + "…", padL + 8, padT + 16);
      return;
    }
    var t0 = points[0].t;
    var t1 = Math.max(points[points.length - 1].t, t0 + 1e-6);
    function scale(v) { return config.log ? Math.log10(Math.max(v, 1e-3)) : v; }
    var lo = Infinity;
    var hi = -Infinity;
    points.forEach(function (p) {
      var v = scale(p.v);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    });
    (config.thresholds || []).forEach(function (t) {
      var v = scale(t.v);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    });
    if (hi - lo < 1e-9) hi = lo + 1;
    function X(t) { return padL + ((t - t0) / (t1 - t0)) * plotW; }
    function Y(v) { return padT + plotH - ((v - lo) / (hi - lo)) * plotH; }
    ctx.strokeStyle = "#1b2635";
    ctx.lineWidth = 1;
    for (var g = 0; g <= 4; g++) {
      var gy = padT + (plotH / 4) * g;
      ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(padL + plotW, gy); ctx.stroke();
    }
    (config.thresholds || []).forEach(function (t) {
      ctx.strokeStyle = t.color;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(padL, Y(scale(t.v))); ctx.lineTo(padL + plotW, Y(scale(t.v))); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = t.color;
      ctx.font = "10px 'JetBrains Mono', ui-monospace, monospace";
      ctx.fillText(t.label, padL + 4, Y(scale(t.v)) - 4);
    });
    ctx.strokeStyle = "#5cd4ff";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    points.forEach(function (p, i) {
      var x = X(p.t);
      var y = Y(scale(p.v));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (chartHover >= 0 && chartHover < points.length) {
      var hp = points[chartHover];
      ctx.strokeStyle = "#e9f1f8";
      ctx.beginPath(); ctx.moveTo(X(hp.t), padT); ctx.lineTo(X(hp.t), padT + plotH); ctx.stroke();
      ctx.fillStyle = "#e9f1f8";
      ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
      ctx.fillText(config.format(hp.v) + " @ " + ObsState.fmtClock(hp.t * 1e9), padL + 6, padT + 14);
    }
    ctx.fillStyle = "#54657a";
    ctx.font = "10px 'JetBrains Mono', ui-monospace, monospace";
    ctx.fillText(config.unit, 6, padT + 12);
  }

  function drawCharts() {
    function byId(id) { return document.getElementById(id); }
    var range = byId("chart-range");
    var speed = byId("chart-speed");
    var fuel = byId("chart-fuel");
    if (!range || !speed || !fuel) return;
    var odom = store.history.odometry || [];
    function norm(payload, key) {
      var v = payload[key];
      return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    drawChart(range, {
      stream: "odometry", log: true, unit: "log m",
      points: odom.map(function (e) { return { t: e.stamp_ns / 1e9, v: norm(e.payload, "position_m") }; }),
      thresholds: [{ v: 5, label: "keep-out 5 m", color: "#FFB454" },
                   { v: 0.5, label: "capture 0.5 m", color: "#3DE08F" }],
      format: ObsState.fmtMeters,
    });
    drawChart(speed, {
      stream: "odometry", log: false, unit: "m/s",
      points: odom.map(function (e) { return { t: e.stamp_ns / 1e9, v: norm(e.payload, "velocity_mps") }; }),
      thresholds: [{ v: 0.2, label: "gate 0.2 m/s", color: "#FFB454" },
                   { v: 0.1, label: "dock 0.1 m/s", color: "#3DE08F" }],
      format: function (v) { return v.toFixed(3) + " m/s"; },
    });
    drawChart(fuel, {
      stream: "fuel", log: false, unit: "fraction",
      points: (store.history.fuel || []).map(function (e) {
        return { t: e.stamp_ns / 1e9, v: e.payload.fraction };
      }),
      thresholds: [{ v: 0.05, label: "floor", color: "#FF5F5F" }],
      format: function (v) { return (v * 100).toFixed(2) + "%"; },
    });
  }

  /* ---------------- shared components ---------------- */
  function hudChips() {
    var odom = store.latest.odometry;
    var fuel = store.latest.fuel;
    var range = null;
    var speed = null;
    if (odom) {
      var p = odom.payload.position_m;
      var v = odom.payload.velocity_mps;
      range = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      speed = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    function chip(label, value, tone) {
      return React.createElement("div", { className: "chip-stat " + (tone || ""), key: label },
        React.createElement("span", { className: "chip-label" }, label),
        React.createElement("span", { className: "chip-value" }, value));
    }
    return React.createElement("div", { className: "hud", role: "status", "aria-label": "current readings" },
      chip("Range", range === null ? "—" : ObsState.fmtMeters(range), range !== null && range < 5 ? "warn" : ""),
      chip("Rel speed", speed === null ? "—" : speed.toFixed(3) + " m/s", speed !== null && speed > 0.2 ? "warn" : ""),
      chip("Fuel", fuel ? (fuel.payload.fraction * 100).toFixed(1) + "%" : "—",
        fuel && fuel.payload.fraction <= 0.05 ? "fault" : ""),
      chip("Clock", odom ? ObsState.fmtClock(odom.stamp_ns) : "—"));
  }

  function transport() {
    if (!player) {
      return React.createElement("p", { className: "muted" },
        "Live stream has no timeline. Load a replay to scrub.");
    }
    var span = player.endNs - player.startNs;
    var pct = span > 0 ? ((player.clockNs - player.startNs) / span) * 100 : 0;
    void pct;
    var marks = (store.history.safety_event || []).map(function (e, i) {
      var left = span > 0 ? ((e.stamp_ns - player.startNs) / span) * 100 : 0;
      return React.createElement("span", {
        key: "m" + i, className: "tick", style: { left: left + "%" },
        title: e.payload.reason_code + " @ " + ObsState.fmtClock(e.stamp_ns),
      });
    });
    return React.createElement("div", { className: "transport" },
      React.createElement("button", {
        onClick: function () { ObsState.playerToggle(player); requestRender(); },
        "aria-pressed": player.playing, "aria-label": player.playing ? "Pause replay" : "Play replay",
      }, player.playing ? "Pause" : "Play"),
      React.createElement("button", {
        onClick: function () { ObsState.playerStep(player, store, -1); requestRender(); },
        "aria-label": "Step back one frame",
      }, "‹"),
      React.createElement("button", {
        onClick: function () { ObsState.playerStep(player, store, 1); requestRender(); },
        "aria-label": "Step forward one frame",
      }, "›"),
      React.createElement("div", { className: "scrub" },
        React.createElement("div", { className: "ticks", "aria-hidden": "true" }, marks),
        React.createElement("input", {
          type: "range", min: player.startNs, max: player.endNs, step: 1e8,
          value: player.clockNs, "aria-label": "Replay position",
          onChange: function (e) { ObsState.playerSeek(player, store, Number(e.target.value)); requestRender(); },
        })),
      React.createElement("span", { className: "clock" },
        ObsState.fmtClock(player.clockNs) + " / " + ObsState.fmtClock(player.endNs)),
      React.createElement("label", { className: "speed" }, "Speed",
        React.createElement("select", {
          value: String(player.speed), "aria-label": "Replay speed",
          onChange: function (e) { ObsState.playerSetSpeed(player, Number(e.target.value)); requestRender(); },
        }, [0.25, 0.5, 1, 2, 4, 8].map(function (s) {
          return React.createElement("option", { key: s, value: s }, s + "×");
        }))));
  }

  function instrumentRail() {
    function row(label, value, tone) {
      return React.createElement("div", { className: "inst-row" + (tone ? " " + tone : ""), key: label },
        React.createElement("span", null, label),
        React.createElement("strong", null, value));
    }
    var odom = store.latest.odometry;
    var applied = store.latest.applied_command;
    var metrics = store.latest.episode_metrics;
    var rows = [
      row("Target", "LVLH origin"),
      row("Range", odom ? ObsState.fmtMeters(Math.sqrt(odom.payload.position_m.reduce(function (a, v) { return a + v * v; }, 0))) : "—"),
      row("Fuel", store.latest.fuel ? (store.latest.fuel.payload.fraction * 100).toFixed(2) + "%" : "—"),
      row("Interlock", applied && applied.payload.override ? applied.payload.override : "quiet",
        applied && applied.payload.override ? "warn" : ""),
      row("Outcome", metrics ? (metrics.payload.success ? "docked" : metrics.payload.terminal_reason || "running") : "—",
        metrics ? (metrics.payload.success ? "go" : "") : ""),
    ];
    var thrust = null;
    if (applied) {
      var f = applied.payload.force_n;
      var mag = Math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
      thrust = React.createElement("div", { className: "thrust" },
        ["Radial", "In-track", "Cross"].map(function (axis, i) {
          var v = Math.min(1, Math.abs(f[i]));
          return React.createElement("div", { className: "axis", key: axis },
            React.createElement("span", { className: "axis-label" }, axis),
            React.createElement("div", { className: "axis-bar" },
              React.createElement("div", {
                className: "axis-fill" + (f[i] < 0 ? " neg" : ""),
                style: { width: (v * 100).toFixed(1) + "%" },
              })),
            React.createElement("span", { className: "axis-val" }, f[i].toFixed(2)));
        }),
        React.createElement("div", { className: "inst-row" },
          React.createElement("span", null, "Thrust magnitude"),
          React.createElement("strong", null, mag.toFixed(3) + " m/s²")));
    }
    return React.createElement("aside", { className: "rail", "aria-label": "Instruments" },
      React.createElement("h2", null, "Instruments"), rows, thrust,
      React.createElement("h2", null, "Camera"),
      React.createElement("div", { className: "btn-row", role: "group", "aria-label": "Camera presets" },
        ["chase", "side", "top"].map(function (preset) {
          return React.createElement("button", {
            key: preset,
            onClick: function () { camPreset = preset; applyPreset(); requestRender(); },
            "aria-pressed": camPreset === preset,
          }, preset);
        })),
      React.createElement("button", {
        onClick: function () { followCam = !followCam; requestRender(); },
        "aria-pressed": followCam,
      }, followCam ? "Following chaser" : "Locked to origin"));
  }

  /* ---------------- views ---------------- */
  var VIEWS = ["mission", "telemetry", "events", "replay"];

  function setView(next) {
    if (VIEWS.indexOf(next) < 0) next = "mission";
    if (next === view) return;
    view = next;
    try { window.history.replaceState(null, "", "#/" + view); } catch (e) { /* file URLs */ }
    requestRender();
  }

  function header() {
    var stale = ObsState.staleStreams(store);
    return React.createElement("header", { className: "command" },
      React.createElement("a", { className: "skip", href: "#content" }, "Skip to content"),
      React.createElement("div", { className: "mark", "aria-label": "SpaceForge mission console" },
        React.createElement("svg", { width: 30, height: 30, viewBox: "0 0 30 30", "aria-hidden": "true" },
          React.createElement("circle", { cx: 15, cy: 15, r: 4, fill: "none", stroke: "#5cd4ff", strokeWidth: 1.6 }),
          React.createElement("ellipse", { cx: 15, cy: 15, rx: 12, ry: 5.5, fill: "none", stroke: "#5cd4ff", strokeWidth: 1.2, transform: "rotate(-24 15 15)" }),
          React.createElement("circle", { cx: 25, cy: 9, r: 2, fill: "#5cd4ff" })),
        React.createElement("div", null,
          React.createElement("div", { className: "mark-name" }, "SpaceForge"),
          React.createElement("div", { className: "mark-sub" }, "Mission console"))),
      React.createElement("nav", { className: "tabs", "aria-label": "Views" },
        VIEWS.map(function (v, i) {
          return React.createElement("button", {
            key: v, onClick: function () { setView(v); },
            "aria-pressed": view === v,
            title: "Keyboard: " + (i + 1),
          }, v.charAt(0).toUpperCase() + v.slice(1));
        })),
      React.createElement("div", {
        className: "pill " + store.connection, role: "status", "aria-live": "polite",
        "aria-label": "Connection: " + store.connection,
      },
        React.createElement("span", { className: "pulse", "aria-hidden": "true" }),
        store.connection === "live" ? (player ? "replay" : "live") : store.connection,
        stale.length ? " · stale: " + stale.join(", ") : ""),
      React.createElement("input", {
        type: "text", defaultValue: wsUrl, "aria-label": "Bridge WebSocket URL",
        spellCheck: false, autoComplete: "off",
        onChange: function (e) {
          wsUrl = e.target.value;
          try { window.localStorage.setItem("spaceforge.bridgeUrl", wsUrl); } catch (err) { /* ignore */ }
        },
      }),
      React.createElement("button", {
        onClick: function () { connect(); },
      }, "Connect"));
  }

  function missionView() {
    var fallback = glFailed || typeof THREE === "undefined";
    return React.createElement("div", { className: "mission-grid" },
      React.createElement("div", { className: "stage" },
        React.createElement("canvas", {
          id: "scene3d", width: 880, height: 520, tabIndex: 0,
          hidden: fallback,
          "aria-label": "3D rendezvous view. Drag to orbit, scroll to zoom, double-click to reset camera.",
        }),
        fallback ? React.createElement("div", null,
          React.createElement("canvas", { id: "scene2d", width: 880, height: 520 }),
          React.createElement("p", { className: "muted" }, "3D unavailable: top-down plot shown.")) : null,
        hudChips(),
        transport()),
      instrumentRail());
  }

  function telemetryView() {
    var metrics = store.latest.episode_metrics;
    return React.createElement("div", null,
      React.createElement("div", { className: "charts" },
        [["chart-range", "Range (log scale)"], ["chart-speed", "Relative speed"],
         ["chart-fuel", "Fuel fraction"]].map(function (pair) {
          return React.createElement("figure", { className: "chart", key: pair[0] },
            React.createElement("figcaption", null, pair[1]),
            React.createElement("canvas", {
              id: pair[0], height: 150, tabIndex: 0,
              role: "img", "aria-label": pair[1] + " over time. Use the replay scrubber to inspect values.",
            }));
        })),
      metrics ? React.createElement("table", { className: "table" },
        React.createElement("caption", null, "Latest episode: " + (metrics.payload.label || "")),
        React.createElement("tbody", null,
          [["Seed", metrics.payload.seed], ["Steps", metrics.payload.steps],
           ["Start range", ObsState.fmtMeters(metrics.payload.start_dist_m)],
           ["End range", ObsState.fmtMeters(metrics.payload.end_dist_m)],
           ["End speed", metrics.payload.end_speed_mps + " m/s"],
           ["Fuel used", metrics.payload.fuel_used],
           ["Delta-v", metrics.payload.dv_mps + " m/s"],
           ["Interlock fires", metrics.payload.safety_overrides],
           ["Result", metrics.payload.success ? "docked" : metrics.payload.terminal_reason],
          ].map(function (kv, i) {
            return React.createElement("tr", { key: i },
              React.createElement("th", { scope: "row" }, kv[0]),
              React.createElement("td", null, String(kv[1] === undefined ? "—" : kv[1])));
          })))
        : React.createElement("p", { className: "muted" }, "No episode summary yet."));
  }

  function eventsView() {
    var events = (store.history.safety_event || []).slice().reverse();
    var traces = (store.history.llm_trace || []).slice().reverse();
    var curriculum = store.latest.curriculum_state;
    var tensorboard = store.latest.tensorboard;
    function seek(e) {
      if (player) { ObsState.playerSeek(player, store, e.stamp_ns); requestRender(); }
    }
    return React.createElement("div", { className: "events-grid" },
      React.createElement("section", { className: "panel" },
        React.createElement("h2", null, "Safety log"),
        events.length ? React.createElement("ol", { className: "log", "aria-live": "polite" },
          events.map(function (e, i) {
            return React.createElement("li", { key: i, className: "log-line" },
              React.createElement("button", {
                className: "log-jump", onClick: function () { seek(e); },
                title: player ? "Seek replay to this event" : "Load a replay to seek",
                "aria-label": "Event " + e.payload.reason_code + " at " + ObsState.fmtClock(e.stamp_ns),
              }, ObsState.fmtClock(e.stamp_ns)),
              React.createElement("span", { className: "code warn" }, e.payload.reason_code),
              React.createElement("span", null, e.payload.reason));
          })) : React.createElement("p", { className: "muted" },
            "No interlock fires recorded. Connect the bridge or load a replay.")),
      React.createElement("div", null,
        React.createElement("section", { className: "panel" },
          React.createElement("h2", null, "Planner traces"),
          traces.length ? traces.slice(0, 5).map(function (t, i) {
            var p = t.payload;
            return React.createElement("div", { className: "trace", key: i },
              React.createElement("span", { className: "code" }, p.status),
              React.createElement("span", null, (p.provider || "unknown") + " · " + p.token_count + " tokens · $" + Number(p.cost_usd).toFixed(4)));
          }) : React.createElement("p", { className: "muted" }, "No planner traces yet.")),
        React.createElement("section", { className: "panel" },
          React.createElement("h2", null, "Curriculum"),
          curriculum ? React.createElement("p", null,
            curriculum.payload.current_stage + " · episode " + curriculum.payload.episode_index +
            " · " + Number(curriculum.payload.success_rate).toFixed(2) + " success")
            : React.createElement("p", { className: "muted" }, "No curriculum state yet.")),
        tensorboard ? React.createElement("section", { className: "panel" },
          React.createElement("h2", null, "Training"),
          React.createElement("a", { href: tensorboard.payload.url }, "Open TensorBoard"),
          React.createElement("p", { className: "muted" }, tensorboard.payload.run_dir)) : null));
  }

  function replayView() {
    return React.createElement("div", null,
      React.createElement("div", { className: "cards" },
        SCENARIOS.map(function (s) {
          return React.createElement("article", { className: "card", key: s.key },
            React.createElement("h2", null, s.name),
            React.createElement("p", { className: "muted" }, s.hint),
            React.createElement("button", { onClick: function () { loadScenario(s.key); } },
              "Load " + s.key));
        }),
        React.createElement("article", { className: "card drop",
          onDragOver: function (e) { e.preventDefault(); },
          onDrop: function (e) {
            e.preventDefault();
            if (e.dataTransfer.files[0]) loadReplayFile(e.dataTransfer.files[0]);
          } },
          React.createElement("h2", null, "Your own trace"),
          React.createElement("p", { className: "muted" },
            "Drop a record_episode JSON or envelope array here, or pick a file."),
          React.createElement("label", { className: "filelike" }, "Choose file",
            React.createElement("input", {
              type: "file", accept: ".json", hidden: true,
              "aria-label": "Choose replay JSON file",
              onChange: function (e) { if (e.target.files[0]) loadReplayFile(e.target.files[0]); },
            })))),
      React.createElement("h2", null, "Transport"),
      transport(),
      store.errors.length ? React.createElement("ul", { className: "notices" },
        store.errors.slice(-5).map(function (msg, i) {
          return React.createElement("li", { key: i }, msg);
        })) : null);
  }

  function app() {
    return React.createElement("div", null, header(),
      React.createElement("div", { className: "filters", role: "group", "aria-label": "Stream filters" },
        ObsState.STREAMS.map(function (s) {
          return React.createElement("button", {
            key: s, className: "chip" + (unsubscribed[s] ? " off" : ""),
            onClick: function () { toggleStream(s); },
            "aria-pressed": !unsubscribed[s],
            title: unsubscribed[s] ? "Resubscribe " + s : "Unsubscribe " + s,
          }, s);
        })),
      React.createElement("main", { id: "content", tabIndex: -1 },
        view === "mission" ? missionView()
        : view === "telemetry" ? telemetryView()
        : view === "events" ? eventsView() : replayView()));
  }

  /* ---------------- render loop ---------------- */
  function requestRender() {
    var now = Date.now();
    if (now - lastRenderAt < 200) {
      if (!requestRender.queued) {
        requestRender.queued = true;
        setTimeout(function () { requestRender.queued = false; requestRender(); }, 220);
      }
      return;
    }
    lastRenderAt = now;
    render();
  }

  function render() {
    if (typeof React === "undefined" || typeof ReactDOM === "undefined") {
      var fallback = document.getElementById("fallback");
      if (fallback) fallback.style.display = "block";
      return;
    }
    ReactDOM.render(React.createElement(app), document.getElementById("root"));
    if (view === "mission") {
      var canvas = document.getElementById("scene3d");
      if (!glFailed && typeof THREE !== "undefined" && canvas && (!three || three.canvas !== canvas)) {
        if (!initThree(canvas)) {
          three = null;
          glFailed = true;
        }
        var note = document.getElementById("cdn-note");
        if (note) note.style.display = glFailed ? "block" : "none";
      }
      if (three && !three.broken) updateThree();
      else {
        var flat = document.getElementById("scene2d");
        if (flat) drawFallback2D(flat);
      }
    }
    if (view === "telemetry") drawCharts();
  }

  var lastFrameAt = 0;
  function loop(nowMs) {
    window.requestAnimationFrame(loop);
    if (player && player.playing) {
      if (!lastFrameAt) lastFrameAt = nowMs;
      var dtNs = (nowMs - lastFrameAt) * 1e6;
      lastFrameAt = nowMs;
      var reduced = window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (!reduced && ObsState.playerTick(player, store, dtNs)) requestRender();
      else if (reduced) { player.playing = false; requestRender(); }
    } else {
      lastFrameAt = nowMs;
    }
    if (view === "mission" && three) updateThree();
  }

  /* ---------------- keyboard + deep links ---------------- */
  document.addEventListener("keydown", function (event) {
    var tag = (event.target && event.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (event.key === " ") {
      if (player) { event.preventDefault(); ObsState.playerToggle(player); requestRender(); }
    } else if (event.key === "ArrowRight" && player) {
      ObsState.playerSeek(player, store, player.clockNs + 5e9); requestRender();
    } else if (event.key === "ArrowLeft" && player) {
      ObsState.playerSeek(player, store, player.clockNs - 5e9); requestRender();
    } else if (event.key >= "1" && event.key <= "4") {
      setView(["mission", "telemetry", "events", "replay"][Number(event.key) - 1]);
    } else if (event.key === "f" || event.key === "F") {
      followCam = !followCam; requestRender();
    }
  });

  function routeFromHash() {
    var hash = (window.location.hash || "").replace("#/", "");
    if (["mission", "telemetry", "events", "replay"].indexOf(hash) >= 0) view = hash;
  }
  window.addEventListener("hashchange", function () { routeFromHash(); requestRender(); });

  window.addEventListener("DOMContentLoaded", function () {
    routeFromHash();
    render();
    window.requestAnimationFrame(loop);
    connect();
  });

  // Debug/testing handle (used by the jsdom smoke harness).
  window.SpaceForgeDashboard = {
    version: 1, store: store,
    get player() { return player; },
    get view() { return view; },
    setView: setView, connect: connect, refresh: requestRender,
    loadTrace: loadTraceObject, loadEnvelopes: loadEnvelopes,
  };
})();
