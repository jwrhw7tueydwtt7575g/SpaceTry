/* SpaceForge docking simulator: user interface.
 * Depends on SFSim (sim.js) and SF_POLICY (policy.js). No frameworks, no network.
 */
(function () {
  'use strict';
  const S = window.SFSim;
  const POLICY = new S.Policy(window.SF_POLICY);
  const C = S.DEFAULTS;
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs = {}, html = '') => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (html) e.innerHTML = html;
    return e;
  };
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  // ---------------------------------------------------------- palette
  // Canvas colours come from the CSS tokens so the light and dark themes match.
  const COL = {};
  const SOURCE_COL = {}, MODE_COL = {}, OUTCOME_COL = {};
  let GRID = '150,160,175';
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name) => cs.getPropertyValue(name).trim();
    Object.assign(COL, {
      bg: v('--scene'), surface: v('--surface'), text: v('--text'), muted: v('--muted'), faint: v('--faint'), line: v('--border'),
      accent: v('--accent'), amber: v('--amber'), orange: v('--orange'), violet: v('--violet'), green: v('--green'), red: v('--red'),
    });
    GRID = v('--grid') || GRID;
    Object.assign(SOURCE_COL, { none: COL.faint, primary: COL.accent, backup: COL.amber, brake: COL.orange, veto: COL.violet, hold: COL.violet });
    Object.assign(MODE_COL, { off: COL.red, v1: COL.violet, v2: COL.accent });
    Object.assign(OUTCOME_COL, { docked: COL.green, collision: COL.red, timeout: COL.faint, fuel: COL.orange, out_of_bounds: COL.violet });
  }
  readTheme();
  const OUTCOME_TXT = { docked: 'Docked', collision: 'Collision', timeout: 'Timed out', fuel: 'Out of fuel', out_of_bounds: 'Out of bounds' };
  const rgba = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };
  const grid = (a) => `rgba(${GRID},${a})`;
  const SANS = getComputedStyle(document.documentElement).getPropertyValue('--sans').trim() || 'system-ui, sans-serif';
  const font = (px, weight = 400) => `${weight} ${px}px ${SANS}`;

  // ---------------------------------------------------------- catalogue
  const PILOTS = {
    glideslope: { label: 'Glideslope', name: 'Glideslope', desc: 'Exponential glideslope with CW feed-forward. The v2 filter also uses it as the backup controller.' },
    ppo: { label: 'PPO', name: 'PPO policy', desc: `Shipped PPO checkpoint (${POLICY.source}), a 10-64-64-3 network evaluated in the page.` },
    hot: { label: 'Direct', name: 'Direct thrust', desc: 'Full thrust along the line of sight to the target, never brakes. A worst case for testing the filters.' },
    manual: { label: 'Manual', name: 'Manual', desc: 'W/S, A/D and Q/E (or the arrow keys) thrust along R-bar, V-bar and H-bar.' },
  };
  const SAFETY = {
    off: { label: 'Off', desc: 'Commands go straight to the thrusters.' },
    v1: { label: 'v1 veto', desc: 'Inside 5 m and above 0.2 m/s, any command that does not reduce speed is replaced by zero thrust.' },
    v2: { label: 'v2 filter', desc: 'Predicts each step against a range-dependent closing-speed limit. Steps over the limit are flown by the glideslope or a brake.' },
  };
  const FAULTS = [
    { key: 'stuck', label: 'Stuck V-bar thruster', sub: 'held at +0.6', set: (f) => { f.stuckAxis = 1; f.stuckValue = 0.6; } },
    { key: 'bias', label: 'Thrust bias', sub: '+0.05 R-bar, −0.03 V-bar', set: (f) => { f.bias = [0.05, -0.03, 0]; } },
    { key: 'dropout', label: 'Command dropout', sub: '20% of steps get zero thrust', set: (f) => { f.dropoutProb = 0.2; } },
    { key: 'freeze', label: 'Sensor freeze', sub: '20% of readings are stale', set: (f) => { f.sensorFreezeProb = 0.2; } },
    { key: 'leak', label: 'Fuel leak', sub: '0.05% of the tank per second', set: (f) => { f.fuelLeakRate = 0.0005; } },
  ];
  const SCENARIOS = [
    { key: 'nominal', title: 'Glideslope approach', sub: 'Glideslope pilot from (12, −7, 3.5) m.', pilot: 'glideslope', safety: 'v2', pos: [12, -7, 3.5] },
    { key: 'ppo', title: 'PPO approach', sub: 'PPO policy from (−14, 11, 6) m with a small initial drift.', pilot: 'ppo', safety: 'v2', pos: [-14, 11, 6], vel: [0.003, 0, -0.002] },
    { key: 'direct', title: 'Direct thrust, R-bar', sub: 'Full thrust at the target from 6 m on R-bar.', pilot: 'hot', safety: 'v2', pos: [6, 0, 0] },
    { key: 'oblique', title: 'Direct thrust, oblique', sub: 'Full thrust at the target from (3, −8, 2) m.', pilot: 'hot', safety: 'v2', pos: [3, -8, 2] },
    { key: 'low-fuel', title: 'Low fuel with leak', sub: '7% fuel, a 0.05%/s leak and the direct-thrust pilot.', pilot: 'hot', safety: 'v2', pos: [7, -2, 1], fuel: 0.07, faults: ['leak'] },
    { key: 'stuck', title: 'Stuck thruster', sub: 'V-bar thruster held at +0.6 with the direct-thrust pilot.', pilot: 'hot', safety: 'v2', pos: [8, 4, 0], faults: ['stuck'] },
    { key: 'manual', title: 'Manual', sub: 'Fly it yourself from (9, −12, 0) m.', pilot: 'manual', safety: 'v2', pos: [9, -12, 0] },
  ];
  // Scenario ids used in earlier links.
  const SCENARIO_ALIASES = { learned: 'ppo', slam: 'direct', fumes: 'low-fuel' };
  const SPEEDS = [5, 15, 40, 100];
  const VIEWS = { '3d': { yaw: -0.62, pitch: 0.42 }, plane: { yaw: 0, pitch: 0 }, side: { yaw: Math.PI / 2, pitch: 0 } };

  // ---------------------------------------------------------- state
  const st = {
    tab: 'fly', scenario: 'nominal', pilot: 'glideslope', safety: 'v2', faults: new Set(),
    start: { pos: [12, -7, 3.5], vel: [0, 0, 0], fuel: 1 },
    speed: 15, playing: false, cursor: null, runs: [], log: [],
    view: '3d', cam: { yaw: VIEWS['3d'].yaw, pitch: VIEWS['3d'].pitch, zoom: 18, panX: 0, panY: 0, auto: true },
    keys: new Set(), dirty: true, doneAt: 0, mc: null, mcMode: 'v2',
  };

  function faultConfig(keys = st.faults) {
    const f = { stuckAxis: null, stuckValue: 0, bias: [0, 0, 0], dropoutProb: 0, sensorFreezeProb: 0, fuelLeakRate: 0 };
    for (const def of FAULTS) if (keys.has(def.key)) def.set(f);
    return f;
  }

  function controllerFor(pilot) {
    if (pilot === 'glideslope') return (env) => S.glideslope(env.pos, env.vel, C.n, C.dt, C.maxAccel);
    if (pilot === 'hot') return (env) => S.hotApproach(env.pos);
    if (pilot === 'ppo') return (env, obs) => POLICY.act(obs);
    return () => {
      const k = st.keys;
      return [(k.has('w') ? 1 : 0) - (k.has('s') ? 1 : 0), (k.has('d') ? 1 : 0) - (k.has('a') ? 1 : 0), (k.has('q') ? 1 : 0) - (k.has('e') ? 1 : 0)];
    };
  }

  // ---------------------------------------------------------- runs
  function snapshot(env, extra) {
    const dist = env.dist, speed = env.speed;
    return Object.assign({
      t: env.t, pos: env.pos.slice(), vel: env.vel.slice(), dist, speed,
      closing: S.closingSpeed(env.pos, env.vel), limit: S.closingLimit(dist, C), fuel: env.fuel,
      dv: env.dv, overrides: env.overrides, cmd: [0, 0, 0], applied: [0, 0, 0], executed: [0, 0, 0],
      source: 'none', override: null, reason: '', fault: null, value: null,
    }, extra || {});
  }

  function makeRun(safety, label) {
    const env = new S.Env({ safety, pos: st.start.pos, vel: st.start.vel, fuel: st.start.fuel, faults: faultConfig(), seed: 7 });
    const obs = env.reset(st.start.pos, st.start.vel, st.start.fuel);
    const run = { env, safety, label, obs, frames: [], lastSource: 'none', lastLogT: -99, enteredKeepOut: env.dist < C.safeDist, floorLogged: false };
    run.frames.push(snapshot(env, { value: st.pilot === 'ppo' ? POLICY.value(obs) : null }));
    return run;
  }

  function rebuild(keepCamera) {
    st.playing = false;
    st.cursor = null;
    st.log = [];
    st.doneAt = 0;
    if (st.tab === 'compare') st.runs = ['off', 'v1', 'v2'].map((m) => makeRun(m, SAFETY[m].label));
    else st.runs = [makeRun(st.safety, PILOTS[st.pilot].label)];
    if (!keepCamera) st.cam.auto = true;
    logEvent(0, '', 'Ready', `${PILOTS[st.pilot].name}, ${st.tab === 'compare' ? 'all three filters' : SAFETY[st.safety].label}, ${fmtLen(norm(st.start.pos))} from the target.`);
    hideBanner();
    syncControls();
    st.dirty = true;
  }

  const norm = S.norm;
  const allDone = () => st.runs.every((r) => r.env.done);
  const atStart = () => st.runs.every((r) => r.frames.length === 1);

  function stepAll() {
    const ctrl = controllerFor(st.pilot);
    for (const run of st.runs) {
      const env = run.env;
      if (env.done) continue;
      const obs = run.obs;
      const r = env.step(ctrl(env, obs));
      run.obs = env.observe();
      const source = r.override === 'emergency_hold' ? 'hold' : r.source;
      run.frames.push(snapshot(env, {
        cmd: r.cmd, applied: r.applied, executed: r.executed, source, override: r.override, reason: r.reason,
        fault: r.fault, value: st.pilot === 'ppo' ? POLICY.value(run.obs) : null,
      }));
      if (st.tab === 'fly') detectEvents(run, r, source);
    }
    if (allDone() && !st.doneAt) {
      st.doneAt = performance.now();
      st.playing = false;
      if (st.tab === 'fly') showBanner(st.runs[0]);
    }
  }

  // ---------------------------------------------------------- flight log
  function logEvent(t, tone, title, text) {
    st.log.unshift({ t, tone, title, text });
    if (st.log.length > 80) st.log.pop();
    st.logDirty = true;
  }

  function detectEvents(run, r, source) {
    const env = run.env, t = env.t;
    if (source !== run.lastSource && source !== 'none') {
      if (source !== 'primary' && t - run.lastLogT > 4) {
        const a = authorityText(source, r.reason);
        logEvent(t, source === 'veto' || source === 'hold' ? 'violet' : 'amber', a.who, a.why);
        run.lastLogT = t;
      } else if (source === 'primary' && run.lastSource !== 'none' && t - run.lastLogT > 4) {
        logEvent(t, '', 'Pilot in control again', 'Predicted step is within the limit.');
        run.lastLogT = t;
      }
      run.lastSource = source;
    }
    if (!run.enteredKeepOut && env.dist < C.safeDist) {
      run.enteredKeepOut = true;
      const cl = S.closingSpeed(env.pos, env.vel);
      logEvent(t, cl > C.maxSafeClosing ? 'red' : '', 'Inside 5 m', `Closing at ${cl.toFixed(3)} m/s (limit 0.200 m/s).`);
    }
    if (!run.floorLogged && env.fuel < C.fuelFloor) {
      run.floorLogged = true;
      const why = { v2: 'The filter now only spends fuel on braking.', v1: 'The interlock now blocks all thrust.', off: 'No filter, so the pilot keeps thrusting.' };
      logEvent(t, 'amber', 'Fuel below 5%', why[run.env.safety]);
    }
    if (env.done) {
      const tone = env.outcome === 'docked' ? 'green' : env.outcome === 'collision' ? 'red' : 'amber';
      logEvent(t, tone, OUTCOME_TXT[env.outcome], outcomeLine(env));
    }
  }

  function outcomeLine(env) {
    const f = env.outcome === 'collision' ? ` at ${env.speed.toFixed(2)} m/s` : '';
    return `t = ${env.t.toFixed(0)} s${f}, Δv ${env.dv.toFixed(2)} m/s, ${env.overrides} interventions`;
  }

  // What the overlay and the event log say about who is flying.
  function authorityText(source, reason) {
    if (source === 'backup') return { who: 'Backup in control', why: 'The pilot command would exceed the closing-speed limit; the glideslope flies this step.' };
    if (source === 'brake') {
      return /fuel/i.test(reason || '')
        ? { who: 'Brake', why: 'Fuel is below 5% and the craft is on a collision course.' }
        : { who: 'Brake', why: 'Closing speed is far over the limit; braking.' };
    }
    if (source === 'veto') return { who: 'Thrust vetoed', why: 'Inside 5 m above 0.2 m/s and the command does not reduce speed.' };
    if (source === 'hold') return { who: 'Hold', why: 'Fuel is below 5%; thrust stopped.' };
    return { who: `${PILOTS[st.pilot].name} in control`, why: st.safety === 'off' ? 'No filter.' : 'Predicted step is within the limit.' };
  }

  // ---------------------------------------------------------- formatting
  function fmtLen(m) {
    if (m >= 100) return m.toFixed(0) + ' m';
    if (m >= 10) return m.toFixed(1) + ' m';
    if (m >= 1) return m.toFixed(2) + ' m';
    return (m * 100).toFixed(1) + ' cm';
  }
  function splitLen(m) {
    if (m >= 100) return [m.toFixed(0), 'm'];
    if (m >= 10) return [m.toFixed(1), 'm'];
    if (m >= 1) return [m.toFixed(2), 'm'];
    return [(m * 100).toFixed(1), 'cm'];
  }
  const fmtT = (t) => `t = ${Math.round(t)} s`;
  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  }

  // ---------------------------------------------------------- canvas plumbing
  const canvases = {};
  function setupCanvas(id) {
    const cv = $(id);
    const ctx = cv.getContext('2d');
    const rec = { cv, ctx, w: 0, h: 0, dpr: 1 };
    const fit = () => {
      const r = cv.getBoundingClientRect();
      rec.dpr = Math.min(window.devicePixelRatio || 1, 2);
      rec.w = Math.max(1, r.width);
      rec.h = Math.max(1, r.height);
      cv.width = Math.round(rec.w * rec.dpr);
      cv.height = Math.round(rec.h * rec.dpr);
      st.dirty = true;
    };
    new ResizeObserver(fit).observe(cv);
    fit();
    canvases[id] = rec;
    return rec;
  }
  function begin(rec) {
    const { ctx, dpr } = rec;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rec.w, rec.h);
    return ctx;
  }

  // ---------------------------------------------------------- 3-D projection
  // World: LVLH Hill frame [x radial (R-bar), y along-track (V-bar), z cross-track (H-bar)].
  // Display: X = V-bar (right), Y = R-bar (up), Z = H-bar (toward viewer), then yaw/pitch.
  function projector(cam, w, h) {
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const s = (Math.min(w, h) * 0.5) / cam.zoom;
    const ox = w / 2 + cam.panX, oy = h / 2 + cam.panY;
    const rot = (p) => {
      const X = p[1], Y = p[0], Z = p[2];
      const X1 = X * cy + Z * sy, Z1 = -X * sy + Z * cy;
      return [X1, Y * cp - Z1 * sp, Y * sp + Z1 * cp];
    };
    const P = (p) => { const r = rot(p); return [ox + r[0] * s, oy - r[1] * s, r[2]]; };
    P.rot = rot; P.s = s; P.ox = ox; P.oy = oy;
    return P;
  }

  function drawBox(ctx, P, c, half, base, alpha = 1) {
    const [hx, hy, hz] = half;
    const v = [];
    for (const dx of [-1, 1]) for (const dy of [-1, 1]) for (const dz of [-1, 1]) v.push(P([c[0] + dx * hx, c[1] + dy * hy, c[2] + dz * hz]));
    // vertex index = (dx>0)*4 + (dy>0)*2 + (dz>0)
    const faces = [
      [[0, 1, 3, 2], [-1, 0, 0]], [[4, 5, 7, 6], [1, 0, 0]],
      [[0, 1, 5, 4], [0, -1, 0]], [[2, 3, 7, 6], [0, 1, 0]],
      [[0, 2, 6, 4], [0, 0, -1]], [[1, 3, 7, 5], [0, 0, 1]],
    ];
    const L = [-0.35, 0.75, 0.56];
    const rgb = parseInt(base.slice(1), 16);
    for (const [idx, n] of faces) {
      const rn = P.rot(n);
      if (rn[2] <= 0.001) continue;
      const lum = 0.38 + 0.62 * Math.max(0, rn[0] * L[0] + rn[1] * L[1] + rn[2] * L[2]);
      const r = Math.round(((rgb >> 16) & 255) * lum), g = Math.round(((rgb >> 8) & 255) * lum), b = Math.round((rgb & 255) * lum);
      ctx.beginPath();
      idx.forEach((k, i) => (i ? ctx.lineTo(v[k][0], v[k][1]) : ctx.moveTo(v[k][0], v[k][1])));
      ctx.closePath();
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
      ctx.fill();
      ctx.strokeStyle = grid(0.3 * alpha);
      ctx.lineWidth = 0.6;
      ctx.stroke();
    }
  }

  function ring(ctx, P, radius, plane, n = 96) {
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const a = (i / n) * Math.PI * 2, c = Math.cos(a) * radius, s = Math.sin(a) * radius;
      const p = plane === 'RV' ? [c, s, 0] : plane === 'VH' ? [0, c, s] : [c, 0, s];
      const q = P(p);
      i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]);
    }
  }

  function coastFrom(pos, vel, horizon = 240, every = 2) {
    const pts = [pos];
    let p = pos, v = vel, impact = null;
    for (let t = every; t <= horizon; t += every) {
      [p, v] = S.cwStep(p, v, [0, 0, 0], every, C.n);
      pts.push(p);
      if (norm(p) < C.dockDist) { impact = { t, speed: norm(v) }; break; }
    }
    return { pts, impact };
  }

  function frameAt(run, cursor) {
    const i = cursor == null ? run.frames.length - 1 : Math.min(cursor, run.frames.length - 1);
    return run.frames[i];
  }
  const cursorIndex = () => (st.cursor == null ? Math.max(...st.runs.map((r) => r.frames.length - 1)) : st.cursor);

  function updateCamera() {
    const cam = st.cam, target = VIEWS[st.view];
    let moving = false;
    const ease = 0.14;
    for (const k of ['yaw', 'pitch']) {
      const d = target[k] - cam[k];
      if (Math.abs(d) > 1e-4 && !cam.free) { cam[k] += d * ease; moving = true; }
    }
    if (cam.auto) {
      const frames = st.runs.map((r) => frameAt(r, st.cursor));
      const far = Math.max(...frames.map((f) => f.dist));
      const want = clamp(Math.max(far * 1.45, 7.2), 7.2, 140);
      const z = Math.exp(lerp(Math.log(cam.zoom), Math.log(want), 0.08));
      if (Math.abs(z - cam.zoom) / cam.zoom > 1e-4) moving = true;
      cam.zoom = z;
      cam.panX = lerp(cam.panX, 0, 0.15); cam.panY = lerp(cam.panY, 0, 0.15);
    }
    return moving;
  }

  // ---------------------------------------------------------- scene
  function drawScene() {
    const rec = canvases.scene;
    const ctx = begin(rec);
    const { w, h } = rec;
    const P = projector(st.cam, w, h);
    const s = P.s;

    ctx.fillStyle = COL.bg; ctx.fillRect(0, 0, w, h);

    // reference grids: orbit-plane wall and V/H floor, faded by viewing angle
    const step = niceStep(st.cam.zoom / 3.2);
    const ext = Math.ceil((st.cam.zoom * 1.9) / step) * step;
    const floorA = clamp(Math.abs(Math.sin(st.cam.pitch)) * 2.2, 0, 1);
    const wallA = clamp(Math.cos(st.cam.pitch) * Math.abs(Math.cos(st.cam.yaw)) - floorA * 0.55, 0, 1);
    const sideA = clamp(Math.cos(st.cam.pitch) * Math.abs(Math.sin(st.cam.yaw)) - floorA * 0.55, 0, 1);
    const drawGrid = (alpha, mk) => {
      if (alpha <= 0.02) return;
      for (let g = -ext; g <= ext + 1e-9; g += step) {
        const major = Math.abs(g) < 1e-9;
        const fade = 1 - Math.abs(g) / (ext + step);
        ctx.strokeStyle = major ? grid(0.38 * alpha) : grid(0.13 * alpha * fade);
        ctx.lineWidth = major ? 1 : 0.8;
        for (const [a, b] of mk(g)) {
          const p = P(a), q = P(b);
          ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
        }
      }
    };
    drawGrid(floorA, (g) => [[[0, g, -ext], [0, g, ext]], [[0, -ext, g], [0, ext, g]]]);
    drawGrid(wallA, (g) => [[[g, -ext, 0], [g, ext, 0]], [[-ext, g, 0], [ext, g, 0]]]);
    drawGrid(sideA, (g) => [[[g, 0, -ext], [g, 0, ext]], [[-ext, 0, g], [ext, 0, g]]]);

    // axis labels on the grid
    ctx.font = font(11);
    ctx.fillStyle = grid(0.85);
    const lab = (p, t) => { const q = P(p); if (q[0] > 8 && q[0] < w - 60 && q[1] > 16 && q[1] < h - 8) ctx.fillText(t, q[0] + 4, q[1] - 4); };
    lab([0, ext * 0.92, 0], '+V-bar');
    lab([ext * 0.85, 0, 0], '+R-bar (zenith)');
    if (floorA > 0.3 || sideA > 0.3) lab([0, 0, ext * 0.9], '+H-bar');

    const origin = P([0, 0, 0]);

    // keep-out sphere
    const ko = C.safeDist * s;
    ctx.fillStyle = rgba(COL.amber, 0.05);
    ctx.beginPath(); ctx.arc(origin[0], origin[1], ko, 0, Math.PI * 2); ctx.fill();
    ctx.setLineDash([5, 5]); ctx.strokeStyle = rgba(COL.amber, 0.7); ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = rgba(COL.amber, 0.22); ctx.lineWidth = 1;
    ring(ctx, P, C.safeDist, 'VH'); ctx.stroke();
    ring(ctx, P, C.safeDist, 'RV'); ctx.stroke();
    ctx.fillStyle = COL.amber;
    ctx.font = font(11);
    ctx.fillText('5 m keep-out', origin[0] + ko * 0.72 + 6, origin[1] - ko * 0.72);

    // capture sphere
    const cr = C.captureRadius * s;
    ctx.beginPath(); ctx.arc(origin[0], origin[1], cr, 0, Math.PI * 2);
    ctx.fillStyle = rgba(COL.green, 0.1); ctx.fill();
    ctx.strokeStyle = COL.green; ctx.lineWidth = 1; ctx.stroke();

    const cursor = st.cursor;
    const mainRun = st.runs[0];

    // coast arc: where the craft goes if thrust stops now
    const coastRuns = st.tab === 'fly' ? [mainRun] : [];
    for (const run of coastRuns) {
      const f = frameAt(run, cursor);
      if (run.env.done && cursor == null) continue;
      const { pts, impact } = coastFrom(f.pos, f.vel, 240);
      ctx.setLineDash([2, 5]);
      ctx.strokeStyle = impact && impact.speed >= C.maxDockSpeed ? rgba(COL.red, 0.7) : rgba(COL.text, 0.3);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      pts.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
      ctx.stroke(); ctx.setLineDash([]);
      if (impact && impact.speed >= C.maxDockSpeed) {
        const q = P(pts[pts.length - 1]);
        ctx.strokeStyle = COL.red; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(q[0] - 4, q[1] - 4); ctx.lineTo(q[0] + 4, q[1] + 4); ctx.moveTo(q[0] + 4, q[1] - 4); ctx.lineTo(q[0] - 4, q[1] + 4); ctx.stroke();
      }
    }

    // trails (+ floor shadow and drop line in 3-D)
    const trailColor = (run, fr) => {
      if (st.tab === 'compare') return MODE_COL[run.safety];
      if (fr.closing > fr.limit + 1e-9) return COL.red;
      return SOURCE_COL[fr.source] || COL.accent;
    };
    for (const run of st.runs) {
      const end = cursor == null ? run.frames.length - 1 : Math.min(cursor, run.frames.length - 1);
      const fr = run.frames;
      if (floorA > 0.25) {
        ctx.strokeStyle = grid(0.25 * floorA); ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i <= end; i++) { const q = P([0, fr[i].pos[1], fr[i].pos[2]]); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }
        ctx.stroke();
      }
      ctx.lineWidth = 2;
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      let i = 1;
      while (i <= end) {
        const col = trailColor(run, fr[i]);
        ctx.strokeStyle = col;
        ctx.beginPath();
        const q0 = P(fr[i - 1].pos); ctx.moveTo(q0[0], q0[1]);
        while (i <= end && trailColor(run, fr[i]) === col) { const q = P(fr[i].pos); ctx.lineTo(q[0], q[1]); i++; }
        ctx.stroke();
      }
      // start marker
      const s0 = P(fr[0].pos);
      ctx.strokeStyle = rgba(COL.text, 0.55); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(s0[0], s0[1], 4.5, 0, Math.PI * 2); ctx.stroke();
    }

    // station (draw after trails so it reads as solid)
    const parts = [
      { c: [0, 0, 1.3], half: [0.025, 0.42, 0.78], col: '#2c5fb8' },
      { c: [0, 0, -1.3], half: [0.025, 0.42, 0.78], col: '#2c5fb8' },
      { c: [0, 0, 0], half: [0.05, 0.05, 0.55], col: '#9aa8ba' },
      { c: [0, 0, 0], half: [0.24, 0.34, 0.24], col: '#d4dce6' },
      { c: [0, -0.4, 0], half: [0.12, 0.06, 0.12], col: '#e8b04a' },
    ];
    parts.sort((a, b) => P(a.c)[2] - P(b.c)[2]);
    for (const p of parts) drawBox(ctx, P, p.c, p.half, p.col);

    // chasers
    const placed = [];
    const chasers = st.runs.map((run) => ({ run, f: frameAt(run, cursor) }));
    chasers.sort((a, b) => P(a.f.pos)[2] - P(b.f.pos)[2]);
    for (const { run, f } of chasers) {
      const q = P(f.pos);
      const col = st.tab === 'compare' ? MODE_COL[run.safety] : SOURCE_COL[f.source === 'none' ? 'primary' : f.source];
      // drop line
      if (floorA > 0.25) {
        const fl = P([0, f.pos[1], f.pos[2]]);
        ctx.setLineDash([2, 3]); ctx.strokeStyle = grid(0.5 * floorA); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(q[0], q[1]); ctx.lineTo(fl[0], fl[1]); ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.ellipse(fl[0], fl[1], 4, 2, 0, 0, Math.PI * 2); ctx.strokeStyle = grid(0.55 * floorA); ctx.stroke();
      }
      // plume opposite to the executed acceleration
      const ex = f.executed, em = norm(ex);
      if (em > 1e-6) {
        const len = Math.max(0.55 * (em / C.maxAccel), 14 / s * (em / C.maxAccel));
        const tail = P([f.pos[0] - (ex[0] / em) * len, f.pos[1] - (ex[1] / em) * len, f.pos[2] - (ex[2] / em) * len]);
        const g = ctx.createLinearGradient(q[0], q[1], tail[0], tail[1]);
        g.addColorStop(0, rgba(col, 0.95)); g.addColorStop(1, rgba(col, 0));
        ctx.strokeStyle = g; ctx.lineWidth = 5; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(q[0], q[1]); ctx.lineTo(tail[0], tail[1]); ctx.stroke();
      }
      const bodyHalf = Math.max(0.17, 3.2 / s);
      drawBox(ctx, P, f.pos, [bodyHalf, bodyHalf * 1.25, bodyHalf], st.tab === 'compare' ? col : COL.text);
      // label
      ctx.font = font(12);
      const label = st.tab === 'compare' ? `${run.label}: ${fmtLen(f.dist)}` : `${fmtLen(f.dist)}, ${f.speed.toFixed(3)} m/s`;
      const tw = ctx.measureText(label).width;
      const lx = clamp(q[0] + 14, 6, w - tw - 16);
      let ly = clamp(q[1] - 14, 18, h - 8);
      for (let guard = 0; guard < 6; guard++) {
        const hit = placed.find((b) => Math.abs(b.y - ly) < 21 && lx < b.x + b.w + 10 && b.x < lx + tw + 10);
        if (!hit) break;
        ly = hit.y + 22;
      }
      placed.push({ x: lx, y: ly, w: tw });
      if (Math.abs(ly - (q[1] - 14)) > 4) {
        ctx.strokeStyle = rgba(COL.text, 0.3); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(q[0] + 4, q[1] - 2); ctx.lineTo(lx - 6, ly - 4); ctx.stroke();
      }
      ctx.fillStyle = rgba(COL.surface, 0.88);
      ctx.beginPath(); ctx.roundRect(lx - 6, ly - 14, tw + 12, 20, 4); ctx.fill();
      ctx.fillStyle = st.tab === 'compare' ? col : COL.text;
      ctx.fillText(label, lx, ly);
    }

    // axis triad + scale bar
    const tx = 44, ty = h - 46;
    const tri = [['R', [1, 0, 0], COL.red], ['V', [0, 1, 0], COL.green], ['H', [0, 0, 1], COL.accent]];
    ctx.font = font(11, 600);
    for (const [name, v, col] of tri) {
      const r = P.rot(v);
      const ex2 = tx + r[0] * 22, ey2 = ty - r[1] * 22;
      ctx.strokeStyle = col; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(ex2, ey2); ctx.stroke();
      ctx.fillStyle = col; ctx.fillText(name, ex2 + (r[0] >= 0 ? 3 : -10), ey2 + (r[1] >= 0 ? -2 : 10));
    }
    const target = st.cam.zoom * 0.5;
    const bar = niceStep(target);
    const bw = bar * s;
    ctx.strokeStyle = rgba(COL.text, 0.7); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(w - 20 - bw, h - 22); ctx.lineTo(w - 20, h - 22);
    ctx.moveTo(w - 20 - bw, h - 26); ctx.lineTo(w - 20 - bw, h - 18); ctx.moveTo(w - 20, h - 26); ctx.lineTo(w - 20, h - 18); ctx.stroke();
    ctx.fillStyle = COL.muted; ctx.font = font(11);
    const bl = fmtLen(bar);
    ctx.fillText(bl, w - 20 - bw / 2 - ctx.measureText(bl).width / 2, h - 30);
  }

  // ---------------------------------------------------------- charts
  const PH = { xmin: 0.2 };
  function phaseScale(rec) {
    const far = Math.max(...st.runs.map((r) => r.frames[0].dist), 6);
    const xmax = [10, 20, 40, 80, 160].find((v) => v >= far * 1.1) || 160;
    let ymax = 0.35;
    for (const r of st.runs) for (const f of r.frames) ymax = Math.max(ymax, f.closing * 1.12);
    ymax = Math.ceil(ymax * 10) / 10;
    const L = 46, R = 16, T = 22, B = 26;
    const pw = rec.w - L - R, ph = rec.h - T - B;
    const lx = (d) => L + ((Math.log(Math.max(d, PH.xmin)) - Math.log(PH.xmin)) / (Math.log(xmax) - Math.log(PH.xmin))) * pw;
    const ymin = -0.1;
    const ly = (v) => T + (1 - (clamp(v, ymin, ymax) - ymin) / (ymax - ymin)) * ph;
    return { L, R, T, B, pw, ph, lx, ly, xmax, ymin, ymax };
  }

  function axesText(ctx) { ctx.font = font(11); ctx.fillStyle = COL.faint; }

  function drawPhase() {
    const rec = canvases.phase, ctx = begin(rec);
    const sc = phaseScale(rec);
    const { lx, ly, L, T, pw, ph } = sc;
    // grid
    axesText(ctx);
    for (const d of [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 160]) {
      if (d > sc.xmax) break;
      const x = lx(d);
      ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ph); ctx.stroke();
      ctx.fillText(d < 1 ? d.toFixed(1) : String(d), x - 8, T + ph + 15);
    }
    const ystep = sc.ymax > 1.2 ? 0.5 : sc.ymax > 0.6 ? 0.2 : 0.1;
    const yticks = ystep < 0.2 ? [-0.1] : [];
    for (let v = 0; v <= sc.ymax + 1e-9; v += ystep) yticks.push(+v.toFixed(2));
    for (const v of yticks) {
      const y = ly(v);
      ctx.strokeStyle = Math.abs(v) < 1e-9 ? grid(0.45) : COL.line;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
      ctx.fillText(v.toFixed(1), 8, y + 3);
    }
    // forbidden region above the envelope (taper inside 5 m, braking curve outside)
    const N = 90;
    const envD = (i) => Math.exp(Math.log(PH.xmin) + (i / N) * (Math.log(sc.xmax) - Math.log(PH.xmin)));
    ctx.beginPath();
    ctx.moveTo(lx(PH.xmin), ly(sc.ymax));
    for (let i = 0; i <= N; i++) ctx.lineTo(lx(envD(i)), ly(S.closingLimit(envD(i), C)));
    ctx.lineTo(lx(sc.xmax), ly(sc.ymax));
    ctx.closePath();
    ctx.fillStyle = rgba(COL.red, 0.08); ctx.fill();
    ctx.strokeStyle = COL.red; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= N; i++) { const x = lx(envD(i)), y = ly(S.closingLimit(envD(i), C)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
    ctx.setLineDash([3, 4]); ctx.strokeStyle = COL.faint; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(lx(PH.xmin), ly(C.maxSafeClosing)); ctx.lineTo(lx(sc.xmax), ly(C.maxSafeClosing)); ctx.stroke();
    ctx.setLineDash([]);
    // keep-out edge and capture box
    ctx.strokeStyle = rgba(COL.amber, 0.6); ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(lx(C.safeDist), T); ctx.lineTo(lx(C.safeDist), T + ph); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = rgba(COL.green, 0.14);
    ctx.fillRect(lx(PH.xmin), ly(C.maxDockSpeed), lx(C.dockDist) - lx(PH.xmin), ly(sc.ymin) - ly(C.maxDockSpeed));
    ctx.font = font(11);
    ctx.fillStyle = COL.green; ctx.fillText('capture', lx(PH.xmin) + 4, ly(sc.ymin) - 5);
    ctx.fillStyle = COL.amber; ctx.fillText('5 m', lx(C.safeDist) + 4, T + 12);
    ctx.fillStyle = COL.red; ctx.fillText('over limit', lx(0.9), T + 12);
    // trajectories
    const cursor = cursorIndex();
    for (const run of st.runs) {
      const fr = run.frames, end = Math.min(cursor, fr.length - 1);
      ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
      let i = 1;
      while (i <= end) {
        const col = st.tab === 'compare' ? MODE_COL[run.safety] : (fr[i].closing > fr[i].limit + 1e-9 ? COL.red : SOURCE_COL[fr[i].source]);
        ctx.strokeStyle = col;
        ctx.beginPath(); ctx.moveTo(lx(fr[i - 1].dist), ly(fr[i - 1].closing));
        while (i <= end) {
          const c2 = st.tab === 'compare' ? MODE_COL[run.safety] : (fr[i].closing > fr[i].limit + 1e-9 ? COL.red : SOURCE_COL[fr[i].source]);
          if (c2 !== col) break;
          ctx.lineTo(lx(fr[i].dist), ly(fr[i].closing)); i++;
        }
        ctx.stroke();
      }
      const f = fr[end];
      const col = st.tab === 'compare' ? MODE_COL[run.safety] : COL.text;
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(lx(f.dist), ly(f.closing), 3.5, 0, Math.PI * 2); ctx.fill();
    }
    axesText(ctx);
    ctx.fillText('m/s', 8, T - 9);
    ctx.fillText('m', L + pw + 4, T + ph + 15);
  }

  function drawProfile() {
    const rec = canvases.profile, ctx = begin(rec);
    const L = 40, R = 40, T = 22, B = 34;
    const pw = rec.w - L - R, ph = rec.h - T - B;
    const tmax = Math.max(60, Math.ceil(Math.max(...st.runs.map((r) => r.frames[r.frames.length - 1].t)) / 30) * 30);
    const far = Math.max(...st.runs.map((r) => Math.max(...r.frames.map((f) => f.dist))), 6);
    const dmax = [10, 20, 40, 80, 160].find((v) => v >= far * 1.05) || 160;
    let vmax = 0.3;
    for (const r of st.runs) for (const f of r.frames) vmax = Math.max(vmax, f.speed * 1.1);
    vmax = Math.ceil(vmax * 10) / 10;
    const lx = (t) => L + (t / tmax) * pw;
    const ly = (d) => T + (1 - (Math.log(Math.max(d, 0.1)) - Math.log(0.1)) / (Math.log(dmax) - Math.log(0.1))) * ph;
    const vy = (v) => T + (1 - v / vmax) * ph;
    st.profileGeom = { L, pw, tmax };
    axesText(ctx);
    for (const d of [0.1, 0.5, 1, 5, 10, 50, 100]) {
      if (d > dmax) break;
      ctx.strokeStyle = COL.line; ctx.beginPath(); ctx.moveTo(L, ly(d)); ctx.lineTo(L + pw, ly(d)); ctx.stroke();
      ctx.fillText(d < 1 ? d.toFixed(1) : String(d), 6, ly(d) + 3);
    }
    const tstep = tmax > 240 ? 60 : 30;
    for (let t = 0; t <= tmax; t += tstep) ctx.fillText(String(t), lx(t) - 6, T + ph + 14);
    for (let v = 0; v <= vmax + 1e-9; v += vmax > 1 ? 0.5 : 0.1) ctx.fillText(v.toFixed(1), L + pw + 6, vy(v) + 3);
    // thresholds
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = rgba(COL.amber, 0.6); ctx.beginPath(); ctx.moveTo(L, ly(C.safeDist)); ctx.lineTo(L + pw, ly(C.safeDist)); ctx.stroke();
    ctx.strokeStyle = rgba(COL.green, 0.7); ctx.beginPath(); ctx.moveTo(L, ly(C.dockDist)); ctx.lineTo(L + pw, ly(C.dockDist)); ctx.stroke();
    ctx.setLineDash([]);
    const cursor = cursorIndex();
    // event band
    const bandY = T + ph + 20;
    st.runs.forEach((run, k) => {
      const fr = run.frames, end = Math.min(cursor, fr.length - 1);
      const y = bandY + k * 4;
      for (let i = 1; i <= end; i++) {
        if (fr[i].source === 'primary' || fr[i].source === 'none') continue;
        ctx.fillStyle = st.tab === 'compare' ? MODE_COL[run.safety] : SOURCE_COL[fr[i].source];
        ctx.fillRect(lx(fr[i - 1].t), y, Math.max(1, lx(fr[i].t) - lx(fr[i - 1].t)), 3);
      }
    });
    for (const run of st.runs) {
      const fr = run.frames, end = Math.min(cursor, fr.length - 1);
      const col = st.tab === 'compare' ? MODE_COL[run.safety] : COL.accent;
      ctx.strokeStyle = rgba(col, 0.45); ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= end; i++) { const x = lx(fr[i].t), y = vy(fr[i].speed); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
      ctx.strokeStyle = col; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= end; i++) { const x = lx(fr[i].t), y = ly(fr[i].dist); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
    }
    if (st.cursor != null) {
      const t = frameAt(st.runs[0], st.cursor).t;
      ctx.strokeStyle = rgba(COL.text, 0.5); ctx.beginPath(); ctx.moveTo(lx(t), T); ctx.lineTo(lx(t), T + ph); ctx.stroke();
    }
    axesText(ctx);
    ctx.fillText('range m', 6, T - 9);
    const sl = 'speed m/s';
    ctx.fillText(sl, rec.w - ctx.measureText(sl).width - 4, T - 9);
  }

  function drawTimeline() {
    const rec = canvases.timelineCanvas, ctx = begin(rec);
    const { w, h } = rec;
    const run = st.runs[0];
    const total = Math.max(run.env.maxSteps, ...st.runs.map((r) => r.frames.length - 1));
    const live = Math.max(...st.runs.map((r) => r.frames.length - 1));
    const x = (i) => 6 + (i / total) * (w - 12);
    ctx.fillStyle = grid(0.18);
    ctx.beginPath(); ctx.roundRect(6, h / 2 - 3, w - 12, 6, 3); ctx.fill();
    ctx.fillStyle = rgba(COL.accent, 0.45);
    ctx.beginPath(); ctx.roundRect(6, h / 2 - 3, Math.max(0, x(live) - 6), 6, 3); ctx.fill();
    st.runs.forEach((r, k) => {
      for (let i = 1; i < r.frames.length; i++) {
        const src = r.frames[i].source;
        if (src === 'primary' || src === 'none') continue;
        ctx.fillStyle = st.tab === 'compare' ? MODE_COL[r.safety] : SOURCE_COL[src];
        ctx.fillRect(x(i - 1), h / 2 - 9 - k * 0, Math.max(1.2, x(i) - x(i - 1)), 4);
      }
    });
    const cur = cursorIndex();
    ctx.fillStyle = COL.text;
    ctx.beginPath(); ctx.arc(x(cur), h / 2, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = COL.surface; ctx.beginPath(); ctx.arc(x(cur), h / 2, 2.5, 0, Math.PI * 2); ctx.fill();
    st.timelineGeom = { total, w };
  }

  function drawCritic() {
    const rec = canvases.criticSpark, ctx = begin(rec);
    const fr = st.runs[0].frames, end = Math.min(cursorIndex(), fr.length - 1);
    const vals = fr.slice(0, end + 1).map((f) => f.value).filter((v) => v != null);
    if (vals.length < 2) return;
    const lo = Math.min(...vals), hi = Math.max(...vals);
    ctx.strokeStyle = COL.accent; ctx.lineWidth = 1.4;
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = (i / Math.max(1, st.runs[0].env.maxSteps)) * rec.w;
      const y = rec.h - 3 - ((v - lo) / Math.max(1e-9, hi - lo)) * (rec.h - 6);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }

  // ---------------------------------------------------------- telemetry DOM
  const axisEls = [];
  function buildAxes() {
    const box = $('axes');
    ['R-bar', 'V-bar', 'H-bar'].forEach((name) => {
      const row = el('div', { class: 'axisrow' }, `<span>${name}</span><div class="track"><i class="fill"></i><i class="cmd"></i></div><b class="num">0.00</b>`);
      box.appendChild(row);
      axisEls.push({ fill: row.querySelector('.fill'), cmd: row.querySelector('.cmd'), num: row.querySelector('.num') });
    });
    box.appendChild(el('div', { class: 'legendline' }, `<span><i style="background:${COL.accent}"></i>applied</span><span><i style="background:${COL.text};opacity:.6;width:2px;height:9px"></i>commanded</span>`));
  }

  function phaseName(run, f) {
    const env = run.env;
    if (env.done && (st.cursor == null || st.cursor >= run.frames.length - 1)) {
      return { text: OUTCOME_TXT[env.outcome], kind: env.outcome === 'docked' ? 'docked' : env.outcome === 'collision' ? 'collision' : 'warn' };
    }
    if (run.frames.length === 1 && !st.playing) return { text: 'Standby', kind: '' };
    if (f.dist < C.safeDist) return { text: 'Inside 5 m', kind: f.closing > f.limit ? 'collision' : '' };
    return { text: 'Approach', kind: '' };
  }

  function updateTelemetry() {
    const run = st.runs[0];
    const f = frameAt(run, st.cursor);
    const ph = phaseName(run, f);
    $('phaseTxt').textContent = ph.text; $('phaseTxt').dataset.kind = ph.kind;
    $('met').textContent = fmtT(f.t);
    const [rv, ru] = splitLen(f.dist);
    $('range').textContent = rv; $('rangeU').textContent = ru;
    const cl = $('closing');
    cl.innerHTML = `${f.closing >= 0 ? '' : '−'}${Math.abs(f.closing).toFixed(3)}<small>m/s</small>`;
    cl.className = f.closing > f.limit ? 'bad' : '';
    $('speed').innerHTML = `${f.speed.toFixed(3)}<small>m/s</small>`;
    $('dv').innerHTML = `${f.dv.toFixed(2)}<small>m/s</small>`;
    const ov = $('ovr');
    ov.textContent = String(f.overrides);
    ov.className = f.overrides ? 'warn' : '';
    const use = f.closing / Math.max(f.limit, 1e-9);
    $('envTxt').textContent = f.closing <= 0 ? 'range opening'
      : `${(use * 100).toFixed(0)}% of ${f.limit.toFixed(3)} m/s`;
    $('envBar').style.background = use > 1 ? COL.red : use > 0.85 ? COL.amber : COL.green;
    $('envBar').style.width = `${clamp(use, 0, 1.25) / 1.25 * 100}%`;
    $('envBar').parentElement.querySelector('em').style.left = `${100 / 1.25}%`;
    $('fuelTxt').textContent = `${(f.fuel * 100).toFixed(2)}%`;
    $('fuelBar').style.width = `${clamp(f.fuel, 0, 1) * 100}%`;
    $('fuelBar').style.background = f.fuel < C.fuelFloor ? COL.red : f.fuel < 0.2 ? COL.amber : COL.green;
    $('floorMark').style.left = `${C.fuelFloor * 100}%`;
    const src = f.source === 'none' ? 'primary' : f.source;
    f.applied.forEach((a, i) => {
      const e = axisEls[i];
      const lo = Math.min(0, a), hi = Math.max(0, a);
      e.fill.style.left = `${50 + lo * 50}%`;
      e.fill.style.width = `${(hi - lo) * 50}%`;
      e.fill.style.background = SOURCE_COL[src];
      e.cmd.style.left = `calc(${50 + clamp(f.cmd[i], -1, 1) * 50}% - 1px)`;
      e.num.textContent = (a >= 0 ? '+' : '−') + Math.abs(a).toFixed(2);
    });
    $('criticPanel').hidden = st.pilot !== 'ppo';
    if (st.pilot === 'ppo' && f.value != null) $('criticV').textContent = f.value.toFixed(1);

    // who is flying
    const auth = $('authority');
    let who, why, data;
    if (run.frames.length === 1 && !st.playing) { who = 'Ready'; why = 'Press Play or Space.'; data = 'none'; }
    else if (run.env.done && st.cursor == null) { who = OUTCOME_TXT[run.env.outcome]; why = outcomeLine(run.env); data = run.env.outcome === 'docked' ? 'docked' : 'failed'; }
    else {
      const src = f.source === 'none' ? 'primary' : f.source;
      ({ who, why } = authorityText(src, f.reason));
      data = src;
    }
    if (f.fault && !(run.env.done && st.cursor == null)) why = `Fault: ${f.fault.replace('_', ' ')}. ` + why;
    $('authWho').textContent = who; $('authWhy').textContent = why; auth.dataset.src = data;
  }

  function renderLog() {
    if (!st.logDirty) return;
    st.logDirty = false;
    const box = $('log');
    box.innerHTML = '';
    if (!st.log.length) { box.appendChild(el('div', { class: 'empty' }, 'None yet.')); return; }
    for (const e of st.log) {
      box.appendChild(el('div', { class: 'ev', 'data-tone': e.tone }, `<span class="t">${Math.round(e.t)} s</span><div><b>${e.title}</b><br>${e.text}</div>`));
    }
  }

  function updateCompare() {
    const auth = $('authority');
    const started = !atStart() || st.playing;
    $('authWho').textContent = started ? `${PILOTS[st.pilot].name}, three filters` : 'Ready';
    $('authWhy').textContent = started ? 'Same pilot, start and faults.' : 'Press Play or Space to fly all three at once.';
    auth.dataset.src = started ? 'primary' : 'none';
    const box = $('cmpCards');
    if (box.children.length !== st.runs.length) {
      box.innerHTML = '';
      for (const run of st.runs) {
        const card = el('div', { class: 'cmpcard', style: `--c:${MODE_COL[run.safety]}` },
          `<header><b><i></i>${SAFETY[run.safety].label}</b><span></span></header><p>${SAFETY[run.safety].desc}</p>
           <dl><div><dt>Range</dt><dd data-k="range"></dd></div><div><dt>Speed</dt><dd data-k="speed"></dd></div><div><dt>Interventions</dt><dd data-k="ovr"></dd></div></dl>`);
        box.appendChild(card);
      }
    }
    st.runs.forEach((run, i) => {
      const card = box.children[i];
      const f = frameAt(run, st.cursor);
      const done = run.env.done && (st.cursor == null || st.cursor >= run.frames.length - 1);
      card.querySelector('header span').textContent = done ? OUTCOME_TXT[run.env.outcome] : f.dist < C.safeDist ? 'inside 5 m' : 'approaching';
      card.querySelector('header span').style.color = done ? OUTCOME_COL[run.env.outcome] : COL.muted;
      card.querySelector('[data-k=range]').textContent = fmtLen(f.dist);
      card.querySelector('[data-k=speed]').textContent = f.speed.toFixed(3);
      card.querySelector('[data-k=ovr]').textContent = String(f.overrides);
    });
  }

  function showBanner(run) {
    const env = run.env;
    $('bannerTitle').textContent = OUTCOME_TXT[env.outcome];
    $('bannerSub').textContent = outcomeLine(env);
    $('banner').dataset.kind = env.outcome;
    $('banner').classList.add('show');
    clearTimeout(st.bannerTimer);
    st.bannerTimer = setTimeout(hideBanner, 3200);
  }
  function hideBanner() { $('banner').classList.remove('show'); }

  // ---------------------------------------------------------- controls
  function buildControls() {
    const sc = $('scenarios');
    for (const s of SCENARIOS) {
      const b = el('button', { class: 'scenario', 'data-key': s.key, 'aria-pressed': 'false', title: s.sub }, s.title);
      b.onclick = () => loadScenario(s.key);
      sc.appendChild(b);
    }
    for (const [k, p] of Object.entries(PILOTS)) {
      const b = el('button', { 'data-key': k, 'aria-pressed': 'false' }, p.label);
      b.onclick = () => setPilot(k);
      $('pilotSeg').appendChild(b);
    }
    for (const [k, s] of Object.entries(SAFETY)) {
      const b = el('button', { 'data-key': k, 'data-tone': k, 'aria-pressed': 'false' }, s.label);
      b.onclick = () => setSafety(k);
      $('safetySeg').appendChild(b);
    }
    for (const f of FAULTS) {
      const lab = el('label', { class: 'switch' }, `<div><b>${f.label}</b><span>${f.sub}</span></div><input type="checkbox" data-key="${f.key}" role="switch">`);
      lab.querySelector('input').onchange = (e) => toggleFault(f.key, e.target.checked);
      $('faults').appendChild(lab);
    }
    for (const sp of SPEEDS) {
      const b = el('button', { 'data-speed': sp, 'aria-pressed': String(sp === st.speed) }, `${sp}×`);
      b.onclick = () => { st.speed = sp; syncControls(); };
      $('speeds').appendChild(b);
    }
    document.querySelectorAll('#views button').forEach((b) => (b.onclick = () => setView(b.dataset.view)));
    document.querySelectorAll('.tab').forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));
    $('fitBtn').onclick = () => { st.cam.auto = true; st.cam.free = false; st.dirty = true; };
    $('playBtn').onclick = togglePlay;
    $('stepBtn').onclick = () => { st.playing = false; st.cursor = null; if (!allDone()) stepAll(); st.dirty = true; syncControls(); };
    $('resetBtn').onclick = () => rebuild(true);
    $('applyStart').onclick = applyStartInputs;
    $('randomStart').onclick = randomStart;
    ['sx', 'sy', 'sz'].forEach((id) => $(id).addEventListener('keydown', (e) => { if (e.key === 'Enter') applyStartInputs(); }));
    $('aboutBtn').onclick = () => $('about').showModal();
    $('aboutClose').onclick = () => $('about').close();
    $('mcRun').onclick = runCampaign;
  }

  function syncControls() {
    document.querySelectorAll('.scenario').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.key === st.scenario)));
    document.querySelectorAll('#pilotSeg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.key === st.pilot)));
    document.querySelectorAll('#safetySeg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.key === st.safety)));
    document.querySelectorAll('#faults input').forEach((i) => (i.checked = st.faults.has(i.dataset.key)));
    document.querySelectorAll('#speeds button').forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.speed) === st.speed)));
    document.querySelectorAll('#views button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === st.view)));
    document.querySelectorAll('.tab').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === st.tab)));
    const scn = SCENARIOS.find((x) => x.key === st.scenario);
    $('scenarioDesc').textContent = scn ? scn.sub : 'Custom start.';
    $('pilotDesc').textContent = PILOTS[st.pilot].desc;
    $('safetyDesc').textContent = SAFETY[st.safety].desc;
    $('sx').value = +st.start.pos[0].toFixed(2); $('sy').value = +st.start.pos[1].toFixed(2); $('sz').value = +st.start.pos[2].toFixed(2);
    $('keysHint').hidden = st.pilot !== 'manual' || st.tab === 'campaign';
    $('playIcon').innerHTML = st.playing ? '<path d="M3 1.5h3v11H3zM8 1.5h3v11H8z" fill="currentColor"/>' : '<path d="M3 1.5v11l9.5-5.5z" fill="currentColor"/>';
    const camp = st.tab === 'campaign';
    $('flyCenter').hidden = camp; $('campaignCenter').hidden = !camp;
    $('flyRight').hidden = st.tab !== 'fly'; $('compareRight').hidden = st.tab !== 'compare'; $('campaignRight').hidden = !camp;
    $('scenarioSection').hidden = camp; $('startSection').hidden = camp; $('safetySection').hidden = st.tab !== 'fly';
    st.dirty = true;
  }

  function loadScenario(key) {
    const s = SCENARIOS.find((x) => x.key === key);
    st.scenario = key; st.pilot = s.pilot; st.safety = s.safety;
    st.faults = new Set(s.faults || []);
    st.start = { pos: s.pos.slice(), vel: (s.vel || [0, 0, 0]).slice(), fuel: s.fuel == null ? 1 : s.fuel };
    if (s.pilot === 'manual') st.speed = 5;
    else if (st.speed === 5) st.speed = 15;
    rebuild();
  }

  const midFlight = () => !atStart() && !allDone();

  function setPilot(k) {
    st.pilot = k;
    st.scenario = null;
    if (k === 'manual') st.speed = 5;
    if (midFlight()) { logEvent(st.runs[0].env.t, '', 'Pilot changed', `${PILOTS[k].name} from the next step.`); syncControls(); }
    else rebuild(true);
  }
  function setSafety(k) {
    st.safety = k;
    st.scenario = null;
    if (midFlight() && st.tab === 'fly') {
      st.runs[0].env.safety = k;
      st.runs[0].safety = k;
      logEvent(st.runs[0].env.t, k === 'off' ? 'red' : '', 'Filter changed', `${SAFETY[k].label} from the next step.`);
      syncControls();
    } else rebuild(true);
  }
  function toggleFault(key, on) {
    on ? st.faults.add(key) : st.faults.delete(key);
    const f = faultConfig();
    if (midFlight()) {
      for (const r of st.runs) r.env.faults = Object.assign({}, f);
      const def = FAULTS.find((x) => x.key === key);
      logEvent(st.runs[0].env.t, on ? 'red' : '', on ? `Fault on: ${def.label}` : `Fault off: ${def.label}`, def.sub);
      syncControls();
    } else rebuild(true);
  }
  function applyStartInputs() {
    const v = ['sx', 'sy', 'sz'].map((id) => Number($(id).value));
    if (v.some((x) => !Number.isFinite(x))) return;
    if (norm(v) < 0.6) return;
    st.start = { pos: v, vel: [0, 0, 0], fuel: st.start.fuel };
    st.scenario = null;
    rebuild();
  }
  function randomStart() {
    const rng = S.mulberry32((Math.random() * 1e9) | 0);
    let p;
    do { p = [0, 1, 2].map(() => rng.uniform(-20, 20)); } while (norm(p) < 6);
    st.start = { pos: p, vel: [0, 1, 2].map(() => rng.uniform(-0.005, 0.005)), fuel: 1 };
    st.scenario = null;
    rebuild();
  }
  function setView(v) { st.view = v; st.cam.free = false; syncControls(); }
  function setTab(t) {
    if (t === st.tab) return;
    st.tab = t;
    if (t !== 'campaign') rebuild(true);
    else { st.playing = false; syncControls(); drawCampaign(); }
  }
  function togglePlay() {
    if (st.tab === 'campaign') return;
    if (allDone()) { rebuild(true); st.playing = true; }
    else { st.cursor = null; st.playing = !st.playing; }
    if (st.playing && st.runs[0].frames.length === 1) logEvent(0, '', 'Started', `${PILOTS[st.pilot].name}, ${st.tab === 'compare' ? 'all three filters' : SAFETY[st.safety].label}.`);
    hideBanner();
    syncControls();
  }

  // ---------------------------------------------------------- pointer input
  function bindStage() {
    const cv = $('scene');
    let drag = null;
    cv.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, moved: false, btn: e.button, shift: e.shiftKey }; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (!drag.moved) return;
      drag.x = e.clientX; drag.y = e.clientY;
      if (st.view === '3d' && !drag.shift && drag.btn === 0) {
        st.cam.free = true;
        st.cam.yaw += dx * 0.006;
        st.cam.pitch = clamp(st.cam.pitch + dy * 0.006, -1.45, 1.45);
        VIEWS['3d'].yaw = st.cam.yaw; VIEWS['3d'].pitch = st.cam.pitch;
      } else {
        st.cam.auto = false; st.cam.panX += dx; st.cam.panY += dy;
      }
      st.dirty = true;
    });
    cv.addEventListener('pointerup', (e) => {
      if (drag && !drag.moved) placeStart(e);
      drag = null;
    });
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      st.cam.auto = false;
      st.cam.zoom = clamp(st.cam.zoom * Math.exp(e.deltaY * 0.0012), 1.2, 400);
      st.dirty = true;
    }, { passive: false });
    cv.addEventListener('dblclick', () => { st.cam.auto = true; st.dirty = true; });
  }

  function placeStart(e) {
    if (st.view === '3d' || midFlight()) return;
    const rec = canvases.scene, r = rec.cv.getBoundingClientRect();
    const P = projector(st.cam, rec.w, rec.h);
    const a = (e.clientX - r.left - P.ox) / P.s, b = -(e.clientY - r.top - P.oy) / P.s;
    const pos = st.start.pos.slice();
    if (st.view === 'plane') { pos[0] = b; pos[1] = a; } else { pos[0] = b; pos[2] = a; }
    if (norm(pos) < 0.8 || norm(pos) > 90) return;
    st.start = { pos: pos.map((v) => Math.round(v * 10) / 10), vel: [0, 0, 0], fuel: st.start.fuel };
    st.scenario = null;
    rebuild(true);
  }

  function bindTimeline() {
    const box = $('timeline');
    let down = false;
    const seek = (e) => {
      const r = box.getBoundingClientRect();
      const g = st.timelineGeom || { total: 300 };
      const live = Math.max(...st.runs.map((x) => x.frames.length - 1));
      const i = Math.round(clamp((e.clientX - r.left - 6) / (r.width - 12), 0, 1) * g.total);
      st.playing = false;
      st.cursor = i >= live ? null : i;
      hideBanner();
      syncControls();
    };
    box.addEventListener('pointerdown', (e) => { down = true; box.setPointerCapture(e.pointerId); seek(e); });
    box.addEventListener('pointermove', (e) => down && seek(e));
    box.addEventListener('pointerup', () => (down = false));
    $('profile').addEventListener('click', (e) => {
      const g = st.profileGeom; if (!g) return;
      const r = e.currentTarget.getBoundingClientRect();
      const t = clamp((e.clientX - r.left - g.L) / g.pw, 0, 1) * g.tmax;
      const live = Math.max(...st.runs.map((x) => x.frames.length - 1));
      const i = Math.round(t / C.dt);
      st.playing = false; st.cursor = i >= live ? null : i; syncControls();
    });
  }

  function bindKeys() {
    const MAN = new Set(['w', 'a', 's', 'd', 'q', 'e']);
    const ARROWS = { ArrowUp: 'w', ArrowDown: 's', ArrowLeft: 'a', ArrowRight: 'd' };
    window.addEventListener('keydown', (e) => {
      if (e.target.matches('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
      let k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (st.pilot === 'manual' && ARROWS[k]) k = ARROWS[k];
      if (st.pilot === 'manual' && MAN.has(k)) {
        st.keys.add(k);
        if (!st.playing && !allDone() && st.tab !== 'campaign') { st.playing = true; st.cursor = null; syncControls(); }
        document.querySelectorAll(`#keysHint kbd[data-k="${k}"]`).forEach((x) => x.classList.add('on'));
        e.preventDefault();
        return;
      }
      if (k === ' ') { togglePlay(); e.preventDefault(); }
      else if (k === 'r') rebuild(true);
      else if (k === '.') $('stepBtn').click();
      else if (k === 'f') $('fitBtn').click();
      else if (k === 'v') setView({ '3d': 'plane', plane: 'side', side: '3d' }[st.view]);
      else if (k === '1') setTab('fly');
      else if (k === '2') setTab('compare');
      else if (k === '3') setTab('campaign');
      else if (k === 'ArrowLeft' || k === 'ArrowRight') {
        const live = Math.max(...st.runs.map((x) => x.frames.length - 1));
        const cur = st.cursor == null ? live : st.cursor;
        const n = clamp(cur + (k === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1), 0, live);
        st.playing = false; st.cursor = n >= live ? null : n; syncControls();
      }
    });
    window.addEventListener('keyup', (e) => {
      let k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (ARROWS[k]) k = ARROWS[k];
      st.keys.delete(k);
      document.querySelectorAll(`#keysHint kbd[data-k="${k}"]`).forEach((x) => x.classList.remove('on'));
    });
    window.addEventListener('blur', () => st.keys.clear());
  }

  // ---------------------------------------------------------- Monte Carlo campaign
  const DOMAINS = {
    nominal: { domain: null, faults: {} },
    randomized: { domain: { massLow: 0.8, massHigh: 1.2, effTol: 0.1, thrustNoise: 0.02, obsPosNoise: 0.05, obsVelNoise: 0.01 }, faults: {} },
    faults: { domain: { massLow: 0.9, massHigh: 1.1, effTol: 0.05, thrustNoise: 0.01 }, faults: { bias: [0.01, 0, 0], dropoutProb: 0.02, sensorFreezeProb: 0.02, fuelLeakRate: 0.0002 } },
  };
  const STARTS = {
    train: { p: 20, v: 0.005 },
    close: { p: 6, v: 0.05 },
    hot: { p: 10, v: 0.3 },
  };

  function runCampaign() {
    if (st.mc && st.mc.running) return;
    const n = Number($('mcN').value), seed = Number($('mcSeed').value) | 0;
    const dom = DOMAINS[$('mcDomain').value], box = STARTS[$('mcStarts').value];
    const rng = S.mulberry32(seed * 2654435761);
    const starts = [];
    while (starts.length < n) {
      const pos = [0, 1, 2].map(() => rng.uniform(-box.p, box.p));
      if (norm(pos) < 1.0) continue;
      starts.push({ pos, vel: [0, 1, 2].map(() => rng.uniform(-box.v, box.v)) });
    }
    const faults = Object.assign(faultConfig(), dom.faults);
    if (st.faults.has('bias')) faults.bias = [0.05, -0.03, 0];
    const modes = ['off', 'v1', 'v2'];
    const pilot = st.pilot === 'manual' ? 'glideslope' : st.pilot;
    const ctrl = controllerFor(pilot);
    st.mc = { running: true, pilot, n, starts, modes, results: { off: [], v1: [], v2: [] }, done: 0, total: n * 3, domain: $('mcDomain').value };
    $('mcRun').disabled = true;
    $('mcNote').textContent = pilot !== st.pilot ? 'The manual pilot cannot be batch-run, so the glideslope flies instead.' : `Running ${PILOTS[pilot].name} on ${n} starts per filter.`;
    const work = () => {
      const t0 = performance.now();
      while (performance.now() - t0 < 14 && st.mc.done < st.mc.total) {
        const k = st.mc.done, mode = modes[Math.floor(k / n)], i = k % n;
        const s0 = starts[i];
        const env = new S.Env({ safety: mode, pos: s0.pos, vel: s0.vel, faults, domain: dom.domain, rng: S.mulberry32(seed * 7919 + i) });
        const res = S.runEpisode(env, ctrl);
        st.mc.results[mode].push(Object.assign({ start: s0 }, res));
        st.mc.done++;
      }
      $('mcProg').style.width = `${(st.mc.done / st.mc.total) * 100}%`;
      drawCampaign();
      if (st.mc.done < st.mc.total) requestAnimationFrame(work);
      else {
        st.mc.running = false;
        $('mcRun').disabled = false;
        $('mcNote').textContent = `${PILOTS[pilot].name}, ${$('mcDomain').selectedOptions[0].text.split(':')[0].toLowerCase()} conditions, ${n} starts per filter, seed ${seed}. Click a start to fly it.`;
      }
    };
    requestAnimationFrame(work);
  }

  function summarize(list) {
    const k = (o) => list.filter((r) => r.outcome === o).length;
    const docked = list.filter((r) => r.outcome === 'docked');
    const mean = (a, f) => (a.length ? a.reduce((s, r) => s + f(r), 0) / a.length : NaN);
    return {
      n: list.length, docked: k('docked'), collision: k('collision'), timeout: k('timeout'), fuel: k('fuel'), out_of_bounds: k('out_of_bounds'),
      tDock: mean(docked, (r) => r.steps * C.dt), fuel_: mean(docked, (r) => r.fuelUsed), ovr: mean(list, (r) => r.overrides),
    };
  }

  function drawCampaign() {
    const box = $('results');
    const mc = st.mc;
    box.innerHTML = '';
    for (const mode of ['off', 'v1', 'v2']) {
      const list = mc ? mc.results[mode] : [];
      const sm = summarize(list);
      const pct = (x) => (sm.n ? (100 * x) / sm.n : 0);
      const card = el('div', { class: 'mode', role: 'button', tabindex: '0', 'aria-pressed': String(st.mcMode === mode), style: `--c:${MODE_COL[mode]}` });
      card.innerHTML = `<header><b><i></i>${SAFETY[mode].label}</b><span>${sm.n ? `${sm.n} episodes` : 'not run yet'}</span></header>
        <div class="stack">${['docked', 'collision', 'fuel', 'out_of_bounds', 'timeout'].map((o) => `<i style="width:${pct(sm[o])}%;background:${OUTCOME_COL[o]}" title="${OUTCOME_TXT[o]} ${sm[o]}"></i>`).join('')}</div>
        <dl><div><dt>Docked</dt><dd class="${sm.n && sm.docked === sm.n ? 'ok' : ''}">${sm.n ? pct(sm.docked).toFixed(1) + '%' : '-'}</dd></div>
        <div><dt>Collisions</dt><dd class="${sm.collision ? 'bad' : ''}">${sm.n ? sm.collision : '-'}</dd></div>
        <div><dt>Mean dock time</dt><dd>${Number.isFinite(sm.tDock) ? sm.tDock.toFixed(0) + ' s' : '-'}</dd></div>
        <div><dt>Interventions</dt><dd>${Number.isFinite(sm.ovr) ? sm.ovr.toFixed(1) : '-'}</dd></div></dl>`;
      card.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.click(); } };
      card.onclick = () => { st.mcMode = mode; drawCampaign(); };
      box.appendChild(card);
    }
    $('scatterLegend').innerHTML = Object.entries(OUTCOME_TXT).map(([k, v]) => `<span><i style="background:${OUTCOME_COL[k]}"></i>${v}</span>`).join('') + `<span style="margin-left:auto">showing ${SAFETY[st.mcMode].label}</span>`;
    drawScatter();
    drawHist();
  }

  function drawScatter() {
    const rec = canvases.scatter;
    if (rec.w < 80 || rec.h < 80) return;
    const ctx = begin(rec);
    const mc = st.mc;
    const L = 40, R = 14, T = 12, B = 26, pw = rec.w - L - R, ph = rec.h - T - B;
    const ext = mc ? Math.max(6, ...mc.starts.map((s) => Math.max(Math.abs(s.pos[0]), Math.abs(s.pos[1])))) * 1.08 : 22;
    const half = Math.min(pw, ph) / 2;
    const cx = L + pw / 2, cy = T + ph / 2, s = half / ext;
    st.scatterGeom = { cx, cy, s };
    axesText(ctx);
    const step = niceStep(ext / 2.5);
    for (let g = -Math.floor(ext / step) * step; g <= ext; g += step) {
      ctx.strokeStyle = Math.abs(g) < 1e-9 ? grid(0.45) : COL.line;
      ctx.beginPath(); ctx.moveTo(cx + g * s, cy - ext * s); ctx.lineTo(cx + g * s, cy + ext * s); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - ext * s, cy - g * s); ctx.lineTo(cx + ext * s, cy - g * s); ctx.stroke();
      ctx.fillText(String(+g.toFixed(1)), cx + g * s - 6, cy + ext * s + 14);
      if (Math.abs(g) > 1e-9) ctx.fillText(String(+g.toFixed(1)), cx - ext * s - 30, cy - g * s + 3);
    }
    ctx.fillText('V-bar, m →', cx + ext * s - 62, cy - 6);
    ctx.fillText('R-bar ↑', cx + 6, cy - ext * s + 10);
    ctx.beginPath(); ctx.arc(cx, cy, C.safeDist * s, 0, Math.PI * 2);
    ctx.setLineDash([4, 5]); ctx.strokeStyle = rgba(COL.amber, 0.6); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = COL.green; ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2); ctx.fill();
    if (!mc) {
      ctx.fillStyle = COL.muted; ctx.font = font(13);
      const t = 'No results yet. Press Run.';
      ctx.fillText(t, cx - ctx.measureText(t).width / 2, cy - 30);
      return;
    }
    const list = mc.results[st.mcMode];
    for (const r of list) {
      const x = cx + r.start.pos[1] * s, y = cy - r.start.pos[0] * s;
      ctx.fillStyle = rgba(OUTCOME_COL[r.outcome], r.outcome === 'docked' ? 0.55 : 0.95);
      ctx.beginPath(); ctx.arc(x, y, r.outcome === 'docked' ? 2.6 : 3.6, 0, Math.PI * 2); ctx.fill();
    }
  }

  function drawHist() {
    const rec = canvases.hist;
    if (rec.w < 80 || rec.h < 40) return;
    const ctx = begin(rec);
    const mc = st.mc;
    const L = 40, R = 14, T = 12, B = 24, pw = rec.w - L - R, ph = rec.h - T - B;
    const bins = 30, tmax = 300, bw = tmax / bins;
    const counts = {};
    let peak = 1;
    for (const mode of ['off', 'v1', 'v2']) {
      counts[mode] = new Array(bins).fill(0);
      if (mc) for (const r of mc.results[mode]) if (r.outcome === 'docked') counts[mode][Math.min(bins - 1, Math.floor((r.steps * C.dt) / bw))]++;
      peak = Math.max(peak, ...counts[mode]);
    }
    axesText(ctx);
    for (let t = 0; t <= tmax; t += 60) {
      const lbl = `${t} s`, tw = ctx.measureText(lbl).width;
      ctx.fillText(lbl, clamp(L + (t / tmax) * pw - tw / 2, 0, rec.w - tw - 2), T + ph + 15);
    }
    ctx.strokeStyle = COL.line; ctx.beginPath(); ctx.moveTo(L, T + ph); ctx.lineTo(L + pw, T + ph); ctx.stroke();
    for (const mode of ['off', 'v1', 'v2']) {
      const c = counts[mode];
      ctx.strokeStyle = MODE_COL[mode]; ctx.lineWidth = 1.8; ctx.fillStyle = rgba(MODE_COL[mode], 0.05);
      ctx.beginPath(); ctx.moveTo(L, T + ph);
      c.forEach((v, i) => { const y = T + ph - (v / peak) * ph; ctx.lineTo(L + (i / bins) * pw, y); ctx.lineTo(L + ((i + 1) / bins) * pw, y); });
      ctx.lineTo(L + pw, T + ph); ctx.fill(); ctx.stroke();
    }
    ctx.font = font(11, 600);
    ['off', 'v1', 'v2'].forEach((m, i) => { ctx.fillStyle = MODE_COL[m]; ctx.fillText(SAFETY[m].label, L + pw - 160 + i * 56, T + 10); });
  }

  function bindScatter() {
    $('scatter').addEventListener('click', (e) => {
      const mc = st.mc, g = st.scatterGeom;
      if (!mc || !g || mc.running) return;
      const r = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top;
      let best = null, bd = 64;
      for (const res of mc.results[st.mcMode]) {
        const px = g.cx + res.start.pos[1] * g.s, py = g.cy - res.start.pos[0] * g.s;
        const d = (px - x) ** 2 + (py - y) ** 2;
        if (d < bd) { bd = d; best = res; }
      }
      if (!best) return;
      st.start = { pos: best.start.pos.slice(), vel: best.start.vel.slice(), fuel: 1 };
      st.pilot = mc.pilot; st.safety = st.mcMode; st.scenario = null;
      st.tab = 'fly';
      rebuild();
      togglePlay();
    });
  }

  // ---------------------------------------------------------- main loop
  let last = performance.now(), acc = 0;
  function frame(now) {
    const dtReal = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (st.tab === 'campaign' && st.dirty && !(st.mc && st.mc.running)) { st.dirty = false; drawCampaign(); }
    if (st.tab !== 'campaign') {
      if (st.playing) {
        acc += dtReal * st.speed;
        let n = 0;
        while (acc >= 1 && n < 12 && !allDone()) { stepAll(); acc -= 1; n++; }
        if (n) st.dirty = true;
      } else acc = 0;
      if (updateCamera()) st.dirty = true;
      if (st.dirty) {
        st.dirty = false;
        drawScene(); drawPhase(); drawProfile(); drawTimeline();
        if (st.tab === 'fly') { updateTelemetry(); renderLog(); if (st.pilot === 'ppo') drawCritic(); }
        else updateCompare();
        const f = frameAt(st.runs[0], st.cursor);
        $('clock').innerHTML = `${fmtT(f.t)}${st.cursor != null ? ' <small>replay</small>' : ''}`;
        if (st.playing) syncPlayIcon();
      }
    }
    requestAnimationFrame(frame);
  }
  let iconPlaying = null;
  function syncPlayIcon() {
    if (iconPlaying === st.playing) return;
    iconPlaying = st.playing;
    $('playIcon').innerHTML = st.playing ? '<path d="M3 1.5h3v11H3zM8 1.5h3v11H8z" fill="currentColor"/>' : '<path d="M3 1.5v11l9.5-5.5z" fill="currentColor"/>';
  }

  // ---------------------------------------------------------- boot
  function boot() {
    buildControls();
    buildAxes();
    ['scene', 'phase', 'profile', 'timelineCanvas', 'criticSpark', 'scatter', 'hist'].forEach(setupCanvas);
    bindStage(); bindTimeline(); bindKeys(); bindScatter();
    const params = new URLSearchParams(location.hash.replace(/^#/, ''));
    const wanted = SCENARIO_ALIASES[params.get('scenario')] || params.get('scenario');
    loadScenario(SCENARIOS.some((s) => s.key === wanted) ? wanted : 'nominal');
    if (params.get('view') && VIEWS[params.get('view')]) setView(params.get('view'));
    if (params.get('tab')) setTab(params.get('tab'));
    const pre = Number(params.get('steps') || 0);
    if (pre > 0 && st.tab !== 'campaign') { for (let i = 0; i < pre && !allDone(); i++) stepAll(); hideBanner(); st.doneAt = st.doneAt && 1; }
    if (params.get('play') === '1') togglePlay();
    for (const [key, id] of [['mcStarts', 'mcStarts'], ['mcDomain', 'mcDomain'], ['mcN', 'mcN']]) if (params.get(key)) $(id).value = params.get(key);
    if (params.get('mc') === '1') { setTab('campaign'); runCampaign(); }
    window.SFLab = { st, stepAll, runCampaign, rebuild };
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        readTheme();
        $('axes').querySelector('.legendline i').style.background = COL.accent;
        st.logDirty = true; st.dirty = true;
        if (st.tab === 'campaign') drawCampaign();
      });
    }
    requestAnimationFrame(frame);
  }
  boot();
})();
