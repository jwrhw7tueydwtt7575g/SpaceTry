"""Emit a self-contained interactive replay of the recorded scenarios.

    python -m spaceforge_mvp.viz.dashboard

Writes spaceforge_mvp/viz/dashboard.html — one file, no server, no CDN. Drop it
in a browser (or on a USB stick in a lecture theatre) and it plays.
"""

import json
from pathlib import Path
from typing import Any, Dict

from spaceforge_mvp.viz.record import SCENARIO_DIR

OUT = Path(__file__).resolve().parent / "dashboard.html"

HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SpaceForge · rendezvous replay</title>
<style>
  :root { --bg:#060910; --panel:#0C141F; --edge:#1D2C3F; --fg:#E8EFF7; --muted:#7A8DA6;
          --faint:#46586E; --cyan:#41D3FF; --green:#3DE08F; --amber:#FFB454; --red:#FF5F5F; }
  * { box-sizing: border-box; }
  body { margin:0; background:transparent; color:var(--foreground, #E8EFF7);
         font-family: ui-sans-serif, system-ui, "DejaVu Sans", sans-serif; }
  #wrap { width: 1000px; }
  .bar { display:flex; align-items:center; gap:8px; margin:0 0 10px 0; flex-wrap:wrap; }
  .tab { border:1px solid var(--border, #1D2C3F); background:transparent; color:inherit;
         padding:6px 11px; border-radius:999px; font-size:12.5px; cursor:pointer; }
  .tab.on { border-color:var(--accent, #41D3FF); color:var(--accent, #41D3FF); }
  .btn { border:1px solid var(--border,#1D2C3F); background:transparent; color:inherit;
         padding:6px 11px; border-radius:6px; font-size:12.5px; cursor:pointer; }
  .spacer { flex:1; }
  #spd { display:inline-flex; gap:6px; }
  .stage { display:grid; grid-template-columns: 664px 336px; gap:0; border-radius:12px;
           overflow:hidden; border:1px solid var(--border,#1D2C3F); background:var(--bg); }
  canvas#scene { display:block; width:664px; height:560px; }
  .hud { padding:14px 16px; background:var(--bg); border-left:1px solid var(--border,#1D2C3F); }
  .title { font-size:14.5px; font-weight:600; letter-spacing:.2px; }
  .sub { font-size:10.5px; color:var(--muted,#7A8DA6); margin-top:2px; line-height:1.35; }
  .status { margin-top:12px; font-size:12.5px; font-weight:600; }
  .rows { margin-top:8px; }
  .row { display:flex; justify-content:space-between; font-size:11px; padding:2.5px 0; }
  .row span:first-child { color:var(--muted,#7A8DA6); }
  .row span:last-child { font-family: ui-monospace, "DejaVu Sans Mono", monospace; }
  .fuelwrap { margin-top:10px; }
  .fuel { height:8px; background:var(--panel,#0C141F); border:1px solid var(--edge,#1D2C3F);
          border-radius:3px; overflow:hidden; }
  .fuelfill { height:100%; background:var(--green,#3DE08F); }
  canvas#strip { display:block; width:304px; height:74px; margin-top:12px;
                 background:var(--panel,#0C141F); border:1px solid var(--edge,#1D2C3F); border-radius:4px; }
  .scrub { display:flex; align-items:center; gap:10px; margin-top:12px; }
  input[type=range] { flex:1; accent-color:#41D3FF; }
  .axes { margin-top: 11px; }
  .axis { display:flex; align-items:center; gap:7px; font-size:9.5px; margin:3.5px 0;
          font-family: ui-monospace, "DejaVu Sans Mono", monospace; }
  .axis .lbl { width: 58px; color: var(--muted,#7A8DA6); letter-spacing:.02em; }
  .axis .bar2 { flex:1; height:7px; position:relative; background:var(--panel,#0C141F);
                border:1px solid var(--edge,#1D2C3F); border-radius:2px; }
  .axis .bar2 i { position:absolute; top:0; bottom:0; display:block; }
  .axis .val { width: 54px; text-align:right; }
  .axishead { font-size:8.5px; color:var(--faint,#46586E); letter-spacing:.06em;
              margin-top:11px; font-family: ui-monospace, monospace; }
  .note { font-size:11px; color:var(--muted,#7A8DA6); margin-top:10px; line-height:1.45; }
  .legend { display:flex; gap:14px; font-size:11px; margin-top:10px; color:var(--muted,#7A8DA6); }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; }
</style>
</head>
<body>
<div id="wrap">
  <div class="bar" id="tabs"></div>
  <div class="stage">
    <canvas id="scene"></canvas>
    <div class="hud">
      <div class="title" id="hTitle">—</div>
      <div class="sub" id="hSub">—</div>
      <div class="status" id="hStatus">—</div>
      <div class="rows" id="hRows"></div>
      <div class="axishead">APPLIED ACCELERATION · BODY AXES (m/s²)</div>
      <div class="axes" id="hAxes"></div>
      <div class="fuelwrap">
        <div class="row" style="padding:0 0 4px 0"><span>FUEL REMAINING</span><span id="hFuelTxt">—</span></div>
        <div class="fuel"><div class="fuelfill" id="hFuel"></div></div>
      </div>
      <canvas id="strip" width="608" height="148"></canvas>
      <div class="axishead" style="margin-top:5px">RANGE · LOG SCALE · ▼ CLOSEST PASS · TICKS = INTERLOCK EVENTS</div>
      <div class="legend" id="hLegend"></div>
      <div class="note" id="hNote"></div>
    </div>
  </div>
  <div class="scrub">
    <button class="btn" id="play">❚❚ pause</button>
    <input type="range" id="seek" min="0" max="100" value="0">
    <span id="clock" style="font-family:ui-monospace,monospace;font-size:11.5px;min-width:64px;text-align:right">00:00</span>
    <span id="spd"></span>
  </div>
</div>
<script>
const DATA = __DATA__;

window.addEventListener('error', (e) => {
  const d = document.createElement('div');
  d.style.cssText = 'color:#FF5F5F;font:12px ui-monospace,monospace;padding:6px 0';
  d.textContent = 'JS ERROR: ' + e.message + ' (line ' + e.lineno + ')';
  document.body.appendChild(d);
});

const Z_RIGHT = -0.36, Z_UP = 0.24;
const KEEP_OUT = 5.0, CAPTURE = 0.5;
const C = {fg:'#E8EFF7', muted:'#7A8DA6', faint:'#46586E', grid:'#15202D', grid2:'#213650',
           cyan:'#41D3FF', green:'#3DE08F', amber:'#FFB454', red:'#FF5F5F', bg:'#060910', edge:'#1D2C3F'};

const proj = (p) => [p[1] + Z_RIGHT * p[2], p[0] + Z_UP * p[2]];
const fmt = (v) => v >= 1000 ? (v/1000).toFixed(2)+' km' : v >= 100 ? v.toFixed(1)+' m'
                : v >= 10 ? v.toFixed(2)+' m' : v >= 1 ? v.toFixed(3)+' m' : (v*100).toFixed(1)+' cm';
const niceBar = (s) => { const n=[0.1,0.25,0.5,1,2,5,10,25,50,100,250,500,1000,2500]; let b=n[0];
  for (const c of n) { if (c <= s) b = c; else break; } return b; };
const niceStep = (s) => { const n=[0.1,0.25,0.5,1,2,5,10,25,50,100,250,500,1000,2500]; 
  for (const c of n) { if (s/c <= 6.5) return c; } return 5000; };

const scene = document.getElementById('scene');
const strip = document.getElementById('strip');
const ctx = scene.getContext('2d');
const sctx = strip.getContext('2d');
const DPR = Math.min(window.devicePixelRatio || 1, 2);

function fitCanvas(cv, c, cssW, cssH) {
  cv.width = cssW * DPR; cv.height = cssH * DPR;
  c.setTransform(DPR, 0, 0, DPR, 0, 0);
}
fitCanvas(scene, ctx, 664, 560);
fitCanvas(strip, sctx, 304, 74);

const stars = Array.from({length: 190}, () => [Math.random(), Math.random(), Math.random()]);
const state = { key: DATA.order[0], i: 0, playing: true, speed: 1, last: performance.now(), scale: null };

function cur() {
  const sc = DATA.scenarios[state.key];
  const i = Math.max(0, Math.min(Math.floor(state.i), sc.frames.length - 1));
  return { sc, f: sc.frames[i], i };
}

function drawScene() {
  const W = 664, H = 560;
  const { sc, f, i } = cur();
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
  for (const s of stars) { ctx.globalAlpha = 0.08 + s[2]*0.4; ctx.fillStyle = '#fff';
    ctx.fillRect(s[0]*W, s[1]*H, 1, 1); }
  ctx.globalAlpha = 1;

  // camera: midpoint between target and craft, log-eased zoom
  const [pu, pv] = proj([f.pos[0]*0.5, f.pos[1]*0.5, f.pos[2]*0.5]);
  let fit = 1.2;
  const [u, v] = proj(f.pos);
  fit = Math.max(fit, Math.abs(u - pu) * 1.6, Math.abs(v - pv) * 1.6);
  const targetHH = Math.min(Math.max(fit, 0.75), 5200);
  state.scale = state.scale === null ? targetHH
    : Math.exp(Math.log(state.scale) + (Math.log(targetHH) - Math.log(state.scale)) * 0.085);
  const hh = state.scale, hw = hh * (W / H);
  const pxPerM = H / (2 * hh);
  const X = (w) => (w - pu) * pxPerM + W/2;
  const Y = (w) => H/2 - (w - pv) * pxPerM;

  // orbital grid
  const step = niceStep(hh);
  const i0 = Math.floor((pu - hh*1.5)/step) - 1, i1 = Math.ceil((pu + hh*1.5)/step) + 1;
  const j0 = Math.floor((pv - hw*1.2)/step) - 1, j1 = Math.ceil((pv + hw*1.2)/step) + 1;
  ctx.lineWidth = 1;
  for (let k = i0; k <= i1 && Math.abs(k) <= 80; k++) {
    const [a, b] = proj([k*step, pv - hh*1.5, 0]), [c2, d2] = proj([k*step, pv + hh*1.5, 0]);
    ctx.strokeStyle = k === 0 ? C.grid2 : C.grid;
    ctx.beginPath(); ctx.moveTo(X(a), Y(b)); ctx.lineTo(X(c2), Y(d2)); ctx.stroke();
  }
  for (let k = j0; k <= j1 && Math.abs(k) <= 80; k++) {
    const [a, b] = proj([pu - hw*1.2, k*step, 0]), [c2, d2] = proj([pu + hw*1.2, k*step, 0]);
    ctx.strokeStyle = k === 0 ? C.grid2 : C.grid;
    ctx.beginPath(); ctx.moveTo(X(a), Y(b)); ctx.lineTo(X(c2), Y(d2)); ctx.stroke();
  }

  // rings
  const ring = (r, color, dash, alpha, label) => {
    if (r * pxPerM < 34) return;
    ctx.save(); ctx.setLineDash(dash); ctx.globalAlpha = alpha; ctx.strokeStyle = color;
    ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(0), Y(0), r*pxPerM, 0, 7); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = Math.min(1, alpha + 0.3); ctx.fillStyle = color;
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillText(label, X(0) + r*pxPerM*0.7 + 6, Y(0) - r*pxPerM*0.7 - 4);
    ctx.restore();
  };
  ring(KEEP_OUT, C.amber, [6,4], 0.55, 'keep-out ' + KEEP_OUT + ' m');
  ring(CAPTURE, C.green, [2,3], 0.85, 'capture ' + CAPTURE + ' m');

  // trail
  for (let k = 0; k <= i; k++) {
    const p = proj(sc.frames[k].pos);
    const prev = k > 0 ? proj(sc.frames[k-1].pos) : p;
    const mag = Math.hypot(...sc.frames[k].applied);
    ctx.globalAlpha = 0.14 + 0.74 * Math.min(mag/0.4, 1);
    ctx.strokeStyle = sc.color; ctx.lineWidth = 2.1; ctx.beginPath();
    ctx.moveTo(X(prev[0]), Y(prev[1])); ctx.lineTo(X(p[0]), Y(p[1])); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // range line + start marker
  ctx.save(); ctx.setLineDash([1,3]); ctx.globalAlpha = 0.35; ctx.strokeStyle = sc.color;
  ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(u), Y(v)); ctx.stroke(); ctx.restore();
  const s0 = proj(sc.frames[0].pos);
  ctx.globalAlpha = 0.75; ctx.strokeStyle = sc.color; ctx.lineWidth = 1.1;
  ctx.beginPath(); ctx.arc(X(s0[0]), Y(s0[1]), 5.5, 0, 7); ctx.stroke(); ctx.globalAlpha = 1;

  // target station (fixed pixel size)
  const s = 30 / pxPerM, tx = X(0), ty = Y(0);
  ctx.fillStyle = '#DCE7F3'; ctx.fillRect(tx - s*0.11, ty - s*0.20, s*0.22, s*0.40);
  ctx.fillStyle = '#17304F'; ctx.strokeStyle = C.cyan; ctx.lineWidth = 0.8;
  for (const sgn of [1, -1]) {
    const x0 = sgn > 0 ? tx + s*0.13 : tx - s*0.13 - s*0.42;
    ctx.beginPath(); ctx.rect(x0, ty - s*0.10, s*0.42, s*0.20); ctx.fill(); ctx.stroke();
  }

  // chaser + plume
  const cs = 26 / pxPerM, ang = Math.atan2(Y(v) - ty, X(u) - tx);
  const rot = (dx, dy) => [X(u) + dx*Math.cos(ang) - dy*Math.sin(ang), Y(v) + dx*Math.sin(ang) + dy*Math.cos(ang)];
  const thrust = Math.hypot(...f.applied);
  if (thrust > 0.012) {
    const plume = cs * (0.35 + 1.9 * Math.min(1, thrust/0.6));
    const tip = rot(-cs*0.16 - plume, 0), bl = rot(-cs*0.16, cs*0.20), br = rot(-cs*0.16, -cs*0.20);
    ctx.fillStyle = C.amber; ctx.beginPath(); ctx.moveTo(...rot(-cs*0.16, 0));
    ctx.lineTo(...bl); ctx.lineTo(...tip); ctx.lineTo(...br); ctx.closePath(); ctx.fill();
  }
  ctx.fillStyle = sc.color; ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.7;
  const nose = rot(cs*0.58, 0), left = rot(-cs*0.34, cs*0.36), tail = rot(-cs*0.16, 0), right = rot(-cs*0.34, -cs*0.36);
  ctx.beginPath(); ctx.moveTo(...nose); ctx.lineTo(...left); ctx.lineTo(...tail); ctx.lineTo(...right);
  ctx.closePath(); ctx.fill(); ctx.stroke();
  if (sc.summary.success && i >= sc.frames.length - 1) {
    ctx.strokeStyle = C.green; ctx.lineWidth = 2.4; ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.arc(tx, ty, CAPTURE*pxPerM, 0, 7); ctx.stroke(); ctx.globalAlpha = 1;
  }

  // scale bar
  const bar = niceBar(0.30 * 2 * hw), barPx = bar * pxPerM;
  ctx.strokeStyle = C.fg; ctx.lineWidth = 2.2;
  ctx.beginPath(); ctx.moveTo(34, H - 26); ctx.lineTo(34 + barPx, H - 26); ctx.stroke();
  ctx.fillStyle = C.fg; ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(fmt(bar), 34, H - 33);

  drawHud(sc, f, i);
}

function drawStrip() {
  const W = 304, H = 74;
  sctx.clearRect(0, 0, W, H);
  sctx.fillStyle = '#0C141F'; sctx.fillRect(0, 0, W, H);
  const all = DATA.order.map(k => DATA.scenarios[k]);
  const nmax = Math.max(...all.map(s => s.frames.length - 1));
  const rmax = Math.max(...all.flatMap(s => s.frames.map(f => f.dist)));
  const ylo = Math.log10(0.05), yhi = Math.max(1.2, Math.log10(rmax) + 0.05);
  const px = (i) => 8 + (i/nmax) * (W - 16);
  const py = (d) => H - 8 - ((Math.log10(Math.max(d, 0.05)) - ylo) / (yhi - ylo)) * (H - 20);
  sctx.font = '8px ui-monospace, monospace';
  for (let d = 0; d < 6; d++) {
    const val = Math.pow(10, d), lv = Math.log10(val);
    if (lv <= ylo || lv >= yhi) continue;
    sctx.strokeStyle = '#213650'; sctx.lineWidth = 0.7;
    sctx.beginPath(); sctx.moveTo(8, py(val)); sctx.lineTo(W-8, py(val)); sctx.stroke();
    sctx.fillStyle = C.faint; sctx.fillText(fmt(val).replace(' ',''), 12, py(val) - 2);
  }
  sctx.save(); sctx.setLineDash([3,3]); sctx.strokeStyle = C.green; sctx.globalAlpha = 0.9;
  sctx.beginPath(); sctx.moveTo(8, py(CAPTURE)); sctx.lineTo(W-8, py(CAPTURE)); sctx.stroke(); sctx.restore();
  for (const sc of all) {
    const active = sc.key === state.key;
    sctx.globalAlpha = active ? 0.35 : 0.16; sctx.strokeStyle = sc.color; sctx.lineWidth = 1.3;
    sctx.beginPath();
    sc.frames.forEach((f, k) => { const X = px(k), Y = py(f.dist); k ? sctx.lineTo(X, Y) : sctx.moveTo(X, Y); });
    sctx.stroke();
    if (active) {
      sctx.globalAlpha = 1; sctx.lineWidth = 2;
      sctx.beginPath();
      sc.frames.slice(0, Math.min(state.i, sc.frames.length-1) + 1)
        .forEach((f, k) => { const X = px(k), Y = py(f.dist); k ? sctx.lineTo(X, Y) : sctx.moveTo(X, Y); });
      sctx.stroke();
    }
  }
  // interlock events on the time axis, then the closest-approach marker
  const act = DATA.scenarios[state.key];
  const EVENTS = ['emergency_hold', 'emergency_brake', 'rejected', 'braking_substituted'];
  sctx.lineWidth = 1;
  act.frames.forEach((fr, k) => {
    if (!EVENTS.includes(fr.override)) return;
    sctx.strokeStyle = fr.override === 'braking_substituted' ? C.amber : C.red;
    sctx.globalAlpha = 0.8;
    sctx.beginPath(); sctx.moveTo(px(k), H - 7); sctx.lineTo(px(k), H - 1); sctx.stroke();
  });
  let mi = 0, md = Infinity;
  act.frames.forEach((fr, k) => { if (fr.dist < md) { md = fr.dist; mi = k; } });
  sctx.globalAlpha = 0.9; sctx.fillStyle = C.fg;
  sctx.beginPath();
  sctx.moveTo(px(mi), py(md) - 1);
  sctx.lineTo(px(mi) - 3.2, py(md) - 6.5);
  sctx.lineTo(px(mi) + 3.2, py(md) - 6.5);
  sctx.closePath(); sctx.fill();
  sctx.globalAlpha = 1;

  sctx.strokeStyle = C.fg; sctx.lineWidth = 1.2;
  sctx.beginPath(); sctx.moveTo(px(Math.min(state.i, nmax)), 6); sctx.lineTo(px(Math.min(state.i, nmax)), H-6); sctx.stroke();
}

const ROWS = [
  ['RANGE', (f) => fmt(f.dist)],
  ['CLOSING RATE', (f) => (f.closing >= 0 ? '+' : '') + f.closing.toFixed(3) + ' m/s'],
  ['REL SPEED', (f) => f.speed.toFixed(3) + ' m/s'],
  ['APPLIED THRUST', (f) => Math.hypot(...f.applied).toFixed(3) + ' m/s²'],
  ['MET', (f) => String(Math.floor(f.t/60)).padStart(2,'0') + ':' + String(Math.floor(f.t%60)).padStart(2,'0')],
  ['TOTAL Δv', (f) => f.dv.toFixed(3) + ' m/s'],
];

function statusOf(sc, f, i) {
  if (sc.summary.success && i >= sc.frames.length - 1) return ['DOCKED — SOFT CAPTURE', C.green];
  if (f.override === 'emergency_hold') return ['INTERLOCK — EMERGENCY HOLD', C.red];
  if (f.override === 'rejected') return ['INTERLOCK — THRUST REJECTED', C.red];
  if (f.override === 'braking_substituted') return ['RTA — CERTIFIED BACKUP FLYING', C.amber];
  if (f.override === 'emergency_brake') return ['RTA — RESERVE BRAKING', C.amber];
  if (f.diverging) return ['DIVERGING — NO CAPTURE', C.red];
  if (f.dist < KEEP_OUT) return ['INSIDE KEEP-OUT — GLIDESLOPE', C.amber];
  return ['APPROACH — CLOSING', C.cyan];
}

const AXIS_NAMES = ['RADIAL', 'IN-TRACK', 'CROSS-TRK'];

function drawAxes(f) {
  const A = f.applied;
  document.getElementById('hAxes').innerHTML = AXIS_NAMES.map((name, k) => {
    const v = A[k] || 0, w = Math.min(1, Math.abs(v)) * 50;      // command ceiling is 1 m/s²
    const left = v >= 0 ? 50 : 50 - w;
    const col = Math.abs(v) < 0.005 ? C.faint : (v >= 0 ? C.amber : C.cyan);
    const sign = (v >= 0 ? '+' : '') + v.toFixed(3);
    return `<div class="axis"><span class="lbl">${name}</span>` +
           `<span class="bar2"><i style="left:${left}%;width:${w}%;background:${col}"></i></span>` +
           `<span class="val" style="color:${col}">${sign}</span></div>`;
  }).join('');
}

function drawHud(sc, f, i) {
  document.getElementById('hTitle').textContent = sc.title;
  document.getElementById('hSub').textContent = sc.subtitle;
  const [txt, col] = statusOf(sc, f, i);
  const st = document.getElementById('hStatus');
  st.textContent = txt; st.style.color = col;
  document.getElementById('hRows').innerHTML = ROWS.map(([k, fn]) =>
    `<div class="row"><span>${k}</span><span>${fn(f)}</span></div>`).join('')
    + `<div class="row"><span>INTERLOCK FIRES</span><span style="color:${sc.summary.overrides ? C.red : C.muted}">${sc.summary.overrides}</span></div>`;
  drawAxes(f);
  document.getElementById('hFuelTxt').textContent = (f.fuel*100).toFixed(2) + '%';
  const fill = document.getElementById('hFuel');
  fill.style.width = (f.fuel*100) + '%';
  fill.style.background = f.fuel > 0.05 ? C.green : C.red;
  document.getElementById('hLegend').innerHTML =
    `<span><i class="dot" style="background:${sc.color}"></i>${sc.name}</span>`;
  document.getElementById('hNote').textContent = sc.note || '';
  document.getElementById('clock').textContent =
    String(Math.floor(f.t/60)).padStart(2,'0') + ':' + String(Math.floor(f.t%60)).padStart(2,'0');
  const seek = document.getElementById('seek');
  seek.max = sc.frames.length - 1; seek.value = i;
}

function tick(now) {
  const dt = (now - state.last) / 1000; state.last = now;
  const sc = DATA.scenarios[state.key];
  if (state.playing) {
    state.i += dt * 15 * state.speed;             // 15 sim steps per second at 1x
    if (state.i >= sc.frames.length - 1) state.i = sc.frames.length - 1;
  }
  drawScene(); drawStrip();
  requestAnimationFrame(tick);
}

function buildTabs() {
  const box = document.getElementById('tabs');
  box.innerHTML = '';
  DATA.order.forEach((k, n) => {
    const sc = DATA.scenarios[k];
    const b = document.createElement('button');
    b.className = 'tab' + (k === state.key ? ' on' : '');
    b.textContent = (n+1) + '. ' + sc.tab;
    b.onclick = () => { state.key = k; state.i = 0; state.scale = null; buildTabs(); drawScene(); drawStrip(); };
    box.appendChild(b);
  });
  const sp = document.createElement('div'); sp.className = 'spacer'; box.appendChild(sp);
  const r = document.createElement('button');
  r.className = 'btn'; r.textContent = 'restart';
  r.onclick = () => { state.i = 0; state.scale = null; };
  box.appendChild(r);
}

document.getElementById('play').onclick = (e) => {
  state.playing = !state.playing;
  e.target.textContent = state.playing ? '❚❚ pause' : '▶ play';
};
document.getElementById('seek').oninput = (e) => {
  state.i = parseFloat(e.target.value); state.playing = false;
  document.getElementById('play').textContent = '▶ play';
};
window.addEventListener('keydown', (e) => {
  const sc = DATA.scenarios[state.key];
  if (e.code === 'Space') { e.preventDefault(); document.getElementById('play').click(); }
  else if (e.code === 'ArrowRight' || e.code === 'ArrowLeft') {
    e.preventDefault(); state.playing = false;
    document.getElementById('play').textContent = '▶ play';
    const step = e.code === 'ArrowRight' ? 4 : -4;
    state.i = Math.max(0, Math.min(sc.frames.length - 1, Math.floor(state.i) + step));
  }
});

function initSpeed() {
  const box = document.getElementById('spd');
  box.innerHTML = '';
  [0.5, 1, 2].forEach(v => {
    const b = document.createElement('button');
    b.className = 'btn';
    b.style.padding = '4px 9px';
    b.style.fontSize = '11px';
    b.textContent = v + '×';
    if (state.speed === v) { b.style.borderColor = '#41D3FF'; b.style.color = '#41D3FF'; }
    b.onclick = () => { state.speed = v; initSpeed(); };
    box.appendChild(b);
  });
}

initSpeed();
buildTabs();
drawScene(); drawStrip();   // paint frame 0 right away: static panes may never fire rAF
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def _prepared(scenario_key: str) -> Dict[str, Any]:
    """Load one trace and add the fields the dashboard wants precomputed."""
    data = json.loads((SCENARIO_DIR / f"{scenario_key}.json").read_text())
    frames = data["frames"]
    for k, f in enumerate(frames):
        pos, vel = f["pos"], f["vel"]
        dist = max(f["dist"], 1e-9)
        f["closing"] = -sum(p * v for p, v in zip(pos, vel)) / dist
        look = min(20, k)
        d_prev = frames[k - look]["dist"] if look > 0 else dist
        f["diverging"] = bool(look > 4 and dist > d_prev + 0.15 and dist > 2.0)
    return data


def _outcome(summary: Dict[str, Any]) -> str:
    """One-line caption computed from the recorded summary (never hand-written)."""
    reason = summary["terminal_reason"]
    fires = summary["overrides"]
    if reason == "docked":
        return (f"Docks in {summary['duration']:.0f} s · {summary['dv']:.2f} m/s Δv · "
                f"{fires} interlock fire{'s' if fires != 1 else ''}")
    if reason == "collision" or summary.get("penetrated"):
        speed = summary.get("continuous_min_speed", summary["end_speed"])
        return f"Collision · fly-through at {speed:.1f} m/s"
    if reason == "out_of_bounds":
        return f"No capture · left the {summary['end_dist']:.0f} m work volume at {summary['end_speed']:.1f} m/s"
    return f"No capture · {reason} at {summary['end_dist']:.1f} m"


META = {
    "docking": ("Certified glideslope", "glideslope", "#41D3FF",
                "The certified law: velocity matching on an exponential glideslope with the natural CW "
                "drift cancelled in feed-forward. Runs to a soft capture, never needs a veto."),
    "rl_policy": ("PPO policy as shipped", "PPO checkpoint", "#41D3FF",
                  "The released PPO checkpoint replayed in the environment it was trained in, through the "
                  "same safety layer as every other run. The caption above is computed from this trace."),
    "hot_open": ("No runtime assurance", "reckless law", "#FF5F5F",
                 "A deliberately reckless law: full thrust straight at the target, no braking logic. "
                 "With nothing between policy and thrusters it strikes the target."),
    "hot_v1": ("Interlock v1 · zeroed veto", "no protection · v1 · v2", "#FFB454",
               "Same law under the v1 interlock: unsafe thrust inside the 5 m bubble is vetoed, and the "
               "veto zeroes it. Denying thrust lowers the impact speed but does not stop the collision -- "
               "the vehicle was already moving when it entered."),
    "hot_v2": ("RTA v2 · predictive filter", "no protection · v1 · v2", "#3DE08F",
               "Same law, same physics: v2 predicts every step against a closing-speed envelope and, when "
               "the policy's command would violate it, flies the certified glideslope backup instead. "
               "The identical approach ends in a soft capture."),
}


def build() -> Path:
    scenarios: Dict[str, Any] = {}
    order = ["docking", "rl_policy", "hot_open", "hot_v1", "hot_v2"]
    for key in order:
        data = _prepared(key)
        title, tab, color, note = META[key]
        sub = _outcome(data["summary"])
        data.update(key=key, name=title, title=title, subtitle=sub, tab=tab, color=color, note=note)
        scenarios[key] = data

    payload = {"order": order, "scenarios": scenarios}
    html = HEAD.replace("__DATA__", json.dumps(payload))
    OUT.write_text(html)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"dashboard written: {path}  ({path.stat().st_size / 1024:.0f} KB)")
