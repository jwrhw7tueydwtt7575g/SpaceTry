// Parity: the browser simulation must reproduce the Python reference stack.
// Fixtures come from `python -m spaceforge_mvp.viz.lab_export`.
//   node spaceforge_mvp/viz/lab/sim.test.js
'use strict';
const assert = require('assert');
const path = require('path');
const fs = require('fs');
const S = require('./sim.js');

const FX = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures.json'), 'utf8'));
const policySpec = require('./policy.js');
const C = Object.assign({}, S.DEFAULTS);

let passed = 0;
const failures = [];
function test(name, fn) {
  try { fn(); passed++; } catch (e) { failures.push(`${name}: ${e.message}`); }
}
function close(a, b, tol, what) {
  const xs = Array.isArray(a) || ArrayBuffer.isView(a) ? Array.from(a) : [a];
  const ys = Array.isArray(b) ? b : [b];
  assert.strictEqual(xs.length, ys.length, `${what}: length`);
  xs.forEach((x, i) => {
    const err = Math.abs(x - ys[i]);
    const bound = tol * Math.max(1, Math.abs(ys[i]));
    if (!(err <= bound)) throw new Error(`${what}[${i}] js=${x} py=${ys[i]} err=${err.toExponential(2)}`);
  });
}

// Config must match the Python defaults the fixtures were produced with.
test('config', () => {
  const c = FX.config;
  const map = { n: 'n', dt: 'dt', maxAccel: 'max_accel', fuelRate: 'fuel_rate', safeDist: 'safe_dist',
    fuelFloor: 'fuel_floor', maxSafeClosing: 'max_safe_closing', maxDockSpeed: 'max_dock_speed',
    captureRadius: 'capture_radius', captureFrac: 'capture_frac', dockDist: 'dock_dist',
    outerBrakingFraction: 'outer_braking_fraction',
    workVolume: 'work_volume', posNorm: 'pos_norm', velNorm: 'vel_norm' };
  for (const [js, py] of Object.entries(map)) assert.strictEqual(S.DEFAULTS[js], c[py], js);
});

const U = FX.unit;
U.cw_step.forEach((k, i) => test(`cw_step#${i}`, () => {
  const [p, v] = S.cwStep(k.pos, k.vel, k.acc, k.dt, C.n);
  close(p, k.next_pos, 1e-12, 'pos'); close(v, k.next_vel, 1e-12, 'vel');
}));
U.limit.forEach((k, i) => test(`limit#${i}`, () => close(S.closingLimit(k.d, C, k.max_accel), k.limit, 1e-15, 'limit')));
U.predict.forEach((k, i) => test(`predict#${i}`, () => {
  assert.strictEqual(S.predictUnsafe(k.pos, k.vel, k.acc, k.dt, C, 8, k.max_accel), k.unsafe);
}));
U.glideslope.forEach((k, i) => test(`glideslope#${i}`, () => {
  close(S.glideslope(k.pos, k.vel, C.n, k.dt, k.max_accel), k.out, 1e-12, 'cmd');
}));
U.brake.forEach((k, i) => test(`brake#${i}`, () => {
  close(S.certifiedBrake(k.vel, k.max_accel, k.dt), k.brake, 1e-12, 'brake');
  close(S.envelopeCorrection(k.pos, k.vel, C, k.max_accel, k.dt), k.envelope, 1e-12, 'envelope');
}));
U.filter.forEach((k, i) => test(`filter#${i}`, () => {
  const r = S.applySafetyFilter(k.cmd, k.pos, k.vel, k.fuel, C, k.dt, k.max_accel, C.n);
  assert.strictEqual(r.override, k.override);
  close(r.applied, k.applied, 1e-12, 'applied');
}));
U.v1.forEach((k, i) => test(`v1#${i}`, () => {
  assert.strictEqual(S.evaluateV1(k.cmd, k.pos, k.vel, k.fuel, C, k.dt, k.max_accel).status, k.status);
}));

// Whole episodes through the environment, step by step.
const policy = new S.Policy(policySpec);
for (const ep of FX.episodes) {
  test(`episode ${ep.name}`, () => {
    const faults = ep.faults ? {
      stuckAxis: ep.faults.stuck_axis, stuckValue: ep.faults.stuck_value,
      bias: ep.faults.bias, fuelLeakRate: ep.faults.fuel_leak_rate,
    } : null;
    const env = new S.Env({ safety: ep.safety, pos: ep.start_pos, vel: ep.start_vel, fuel: ep.fuel,
      maxSteps: ep.max_steps, faults });
    ep.steps.forEach((ref, i) => {
      let cmd;
      if (ep.controller === 'glideslope') cmd = S.glideslope(env.pos, env.vel, C.n, C.dt, C.maxAccel);
      else if (ep.controller === 'hot') cmd = S.hotApproach(env.pos);
      else if (ep.controller === 'away') cmd = env.pos.map((v) => v / Math.max(env.dist, 1e-9));
      else if (ep.controller === 'ppo') {
        cmd = policy.act(env.observe());
        close(cmd, ref.action, 2e-5, `step ${i} action`);
        cmd = ref.action; // drive with the reference action so float32 drift cannot compound
      }
      const r = env.step(cmd);
      assert.strictEqual(r.override, ref.override, `step ${i} override`);
      close(env.pos, ref.pos, 1e-9, `step ${i} pos`);
      close(env.vel, ref.vel, 1e-9, `step ${i} vel`);
      close(env.fuel, ref.fuel, 1e-12, `step ${i} fuel`);
    });
    assert.strictEqual(env.outcome, ep.outcome, 'outcome');
  });
}

// Free-running PPO (no reference actions fed back) must still reach the same outcome.
for (const ep of FX.episodes.filter((e) => e.controller === 'ppo')) {
  test(`free-running ${ep.name}`, () => {
    const env = new S.Env({ safety: ep.safety, pos: ep.start_pos, vel: ep.start_vel, maxSteps: ep.max_steps });
    const sum = S.runEpisode(env, (e, obs) => policy.act(obs));
    assert.strictEqual(sum.outcome, ep.outcome);
    assert.ok(Math.abs(sum.steps - ep.steps.length) <= 2, `steps ${sum.steps} vs ${ep.steps.length}`);
  });
}

test('critic value is finite', () => {
  const env = new S.Env({});
  assert.ok(Number.isFinite(policy.value(env.observe())));
});

test('seeded rng is reproducible', () => {
  const a = S.mulberry32(7), b = S.mulberry32(7);
  for (let i = 0; i < 100; i++) assert.strictEqual(a.normal(), b.normal());
});

const total = passed + failures.length;
if (failures.length) {
  console.error(failures.slice(0, 25).join('\n'));
  console.error(`\n${failures.length}/${total} parity checks FAILED`);
  process.exit(1);
}
console.log(`sim parity: ${passed}/${total} checks passed (${FX.episodes.length} episodes, ${U.filter.length} filter cases)`);
