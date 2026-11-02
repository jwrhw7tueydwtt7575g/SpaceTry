/* SpaceForge Mission Lab -- simulation core.
 *
 * A line-for-line port of the Python reference stack:
 *   core/cw_dynamics.py       -> cwStep, cwAccel
 *   safety/mission_safety.py  -> closingSpeed, closingLimit, predictUnsafe, evaluateV1
 *   control/glideslope.py     -> glideslope, hotApproach
 *   control/rta.py            -> certifiedBrake, envelopeCorrection, applySafetyFilter
 *   env/spacecraft_env.py     -> Env (step order, faults, fuel, termination)
 * plus a forward pass for the shipped SB3 PPO policy (policy.js).
 *
 * The port is only trusted because sim.test.js replays fixtures produced by
 * `python -m spaceforge_mvp.viz.lab_export` and checks every number.
 * No DOM, no network: runs in node and in the browser (UMD).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.SFSim = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const EPS = 1e-9;

  const DEFAULTS = Object.freeze({
    n: 0.00113,            // mean motion, rad/s (~400 km LEO)
    dt: 1.0,               // control step, s
    maxAccel: 0.1,         // m/s^2 per unit command
    fuelRate: 0.001,       // tank fraction per unit |throttle|_1 per step
    safeDist: 5.0,         // keep-out bubble radius, m
    fuelFloor: 0.05,
    maxSafeClosing: 0.2,   // closing-speed limit at the bubble edge, m/s
    maxDockSpeed: 0.1,     // capture speed, m/s
    captureRadius: 0.5,
    captureFrac: 0.5,
    outerBrakingFraction: 0.5, // braking curve outside the keep-out sphere (null = legacy inside-only)
    dockDist: 0.5,
    workVolume: 100.0,
    maxSteps: 300,
    posNorm: 20.0,
    velNorm: 2.0,
  });

  // ------------------------------------------------------------ vectors
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const clip1 = (v) => Math.min(1, Math.max(-1, v));
  const clipVec = (a, lo = -1, hi = 1) => a.map((v) => Math.min(hi, Math.max(lo, v)));
  const ZERO = () => [0, 0, 0];

  // ------------------------------------------------------- CW dynamics
  function cwAccel(pos, vel, n) {
    return [2 * n * vel[1] + 3 * n * n * pos[0], -2 * n * vel[0], -(n * n) * pos[2]];
  }

  const stmCache = new Map();
  /** 6x6 Phi and 6x3 Gamma of the exact CW state transition (row-major). */
  function cwStm(dt, n) {
    const key = dt + ':' + n;
    let m = stmCache.get(key);
    if (m) return m;
    const nt = n * dt, s = Math.sin(nt), c = Math.cos(nt), n2 = n * n;
    const rv = [
      [s / n, (2 / n) * (1 - c), 0],
      [(2 / n) * (c - 1), (1 / n) * (4 * s - 3 * nt), 0],
      [0, 0, s / n],
    ];
    const Phi = [
      [4 - 3 * c, 0, 0, rv[0][0], rv[0][1], rv[0][2]],
      [6 * (s - nt), 1, 0, rv[1][0], rv[1][1], rv[1][2]],
      [0, 0, c, rv[2][0], rv[2][1], rv[2][2]],
      [3 * n * s, 0, 0, c, 2 * s, 0],
      [6 * n * (c - 1), 0, 0, -2 * s, 4 * c - 3, 0],
      [0, 0, -n * s, 0, 0, c],
    ];
    const Gamma = [
      [(1 - c) / n2, (2 / n2) * (nt - s), 0],
      [(2 / n2) * (s - nt), (4 * (1 - c) - 1.5 * nt * nt) / n2, 0],
      [0, 0, (1 - c) / n2],
      rv[0].slice(), rv[1].slice(), rv[2].slice(),
    ];
    m = { Phi, Gamma };
    stmCache.set(key, m);
    return m;
  }

  /** Propagate (pos, vel) over dt under constant acceleration acc (m/s^2). */
  function cwStep(pos, vel, acc, dt, n) {
    const { Phi, Gamma } = cwStm(dt, n);
    const x = [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]];
    const out = new Array(6);
    for (let i = 0; i < 6; i++) {
      const P = Phi[i], G = Gamma[i];
      let a = 0;
      for (let j = 0; j < 6; j++) a += P[j] * x[j];
      let b = 0;
      for (let k = 0; k < 3; k++) b += G[k] * acc[k];
      out[i] = a + b;
    }
    return [[out[0], out[1], out[2]], [out[3], out[4], out[5]]];
  }

  // --------------------------------------------------- safety envelope
  function closingSpeed(pos, vel) {
    const d = norm(pos);
    if (d < EPS) return norm(vel);
    return -dot(vel, pos) / d;
  }

  const envelopeApplies = (d, c) => d < c.safeDist || c.outerBrakingFraction != null;

  /** Inside 5 m: linear taper to capture speed. Outside: braking curve down to the gate at the edge. */
  function closingLimit(d, c, maxAccel = c.maxAccel) {
    const edge = c.maxSafeClosing;
    const floor = c.captureFrac * c.maxDockSpeed;
    if (d >= c.safeDist) {
      if (c.outerBrakingFraction == null) return edge;
      return Math.sqrt(edge * edge + 2.0 * c.outerBrakingFraction * maxAccel * (d - c.safeDist));
    }
    const scaled = (edge * d) / c.safeDist;
    return Math.min(edge, Math.max(floor, scaled));
  }

  /** Sample one step of constant-acceleration motion (mirrors numpy linspace). */
  function sampleStep(pos, vel, acc, dt, samples) {
    const start = dt / samples;
    const step = (dt - start) / (samples - 1);
    const out = [];
    for (let i = 0; i < samples; i++) {
      const tau = i === samples - 1 ? dt : start + i * step;
      const t2 = tau * tau;
      const p = [0, 1, 2].map((k) => pos[k] + vel[k] * tau + 0.5 * acc[k] * t2);
      const v = [0, 1, 2].map((k) => vel[k] + acc[k] * tau);
      out.push([p, v]);
    }
    return out;
  }

  function predictUnsafe(pos, vel, acc, dt, c, samples = 8, maxAccel = c.maxAccel) {
    for (const [p, v] of sampleStep(pos, vel, acc, dt, samples)) {
      const d = norm(p), s = norm(v);
      if (d < c.captureRadius && s >= c.maxDockSpeed) return true;
      if (envelopeApplies(d, c) && closingSpeed(p, v) > closingLimit(d, c, maxAccel) + 1e-12) return true;
    }
    return false;
  }

  function isBrakingCommand(cmd, vel, dt, maxAccel) {
    const speed = norm(vel);
    if (speed <= 1e-6 || dot(cmd, vel) >= 0) return false;
    const next = norm(add(vel, scale(cmd, maxAccel * dt)));
    return next <= speed + 1e-12;
  }

  /** v1 interlock: accept or veto (a veto zeroes thrust). */
  function evaluateV1(cmd, pos, vel, fuel, c, dt, maxAccel) {
    if (fuel < c.fuelFloor) return { status: 'emergency_hold', reason: 'Fuel below the safety floor: emergency hold.' };
    const d = norm(pos), speed = norm(vel);
    if (d < c.safeDist && speed > c.maxSafeClosing) {
      if (isBrakingCommand(cmd, vel, dt, maxAccel)) {
        return { status: 'accepted', reason: 'Braking thrust permitted inside the keep-out bubble.' };
      }
      return { status: 'rejected', reason: 'Inside the keep-out bubble above the speed gate: thrust vetoed.' };
    }
    return { status: 'accepted', reason: 'Command satisfies the interlock.' };
  }

  // ---------------------------------------------------------- guidance
  const GLIDE = { glideTime: 30.0, closingCap: 0.5, velGain: 0.7, brakingMargin: 0.5 };

  /** Certified glideslope (normalized command, 1.0 == maxAccel). */
  function glideslope(pos, vel, n, dt, maxAccel, o = GLIDE) {
    const d = norm(pos);
    let vDes = ZERO();
    if (d >= EPS) {
      const stoppable = o.brakingMargin * Math.sqrt(2.0 * maxAccel * d);
      const closing = Math.min(o.closingCap, d / o.glideTime, stoppable);
      // numpy evaluates (-pos / d) * closing; keep the same rounding order.
      vDes = [(-pos[0] / d) * closing, (-pos[1] / d) * closing, (-pos[2] / d) * closing];
    }
    const gain = Math.min(o.velGain, 1.0 / dt);
    let accel = scale(sub(vDes, vel), gain);
    accel = sub(accel, cwAccel(pos, vel, n));
    return clipVec(accel.map((a) => a / maxAccel));
  }

  /** The deliberately reckless law: full thrust at the target, never brakes. */
  function hotApproach(pos) {
    const d = norm(pos);
    if (d < EPS) return ZERO();
    return [(-pos[0] / d) * 1.0, (-pos[1] / d) * 1.0, (-pos[2] / d) * 1.0];
  }

  // ------------------------------------------------- runtime assurance
  function certifiedBrake(vel, maxAccel, dt) {
    const speed = norm(vel);
    if (speed <= EPS) return ZERO();
    const mag = Math.min(1.0, speed / (maxAccel * dt));
    return [(-vel[0] / speed) * mag, (-vel[1] / speed) * mag, (-vel[2] / speed) * mag];
  }

  function envelopeCorrection(pos, vel, c, maxAccel, dt, margin = 0.9) {
    const d = norm(pos);
    if (d < EPS) return certifiedBrake(vel, maxAccel, dt);
    const away = [pos[0] / d, pos[1] / d, pos[2] / d];
    const closing = closingSpeed(pos, vel);
    const limitNow = closingLimit(d, c, maxAccel);
    const lookAhead = Math.max(0.0, d - limitNow * dt);
    const target = margin * Math.min(limitNow, closingLimit(lookAhead, c, maxAccel));
    if (closing <= target) return ZERO();
    const dv = scale(away, closing - target);
    let cmd = dv.map((v) => v / (maxAccel * dt));
    const m = norm(cmd);
    if (m > 1.0) cmd = cmd.map((v) => v / m);
    return cmd;
  }

  /** v2 predictive filter (Simplex): policy keeps authority while the predicted step is safe. */
  function applySafetyFilter(cmdIn, pos, vel, fuel, c, dt, maxAccel, n) {
    const cmd = clipVec(cmdIn);
    if (fuel < c.fuelFloor) {
      if (predictUnsafe(pos, vel, ZERO(), dt, c, 8, maxAccel)) {
        return { applied: envelopeCorrection(pos, vel, c, maxAccel, dt), override: 'emergency_brake',
          reason: 'Fuel below floor on a collision course: reserve spent on certified braking.', source: 'brake' };
      }
      return { applied: ZERO(), override: 'emergency_hold', reason: 'Fuel below floor: discretionary thrust stopped.', source: 'hold' };
    }
    if (!predictUnsafe(pos, vel, scale(cmd, maxAccel), dt, c, 8, maxAccel)) {
      return { applied: cmd, override: null, reason: 'Predicted step stays inside the approach envelope.', source: 'primary' };
    }
    const backup = glideslope(pos, vel, n, dt, maxAccel);
    if (!predictUnsafe(pos, vel, scale(backup, maxAccel), dt, c, 8, maxAccel)) {
      return { applied: backup, override: 'braking_substituted',
        reason: 'Command would breach the envelope: certified glideslope backup is flying.', source: 'backup' };
    }
    return { applied: envelopeCorrection(pos, vel, c, maxAccel, dt), override: 'braking_substituted',
      reason: 'Far above the envelope: envelope-tracking brake substituted.', source: 'brake' };
  }

  // -------------------------------------------------------------- RNG
  function mulberry32(seed) {
    let a = seed >>> 0;
    const rng = function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    rng.uniform = (lo, hi) => lo + (hi - lo) * rng();
    rng.normal = (mu = 0, sd = 1) => {
      let u = 0;
      while (u <= 1e-12) u = rng();
      return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
    };
    return rng;
  }

  // ----------------------------------------------------- environment
  const NO_FAULTS = Object.freeze({ stuckAxis: null, stuckValue: 0, bias: [0, 0, 0], dropoutProb: 0, sensorFreezeProb: 0, fuelLeakRate: 0 });

  function faultsIdentity(f) {
    return !f || (f.stuckAxis == null && !f.bias.some((b) => b !== 0) && !f.dropoutProb && !f.sensorFreezeProb && !f.fuelLeakRate);
  }
  function domainIdentity(d) {
    return !d || (d.massLow === d.massHigh && d.massLow === 1 && !d.effTol && !d.thrustNoise && !d.obsPosNoise && !d.obsVelNoise);
  }

  /**
   * The rendezvous environment (same step order as SpacecraftEnv.step):
   * safety filter -> faults -> throttle clip -> domain -> exact CW -> fuel -> termination.
   */
  class Env {
    constructor(opts = {}) {
      this.c = Object.assign({}, DEFAULTS, opts.config || {});
      this.safety = opts.safety || 'v2';
      this.faults = Object.assign({}, NO_FAULTS, opts.faults || {});
      this.domain = opts.domain || null;
      this.rng = opts.rng || mulberry32(opts.seed == null ? 1 : opts.seed);
      this.maxSteps = opts.maxSteps || this.c.maxSteps;
      this.reset(opts.pos || [12, -7, 3.5], opts.vel || [0, 0, 0], opts.fuel == null ? 1 : opts.fuel);
    }

    reset(pos, vel, fuel = 1) {
      this.pos = pos.slice();
      this.vel = vel.slice();
      this.fuel = fuel;
      this.t = 0;
      this.step_ = 0;
      this.done = false;
      this.outcome = null;
      this.dv = 0;
      this.overrides = 0;
      this.minDist = norm(this.pos);
      this.lastObs = null;
      this.mass = 1;
      this.eff = [1, 1, 1];
      const d = this.domain;
      if (!domainIdentity(d)) {
        this.mass = this.rng.uniform(d.massLow, d.massHigh);
        this.eff = [0, 1, 2].map(() => this.rng.uniform(1 - (d.effTol || 0), 1 + (d.effTol || 0)));
      }
      return this.observe();
    }

    /** 10-D observation, float32-rounded exactly like the gym env. */
    observe() {
      let pos = this.pos, vel = this.vel;
      const d = this.domain;
      if (d && (d.obsPosNoise > 0 || d.obsVelNoise > 0)) {
        pos = pos.map((v) => v + this.rng.normal(0, d.obsPosNoise || 0));
        vel = vel.map((v) => v + this.rng.normal(0, d.obsVelNoise || 0));
      }
      const pn = Math.hypot(pos[0], pos[1]);
      const bearing = pn > EPS ? [-pos[0] / pn, -pos[1] / pn] : [0, 0];
      const obs = new Float32Array([
        pos[0] / this.c.posNorm, pos[1] / this.c.posNorm, pos[2] / this.c.posNorm,
        vel[0] / this.c.velNorm, vel[1] / this.c.velNorm, vel[2] / this.c.velNorm,
        this.fuel, this.step_ / this.maxSteps, bearing[0], bearing[1],
      ]);
      const f = this.faults;
      if (f.sensorFreezeProb > 0 && this.lastObs && this.rng() < f.sensorFreezeProb) return this.lastObs;
      this.lastObs = obs;
      return obs;
    }

    get dist() { return norm(this.pos); }
    get speed() { return norm(this.vel); }

    step(action) {
      if (this.done) throw new Error('episode finished; call reset()');
      const c = this.c, dt = c.dt, A = c.maxAccel;
      const cmd = clipVec(action);
      const pos0 = this.pos.slice(), vel0 = this.vel.slice();

      let applied, override = null, reason, source = 'primary';
      if (this.safety === 'off') {
        applied = cmd.slice();
        reason = 'Safety layer disabled.';
      } else if (this.safety === 'v2') {
        const r = applySafetyFilter(cmd, pos0, vel0, this.fuel, c, dt, A, c.n);
        applied = r.applied.slice(); override = r.override; reason = r.reason; source = r.source;
      } else {
        const r = evaluateV1(cmd, pos0, vel0, this.fuel, c, dt, A);
        if (r.status === 'accepted') applied = cmd.slice();
        else { applied = ZERO(); override = r.status; source = 'veto'; }
        reason = r.reason;
      }

      // Faults act on what the thrusters do, not on what the filter decided.
      const f = this.faults;
      let fault = null;
      let disturbed = applied.slice();
      if (!faultsIdentity(f)) {
        if (f.stuckAxis != null) { disturbed[f.stuckAxis] = f.stuckValue; fault = 'stuck_thruster'; }
        disturbed = add(disturbed, f.bias);
        if (f.dropoutProb > 0 && this.rng() < f.dropoutProb) { disturbed = ZERO(); fault = 'command_dropout'; }
      }
      const throttle = clipVec(disturbed);

      let massScale = 1;
      const d = this.domain;
      if (!domainIdentity(d)) {
        massScale = 1 / this.mass;
        disturbed = disturbed.map((v, i) => v * this.eff[i]);
        if (d.thrustNoise > 0) disturbed = disturbed.map((v) => v + this.rng.normal(0, d.thrustNoise));
      }
      const executed = clipVec(disturbed.map((v) => v * massScale * A), -A, A);

      const [np, nv] = cwStep(pos0, vel0, executed, dt, c.n);
      this.pos = np; this.vel = nv;

      const thrustMag = Math.abs(throttle[0]) + Math.abs(throttle[1]) + Math.abs(throttle[2]);
      const fuelUsed = Math.max(0, c.fuelRate * thrustMag);
      let debit = fuelUsed;
      if (f.fuelLeakRate > 0) { debit += f.fuelLeakRate; fault = fault || 'fuel_leak'; }
      this.fuel = Math.max(0, this.fuel - debit);
      this.step_ += 1;
      this.t = this.step_ * dt;
      this.dv += (Math.abs(executed[0]) + Math.abs(executed[1]) + Math.abs(executed[2])) * dt;
      if (override) this.overrides += 1;

      const dist = norm(this.pos), speed = norm(this.vel);
      this.minDist = Math.min(this.minDist, dist);
      const success = dist < c.dockDist && speed < c.maxDockSpeed;
      let collision = false;
      if (!success) {
        for (const [p, v] of sampleStep(pos0, vel0, executed, dt, 10)) {
          if (norm(p) < c.dockDist && norm(v) >= c.maxDockSpeed) { collision = true; break; }
        }
      }
      let outcome = null;
      if (success) outcome = 'docked';
      else if (collision) outcome = 'collision';
      else if (this.fuel <= 0) outcome = 'fuel';
      else if (dist > c.workVolume) outcome = 'out_of_bounds';
      else if (this.step_ >= this.maxSteps) outcome = 'timeout';
      if (outcome) { this.done = true; this.outcome = outcome; }

      return {
        cmd, applied, executed, throttle, override, reason, source, fault, fuelUsed,
        dist, speed, closing: closingSpeed(this.pos, this.vel), limit: closingLimit(dist, c, c.maxAccel),
        outcome, done: this.done, collision,
      };
    }

    /** Where the vehicle goes if every thruster stops now (exact CW coast). */
    coast(seconds, every = 2) {
      const pts = [];
      let p = this.pos, v = this.vel;
      for (let t = 0; t < seconds; t += every) {
        [p, v] = cwStep(p, v, ZERO(), every, this.c.n);
        pts.push(p);
      }
      return pts;
    }
  }

  // ------------------------------------------------------------ policy
  function decodeF32(b64) {
    let bytes;
    if (typeof atob === 'function') {
      const s = atob(b64);
      bytes = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
    } else {
      bytes = new Uint8Array(Buffer.from(b64, 'base64'));
    }
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }

  /** SB3 MlpPolicy: two tanh hidden layers for the actor and for the critic. */
  class Policy {
    constructor(spec) {
      const load = (layers) => layers.map((l) => ({ out: l.out, in: l.in, w: decodeF32(l.w), b: decodeF32(l.b) }));
      this.actor = load(spec.actor);
      this.critic = load(spec.critic);
      this.source = spec.source;
      this.snapshot = spec.selected_snapshot;
    }
    static forward(layers, x) {
      let h = x;
      for (let li = 0; li < layers.length; li++) {
        const L = layers[li], y = new Float64Array(L.out);
        for (let o = 0; o < L.out; o++) {
          let s = L.b[o];
          const row = o * L.in;
          for (let i = 0; i < L.in; i++) s += L.w[row + i] * h[i];
          y[o] = li < layers.length - 1 ? Math.tanh(s) : s;
        }
        h = y;
      }
      return h;
    }
    /** Deterministic action (distribution mean), clipped to the action box. */
    act(obs) { return Array.from(Policy.forward(this.actor, obs), clip1); }
    /** Critic's value estimate V(s). */
    value(obs) { return Policy.forward(this.critic, obs)[0]; }
  }

  // ------------------------------------------------------- utilities
  /** Run one full episode; returns the summary (used by campaigns). */
  function runEpisode(env, controller) {
    let obs = env.observe();
    let last = null;
    while (!env.done) {
      last = env.step(controller(env, obs));
      obs = env.observe();
    }
    return { outcome: env.outcome, steps: env.step_, fuelUsed: 1 - env.fuel, dv: env.dv,
      overrides: env.overrides, minDist: env.minDist, last };
  }

  return {
    DEFAULTS, cwAccel, cwStm, cwStep, closingSpeed, closingLimit, envelopeApplies, predictUnsafe, isBrakingCommand,
    evaluateV1, glideslope, hotApproach, certifiedBrake, envelopeCorrection, applySafetyFilter,
    mulberry32, Env, Policy, runEpisode, norm, dot,
  };
});
