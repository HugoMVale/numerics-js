// Chamber II: gravity sandbox driven by ode.createVelocityVerlet (symplectic, energy stays bounded).
import { ode, math, linalg } from 'numerics-js';
import {
  $,
  createStage,
  onFrame,
  pointerPos,
  segmented,
  rgba,
  PALETTE,
  fmt,
  clamp,
  lineStep
} from './util.js';

const G = 1;
const TRAIL = 260;
const COLORS = [
  PALETTE.cyan,
  PALETTE.rose,
  PALETTE.violet,
  PALETTE.blue,
  PALETTE.mint,
  PALETTE.amber
];

export function mountOrbits() {
  const canvas = $('#orbit-canvas');
  const hud = $('#hud-orbit');
  const speedInput = $('#orbit-speed');
  const st = createStage(canvas);

  let bodies = []; // {m, trail:Float32Array, head, len, hue}
  let state;
  let step;
  let eps2 = 1e-6;
  let viewR = 1.6;
  let E0 = 1;
  let seed = 11;
  let preset = 'figure8';
  let launchMass = 0.03;
  let aim = null;
  let simTime = 0;
  const err = new Float32Array(240).fill(-9);
  let errHead = 0;

  const makeAcc = (masses, e2) => (u, _v, a) => {
    const ud = u.data;
    const ad = a.data;
    const n = masses.length;
    ad.fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = ud[2 * j] - ud[2 * i];
        const dy = ud[2 * j + 1] - ud[2 * i + 1];
        const r2 = dx * dx + dy * dy + e2;
        const inv = G / (r2 * Math.sqrt(r2));
        const fi = masses[j] * inv;
        const fj = masses[i] * inv;
        ad[2 * i] += dx * fi;
        ad[2 * i + 1] += dy * fi;
        ad[2 * j] -= dx * fj;
        ad[2 * j + 1] -= dy * fj;
      }
    }
    return a;
  };

  function energy() {
    const { u, v } = state;
    const ud = u.data;
    const vd = v.data;
    const n = bodies.length;
    let e = 0;
    for (let i = 0; i < n; i++) {
      e += 0.5 * bodies[i].m * (vd[2 * i] ** 2 + vd[2 * i + 1] ** 2);
      for (let j = i + 1; j < n; j++) {
        const r = Math.hypot(ud[2 * j] - ud[2 * i], ud[2 * j + 1] - ud[2 * i + 1]);
        e -= (G * bodies[i].m * bodies[j].m) / Math.sqrt(r * r + eps2);
      }
    }
    return e;
  }

  /** (Re)build integrator state from a list of {x,y,vx,vy,m}. */
  function build(list, keepTrails = false) {
    const old = bodies;
    const n = list.length;
    const masses = list.map((b) => b.m);
    state = {
      u: new linalg.Vector(2 * n),
      v: new linalg.Vector(2 * n),
      a: new linalg.Vector(2 * n),
      aNext: new linalg.Vector(2 * n)
    };
    list.forEach((b, i) => {
      state.u.data[2 * i] = b.x;
      state.u.data[2 * i + 1] = b.y;
      state.v.data[2 * i] = b.vx;
      state.v.data[2 * i + 1] = b.vy;
    });
    const acc = makeAcc(masses, eps2);
    acc(state.u, state.v, state.a);
    step = ode.createVelocityVerlet(acc);
    bodies = list.map((b, i) => ({
      m: b.m,
      hue: b.hue ?? i,
      trail: keepTrails && old[b.from] ? old[b.from].trail : new Float32Array(TRAIL * 2),
      head: keepTrails && old[b.from] ? old[b.from].head : 0,
      len: keepTrails && old[b.from] ? old[b.from].len : 0
    }));
    E0 = energy();
    err.fill(-9);
  }

  function snapshot() {
    const u = state.u.data;
    const v = state.v.data;
    return bodies.map((b, i) => ({
      x: u[2 * i],
      y: u[2 * i + 1],
      vx: v[2 * i],
      vy: v[2 * i + 1],
      m: b.m,
      hue: b.hue,
      from: i
    }));
  }

  function load(name) {
    preset = name;
    simTime = 0;
    const list = [];
    if (name === 'figure8') {
      eps2 = 1e-6;
      viewR = 1.3;
      const v3 = [-0.93240737, -0.86473146];
      list.push({ x: 0.97000436, y: -0.24308753, vx: -v3[0] / 2, vy: -v3[1] / 2, m: 1 });
      list.push({ x: -0.97000436, y: 0.24308753, vx: -v3[0] / 2, vy: -v3[1] / 2, m: 1 });
      list.push({ x: 0, y: 0, vx: v3[0], vy: v3[1], m: 1 });
      launchMass = 0.05;
    } else if (name === 'solar') {
      eps2 = 1e-4;
      viewR = 2.3;
      const rnd = math.rand(seed++);
      const radii = [0.32, 0.55, 0.85, 1.25, 1.75];
      const masses = [2e-4, 5e-4, 7e-4, 3e-4, 1.2e-3];
      let px = 0;
      let py = 0;
      const planets = radii.map((r, i) => {
        const a = rnd() * 6.2832;
        const sp = Math.sqrt((G * 1) / r);
        const p = {
          x: r * Math.cos(a),
          y: r * Math.sin(a),
          vx: -sp * Math.sin(a),
          vy: sp * Math.cos(a),
          m: masses[i],
          hue: i
        };
        px += p.m * p.vx;
        py += p.m * p.vy;
        return p;
      });
      const a = rnd() * 6.2832;
      const comet = {
        x: 2.2 * Math.cos(a),
        y: 2.2 * Math.sin(a),
        vx: 0.42 * Math.sin(a),
        vy: -0.42 * Math.cos(a),
        m: 1e-5,
        hue: 5
      };
      px += comet.m * comet.vx;
      py += comet.m * comet.vy;
      list.push({ x: 0, y: 0, vx: -px, vy: -py, m: 1, hue: 5 }, ...planets, comet);
      launchMass = 0.001;
    } else {
      eps2 = 2e-3;
      viewR = 2.0;
      const rnd = math.rand(seed++);
      let mx = 0;
      let my = 0;
      let mt = 0;
      for (let i = 0; i < 7; i++) {
        const a = rnd() * 6.2832;
        const r = 0.25 + rnd() * 1.0;
        const m = 0.15 + rnd() * 0.45;
        const vv = 0.35 + rnd() * 0.35;
        const b = {
          x: r * Math.cos(a),
          y: r * Math.sin(a),
          vx: -vv * Math.sin(a),
          vy: vv * Math.cos(a),
          m,
          hue: i
        };
        mx += m * b.vx;
        my += m * b.vy;
        mt += m;
        list.push(b);
      }
      list.forEach((b) => {
        b.vx -= mx / mt;
        b.vy -= my / mt;
      });
      launchMass = 0.05;
    }
    build(list);
    massSeg(launchMass < 0.01 ? 'dust' : launchMass < 0.1 ? 'moon' : 'star');
  }

  segmented($('#orbit-preset'), (v) => load(v));
  const massVals = { dust: 0.001, moon: 0.05, star: 0.6 };
  const massSeg = segmented($('#orbit-mass'), (v) => (launchMass = massVals[v]));

  // ---- pointer: drag to launch -------------------------------------------------------------
  const toWorld = (p) => {
    const s = Math.min(st.w, st.h) / (2 * viewR);
    return { x: (p.x - st.w / 2) / s, y: -(p.y - st.h / 2) / s, s };
  };
  canvas.style.touchAction = 'none';
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    const p = pointerPos(e, canvas);
    aim = { p0: p, p1: p };
  });
  canvas.addEventListener('pointermove', (e) => {
    if (aim) aim.p1 = pointerPos(e, canvas);
  });
  const release = () => {
    if (!aim) return;
    const w0 = toWorld(aim.p0);
    const v = launchVelocity();
    aim = null;
    let list = snapshot();
    if (list.length >= 14) {
      const i = list.findIndex((b) => b.m < 0.1);
      if (i >= 0) list.splice(i, 1);
    }
    list.push({
      x: w0.x,
      y: w0.y,
      vx: v.vx,
      vy: v.vy,
      m: launchMass,
      hue: list.length + 2,
      from: -1
    });
    build(list, true);
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', () => (aim = null));

  function launchVelocity() {
    const s = toWorld(aim.p0).s;
    const k = 1.8 / s;
    return { vx: (aim.p1.x - aim.p0.x) * k, vy: -(aim.p1.y - aim.p0.y) * k };
  }

  function previewPath() {
    const w0 = toWorld(aim.p0);
    const v = launchVelocity();
    const list = snapshot();
    list.push({ x: w0.x, y: w0.y, vx: v.vx, vy: v.vy, m: launchMass });
    const n = list.length;
    const masses = list.map((b) => b.m);
    const s = {
      u: new linalg.Vector(2 * n),
      v: new linalg.Vector(2 * n),
      a: new linalg.Vector(2 * n),
      aNext: new linalg.Vector(2 * n)
    };
    list.forEach((b, i) => {
      s.u.data[2 * i] = b.x;
      s.u.data[2 * i + 1] = b.y;
      s.v.data[2 * i] = b.vx;
      s.v.data[2 * i + 1] = b.vy;
    });
    const acc = makeAcc(masses, eps2);
    acc(s.u, s.v, s.a);
    const stp = ode.createVelocityVerlet(acc);
    const pts = [];
    for (let k = 0; k < 360; k++) {
      stp(s, 0.006);
      if (k % 6 === 0) pts.push([s.u.data[2 * (n - 1)], s.u.data[2 * (n - 1) + 1]]);
    }
    return pts;
  }

  // ---- frame -------------------------------------------------------------------------------
  let hudTick = 0;
  load('figure8');
  onFrame(st, () => {
    const sub = Math.max(1, Math.round(+speedInput.value * 6));
    const dt = 0.002;
    for (let k = 0; k < sub; k++) step(state, dt);
    simTime += sub * dt;

    // drop escapees
    let escaped = false;
    const u = state.u.data;
    for (let i = 0; i < bodies.length; i++) {
      if (Math.hypot(u[2 * i], u[2 * i + 1]) > viewR * 14 || !Number.isFinite(u[2 * i]))
        escaped = true;
    }
    if (escaped) {
      const list = snapshot().filter(
        (b) => Number.isFinite(b.x) && Math.hypot(b.x, b.y) <= viewR * 14
      );
      if (list.length) build(list, true);
    }

    const sc = Math.min(st.w, st.h) / (2 * viewR);
    const cx = st.w / 2;
    const cy = st.h / 2;
    const uNow = state.u.data;
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      b.trail[2 * b.head] = uNow[2 * i];
      b.trail[2 * b.head + 1] = uNow[2 * i + 1];
      b.head = (b.head + 1) % TRAIL;
      b.len = Math.min(TRAIL, b.len + 1);
    }

    const e = energy();
    const rel = Math.abs((e - E0) / E0);
    err[errHead] = Math.log10(Math.max(rel, 1e-12));
    errHead = (errHead + 1) % err.length;

    // draw
    const { ctx, w, h } = st;
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, w, h);
    // faint polar grid
    ctx.strokeStyle = 'rgba(255,255,255,.045)';
    ctx.lineWidth = 1;
    for (let r = 0.5; r < viewR * 1.6; r += 0.5) {
      ctx.beginPath();
      ctx.arc(cx, cy, r * sc, 0, 6.2832);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.moveTo(0, cy);
    ctx.lineTo(w, cy);
    ctx.stroke();

    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      const col =
        i === 0 && b.m >= 0.9 && preset === 'solar' ? PALETTE.amber : COLORS[b.hue % COLORS.length];
      const G_ = 14;
      const per = Math.ceil(b.len / G_);
      for (let g = 0; g < G_; g++) {
        const i0 = g * per;
        const i1 = Math.min(b.len - 1, (g + 1) * per);
        if (i1 - i0 < 1) continue;
        const f = (g + 1) / G_;
        ctx.beginPath();
        for (let k = i0; k <= i1; k++) {
          const q = (b.head - b.len + k + TRAIL * 2) % TRAIL;
          const x = cx + b.trail[2 * q] * sc;
          const y = cy - b.trail[2 * q + 1] * sc;
          lineStep(ctx, k === i0, x, y);
        }
        ctx.strokeStyle = rgba(col, 0.08 * f * f);
        ctx.lineWidth = 6 * f + 1;
        ctx.stroke();
        ctx.strokeStyle = rgba(col, 0.12 + 0.7 * f * f);
        ctx.lineWidth = 0.6 + 1.6 * f;
        ctx.stroke();
      }
    }
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      const col =
        i === 0 && b.m >= 0.9 && preset === 'solar' ? PALETTE.amber : COLORS[b.hue % COLORS.length];
      const x = cx + uNow[2 * i] * sc;
      const y = cy - uNow[2 * i + 1] * sc;
      const r = 2.2 + 5.5 * Math.cbrt(b.m);
      const g = ctx.createRadialGradient(x, y, 0, x, y, r * 4.2);
      g.addColorStop(0, rgba(col, 0.75));
      g.addColorStop(0.3, rgba(col, 0.2));
      g.addColorStop(1, rgba(col, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r * 4.2, 0, 6.2832);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(x, y, r * 0.62, 0, 6.2832);
      ctx.fill();
    }

    // aiming
    ctx.globalCompositeOperation = 'source-over';
    if (aim) {
      const pts = previewPath();
      ctx.fillStyle = 'rgba(255,255,255,.55)';
      for (const p of pts) {
        ctx.beginPath();
        ctx.arc(cx + p[0] * sc, cy - p[1] * sc, 1.6, 0, 6.2832);
        ctx.fill();
      }
      ctx.strokeStyle = 'rgba(255,255,255,.8)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(aim.p0.x, aim.p0.y);
      ctx.lineTo(aim.p1.x, aim.p1.y);
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(aim.p0.x, aim.p0.y, 4, 0, 6.2832);
      ctx.fill();
    }

    // energy sparkline (log10 |ΔE/E0|)
    const sw = Math.min(170, w * 0.36);
    const sh = 40;
    const sx = w - sw - 16;
    const sy = h - sh - 38;
    ctx.fillStyle = 'rgba(6,8,15,.55)';
    ctx.fillRect(sx - 8, sy - 22, sw + 16, sh + 34);
    ctx.font = '10.5px ui-monospace, Menlo, monospace';
    ctx.fillStyle = PALETTE.dim;
    ctx.fillText('energy error |ΔE/E₀|', sx, sy - 8);
    ctx.strokeStyle = 'rgba(255,255,255,.12)';
    ctx.strokeRect(sx, sy, sw, sh);
    ctx.beginPath();
    for (let k = 0; k < err.length; k++) {
      const v = err[(errHead + k) % err.length];
      const x = sx + (sw * k) / (err.length - 1);
      const y = sy + sh - ((clamp(v, -10, -1) + 10) / 9) * sh;
      lineStep(ctx, k === 0, x, y);
    }
    ctx.strokeStyle = rgba(PALETTE.mint, 0.95);
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.fillStyle = rgba(PALETTE.mint, 1);
    ctx.fillText(fmt(rel, 2), sx + sw - 54, sy + 12);

    if (++hudTick % 8 === 0) {
      hud.innerHTML = `<b>createVelocityVerlet</b> · ${bodies.length} bodies · t = ${simTime.toFixed(1)} · dt = 0.002`;
    }
  });
}
