// Chamber 0 (hero): the Lorenz attractor, integrated by ode.rungeKuttaAdaptive('rk45').
import { linalg, ode } from 'numerics-js';
import { $, createStage, onFrame, rgba, PALETTE, fmt, int, reducedMotion, clamp } from './util.js';

const SIGMA = 10;
const BETA = 8 / 3;
const T_END = 60;
const DT = 0.004; // sampling step of the dense output used for drawing
const WINDOW = 6; // length of the glowing tail, in time units
const TAIL = Math.round(WINDOW / DT);

export function mountHero() {
  const canvas = $('#lorenz');
  const hud = $('#hud-lorenz');
  const rhoInput = $('#rho');
  const tolInput = $('#tol');
  const rhoVal = $('#rho-val');
  const tolVal = $('#tol-val');

  const copy = $('.hero-copy');
  const ctrl = $('.hero-ctrl');

  let layout = { cx: 0, cy: 0, s: 1 };
  let stage;
  const computeLayout = () => {
    const s = stage;
    if (!s || !s.w) return;
    if (s.w > 900) {
      layout = { cx: s.w * 0.66, cy: s.h * 0.5, s: Math.min((s.h * 0.8) / 56, (s.w * 0.55) / 46) };
      return;
    }
    // Narrow screens: fit the attractor into the free band between the text block and the controls.
    const top = copy.getBoundingClientRect().bottom - canvas.getBoundingClientRect().top;
    const bottom = ctrl.getBoundingClientRect().top - canvas.getBoundingClientRect().top;
    const band = Math.max(bottom - top, 120);
    layout = {
      cx: s.w * 0.5,
      cy: top + band * 0.5,
      s: Math.min((band * 0.92) / 56, (s.w * 0.8) / 46)
    };
  };
  const st = createStage(canvas, computeLayout);
  stage = st;
  computeLayout();
  // Copy and controls can change height (font load, text reflow) without the canvas resizing.
  const ro = new ResizeObserver(computeLayout);
  ro.observe(copy);
  ro.observe(ctrl);

  let rho = 28;
  let rtol = 1e-6;
  let A, B, beads, stats;
  let tau = 9;
  let yaw = 0.7;
  let pitch = 0.4;
  let dragging = false;
  let userTouched = false;
  let resolveQueued = false;

  const lorenz = (r) => (_t, y, d) => {
    const x = y.data[0];
    const yy = y.data[1];
    const z = y.data[2];
    d.data[0] = SIGMA * (yy - x);
    d.data[1] = x * (r - z) - yy;
    d.data[2] = x * yy - BETA * z;
    return d;
  };

  function sample(sol) {
    const n = Math.floor(T_END / DT) + 1;
    const out = new Float32Array(3 * n);
    const dense = sol.denseOutput;
    for (let i = 0; i < n; i++) {
      const v = dense.eval(Math.min(i * DT, T_END)).data;
      out[3 * i] = v[0];
      out[3 * i + 1] = v[1];
      out[3 * i + 2] = v[2];
    }
    return out;
  }

  function solve() {
    resolveQueued = false;
    const opts = { rtol, atol: rtol * 1e-3, denseOutput: true, maxSteps: 400000 };
    const f = lorenz(rho);
    const a = ode.rungeKuttaAdaptive(
      'rk45',
      f,
      0,
      T_END,
      new linalg.Vector([-6.5, -7.2, 24]),
      opts
    );
    const b = ode.rungeKuttaAdaptive(
      'rk45',
      f,
      0,
      T_END,
      new linalg.Vector([-6.4999, -7.2, 24]),
      opts
    );
    A = sample(a);
    B = sample(b);
    const m = a.t.size;
    beads = new Float32Array(4 * m);
    for (let i = 0; i < m; i++) {
      beads[4 * i] = a.t.data[i];
      beads[4 * i + 1] = a.y.get(i, 0);
      beads[4 * i + 2] = a.y.get(i, 1);
      beads[4 * i + 3] = a.y.get(i, 2);
    }
    stats = { steps: m - 1, evals: a.evaluations, ok: a.success };
  }

  function queueSolve() {
    if (!resolveQueued) {
      resolveQueued = true;
      requestAnimationFrame(solve);
    }
  }

  const label = () => {
    rhoVal.textContent = fmt(rho, 1);
    tolVal.textContent = '1e' + Math.round(Math.log10(rtol) * 2) / 2;
  };
  rhoInput.addEventListener('input', () => {
    rho = +rhoInput.value;
    label();
    queueSolve();
  });
  tolInput.addEventListener('input', () => {
    rtol = Math.pow(10, +tolInput.value);
    label();
    queueSolve();
  });
  label();

  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    userTouched = true;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    yaw += e.movementX * 0.008;
    pitch = clamp(pitch + e.movementY * 0.006, -0.2, 1.3);
  });
  const up = () => (dragging = false);
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);

  // Fixed star field for depth.
  const stars = Array.from({ length: 150 }, (_, i) => ({
    x: Math.abs(Math.sin(i * 12.9898) * 43758.5453) % 1,
    y: Math.abs(Math.sin(i * 78.233) * 12345.6789) % 1,
    z: 0.2 + (Math.abs(Math.sin(i * 3.1)) % 1) * 0.8,
    p: i * 0.7
  }));

  solve();

  const px = new Float32Array(TAIL + 1);
  const py = new Float32Array(TAIL + 1);

  function project(x, y, z, cy, sy, cp, sp) {
    const x1 = x * cy - y * sy;
    const y1 = x * sy + y * cy;
    const z1 = z - 25;
    const y2 = y1 * cp - z1 * sp;
    const z2 = y1 * sp + z1 * cp;
    const k = 1 / (1 + y2 * 0.004);
    return [layout.cx + x1 * layout.s * k, layout.cy - z2 * layout.s * k];
  }

  let hudTick = 0;
  onFrame(st, (dt, time) => {
    const speed = reducedMotion ? 0.5 : 1.8;
    tau += dt * speed;
    if (tau > T_END) tau = 4;
    if (!dragging && !userTouched && !reducedMotion) yaw += dt * 0.07;
    if (!dragging && userTouched && !reducedMotion) yaw += dt * 0.02;

    const { ctx, w, h } = st;
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, w, h);

    // stars
    for (const s of stars) {
      const tw = 0.5 + 0.5 * Math.sin(time * 0.8 + s.p);
      ctx.fillStyle = `rgba(200,215,255,${0.12 + 0.22 * s.z * tw})`;
      const sx = (((s.x * w + yaw * 18 * s.z) % w) + w) % w;
      ctx.fillRect(sx, s.y * h, s.z * 1.6, s.z * 1.6);
    }

    const cy = Math.cos(yaw);
    const sy = Math.sin(yaw);
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);
    ctx.globalCompositeOperation = 'lighter';

    // dim silhouette of the whole attractor
    ctx.fillStyle = rgba(PALETTE.cyan, 0.14);
    ctx.beginPath();
    for (let i = 0; i < A.length / 3; i += 2) {
      const p = project(A[3 * i], A[3 * i + 1], A[3 * i + 2], cy, sy, cp, sp);
      ctx.rect(p[0], p[1], 1.2, 1.2);
    }
    ctx.fill();

    const head = Math.min(Math.floor(tau / DT), A.length / 3 - 1);
    const tail0 = Math.max(0, head - TAIL);
    const n = head - tail0 + 1;

    const drawTail = (S, col) => {
      for (let i = 0; i < n; i++) {
        const k = 3 * (tail0 + i);
        const p = project(S[k], S[k + 1], S[k + 2], cy, sy, cp, sp);
        px[i] = p[0];
        py[i] = p[1];
      }
      const G = 20;
      const per = Math.ceil(n / G);
      for (let g = 0; g < G; g++) {
        const i0 = g * per;
        const i1 = Math.min(n - 1, (g + 1) * per);
        if (i1 - i0 < 1) continue;
        const f = (g + 1) / G;
        ctx.beginPath();
        ctx.moveTo(px[i0], py[i0]);
        for (let i = i0 + 1; i <= i1; i++) ctx.lineTo(px[i], py[i]);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = rgba(col, 0.1 * f * f);
        ctx.lineWidth = 2 + 7 * f;
        ctx.stroke();
        ctx.strokeStyle = rgba(col, 0.25 + 0.75 * f * f);
        ctx.lineWidth = 0.6 + 1.8 * f;
        ctx.stroke();
      }
      // head glow
      if (n > 1) {
        const hx = px[n - 1];
        const hy = py[n - 1];
        const g = ctx.createRadialGradient(hx, hy, 0, hx, hy, 22);
        g.addColorStop(0, rgba(col, 0.9));
        g.addColorStop(0.25, rgba(col, 0.28));
        g.addColorStop(1, rgba(col, 0));
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(hx, hy, 22, 0, 6.2832);
        ctx.fill();
        ctx.fillStyle = 'rgba(255,255,255,.95)';
        ctx.beginPath();
        ctx.arc(hx, hy, 2, 0, 6.2832);
        ctx.fill();
      }
    };
    drawTail(B, PALETTE.rose);
    drawTail(A, PALETTE.cyan);

    // solver steps ("beads") along the tail of trajectory A
    const tLo = tau - WINDOW;
    ctx.fillStyle = 'rgba(255,255,255,.55)';
    ctx.beginPath();
    for (let i = 0; i < beads.length; i += 4) {
      const t = beads[i];
      if (t < tLo) continue;
      if (t > tau) break;
      const p = project(beads[i + 1], beads[i + 2], beads[i + 3], cy, sy, cp, sp);
      ctx.rect(p[0] - 1.3, p[1] - 1.3, 2.6, 2.6);
    }
    ctx.fill();

    if (++hudTick % 6 === 0) {
      const k = 3 * head;
      const sep = Math.hypot(A[k] - B[k], A[k + 1] - B[k + 1], A[k + 2] - B[k + 2]);
      hud.innerHTML =
        `<b>rk45</b> · rtol ${tolVal.textContent} · <b>${int(stats.steps)}</b> steps · ` +
        `<b>${int(stats.evals)}</b> evaluations · t = ${tau.toFixed(1).padStart(4, ' ')} · ` +
        `separation <b>${fmt(sep, 3)}</b>`;
    }
  });
}
