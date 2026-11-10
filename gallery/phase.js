// Chamber V: phase portraits. The eigen-structure comes from Matrix.eig(), every tracer is advanced by ode.rungeKuttaFixed('rk4').
import { linalg, ode, math } from 'numerics-js';
import {
  $,
  createStage,
  onFrame,
  pointerPos,
  rgba,
  PALETTE,
  fmt,
  clamp,
  smooth,
  reducedMotion,
  lineStep
} from './util.js';

const V = 3; // phase-plane half extent
const TAU_RANGE = [-4, 4];
const DET_RANGE = [-2.5, 5];
const NP = 220;
const TRAIL = 12;

export function mountPhase() {
  const canvas = $('#phase-canvas');
  const tdCanvas = $('#td-canvas');
  const hud = $('#hud-phase');
  const readout = $('#phase-readout');
  const presetsEl = $('#phase-presets');
  const st = createStage(canvas);
  const tst = createStage(tdCanvas);

  let tau = -0.5;
  let det = 2.2;
  let eigs = [];
  let tween = null;

  function solveEig() {
    const A = linalg.Matrix.from([
      [0, 1],
      [-det, tau]
    ]);
    eigs = A.eig();
    describe();
  }

  function classify() {
    const disc = tau * tau - 4 * det;
    if (det < -1e-6) return 'Saddle';
    if (Math.abs(det) <= 1e-6) return 'Degenerate';
    if (disc > 1e-6) return tau < 0 ? 'Stable node' : 'Unstable node';
    if (Math.abs(disc) <= 1e-6)
      return tau < 0 ? 'Stable degenerate node' : 'Unstable degenerate node';
    if (Math.abs(tau) < 0.03) return 'Center';
    return tau < 0 ? 'Stable spiral' : 'Unstable spiral';
  }

  function describe() {
    const parts = eigs.map((e) => {
      const re = e.value.re;
      const im = e.value.im;
      return Math.abs(im) < 1e-9
        ? fmt(re, 3)
        : `${fmt(re, 3)} ${im < 0 ? '−' : '+'} ${fmt(Math.abs(im), 3)}i`;
    });
    const uniq = [...new Set(parts)];
    readout.innerHTML =
      `<b>${classify()}</b> · trace τ = ${fmt(tau, 2)}, det Δ = ${fmt(det, 2)}<br>` +
      `<span class="dim">λ =</span> ${uniq.join(', ')}`;
  }

  // ---- presets ------------------------------------------------------------------------------
  const presets = [
    ['Saddle', 0.4, -1.4],
    ['Stable node', -3, 1.6],
    ['Stable spiral', -0.6, 2.4],
    ['Center', 0, 2.2],
    ['Unstable spiral', 0.7, 3.2]
  ];
  presets.forEach(([name, t, d]) => {
    const b = document.createElement('button');
    b.className = 'chip wide';
    b.textContent = name;
    b.addEventListener('click', () => goTo(t, d));
    presetsEl.appendChild(b);
  });
  function goTo(t, d) {
    tween = { t0: tau, d0: det, t1: t, d1: d, p: 0 };
  }

  // ---- trace–determinant plane --------------------------------------------------------------
  const tpad = 14;
  const TX = (t) =>
    tpad + ((t - TAU_RANGE[0]) / (TAU_RANGE[1] - TAU_RANGE[0])) * (tst.w - 2 * tpad);
  const TY = (d) =>
    tpad + ((DET_RANGE[1] - d) / (DET_RANGE[1] - DET_RANGE[0])) * (tst.h - 2 * tpad);
  const iTX = (x) =>
    TAU_RANGE[0] + ((x - tpad) / (tst.w - 2 * tpad)) * (TAU_RANGE[1] - TAU_RANGE[0]);
  const iTY = (y) =>
    DET_RANGE[1] - ((y - tpad) / (tst.h - 2 * tpad)) * (DET_RANGE[1] - DET_RANGE[0]);
  let tdDrag = false;
  tdCanvas.style.touchAction = 'none';
  const setFromPointer = (e) => {
    const p = pointerPos(e, tdCanvas);
    tween = null;
    tau = clamp(iTX(p.x), TAU_RANGE[0], TAU_RANGE[1]);
    det = clamp(iTY(p.y), DET_RANGE[0], DET_RANGE[1]);
    if (Math.abs(tau) < 0.08) tau = 0; // snap to the centre axis
    solveEig();
  };
  tdCanvas.addEventListener('pointerdown', (e) => {
    tdDrag = true;
    tdCanvas.setPointerCapture(e.pointerId);
    setFromPointer(e);
  });
  tdCanvas.addEventListener('pointermove', (e) => tdDrag && setFromPointer(e));
  tdCanvas.addEventListener('pointerup', () => (tdDrag = false));
  tdCanvas.addEventListener('pointercancel', () => (tdDrag = false));

  function drawTD() {
    const { ctx, w, h } = tst;
    if (!w) return;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(8,11,20,.82)';
    ctx.fillRect(0, 0, w, h);
    // saddle region
    ctx.fillStyle = rgba(PALETTE.rose, 0.1);
    ctx.fillRect(tpad, TY(0), w - 2 * tpad, h - tpad - TY(0));
    // spiral regions (above the parabola)
    const para = (t) => (t * t) / 4;
    ctx.beginPath();
    ctx.moveTo(TX(TAU_RANGE[0]), TY(DET_RANGE[1]));
    ctx.lineTo(TX(0), TY(DET_RANGE[1]));
    ctx.lineTo(TX(0), TY(0));
    for (let t = 0; t >= TAU_RANGE[0]; t -= 0.05) ctx.lineTo(TX(t), TY(para(t)));
    ctx.closePath();
    ctx.fillStyle = rgba(PALETTE.cyan, 0.08);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(TX(0), TY(DET_RANGE[1]));
    ctx.lineTo(TX(TAU_RANGE[1]), TY(DET_RANGE[1]));
    for (let t = TAU_RANGE[1]; t >= 0; t -= 0.05) ctx.lineTo(TX(t), TY(para(t)));
    ctx.closePath();
    ctx.fillStyle = rgba(PALETTE.rose, 0.08);
    ctx.fill();
    // axes
    ctx.strokeStyle = 'rgba(255,255,255,.22)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(tpad, TY(0));
    ctx.lineTo(w - tpad, TY(0));
    ctx.moveTo(TX(0), tpad);
    ctx.lineTo(TX(0), h - tpad);
    ctx.stroke();
    // parabola
    ctx.beginPath();
    for (let t = TAU_RANGE[0]; t <= TAU_RANGE[1]; t += 0.05) {
      const x = TX(t);
      const y = TY(para(t));
      lineStep(ctx, t === TAU_RANGE[0], x, y);
    }
    ctx.strokeStyle = rgba(PALETTE.amber, 0.85);
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.font = '10px ui-monospace, Menlo, monospace';
    ctx.fillStyle = PALETTE.dim;
    ctx.fillText('saddle', tpad + 6, h - tpad - 6);
    ctx.fillText('spiral', TX(-3.6), TY(4.4));
    ctx.fillText('spiral', TX(1.5), TY(4.4));
    const small = w < 190;
    if (!small) {
      ctx.fillText('node', TX(-3.6), TY(0.5));
      ctx.fillText('node', TX(2.6), TY(0.5));
    }
    if (!small) {
      ctx.fillText('τ²=4Δ', TX(2.7), TY(2.0));
    }
    ctx.fillText('Δ', TX(0) + 5, tpad + 8);
    // puck
    const x = TX(tau);
    const y = TY(det);
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = 'rgba(255,255,255,.35)';
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, TY(0));
    ctx.moveTo(x, y);
    ctx.lineTo(TX(0), y);
    ctx.stroke();
    ctx.setLineDash([]);
    const g = ctx.createRadialGradient(x, y, 0, x, y, 16);
    g.addColorStop(0, 'rgba(255,255,255,.9)');
    g.addColorStop(0.3, rgba(PALETTE.violet, 0.5));
    g.addColorStop(1, rgba(PALETTE.violet, 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, 16, 0, 6.2832);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 6.2832);
    ctx.fill();
  }

  // ---- particles ----------------------------------------------------------------------------
  const rnd = math.rand(5);
  const P = Array.from({ length: NP }, () => spawn({}));
  function spawn(p) {
    p.x = (rnd() * 2 - 1) * V;
    p.y = (rnd() * 2 - 1) * V;
    p.life = 2 + rnd() * 5;
    p.age = rnd() * 2;
    p.tr = new Float32Array(TRAIL * 2);
    p.n = 0;
    p.h = 0;
    return p;
  }

  const rhs = (_t, y, d) => {
    d.data[0] = y.data[1];
    d.data[1] = -det * y.data[0] + tau * y.data[1];
    return d;
  };

  // ---- frame --------------------------------------------------------------------------------
  let hudTick = 0;
  solveEig();
  onFrame(st, (dt, time) => {
    if (tween) {
      tween.p = Math.min(1, tween.p + dt / 0.9);
      const e = smooth(tween.p);
      tau = tween.t0 + (tween.t1 - tween.t0) * e;
      det = tween.d0 + (tween.d1 - tween.d0) * e;
      solveEig();
      if (tween.p >= 1) tween = null;
    }
    drawTD();

    const h = Math.max(1e-4, Math.min(dt, 0.033) * (reducedMotion ? 0.4 : 1));
    const s = Math.min(st.w, st.h) / (2 * V);
    const cx = st.w / 2;
    const cy = st.h / 2;

    for (const p of P) {
      const r = ode.rungeKuttaFixed('rk4', rhs, 0, h, [p.x, p.y], h);
      const last = r.t.size - 1;
      p.x = r.y.get(last, 0);
      p.y = r.y.get(last, 1);
      p.age += h;
      p.tr[2 * p.h] = p.x;
      p.tr[2 * p.h + 1] = p.y;
      p.h = (p.h + 1) % TRAIL;
      p.n = Math.min(TRAIL, p.n + 1);
      if (
        p.age > p.life ||
        Math.abs(p.x) > V * 1.02 ||
        Math.abs(p.y) > V * 1.02 ||
        !Number.isFinite(p.x)
      ) {
        spawn(p);
      }
    }

    const { ctx, w, h: H } = st;
    ctx.clearRect(0, 0, w, H);
    // axes
    ctx.strokeStyle = 'rgba(255,255,255,.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(w, cy);
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, H);
    ctx.stroke();
    for (let r = 1; r <= V; r++) {
      ctx.beginPath();
      ctx.arc(cx, cy, r * s, 0, 6.2832);
      ctx.stroke();
    }

    // eigen-directions
    ctx.font = '11px ui-monospace, Menlo, monospace';
    eigs.forEach((e, k) => {
      if (Math.abs(e.value.im) > 1e-9) return;
      const vx = e.vectorRe.get(0);
      const vy = e.vectorRe.get(1);
      const L = Math.hypot(w, H);
      const col = e.value.re < 0 ? PALETTE.cyan : PALETTE.rose;
      ctx.strokeStyle = rgba(col, 0.55);
      ctx.lineWidth = 1.6;
      ctx.setLineDash([8, 7]);
      ctx.lineDashOffset = -time * 14 * (e.value.re < 0 ? -1 : 1);
      ctx.beginPath();
      ctx.moveTo(cx - vx * L, cy + vy * L);
      ctx.lineTo(cx + vx * L, cy - vy * L);
      ctx.stroke();
      ctx.setLineDash([]);
      // label on the +v side, clamped to view
      const tt = Math.min(
        Math.abs(w / 2 - 70) / (Math.abs(vx) + 1e-9),
        Math.abs(H / 2 - 24) / (Math.abs(vy) + 1e-9),
        220
      );
      ctx.fillStyle = rgba(col, 1);
      ctx.fillText(
        `λ${k + 1} = ${fmt(e.value.re, 2)}`,
        cx + vx * tt * (k % 2 ? -1 : 1) + 6,
        cy - vy * tt * (k % 2 ? -1 : 1)
      );
    });

    // particles
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    for (const p of P) {
      const a = Math.min(1, p.age / 0.6) * Math.min(1, (p.life - p.age) / 0.8);
      if (a <= 0 || p.n < 2) continue;
      const vx = p.y;
      const vy = -det * p.x + tau * p.y;
      const outward = p.x * vx + p.y * vy > 0;
      const speed = Math.min(1, Math.hypot(vx, vy) / 3.5);
      const col = outward ? PALETTE.rose : PALETTE.cyan;
      const q0 = (p.h - p.n + TRAIL) % TRAIL;
      ctx.beginPath();
      for (let k = 0; k < p.n; k++) {
        const q = (q0 + k) % TRAIL;
        const x = cx + p.tr[2 * q] * s;
        const y = cy - p.tr[2 * q + 1] * s;
        lineStep(ctx, k === 0, x, y);
      }
      ctx.strokeStyle = rgba(col, a * (0.25 + 0.5 * speed));
      ctx.lineWidth = 1.2 + speed * 1.2;
      ctx.stroke();
      ctx.fillStyle = rgba(col, a * 0.95);
      ctx.beginPath();
      ctx.arc(cx + p.x * s, cy - p.y * s, 1.4 + speed * 1.2, 0, 6.2832);
      ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';

    // origin
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, 14);
    g.addColorStop(0, 'rgba(255,255,255,.8)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, 14, 0, 6.2832);
    ctx.fill();

    if (++hudTick % 10 === 0) {
      hud.innerHTML = `<b>Matrix.eig</b> · <b>rungeKuttaFixed('rk4')</b> × ${NP} tracers per frame`;
    }
  });
}
