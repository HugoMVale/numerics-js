// Chamber IV: a curve atelier. interpolate (PCHIP) + integrate.quad + numdiff + roots.brent, all live.
import { interpolate, integrate, numdiff, roots } from 'numerics-js';
import {
  $,
  createStage,
  onFrame,
  pointerPos,
  rgba,
  PALETTE,
  fmt,
  clamp,
  lineStep
} from './util.js';

const XMIN = 0;
const XMAX = 10;
const YMIN = -1.2;
const YMAX = 5.6;

export function mountCurve() {
  const canvas = $('#curve-canvas');
  const hud = $('#hud-curve');
  const readout = $('#curve-readout');
  const splineBtn = $('#curve-spline');
  const st = createStage(canvas);

  const knots = [
    [0.4, 0.5],
    [1.8, 1.7],
    [3.0, 1.8],
    [4.1, 1.85],
    [5.1, 4.4],
    [6.9, 4.3],
    [9.6, 1.9]
  ].map(([x, y]) => ({ x, y }));
  let level = 3.0;
  let a = 1.2;
  let b = 8.2;
  let showSpline = false;
  let hoverX = null;
  let drag = null;
  let f;
  let spline;

  splineBtn.addEventListener('click', () => {
    showSpline = !showSpline;
    splineBtn.setAttribute('aria-pressed', String(showSpline));
  });

  const pad = { l: 44, r: 30, t: 22, b: 40 };
  const X = (x) => pad.l + ((x - XMIN) / (XMAX - XMIN)) * (st.w - pad.l - pad.r);
  const Y = (y) => pad.t + ((YMAX - y) / (YMAX - YMIN)) * (st.h - pad.t - pad.b);
  const iX = (px) => XMIN + ((px - pad.l) / (st.w - pad.l - pad.r)) * (XMAX - XMIN);
  const iY = (py) => YMAX - ((py - pad.t) / (st.h - pad.t - pad.b)) * (YMAX - YMIN);

  let stats = { evalsQuad: 0, sub: 0, err: 0, value: 0, rootEvals: 0, roots: [] };

  function rebuild() {
    const xs = knots.map((k) => k.x);
    const ys = knots.map((k) => k.y);
    f = new interpolate.PchipInterpolator1D(xs, ys);
    spline = new interpolate.CubicSplineInterpolator1D(xs, ys);
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const q = integrate.quad((x) => f.eval(x), lo, hi, { atol: 1e-10, rtol: 1e-10 });
    stats.value = (a <= b ? 1 : -1) * q.value;
    stats.err = q.error;
    stats.evalsQuad = q.evaluations;
    stats.sub = q.subintervals;
    // roots of f(x) - level: bracket by scanning, refine with Brent
    const found = [];
    let evals = 0;
    const g = (x) => f.eval(x) - level;
    const N = 400;
    const x0 = knots[0].x;
    const x1 = knots[knots.length - 1].x;
    let xp = x0;
    let gp = g(xp);
    for (let i = 1; i <= N; i++) {
      const xc = x0 + ((x1 - x0) * i) / N;
      const gc = g(xc);
      if (gp === 0) found.push(xp);
      else if (gp * gc < 0) {
        const r = roots.brent(g, xp, xc, { tolx: 1e-12, tolf: 1e-12 });
        evals += r.evaluations;
        found.push(r.x);
      }
      xp = xc;
      gp = gc;
    }
    stats.rootEvals = evals;
    stats.roots = found;
    describe();
  }

  function describe() {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    readout.innerHTML =
      `<b>∫</b> from ${fmt(lo, 2)} to ${fmt(hi, 2)} = <b>${fmt(stats.value, 6)}</b> ` +
      `<span class="dim">(quad: ${stats.sub} subintervals, ${stats.evalsQuad} evaluations, error ≈ ${fmt(stats.err, 1)})</span><br>` +
      `<b>${stats.roots.length}</b> crossing${stats.roots.length === 1 ? '' : 's'} of <i>y</i> = ${fmt(level, 2)} ` +
      `<span class="dim">(brent: ${stats.rootEvals} evaluations)</span>`;
  }

  // ---- interaction -------------------------------------------------------------------------
  canvas.style.touchAction = 'none';
  const hit = (p) => {
    for (let i = 0; i < knots.length; i++) {
      if (Math.hypot(p.x - X(knots[i].x), p.y - Y(knots[i].y)) < 16) return { kind: 'knot', i };
    }
    if (Math.hypot(p.x - X(a), p.y - (st.h - pad.b + 14)) < 16) return { kind: 'a' };
    if (Math.hypot(p.x - X(b), p.y - (st.h - pad.b + 14)) < 16) return { kind: 'b' };
    if (Math.hypot(p.x - (st.w - pad.r + 2), p.y - Y(level)) < 18) return { kind: 'level' };
    return null;
  };
  canvas.addEventListener('pointerdown', (e) => {
    const p = pointerPos(e, canvas);
    const h = hit(p);
    if (h) {
      drag = h;
      canvas.setPointerCapture(e.pointerId);
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    const p = pointerPos(e, canvas);
    if (drag) {
      if (drag.kind === 'knot') {
        const i = drag.i;
        const lo = i > 0 ? knots[i - 1].x + 0.15 : XMIN;
        const hi = i < knots.length - 1 ? knots[i + 1].x - 0.15 : XMAX;
        knots[i].x = clamp(iX(p.x), lo, hi);
        knots[i].y = clamp(iY(p.y), YMIN + 0.1, YMAX - 0.1);
      } else if (drag.kind === 'a') a = clamp(iX(p.x), XMIN, XMAX);
      else if (drag.kind === 'b') b = clamp(iX(p.x), XMIN, XMAX);
      else if (drag.kind === 'level') level = clamp(iY(p.y), YMIN + 0.2, YMAX - 0.2);
      rebuild();
    } else {
      canvas.style.cursor = hit(p) ? 'grab' : 'crosshair';
      hoverX = p.x > pad.l && p.x < st.w - pad.r ? iX(p.x) : null;
    }
  });
  const end = () => (drag = null);
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('pointerleave', () => (hoverX = null));

  rebuild();

  // ---- frame -------------------------------------------------------------------------------
  let hudTick = 0;
  let demoT = 0;
  let touched = false;
  canvas.addEventListener('pointerdown', () => (touched = true));
  onFrame(st, (dt, time) => {
    // gentle auto-scan of the tangent until the visitor touches the plot
    if (!touched && hoverX === null) {
      demoT += dt;
    }
    const tx =
      hoverX !== null ? hoverX : !touched ? 0.8 + ((Math.sin(demoT * 0.5) + 1) / 2) * 8.4 : null;

    const { ctx, w, h } = st;
    ctx.clearRect(0, 0, w, h);
    const x0 = knots[0].x;
    const x1 = knots[knots.length - 1].x;

    // grid
    ctx.font = '10.5px ui-monospace, Menlo, monospace';
    ctx.fillStyle = PALETTE.dim;
    ctx.strokeStyle = 'rgba(255,255,255,.06)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= 10; x += 1) {
      ctx.beginPath();
      ctx.moveTo(X(x), pad.t);
      ctx.lineTo(X(x), h - pad.b);
      ctx.stroke();
      ctx.fillText(String(x), X(x) - 3, h - pad.b + 14);
    }
    for (let y = 0; y <= 5; y += 1) {
      ctx.beginPath();
      ctx.moveTo(pad.l, Y(y));
      ctx.lineTo(w - pad.r, Y(y));
      ctx.stroke();
      ctx.fillText(String(y), pad.l - 18, Y(y) + 4);
    }
    ctx.strokeStyle = 'rgba(255,255,255,.28)';
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(0));
    ctx.lineTo(w - pad.r, Y(0));
    ctx.stroke();

    // area under the curve between a and b
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const xa = Math.max(lo, x0);
    const xb = Math.min(hi, x1);
    if (xb > xa) {
      const grad = ctx.createLinearGradient(0, Y(YMAX), 0, Y(0));
      grad.addColorStop(0, rgba(PALETTE.cyan, 0.42));
      grad.addColorStop(1, rgba(PALETTE.cyan, 0.04));
      ctx.beginPath();
      ctx.moveTo(X(xa), Y(0));
      const n = 160;
      for (let i = 0; i <= n; i++) {
        const x = xa + ((xb - xa) * i) / n;
        ctx.lineTo(X(x), Y(f.eval(x)));
      }
      ctx.lineTo(X(xb), Y(0));
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
    }

    // level line
    ctx.setLineDash([5, 6]);
    ctx.strokeStyle = rgba(PALETTE.amber, 0.75);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(level));
    ctx.lineTo(w - pad.r, Y(level));
    ctx.stroke();
    ctx.setLineDash([]);

    const path = (fn, from, to) => {
      ctx.beginPath();
      const n = 300;
      for (let i = 0; i <= n; i++) {
        const x = from + ((to - from) * i) / n;
        const y = fn(x);
        lineStep(ctx, i === 0, X(x), Y(y));
      }
    };

    // ghost spline for contrast
    if (showSpline) {
      ctx.setLineDash([6, 5]);
      path((x) => spline.eval(x), x0, x1);
      ctx.strokeStyle = rgba(PALETTE.rose, 0.9);
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = rgba(PALETTE.rose, 1);
      ctx.fillText('cubic spline overshoots', X(3.25), Y(0.9));
    }

    // PCHIP curve with glow
    path((x) => f.eval(x), x0, x1);
    ctx.lineCap = 'round';
    ctx.strokeStyle = rgba(PALETTE.cyan, 0.2);
    ctx.lineWidth = 9;
    ctx.stroke();
    ctx.strokeStyle = rgba(PALETTE.cyan, 1);
    ctx.lineWidth = 2.4;
    ctx.stroke();

    // tangent via numdiff
    if (tx !== null && tx > x0 && tx < x1) {
      const [dfdx] = numdiff.derivativeCentered((x) => f.eval(x), tx);
      const exact = f.derivative(tx);
      const y = f.eval(tx);
      const half = 1.1;
      ctx.strokeStyle = rgba(PALETTE.violet, 0.95);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(X(tx - half), Y(y - dfdx * half));
      ctx.lineTo(X(tx + half), Y(y + dfdx * half));
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(X(tx), Y(y), 4, 0, 6.2832);
      ctx.fill();
      ctx.fillStyle = rgba(PALETTE.violet, 1);
      ctx.font = '11.5px ui-monospace, Menlo, monospace';
      const bx = w - pad.r - 6;
      ctx.textAlign = 'right';
      ctx.fillText(`numdiff   f′(${fmt(tx, 2)}) = ${fmt(dfdx, 6)}`, bx, pad.t + 12);
      ctx.fillStyle = PALETTE.dim;
      ctx.fillText(`|Δ| vs PCHIP derivative: ${fmt(Math.abs(dfdx - exact), 1)}`, bx, pad.t + 28);
      ctx.textAlign = 'left';
      ctx.font = '10.5px ui-monospace, Menlo, monospace';
    }

    // roots
    stats.roots.forEach((r, i) => {
      const pulse = (time * 0.9 + i * 0.3) % 1;
      ctx.strokeStyle = rgba(PALETTE.amber, 0.8 * (1 - pulse));
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(X(r), Y(level), 5 + 14 * pulse, 0, 6.2832);
      ctx.stroke();
      ctx.fillStyle = rgba(PALETTE.amber, 1);
      ctx.beginPath();
      ctx.arc(X(r), Y(level), 4, 0, 6.2832);
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#070a13';
      ctx.fillStyle = rgba(PALETTE.amber, 1);
      const tx2 = X(r) - 14;
      const ty2 = Y(level) + 24;
      ctx.strokeText(fmt(r, 3), tx2, ty2);
      ctx.fillText(fmt(r, 3), tx2, ty2);
    });

    // handles: knots
    knots.forEach((k, i) => {
      const act = drag && drag.kind === 'knot' && drag.i === i;
      ctx.fillStyle = '#06080f';
      ctx.strokeStyle = act ? '#fff' : rgba(PALETTE.cyan, 1);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(X(k.x), Y(k.y), act ? 8 : 6.2, 0, 6.2832);
      ctx.fill();
      ctx.stroke();
    });
    // handles: a, b
    for (const [v, name] of [
      [a, 'a'],
      [b, 'b']
    ]) {
      const x = X(v);
      const y = h - pad.b + 14;
      ctx.fillStyle = rgba(PALETTE.cyan, 0.95);
      ctx.beginPath();
      ctx.moveTo(x, y - 8);
      ctx.lineTo(x + 7, y + 5);
      ctx.lineTo(x - 7, y + 5);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = PALETTE.ink;
      ctx.fillText(name, x - 3, y + 18);
      ctx.strokeStyle = rgba(PALETTE.cyan, 0.4);
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, y - 9);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // handle: level
    ctx.fillStyle = rgba(PALETTE.amber, 1);
    ctx.beginPath();
    ctx.moveTo(w - pad.r + 12, Y(level));
    ctx.lineTo(w - pad.r + 2, Y(level) - 7);
    ctx.lineTo(w - pad.r + 2, Y(level) + 7);
    ctx.closePath();
    ctx.fill();

    if (++hudTick % 10 === 0) {
      hud.innerHTML = `<b>PchipInterpolator1D</b> · <b>quad</b> · <b>derivativeCentered</b> · <b>brent</b>`;
    }
  });
}
