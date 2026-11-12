// Chamber VI: reaction–diffusion. A ring of Brusselator cells (see brusselator.js) is advanced by
// the implicit ode.trBdf2Adaptive; the limit cycle in the inset comes from ode.rungeKuttaAdaptive
// with events and dense output.
import { math } from 'numerics-js';
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
  lineStep,
  reducedMotion
} from './util.js';
import { N, REGIMES, createField, hopfB, turingB } from './brusselator.js';

const KH = 200; // kymograph rows kept on screen
const BAND = 0.24; // fraction of the stage height used by the live profile
const KICK = 0.25; // how hard a pointer drag pushes u, per kymograph row
const SLIDER = {
  waves: { min: 2.2, max: 4.4 },
  spots: { min: 2.5, max: 5.2 }
};

// colour table for the kymograph: deep navy → blue → cyan → near white (little-endian ABGR)
const lut = new Uint32Array(256);
{
  const stops = [
    [0, [6, 10, 24]],
    [0.3, [18, 52, 138]],
    [0.62, [70, 168, 255]],
    [0.86, [170, 236, 255]],
    [1, [255, 250, 232]]
  ];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 1;
    while (k < stops.length - 1 && t > stops[k][0]) k++;
    const [t0, c0] = stops[k - 1];
    const [t1, c1] = stops[k];
    const f = (t - t0) / (t1 - t0);
    const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * f));
    lut[i] = (255 << 24) | (c[2] << 16) | (c[1] << 8) | c[0];
  }
}

export function mountReaction() {
  const canvas = $('#rd-canvas');
  const cycleCanvas = $('#rd-cycle');
  const hud = $('#hud-rd');
  const readout = $('#rd-readout');
  const slider = $('#rd-b');
  const sliderVal = $('#rd-b-val');
  const reseed = $('#rd-reseed');
  const st = createStage(canvas);
  const cst = createStage(cycleCanvas);

  const field = createField('waves');
  const rnd = math.rand(7);
  let regime = 'waves';

  // ---- kymograph buffer: one row per solver call, newest at the bottom -----------------------
  const buf = document.createElement('canvas');
  buf.width = N;
  buf.height = KH;
  const bctx = buf.getContext('2d');
  const img = bctx.createImageData(N, KH);
  const px32 = new Uint32Array(img.data.buffer);
  const clearHistory = () => px32.fill(lut[0]);
  let lo = 0;
  let hi = 1;

  const pushRow = () => {
    const u = field.y;
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < N; i++) {
      if (u[i] < mn) mn = u[i];
      if (u[i] > mx) mx = u[i];
    }
    // follow the row's range slowly, but never stretch noise over the whole colour table
    lo += (mn - lo) * 0.08;
    hi += (mx - hi) * 0.08;
    const span = Math.max(hi - lo, 0.8);
    const mid = (hi + lo) / 2;
    px32.copyWithin(0, N);
    const base = (KH - 1) * N;
    for (let i = 0; i < N; i++) {
      const t = clamp((u[i] - (mid - span / 2)) / span, 0, 1);
      px32[base + i] = lut[(t * 255) | 0];
    }
  };

  // ---- regime handling ---------------------------------------------------------------------
  let acc = 0;
  let last = { steps: 0, decompositions: 0, evaluations: 0 };
  let rows = 0;

  function seed() {
    if (regime === 'waves') field.seedWaves();
    else field.seedNoise(rnd, 0.06);
    clearHistory();
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < N; i++) {
      mn = Math.min(mn, field.y[i]);
      mx = Math.max(mx, field.y[i]);
    }
    lo = mn;
    hi = mx;
    acc = 0;
    rows = 0;
  }

  function describe() {
    const { A, B, Du, Dv } = field.params;
    const bH = hopfB(A);
    const bT = turingB(A, Du, Dv);
    const fp = `<span class="dim">steady state (u*, v*) = (A, B/A) = (${fmt(A, 2)}, ${fmt(B / A, 2)})</span>`;
    if (B > bH && field.cycle) {
      readout.innerHTML =
        `<b>Hopf regime</b> · <i>B</i> = ${fmt(B, 2)} &gt; 1 + <i>A</i>² = ${fmt(bH, 2)}: every cell is a ` +
        `chemical clock with period <i>T</i> = ${fmt(field.cycle.period, 3)}. Diffusion couples ` +
        `neighbours, so the ring keeps one wave circulating.<br>${fp}`;
    } else if (B > bT && Dv > Du) {
      const kc = Math.sqrt(A / Math.sqrt(Du * Dv));
      readout.innerHTML =
        `<b>Turing regime</b> · ${fmt(bT, 2)} &lt; <i>B</i> = ${fmt(B, 2)} &lt; ${fmt(bH, 2)}: a quiet ` +
        `cell is stable, but fast-diffusing <i>v</i> destabilises the ring into stationary spots, ` +
        `wavelength ≈ 2π/<i>k<sub>c</sub></i> = ${fmt((2 * Math.PI) / kc, 2)}.<br>${fp}`;
    } else {
      readout.innerHTML =
        `<b>Stable</b> · <i>B</i> = ${fmt(B, 2)} is below both thresholds (Turing ${fmt(bT, 2)}, ` +
        `Hopf ${fmt(bH, 2)}): disturbances die away.<br>${fp}`;
    }
  }

  function setRegime(name) {
    regime = name;
    setSeg(name);
    field.configure(name);
    const s = SLIDER[name];
    slider.min = s.min;
    slider.max = s.max;
    slider.value = field.params.B;
    sliderVal.textContent = fmt(field.params.B, 2);
    seed();
    describe();
    drawCycle();
  }
  const setSeg = segmented($('#rd-regime'), setRegime);

  slider.addEventListener('input', () => {
    const B = +slider.value;
    sliderVal.textContent = fmt(B, 2);
    field.setB(B);
    describe();
  });
  reseed.addEventListener('click', seed);

  // ---- pointer: drag across the field to disturb u ------------------------------------------
  const ptr = { x: -1, down: false };
  canvas.style.touchAction = 'none';
  const setX = (e) => {
    ptr.x = clamp((pointerPos(e, canvas).x / st.w) * N, 0, N - 1e-6);
  };
  canvas.addEventListener('pointerdown', (e) => {
    ptr.down = true;
    canvas.setPointerCapture(e.pointerId);
    setX(e);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (ptr.down || e.pointerType === 'mouse') setX(e);
  });
  canvas.addEventListener('pointerup', () => (ptr.down = false));
  canvas.addEventListener('pointercancel', () => (ptr.down = false));
  canvas.addEventListener('pointerleave', () => {
    if (!ptr.down) ptr.x = -1;
  });

  // ---- inset: the (u, v) plane with the limit cycle from rungeKuttaAdaptive -----------------
  // Axis extents follow the limit cycle when there is one, the steady state otherwise.
  const extent = () => {
    const { A, B } = field.params;
    const c = field.cycle;
    return c && B > hopfB(A) ? [1.12 * c.umax, 1.12 * c.vmax] : [2.6 * A, 2.2 * (B / A)];
  };
  let UM = 4;
  let VM = 5;
  const IX = (u) => 12 + (u / UM) * (cst.w - 24);
  const IY = (v) => cst.h - 14 - (v / VM) * (cst.h - 26);

  function drawCycle() {
    const { ctx, w, h } = cst;
    if (!w) return;
    [UM, VM] = extent();
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(8,11,20,.82)';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(255,255,255,.14)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(IX(0), 8);
    ctx.lineTo(IX(0), IY(0));
    ctx.lineTo(w - 8, IY(0));
    ctx.stroke();
    const { A, B } = field.params;
    // limit cycle of an isolated cell
    if (field.cycle && B > hopfB(A)) {
      const { path, samples } = field.cycle;
      ctx.beginPath();
      for (let i = 0; i <= samples; i++)
        lineStep(ctx, i === 0, IX(path[2 * i]), IY(path[2 * i + 1]));
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = rgba(PALETTE.amber, 0.75);
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // steady state
    ctx.strokeStyle = 'rgba(255,255,255,.7)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(IX(A), IY(B / A), 3.6, 0, 6.2832);
    ctx.stroke();
    // one dot per cell, hue follows position along the ring
    const y = field.y;
    for (let i = 0; i < N; i++) {
      const u = clamp(y[i], 0, UM);
      const v = clamp(y[N + i], 0, VM);
      ctx.fillStyle = `hsl(${(360 * i) / N} 85% 68%)`;
      ctx.beginPath();
      ctx.arc(IX(u), IY(v), 2.2, 0, 6.2832);
      ctx.fill();
    }
    ctx.font = '10px ui-monospace, Menlo, monospace';
    ctx.fillStyle = PALETTE.dim;
    ctx.fillText('v', IX(0) + 4, 16);
    ctx.fillText('u', w - 18, IY(0) - 5);
  }

  // ---- frame ---------------------------------------------------------------------------------
  let hudTick = 0;
  setRegime('waves');
  onFrame(st, (dt) => {
    const R = REGIMES[regime];
    acc += dt * R.rate * (reducedMotion ? 0.4 : 1);
    let n = 0;
    while (acc >= R.row && n < 3) {
      if (ptr.down && ptr.x >= 0) field.kick(ptr.x - 0.5, KICK);
      last = field.step(R.row);
      pushRow();
      acc -= R.row;
      rows++;
      n++;
    }
    if (acc > R.row * 3) acc = 0; // fell behind (tab was hidden): drop the backlog

    const { ctx, w, h } = st;
    if (!w) return;
    ctx.clearRect(0, 0, w, h);
    const ky = h * (1 - BAND);

    // kymograph
    bctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(buf, 0, 0, w, ky);
    ctx.fillStyle = 'rgba(6,9,18,.28)';
    ctx.fillRect(0, 0, w, ky);

    // live profile
    ctx.fillStyle = 'rgba(6,9,18,.86)';
    ctx.fillRect(0, ky, w, h - ky);
    ctx.strokeStyle = 'rgba(255,255,255,.14)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, ky + 0.5);
    ctx.lineTo(w, ky + 0.5);
    ctx.stroke();
    const padT = 18;
    const padB = 22;
    const [um, vm] = extent();
    const top = Math.max(um, vm);
    const PY = (v) => ky + padT + (1 - v / top) * (h - ky - padT - padB);
    const { A, B } = field.params;
    ctx.setLineDash([2, 5]);
    ctx.strokeStyle = 'rgba(255,255,255,.2)';
    ctx.beginPath();
    for (const lvl of [A, B / A]) {
      ctx.moveTo(0, PY(lvl));
      ctx.lineTo(w, PY(lvl));
    }
    ctx.stroke();
    ctx.setLineDash([]);
    const y = field.y;
    for (const [off, col] of [
      [N, PALETTE.rose],
      [0, PALETTE.blue]
    ]) {
      ctx.beginPath();
      for (let i = 0; i < N; i++) lineStep(ctx, i === 0, ((i + 0.5) / N) * w, PY(y[off + i]));
      ctx.strokeStyle = rgba(col, 0.95);
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
    ctx.font = '10px ui-monospace, Menlo, monospace';
    ctx.fillStyle = rgba(PALETTE.blue, 1);
    ctx.fillText('u', 12, ky + 14);
    ctx.fillStyle = rgba(PALETTE.rose, 1);
    ctx.fillText('v', 26, ky + 14);
    ctx.fillStyle = PALETTE.dim;
    ctx.fillText('x → (ring, periodic)', 44, ky + 14);
    ctx.fillText('now', w - 30, ky - 6);
    ctx.save();
    ctx.translate(12, ky - 10);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('time →', 0, 0);
    ctx.restore();

    // pointer
    if (ptr.x >= 0) {
      const x = (ptr.x / N) * w;
      ctx.strokeStyle = rgba(PALETTE.cyan, ptr.down ? 0.9 : 0.4);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    drawCycle();

    if (++hudTick % 8 === 0) {
      hud.innerHTML =
        `<b>trBdf2Adaptive</b> · ${2 * N} coupled ODEs · analytic Jacobian · ` +
        `${last.steps} step${last.steps === 1 ? '' : 's'}, ${last.decompositions} LU per row`;
    }
  });
  new ResizeObserver(drawCycle).observe(cycleCanvas);
}
