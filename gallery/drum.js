// Chamber I: a circular drum. Modes are J_n(j_nm r)·cos(nθ), built from special.besselJ / besselJZero.
import { special } from 'numerics-js';
import {
  $,
  createStage,
  onFrame,
  pointerPos,
  segmented,
  fmt,
  lineStep,
  rgba,
  PALETTE,
  reducedMotion
} from './util.js';

const NMAX = 9;
const MMAX = 6;
const NR = 400; // radial table resolution
const BUF = 240; // internal field resolution
const C0 = (2 * Math.PI * 0.8) / 2.404825557695773; // wave speed: fundamental rings at 0.8 Hz
const sub = (n, m) => `<i>j</i><sub>${n},${m}</sub>`;

export function mountDrum() {
  const canvas = $('#drum-canvas');
  const mini = $('#drum-mini');
  const hud = $('#hud-drum');
  const readout = $('#drum-readout');
  const chipsEl = $('#drum-chips');
  const soundBtn = $('#drum-sound');
  const st = createStage(canvas);
  const mst = createStage(mini);

  // ---- tables from the library -------------------------------------------------------------
  const NM = (NMAX + 1) * MMAX;
  const idx = (n, m) => n * MMAX + (m - 1);
  const jz = new Float64Array(NM);
  const norm = new Float64Array(NM); // J_{n+1}(j)^2
  const omega = new Float64Array(NM);
  const gamma = new Float64Array(NM);
  for (let n = 0; n <= NMAX; n++) {
    for (let m = 1; m <= MMAX; m++) {
      const k = idx(n, m);
      jz[k] = special.besselJZero(n, m);
      norm[k] = special.besselJ(n + 1, jz[k]) ** 2;
      omega[k] = jz[k] * C0;
      gamma[k] = 0.22 + 0.035 * jz[k];
    }
  }
  const R = new Float32Array(NM * NR);
  for (let n = 0; n <= NMAX; n++) {
    for (let m = 1; m <= MMAX; m++) {
      const k = idx(n, m);
      for (let i = 0; i < NR; i++) R[k * NR + i] = special.besselJ(n, (jz[k] * i) / (NR - 1));
    }
  }

  // ---- pixel tables -------------------------------------------------------------------------
  const pix = [];
  for (let y = 0; y < BUF; y++) {
    for (let x = 0; x < BUF; x++) {
      const dx = (x + 0.5 - BUF / 2) / (BUF / 2);
      const dy = (y + 0.5 - BUF / 2) / (BUF / 2);
      const r = Math.hypot(dx, dy);
      if (r < 1) pix.push([y * BUF + x, r, Math.atan2(dy, dx)]);
    }
  }
  const count = pix.length;
  const pIndex = new Uint32Array(count);
  const pRad = new Uint16Array(count);
  const N1 = NMAX + 1;
  const cosT = new Float32Array(count * N1);
  const sinT = new Float32Array(count * N1);
  pix.forEach(([i, r, th], p) => {
    pIndex[p] = i;
    pRad[p] = Math.min(NR - 1, Math.round(r * (NR - 1)));
    const c = Math.cos(th);
    const s = Math.sin(th);
    cosT[p * N1] = 1;
    sinT[p * N1] = 0;
    cosT[p * N1 + 1] = c;
    sinT[p * N1 + 1] = s;
    for (let n = 2; n <= NMAX; n++) {
      cosT[p * N1 + n] = 2 * c * cosT[p * N1 + n - 1] - cosT[p * N1 + n - 2];
      sinT[p * N1 + n] = 2 * c * sinT[p * N1 + n - 1] - sinT[p * N1 + n - 2];
    }
  });

  // ---- colour map ---------------------------------------------------------------------------
  const lut = new Uint32Array(512);
  const ramp = (t, stops) => {
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const a = stops[i - 1];
        const b = stops[i];
        const f = (t - a[0]) / (b[0] - a[0] || 1);
        return [0, 1, 2].map((k) => a[1][k] + (b[1][k] - a[1][k]) * f);
      }
    }
    return stops[stops.length - 1][1];
  };
  const warm = [
    [0, [9, 12, 22]],
    [0.45, [150, 62, 40]],
    [0.75, [255, 170, 80]],
    [1, [255, 240, 210]]
  ];
  const cool = [
    [0, [9, 12, 22]],
    [0.45, [34, 74, 170]],
    [0.75, [90, 190, 255]],
    [1, [210, 250, 255]]
  ];
  for (let i = 0; i < 512; i++) {
    const v = i / 255.5 - 1;
    const c = ramp(Math.pow(Math.abs(v), 0.85), v >= 0 ? warm : cool);
    lut[i] = (255 << 24) | (c[2] << 16) | (c[1] << 8) | c[0];
  }

  const buf = document.createElement('canvas');
  buf.width = buf.height = BUF;
  const bctx = buf.getContext('2d');
  const img = bctx.createImageData(BUF, BUF);
  const px32 = new Uint32Array(img.data.buffer);

  // ---- modal state --------------------------------------------------------------------------
  const qc = new Float64Array(NM);
  const vc = new Float64Array(NM);
  const qs = new Float64Array(NM);
  const vs = new Float64Array(NM);
  const field = new Float32Array(count);
  const Fc = new Float32Array(N1 * NR);
  const Fs = new Float32Array(N1 * NR);
  let mode = 'strike';
  let pure = { n: 2, m: 1 };
  let pureT = 0;
  let peak = 0;
  let ripples = [];
  let interacted = false;
  let autoTimer = 0.3;
  let autoCount = 0;
  let soundOn = false;

  // ---- audio --------------------------------------------------------------------------------
  let actx = null;
  let master = null;
  const ensureAudio = () => {
    if (!actx) {
      actx = new (window.AudioContext || window.webkitAudioContext)();
      master = actx.createGain();
      master.gain.value = 0.55;
      const comp = actx.createDynamicsCompressor();
      master.connect(comp);
      comp.connect(actx.destination);
    }
    if (actx.state === 'suspended') actx.resume();
  };
  const ping = (voices) => {
    if (!soundOn || !actx) return;
    const t = actx.currentTime;
    for (const v of voices) {
      const o = actx.createOscillator();
      const g = actx.createGain();
      o.type = 'sine';
      o.frequency.value = v.f;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(v.a, t + 0.004);
      g.gain.exponentialRampToValueAtTime(0.0001, t + v.d);
      o.connect(g);
      g.connect(master);
      o.start(t);
      o.stop(t + v.d + 0.05);
    }
  };
  const F0 = 98; // Hz of the fundamental mode
  soundBtn.addEventListener('click', () => {
    soundOn = !soundOn;
    if (soundOn) ensureAudio();
    soundBtn.setAttribute('aria-pressed', String(soundOn));
    soundBtn.textContent = soundOn ? 'Sound on' : 'Sound off';
    if (soundOn) ping([{ f: F0, a: 0.15, d: 0.6 }]);
  });

  // ---- strike -------------------------------------------------------------------------------
  function strike(r0, th0) {
    const rIdx = r0 * (NR - 1);
    const i0 = Math.floor(rIdx);
    const fr = rIdx - i0;
    const voices = [];
    for (let n = 0; n <= NMAX; n++) {
      const w = n === 0 ? 1 : 2;
      for (let m = 1; m <= MMAX; m++) {
        const k = idx(n, m);
        const Jr = R[k * NR + i0] * (1 - fr) + R[k * NR + Math.min(NR - 1, i0 + 1)] * fr;
        const c = (w * Jr) / norm[k];
        vc[k] += c * Math.cos(n * th0);
        vs[k] += c * Math.sin(n * th0);
        voices.push({ k, amp: Math.abs(c) / omega[k] });
      }
    }
    voices.sort((a, b) => b.amp - a.amp);
    const top = voices.slice(0, 9);
    const amax = top[0].amp || 1;
    ping(
      top.map((v) => ({
        f: (F0 * jz[v.k]) / jz[0],
        a: (0.2 * v.amp) / amax,
        d: 1.4 / (1 + 0.12 * jz[v.k])
      }))
    );
  }

  function setMode(m) {
    mode = m;
    setSeg(m);
    qc.fill(0);
    vc.fill(0);
    qs.fill(0);
    vs.fill(0);
    pureT = 0;
    chipsEl
      .querySelectorAll('button')
      .forEach((b) =>
        b.setAttribute(
          'aria-pressed',
          String(m === 'mode' && +b.dataset.n === pure.n && +b.dataset.m === pure.m)
        )
      );
    drawMini();
    describe();
  }
  const setSeg = segmented($('#drum-mode'), (v) => {
    interacted = true;
    setMode(v);
  });

  // chips: one per (n, m) with its Bessel zero
  for (let n = 0; n <= 5; n++) {
    const row = document.createElement('div');
    row.className = 'chip-row';
    row.innerHTML = `<span class="chip-n">n=${n}</span>`;
    for (let m = 1; m <= 4; m++) {
      const b = document.createElement('button');
      b.className = 'chip';
      b.dataset.n = n;
      b.dataset.m = m;
      b.title = `mode (n=${n}, m=${m})`;
      b.textContent = fmt(jz[idx(n, m)], 3);
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => {
        interacted = true;
        pure = { n, m };
        setMode('mode');
        ensureAudioIfOn();
        ping([{ f: (F0 * jz[idx(n, m)]) / jz[0], a: 0.18, d: 1.2 }]);
      });
      row.appendChild(b);
    }
    chipsEl.appendChild(row);
  }
  const ensureAudioIfOn = () => soundOn && ensureAudio();

  function describe() {
    if (mode === 'strike') {
      readout.innerHTML =
        'Strike the membrane anywhere. A hit at radius <i>r</i><sub>0</sub> excites each mode ' +
        '<i>J<sub>n</sub></i>(<i>j<sub>n,m</sub> r</i>)·cos <i>nθ</i> in proportion to ' +
        '<i>J<sub>n</sub></i>(<i>j<sub>n,m</sub> r</i><sub>0</sub>); the zeros <i>j<sub>n,m</sub></i> set the pitches, which is why a drum is inharmonic.';
    } else {
      const k = idx(pure.n, pure.m);
      readout.innerHTML =
        `Mode (<i>n</i>=${pure.n}, <i>m</i>=${pure.m}): ${sub(pure.n, pure.m)} = <b>${fmt(jz[k], 5)}</b>, ` +
        `pitch ×<b>${fmt(jz[k] / jz[0], 3)}</b> the fundamental. ` +
        `${pure.n} nodal diameter${pure.n === 1 ? '' : 's'}, ${pure.m - 1} nodal circle${pure.m === 2 ? '' : 's'}.`;
    }
  }

  // ---- mini plot: radial profiles straight from besselJ -------------------------------------
  function drawMini() {
    const { ctx, w, h } = mst;
    if (!w) return;
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 8, r: 8, t: 10, b: 16 };
    const iw = w - pad.l - pad.r;
    const ih = h - pad.t - pad.b;
    ctx.strokeStyle = 'rgba(255,255,255,.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t + ih / 2);
    ctx.lineTo(pad.l + iw, pad.t + ih / 2);
    ctx.stroke();
    ctx.font = '10px ui-monospace, Menlo, monospace';
    ctx.fillStyle = PALETTE.dim;
    if (mode === 'mode') {
      const { n, m } = pure;
      const k = idx(n, m);
      ctx.beginPath();
      for (let i = 0; i < NR; i++) {
        const x = pad.l + (iw * i) / (NR - 1);
        const y = pad.t + ih / 2 - (R[k * NR + i] / 0.6) * (ih / 2) * 0.95;
        lineStep(ctx, i === 0, x, y);
      }
      ctx.strokeStyle = rgba(PALETTE.amber, 0.95);
      ctx.lineWidth = 2;
      ctx.stroke();
      for (let q = 1; q < m; q++) {
        const rr = jz[idx(n, q)] / jz[k];
        const x = pad.l + iw * rr;
        ctx.fillStyle = rgba(PALETTE.cyan, 1);
        ctx.beginPath();
        ctx.arc(x, pad.t + ih / 2, 3.2, 0, 6.2832);
        ctx.fill();
      }
      ctx.fillStyle = PALETTE.dim;
      ctx.fillText(`J${n}(${fmt(jz[k], 3)}·r)  r = 0 → 1`, pad.l, h - 3);
    } else {
      const cols = [PALETTE.amber, PALETTE.cyan, PALETTE.violet, PALETTE.rose];
      const X = 14;
      for (let n = 0; n < 4; n++) {
        ctx.beginPath();
        for (let i = 0; i <= 200; i++) {
          const x = (X * i) / 200;
          const v = special.besselJ(n, x);
          const sx = pad.l + (iw * x) / X;
          const sy = pad.t + ih / 2 - v * (ih / 2) * 0.95;
          lineStep(ctx, i === 0, sx, sy);
        }
        ctx.strokeStyle = rgba(cols[n], 0.9);
        ctx.lineWidth = 1.6;
        ctx.stroke();
        for (let m = 1; m <= 4; m++) {
          const z = jz[idx(n, m)];
          if (z > X) break;
          ctx.fillStyle = rgba(cols[n], 1);
          ctx.beginPath();
          ctx.arc(pad.l + (iw * z) / X, pad.t + ih / 2, 2.6, 0, 6.2832);
          ctx.fill();
        }
      }
      ctx.fillStyle = PALETTE.dim;
      ctx.fillText('J0 J1 J2 J3 and their zeros, x = 0 → 14', pad.l, h - 3);
    }
  }

  // ---- interaction --------------------------------------------------------------------------
  let geo = { cx: 0, cy: 0, R: 1 };
  const place = () => {
    geo = { cx: st.w / 2, cy: st.h / 2, R: Math.min(st.w, st.h) * 0.43 };
  };
  canvas.style.touchAction = 'none';
  canvas.addEventListener('pointerdown', (e) => {
    const p = pointerPos(e, canvas);
    place();
    const dx = (p.x - geo.cx) / geo.R;
    const dy = (p.y - geo.cy) / geo.R;
    const r = Math.hypot(dx, dy);
    if (r > 1) return;
    interacted = true;
    if (mode !== 'strike') setMode('strike');
    ensureAudioIfOn();
    strike(r, Math.atan2(dy, dx));
    ripples.push({ x: p.x, y: p.y, t: 0 });
  });

  // ---- frame --------------------------------------------------------------------------------
  let clock = 0;
  let hudTick = 0;
  setMode('strike');
  onFrame(st, (dt) => {
    place();
    clock += dt;
    if (!interacted && !reducedMotion) {
      autoTimer -= dt;
      if (autoTimer <= 0) {
        const spots = [
          [0.42, 0.9],
          [0.0, 0],
          [0.7, -2.2],
          [0.28, 2.4],
          [0.55, 0.2]
        ];
        const s = spots[autoCount++ % spots.length];
        strike(s[0], s[1]);
        ripples.push({
          x: geo.cx + Math.cos(s[1]) * s[0] * geo.R,
          y: geo.cy + Math.sin(s[1]) * s[0] * geo.R,
          t: 0
        });
        autoTimer = 3.4;
      }
    }

    // advance modal oscillators analytically
    if (mode === 'strike') {
      for (let k = 0; k < NM; k++) {
        const w = omega[k];
        const g = gamma[k];
        const e = Math.exp(-g * dt);
        const c = Math.cos(w * dt);
        const s = Math.sin(w * dt);
        let q = qc[k];
        let v = vc[k];
        qc[k] = e * (q * c + ((v + g * q) / w) * s);
        vc[k] = e * (v * c - ((w * w * q + g * v) / w) * s);
        q = qs[k];
        v = vs[k];
        qs[k] = e * (q * c + ((v + g * q) / w) * s);
        vs[k] = e * (v * c - ((w * w * q + g * v) / w) * s);
      }
    } else {
      pureT += dt;
      qc.fill(0);
      qs.fill(0);
      qc[idx(pure.n, pure.m)] = Math.cos(omega[idx(pure.n, pure.m)] * pureT);
    }

    // radial functions F_n(r) = Σ_m q_nm R_nm(r)
    Fc.fill(0);
    Fs.fill(0);
    for (let n = 0; n <= NMAX; n++) {
      for (let m = 1; m <= MMAX; m++) {
        const k = idx(n, m);
        const a = qc[k];
        const b = qs[k];
        if (a === 0 && b === 0) continue;
        const base = k * NR;
        const fo = n * NR;
        for (let i = 0; i < NR; i++) {
          const rv = R[base + i];
          Fc[fo + i] += a * rv;
          Fs[fo + i] += b * rv;
        }
      }
    }

    // field → pixels
    let maxAbs = 0;
    for (let p = 0; p < count; p++) {
      const ri = pRad[p];
      const o = p * N1;
      let u = Fc[ri];
      for (let n = 1; n <= NMAX; n++)
        u += Fc[n * NR + ri] * cosT[o + n] + Fs[n * NR + ri] * sinT[o + n];
      field[p] = u;
      const au = u < 0 ? -u : u;
      if (au > maxAbs) maxAbs = au;
    }
    peak = Math.max(maxAbs, peak * Math.exp(-0.3 * dt));
    if (mode === 'mode') peak = Math.max(peak, 0.55);
    const ref = Math.max(1.1 * maxAbs, 0.25 * peak, 1e-12);
    for (let p = 0; p < count; p++) {
      let s = field[p] / ref;
      s = s / (1 + 0.35 * Math.abs(s));
      px32[pIndex[p]] = lut[Math.max(0, Math.min(511, ((s * 1.35 + 1) * 255.5) | 0))];
    }
    bctx.putImageData(img, 0, 0);

    // draw
    const { ctx, w, h } = st;
    ctx.clearRect(0, 0, w, h);
    const R0 = geo.R;
    // soft halo
    const halo = ctx.createRadialGradient(geo.cx, geo.cy, R0 * 0.8, geo.cx, geo.cy, R0 * 1.35);
    halo.addColorStop(0, 'rgba(255,190,100,0.09)');
    halo.addColorStop(1, 'rgba(255,190,100,0)');
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, w, h);
    ctx.save();
    ctx.beginPath();
    ctx.arc(geo.cx, geo.cy, R0, 0, 6.2832);
    ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(buf, geo.cx - R0, geo.cy - R0, 2 * R0, 2 * R0);
    ctx.restore();
    // hoop + lugs
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(233,237,246,.55)';
    ctx.beginPath();
    ctx.arc(geo.cx, geo.cy, R0 + 3, 0, 6.2832);
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(233,237,246,.14)';
    ctx.beginPath();
    ctx.arc(geo.cx, geo.cy, R0 + 11, 0, 6.2832);
    ctx.stroke();
    ctx.fillStyle = 'rgba(233,237,246,.5)';
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * 6.2832;
      ctx.beginPath();
      ctx.arc(geo.cx + Math.cos(a) * (R0 + 11), geo.cy + Math.sin(a) * (R0 + 11), 2.1, 0, 6.2832);
      ctx.fill();
    }
    // ripples
    ripples = ripples.filter((r) => (r.t += dt) < 1);
    for (const r of ripples) {
      ctx.strokeStyle = `rgba(255,255,255,${0.7 * (1 - r.t)})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(r.x, r.y, 4 + r.t * 70, 0, 6.2832);
      ctx.stroke();
    }

    if (++hudTick % 8 === 0) {
      hud.innerHTML = `<b>besselJ</b> · <b>besselJZero</b> · ${NM} modes · ${fmt((NM * NR) / 1000, 0)}k Bessel evaluations at start-up`;
    }
  });

  new ResizeObserver(drawMini).observe(mini);
}
