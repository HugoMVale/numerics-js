// Chamber III: Nelder–Mead on Himmelblau's landscape, plus a basin-of-attraction map of thousands of runs.
import { optimize } from 'numerics-js';
import {
  $,
  createStage,
  onFrame,
  pointerPos,
  rgba,
  PALETTE,
  fmt,
  int,
  reducedMotion,
  lineStep
} from './util.js';

const DOM = 5.5; // domain half-width
const MINIMA = [
  [3, 2],
  [-2.805118, 3.131312],
  [-3.77931, -3.283186],
  [3.584428, -1.848126]
];
const BASIN_COLORS = [PALETTE.cyan, PALETTE.amber, PALETTE.rose, PALETTE.mint];
const himmelblau = (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2;

export function mountSimplex() {
  const canvas = $('#nm-canvas');
  const hud = $('#hud-nm');
  const readout = $('#nm-readout');
  const basinBtn = $('#nm-basins');
  const clearBtn = $('#nm-clear');
  const st = createStage(canvas);

  // ---- heat map ----------------------------------------------------------------------------
  const HM = 440;
  const heat = document.createElement('canvas');
  heat.width = heat.height = HM;
  {
    const c = heat.getContext('2d');
    const im = c.createImageData(HM, HM);
    const stops = [
      [0, [7, 9, 26]],
      [0.22, [28, 26, 92]],
      [0.5, [98, 42, 142]],
      [0.76, [205, 78, 140]],
      [1, [255, 176, 112]]
    ];
    const colorAt = (t) => {
      for (let i = 1; i < stops.length; i++) {
        if (t <= stops[i][0]) {
          const a = stops[i - 1];
          const b = stops[i];
          const f = (t - a[0]) / (b[0] - a[0]);
          return [0, 1, 2].map((k) => a[1][k] + (b[1][k] - a[1][k]) * f);
        }
      }
      return stops[stops.length - 1][1];
    };
    const lmax = Math.log10(1 + 1500);
    for (let j = 0; j < HM; j++) {
      for (let i = 0; i < HM; i++) {
        const x = -DOM + ((i + 0.5) / HM) * 2 * DOM;
        const y = DOM - ((j + 0.5) / HM) * 2 * DOM;
        const t = Math.min(1, Math.log10(1 + himmelblau(x, y)) / lmax);
        const col = colorAt(Math.pow(t, 1.7));
        const band = (t * 16) % 1;
        const line = band < 0.045 ? 1.5 : 1;
        const o = 4 * (j * HM + i);
        im.data[o] = Math.min(255, col[0] * line);
        im.data[o + 1] = Math.min(255, col[1] * line);
        im.data[o + 2] = Math.min(255, col[2] * line);
        im.data[o + 3] = 255;
      }
    }
    c.putImageData(im, 0, 0);
  }

  // ---- geometry ----------------------------------------------------------------------------
  let g = { ox: 0, oy: 0, side: 1 };
  const place = () => {
    const side = Math.min(st.w - 8, st.h - 84);
    g = { ox: (st.w - side) / 2, oy: 46, side };
  };
  const sx = (x) => g.ox + ((x + DOM) / (2 * DOM)) * g.side;
  const sy = (y) => g.oy + ((DOM - y) / (2 * DOM)) * g.side;
  const wx = (px) => ((px - g.ox) / g.side) * 2 * DOM - DOM;
  const wy = (py) => DOM - ((py - g.oy) / g.side) * 2 * DOM;

  // ---- runs --------------------------------------------------------------------------------
  const runs = [];
  let hover = null;
  let colorCursor = 0;

  function startRun(x0, y0) {
    const frames = [];
    const res = optimize.nelderMead((v) => himmelblau(v.data[0], v.data[1]), [x0, y0], {
      tolx: 1e-7,
      tolf: 1e-9,
      callback: (_n, X, fx) => {
        frames.push({ X: Array.from(X.data), fx: Array.from(fx.data) });
        return { stop: false, success: true };
      }
    });
    const color = BASIN_COLORS[colorCursor++ % 4];
    runs.push({ frames, i: 0, acc: 0, color, res, x0, y0, age: 0 });
    if (runs.length > 8) runs.shift();
    readout.innerHTML =
      `Started at (${fmt(x0, 2)}, ${fmt(y0, 2)}) → minimum at <b>(${fmt(res.x.data[0], 4)}, ${fmt(res.x.data[1], 4)})</b> ` +
      `after <b>${frames.length}</b> iterations and <b>${res.evaluations}</b> function evaluations. No gradient was used.`;
  }

  canvas.style.touchAction = 'none';
  canvas.addEventListener('pointerdown', (e) => {
    const p = pointerPos(e, canvas);
    const x = wx(p.x);
    const y = wy(p.y);
    if (Math.abs(x) > DOM || Math.abs(y) > DOM) return;
    startRun(x, y);
  });
  canvas.addEventListener('pointermove', (e) => (hover = pointerPos(e, canvas)));
  canvas.addEventListener('pointerleave', () => (hover = null));
  clearBtn.addEventListener('click', () => {
    runs.length = 0;
    readout.textContent = 'Click anywhere on the landscape to release a simplex.';
  });

  // ---- basin map ---------------------------------------------------------------------------
  const GN = 96;
  const basin = document.createElement('canvas');
  basin.width = basin.height = GN;
  const bctx = basin.getContext('2d');
  const bimg = bctx.createImageData(GN, GN);
  let basinOn = false;
  let basinPos = 0;
  let basinEvals = 0;
  const share = [0, 0, 0, 0, 0];
  const basinDone = () => basinPos >= GN * GN;

  basinBtn.addEventListener('click', () => {
    basinOn = !basinOn;
    basinBtn.setAttribute('aria-pressed', String(basinOn));
    if (basinOn && basinDone()) {
      // restart sweep
      basinPos = 0;
      basinEvals = 0;
      share.fill(0);
      bimg.data.fill(0);
    }
  });

  function stepBasins(batch) {
    for (let k = 0; k < batch && basinPos < GN * GN; k++, basinPos++) {
      const i = basinPos % GN;
      const j = (basinPos / GN) | 0;
      const x = -DOM + ((i + 0.5) / GN) * 2 * DOM;
      const y = DOM - ((j + 0.5) / GN) * 2 * DOM;
      const r = optimize.nelderMead((v) => himmelblau(v.data[0], v.data[1]), [x, y], {
        tolx: 1e-5,
        tolf: 1e-7
      });
      basinEvals += r.evaluations;
      let lab = 4;
      for (let m = 0; m < 4; m++) {
        if (Math.hypot(r.x.data[0] - MINIMA[m][0], r.x.data[1] - MINIMA[m][1]) < 0.1) lab = m;
      }
      share[lab]++;
      const col = lab < 4 ? BASIN_COLORS[lab] : [255, 255, 255];
      const o = 4 * basinPos;
      bimg.data[o] = col[0];
      bimg.data[o + 1] = col[1];
      bimg.data[o + 2] = col[2];
      bimg.data[o + 3] = 150;
    }
    bctx.putImageData(bimg, 0, 0);
  }

  // ---- frame -------------------------------------------------------------------------------
  const RATE = reducedMotion ? 6 : 11; // simplex iterations per second
  let hudTick = 0;
  onFrame(st, (dt, time) => {
    place();
    if (basinOn && !basinDone()) stepBasins(520);

    const { ctx, w, h } = st;
    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heat, g.ox, g.oy, g.side, g.side);
    if (basinOn || basinPos > 0) {
      ctx.imageSmoothingEnabled = false;
      ctx.globalAlpha = basinOn ? 1 : 0;
      ctx.drawImage(basin, g.ox, g.oy, g.side, g.side);
      ctx.globalAlpha = 1;
      ctx.imageSmoothingEnabled = true;
    }
    ctx.strokeStyle = 'rgba(255,255,255,.18)';
    ctx.lineWidth = 1;
    ctx.strokeRect(g.ox + 0.5, g.oy + 0.5, g.side - 1, g.side - 1);

    // minima
    ctx.lineWidth = 1.4;
    MINIMA.forEach((m, k) => {
      const x = sx(m[0]);
      const y = sy(m[1]);
      ctx.strokeStyle = rgba(BASIN_COLORS[k], 0.95);
      ctx.beginPath();
      ctx.moveTo(x, y - 7);
      ctx.lineTo(x + 7, y);
      ctx.lineTo(x, y + 7);
      ctx.lineTo(x - 7, y);
      ctx.closePath();
      ctx.stroke();
    });

    // hover marker
    if (hover && Math.abs(wx(hover.x)) < DOM && Math.abs(wy(hover.y)) < DOM) {
      ctx.strokeStyle = 'rgba(255,255,255,.55)';
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.arc(hover.x, hover.y, 9 + 2 * Math.sin(time * 4), 0, 6.2832);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // runs
    for (const r of runs) {
      r.acc += dt * RATE;
      const last = r.frames.length - 1;
      r.i = Math.min(last, Math.floor(r.acc));
      const done = r.i >= last;
      const col = r.color;

      // best-vertex trail
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let k = 0; k <= r.i; k++) {
        const F = r.frames[k];
        const b = F.fx.indexOf(Math.min(...F.fx));
        const x = sx(F.X[2 * b]);
        const y = sy(F.X[2 * b + 1]);
        lineStep(ctx, k === 0, x, y);
      }
      ctx.strokeStyle = rgba(col, 0.55);
      ctx.lineWidth = 1.4;
      ctx.stroke();

      // ghost triangles
      for (let back = 10; back >= 0; back--) {
        const k = r.i - back;
        if (k < 0) continue;
        const F = r.frames[k];
        const a = back === 0 ? 1 : 0.28 * (1 - back / 11);
        ctx.beginPath();
        ctx.moveTo(sx(F.X[0]), sy(F.X[1]));
        ctx.lineTo(sx(F.X[2]), sy(F.X[3]));
        ctx.lineTo(sx(F.X[4]), sy(F.X[5]));
        ctx.closePath();
        if (back === 0) {
          ctx.fillStyle = rgba(col, 0.22);
          ctx.fill();
        }
        ctx.strokeStyle = rgba(col, back === 0 ? 0.98 : a);
        ctx.lineWidth = back === 0 ? 1.8 : 1;
        ctx.stroke();
      }
      // vertices
      const F = r.frames[r.i];
      const best = F.fx.indexOf(Math.min(...F.fx));
      for (let v = 0; v < 3; v++) {
        ctx.fillStyle = v === best ? '#fff' : rgba(col, 1);
        ctx.beginPath();
        ctx.arc(sx(F.X[2 * v]), sy(F.X[2 * v + 1]), v === best ? 3.4 : 2.4, 0, 6.2832);
        ctx.fill();
      }
      // start flag
      ctx.strokeStyle = rgba(col, 0.8);
      ctx.beginPath();
      ctx.arc(sx(r.x0), sy(r.y0), 4.5, 0, 6.2832);
      ctx.stroke();
      if (done) {
        r.age += dt;
        const x = sx(r.res.x.data[0]);
        const y = sy(r.res.x.data[1]);
        const pulse = (r.age * 1.4) % 1;
        ctx.strokeStyle = rgba(col, 0.9 * (1 - pulse));
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 6 + 22 * pulse, 0, 6.2832);
        ctx.stroke();
      }
    }

    if (++hudTick % 6 === 0) {
      if (basinOn && !basinDone()) {
        hud.innerHTML = `<b>nelderMead</b> · basin map ${int(basinPos)} / ${int(GN * GN)} runs · ${int(basinEvals)} evaluations`;
      } else if (basinOn || basinPos > 0) {
        const tot = share.reduce((a, b) => a + b, 0) || 1;
        hud.innerHTML =
          `<b>nelderMead</b> · ${int(GN * GN)} runs · ${int(basinEvals)} evaluations · basin shares ` +
          share
            .slice(0, 4)
            .map(
              (s, k) =>
                `<span style="color:${rgba(BASIN_COLORS[k], 1)}">${Math.round((100 * s) / tot)}%</span>`
            )
            .join(' ');
      } else {
        const r = runs[runs.length - 1];
        hud.innerHTML = r
          ? `<b>nelderMead</b> · ${r.res.evaluations} evaluations · f* = ${fmt(r.res.fx, 2)}`
          : '<b>nelderMead</b> · derivative-free simplex search';
      }
    }
  });

  // a first run so the stage is alive on arrival
  setTimeout(() => startRun(-0.5, -0.3), 400);
}
