// Shared helpers: canvas stages, pointer handling, formatting, palette.

export const PALETTE = {
  bg: '#06080f',
  ink: '#e9edf6',
  dim: '#8a94ab',
  cyan: [94, 230, 208],
  rose: [255, 122, 170],
  amber: [255, 191, 94],
  violet: [169, 139, 255],
  blue: [110, 168, 255],
  mint: [140, 240, 160]
};

export const rgba = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

export const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Create a DPR-aware canvas stage that tracks its size and on-screen visibility. */
export function createStage(canvas, onResize) {
  const st = {
    canvas,
    ctx: canvas.getContext('2d'),
    w: 0,
    h: 0,
    dpr: 1,
    visible: false
  };
  const resize = () => {
    const r = canvas.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    st.w = r.width;
    st.h = r.height;
    st.dpr = dpr;
    st.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (onResize) onResize(st);
  };
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver((es) => (st.visible = es[es.length - 1].isIntersecting), {
    rootMargin: '80px'
  }).observe(canvas);
  resize();
  return st;
}

/** Pointer position in CSS pixels relative to the canvas. */
export function pointerPos(ev, canvas) {
  const r = canvas.getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
}

const frames = [];
/** Register a per-frame callback; it is invoked only while its stage is visible. */
export function onFrame(stage, fn) {
  frames.push({ stage, fn });
}
export function startLoop() {
  let last = performance.now();
  const tick = (now) => {
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    for (const f of frames) {
      if (!f.stage || f.stage.visible) f.fn(dt, now / 1000);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

export function fmt(x, digits = 4) {
  if (!Number.isFinite(x)) return '—';
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(Math.max(digits - 2, 1));
  return x.toFixed(digits).replace(/\.?0+$/, (m) => (m.includes('.') ? '' : m));
}
export const int = (n) => Math.round(n).toLocaleString('en-US').replace(/,/g, ' ');

export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smooth = (t) => t * t * (3 - 2 * t);

/** Tiny DOM helper. */
export function $(sel, root = document) {
  return root.querySelector(sel);
}

/** Bind a segmented control (buttons with data-value). */
export function segmented(root, onChange) {
  const btns = [...root.querySelectorAll('button')];
  const set = (val) => {
    btns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.value === val)));
  };
  btns.forEach((b) =>
    b.addEventListener('click', () => {
      set(b.dataset.value);
      onChange(b.dataset.value);
    })
  );
  return set;
}

/** Start a sub-path at the first point, extend it afterwards. */
export function lineStep(ctx, first, x, y) {
  if (first) ctx.moveTo(x, y);
  else ctx.lineTo(x, y);
}
