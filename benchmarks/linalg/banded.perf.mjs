// Timing check of the banded LU: the cost of `BandedMatrix.lu()` should grow like
// O(n * kl * (kl + ku)) and that of `BandedMatrix.solveLU()` like O(n * (kl + ku)).
//
// Hand-run, not part of the test suite (timings depend on the machine). Build first:
//
//   npm run build
//   node benchmarks/linalg/banded.perf.mjs [dist]        # report
//   node benchmarks/linalg/banded.perf.mjs dist --check  # exit code 1 if a slope is off
//
// The script fits the slope of log(time) against log(size) for each sweep, and compares it with
// the exponent that the operation count predicts. It also prints the motivating dense-vs-banded
// comparison for a method-of-lines Jacobian (1D tridiagonal and 2D five-point grids).
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const args = process.argv.slice(2);
const check = args.includes('--check');
const dist = resolve(args.find((a) => !a.startsWith('--')) ?? 'dist');
const { BandedMatrix, Matrix, Vector } = await import(pathToFileURL(`${dist}/linalg.js`).href);

function banded(n, kl, ku) {
  let seed = 12345;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;
  const B = new BandedMatrix(n, kl, ku);
  for (let i = 0; i < n; i++) {
    for (let j = Math.max(0, i - kl); j <= Math.min(n - 1, i + ku); j++) {
      B.set(i, j, i === j ? 0.1 * rnd() : rnd()); // small diagonal: rows are interchanged
    }
  }
  return B;
}

/**
 * Best (minimum) wall time in seconds of `fn` over several samples: the minimum is the estimate
 * least disturbed by other load. Each sample repeats `fn` until at least ~20 ms have elapsed.
 */
function time(fn, samples = 15) {
  fn(); // warm up
  let reps = 1;
  for (;;) {
    const t0 = performance.now();
    for (let r = 0; r < reps; r++) fn();
    if (performance.now() - t0 >= 20) break;
    reps *= 2;
  }
  const ts = [];
  for (let s = 0; s < samples; s++) {
    const t0 = performance.now();
    for (let r = 0; r < reps; r++) fn();
    ts.push((performance.now() - t0) / reps / 1000);
  }
  return Math.min(...ts);
}

/** Least-squares slope of log(y) on log(x). */
function slope(xs, ys) {
  const lx = xs.map(Math.log);
  const ly = ys.map(Math.log);
  const mx = lx.reduce((a, b) => a + b) / lx.length;
  const my = ly.reduce((a, b) => a + b) / ly.length;
  let num = 0;
  let den = 0;
  for (let i = 0; i < lx.length; i++) {
    num += (lx[i] - mx) * (ly[i] - my);
    den += (lx[i] - mx) ** 2;
  }
  return num / den;
}

const fmt = (t) => (t < 1e-3 ? `${(t * 1e6).toFixed(1)} us` : `${(t * 1e3).toFixed(2)} ms`);
let failed = false;

function sweep(title, sizes, make, expected, tolerance = 0.3) {
  const ts = sizes.map((s) => time(make(s)));
  const m = slope(sizes, ts);
  const ok = Math.abs(m - expected) <= tolerance;
  failed ||= !ok;
  console.log(`\n${title}`);
  sizes.forEach((s, i) => console.log(`  ${String(s).padStart(8)}  ${fmt(ts[i])}`));
  console.log(`  slope ${m.toFixed(2)} (expected ${expected}) ${ok ? 'ok' : 'OUT OF RANGE'}`);
}

// lu(): linear in n for fixed bandwidths.
sweep(
  'lu(), kl = ku = 4, n varies  [O(n)]',
  [2000, 4000, 8000, 16000, 32000],
  (n) => {
    const B = banded(n, 4, 4);
    const dec = B.lu();
    return () => B.lu(dec);
  },
  1
);
// lu(): quadratic in the bandwidth when kl = ku = k. The lower-order O(n * k) terms (workspace
// copy, pivot search) are still visible below k ~ 64, so the sweep starts there.
sweep(
  'lu(), n = 2000, kl = ku = k varies  [O(k^2)]',
  [64, 128, 256, 512],
  (k) => {
    const B = banded(2000, k, k);
    const dec = B.lu();
    return () => B.lu(dec);
  },
  2,
  0.35
);
// lu(): linear in kl for fixed, much larger ku, where kl * (kl + ku) ~ kl * ku. (With a small
// kl the O(n * (kl + ku)) copy into the factor workspace would hide the trend.)
sweep(
  'lu(), n = 2000, ku = 1000, kl varies  [O(kl) while kl << ku]',
  [16, 32, 64, 128],
  (kl) => {
    const B = banded(2000, kl, 1000);
    const dec = B.lu();
    return () => B.lu(dec);
  },
  1,
  0.3
);
// solveLU(): linear in n, and linear in the bandwidth.
sweep(
  'solveLU(), kl = ku = 4, n varies  [O(n)]',
  [4000, 8000, 16000, 32000, 64000],
  (n) => {
    const dec = banded(n, 4, 4).lu();
    const b = Vector.from(Array.from({ length: n }, (_, i) => Math.sin(i)));
    const out = new Vector(n);
    return () => BandedMatrix.solveLU(dec, b, out);
  },
  1
);
sweep(
  'solveLU(), n = 4000, kl = ku = k varies  [O(k)]',
  [16, 32, 64, 128],
  (k) => {
    const dec = banded(4000, k, k).lu();
    const b = Vector.from(Array.from({ length: 4000 }, (_, i) => Math.sin(i)));
    const out = new Vector(4000);
    return () => BandedMatrix.solveLU(dec, b, out);
  },
  1
);

// Motivating comparison: Jacobian-like matrices of a method-of-lines discretisation.
function dense1D(n) {
  const A = new Matrix(n, n);
  for (let i = 0; i < n; i++) {
    A.set(i, i, 1 + 2.1);
    if (i > 0) A.set(i, i - 1, -1.05);
    if (i < n - 1) A.set(i, i + 1, -1.05);
  }
  return A;
}
function dense2D(m) {
  const n = m * m;
  const A = new Matrix(n, n);
  for (let r = 0; r < m; r++) {
    for (let c = 0; c < m; c++) {
      const i = r * m + c;
      A.set(i, i, 1 + 4.2);
      if (c > 0) A.set(i, i - 1, -1.05);
      if (c < m - 1) A.set(i, i + 1, -1.05);
      if (r > 0) A.set(i, i - m, -1.05);
      if (r < m - 1) A.set(i, i + m, -1.05);
    }
  }
  return A;
}
console.log('\nDense Matrix.solve vs BandedMatrix.solve (matrix of I - gamma*h*J)');
for (const [label, A, kl] of [
  ['1D  n = 800 (kl = ku = 1)', dense1D(800), 1],
  ['1D  n = 1600 (kl = ku = 1)', dense1D(1600), 1],
  ['2D  50x50 grid, n = 2500 (kl = ku = 50)', dense2D(50), 50]
]) {
  const n = A.rows;
  const b = Vector.from(Array.from({ length: n }, (_, i) => Math.cos(i)));
  const B = BandedMatrix.fromMatrix(A, kl, kl);
  const tBand = time(() => B.solve(b), 5);
  const tDense = time(() => A.solve(b), 3);
  console.log(
    `  ${label}: dense ${fmt(tDense)}, banded ${fmt(tBand)} (x${(tDense / tBand).toFixed(0)}); ` +
      `storage ${((n * n * 8) / 1e6).toFixed(1)} MB vs ${((B.data.length * 8) / 1e6).toFixed(2)} MB`
  );
}

if (check && failed) {
  console.error('\nTiming check failed: a slope is outside the expected range.');
  process.exit(1);
}
