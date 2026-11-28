// Timing check of the banded mode of `trBdf2Adaptive` on method-of-lines problems: the Fisher-KPP
// reaction-diffusion equation u_t = d * Laplacian(u) + u * (1 - u) with zero-flux boundaries, on a
// 1D grid (tridiagonal Jacobian, kl = ku = 1) and on a 2D grid (five-point stencil, kl = ku = m).
//
// Hand-run, not part of the test suite (timings depend on the machine). Build first:
//
//   npm run build
//   node benchmarks/ode/trBdf2Adaptive.banded.perf.mjs [dist]
//   node benchmarks/ode/trBdf2Adaptive.banded.perf.mjs dist --skip-dense   # banded modes only
//
// For each case it prints the wall time, the number of steps and the counters of the dense mode
// with an analytic Jacobian, and of the banded mode with an analytic and with a finite-difference
// Jacobian. The dense 2D case (n = 2500) allocates 3 dense matrices of 50 MB and takes tens of
// seconds. Settings: rtol 1e-4, atol 1e-6, t in [0, 30], d = 1e-4.
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const args = process.argv.slice(2);
const skipDense = args.includes('--skip-dense');
const dist = resolve(args.find((a) => !a.startsWith('--')) ?? 'dist');
const { Vector } = await import(pathToFileURL(`${dist}/linalg.js`).href);
const { trBdf2Adaptive } = await import(pathToFileURL(`${dist}/ode.js`).href);

const DIFFUSION = 1e-4;
const T_END = 30;

/** Fisher-KPP on `m` cells (1D) or `m x m` cells (2D), cell-centred finite volumes. */
function fisherKpp(dim, m) {
  const n = dim === 1 ? m : m * m;
  const k = DIFFUSION * m * m;
  const neighbours = Array.from({ length: n }, (_, i) => {
    const list = [];
    if (dim === 1) {
      if (i > 0) list.push(i - 1);
      if (i < n - 1) list.push(i + 1);
    } else {
      const row = Math.floor(i / m);
      const col = i % m;
      if (col > 0) list.push(i - 1);
      if (col < m - 1) list.push(i + 1);
      if (row > 0) list.push(i - m);
      if (row < m - 1) list.push(i + m);
    }
    return list;
  });
  const f = (_t, y, out) => {
    const u = y.data;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (const j of neighbours[i]) s += u[j] - u[i];
      out.data[i] = k * s + u[i] * (1 - u[i]);
    }
    return out;
  };
  // The same entries for a dense and a banded buffer: both only call `set` inside the band.
  const jac = (_t, y, J) => {
    const u = y.data;
    for (let i = 0; i < n; i++) {
      J.set(i, i, -neighbours[i].length * k + (1 - 2 * u[i]));
      for (const j of neighbours[i]) J.set(i, j, k);
    }
    return J;
  };
  const y0 = new Vector(n);
  for (let i = 0; i < n; i++) {
    const rho =
      dim === 1 ? (i + 0.5) / m : Math.hypot(((i % m) + 0.5) / m, (Math.floor(i / m) + 0.5) / m);
    y0.data[i] = 1 / (1 + Math.exp((rho - (dim === 1 ? 0.1 : 0.15)) / 0.02));
  }
  return { n, band: dim === 1 ? 1 : m, f, jac, y0 };
}

function run(label, problem, options) {
  const t0 = performance.now();
  const r = trBdf2Adaptive(problem.f, 0, T_END, problem.y0, { rtol: 1e-4, atol: 1e-6, ...options });
  const s = ((performance.now() - t0) / 1000).toFixed(3).padStart(8);
  console.log(
    `  ${label.padEnd(26)} ${s} s  steps ${String(r.t.size - 1).padStart(4)}  ` +
      `f ${String(r.evaluations).padStart(6)}  jac ${r.evaluationsJacobian}  lu ${r.decompositions}` +
      (r.success ? '' : `  FAILED: ${r.message}`)
  );
  return r;
}

for (const [dim, m] of [
  [1, 800],
  [1, 1600],
  [2, 50]
]) {
  const p = fisherKpp(dim, m);
  console.log(`${dim}D Fisher-KPP, n = ${p.n}, kl = ku = ${p.band}`);
  const band = { kl: p.band, ku: p.band };
  if (!skipDense) run('dense, analytic Jacobian', p, { jac: p.jac });
  run('banded, analytic Jacobian', p, { ...band, jac: p.jac });
  run('banded, finite differences', p, band);
}
