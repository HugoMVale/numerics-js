// Timing microbenchmark of the common Vector/Matrix operations and of the LU family.
//
// Not part of the test suite (timings are machine dependent): run it by hand, on a built
// library, and compare two builds of the library by running it once per build:
//
//   npm run build
//   node benchmarks/linalg/perf.mjs                  # current build, ./dist
//   node benchmarks/linalg/perf.mjs path/to/dist     # any other build, e.g. an older commit
//   node benchmarks/linalg/perf.mjs a/dist b/dist    # several builds, interleaved, with ratios
//   FILTER='lu|solve' ROUNDS=40 node benchmarks/linalg/perf.mjs ...   # only some cases, more rounds
//
// Only API that exists in old and new versions is used, so that old commits can be measured.
// Each case is timed as the best of `ROUNDS` rounds of `REPS` calls (the best round is the least
// disturbed by the garbage collector and the scheduler); with several builds the rounds are
// interleaved, so slow drifts of the machine hit all builds alike.

import { performance } from 'node:perf_hooks';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const dists = process.argv.length > 2 ? process.argv.slice(2) : ['dist'];
const ROUNDS = Number(process.env.ROUNDS ?? 15);
const TARGET_MS = Number(process.env.TARGET_MS ?? 20); // duration of one round

// Deterministic pseudo-random numbers in (-1, 1), the same for every build.
function makeRng() {
  let seed = 123456789;
  return () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296) * 2 - 1;
}

/** Builds the cases for one build of the library: `[name, fn]`, `fn` being the operation. */
async function cases(dist) {
  const { Matrix, Vector } = await import(pathToFileURL(path.resolve(dist, 'linalg.js')).href);
  const rnd = makeRng();
  const vec = (n) => Vector.from(Array.from({ length: n }, rnd));
  const mat = (n) => Matrix.from(Array.from({ length: n }, () => Array.from({ length: n }, rnd)));
  // Diagonally dominant, so that nothing is singular or ill conditioned.
  const dominant = (n) => mat(n).add(Matrix.identity(n).mult(n));
  const spd = (n) => {
    const A = mat(n);
    return A.matmul(A.transpose()).add(Matrix.identity(n).mult(n));
  };

  const list = [];
  const add = (name, fn) => list.push([name, fn]);

  for (const n of [3, 100, 10000]) {
    const a = vec(n);
    const b = vec(n);
    add(`Vector(${n}) add`, () => a.add(b));
    add(`Vector(${n}) addSelf`, () => a.copy().addSelf(b));
    add(`Vector(${n}) mult(scalar)`, () => a.mult(1.5));
    add(`Vector(${n}) dot`, () => a.dot(b));
    add(`Vector(${n}) norm`, () => a.norm());
    add(`Vector(${n}) sum`, () => a.sum());
  }
  for (const n of [3, 20, 100]) {
    const A = mat(n);
    const B = mat(n);
    const x = vec(n);
    add(`Matrix(${n}) add`, () => A.add(B));
    add(`Matrix(${n}) transpose`, () => A.transpose());
    add(`Matrix(${n}) mulVec`, () => A.mulVec(x));
    add(`Matrix(${n}) matmul`, () => A.matmul(B));
  }
  for (const n of [3, 10, 50, 200]) {
    const A = dominant(n);
    const b = vec(n);
    add(`Matrix(${n}) lu`, () => A.lu());
    add(`Matrix(${n}) solve`, () => A.solve(b));
    add(`Matrix(${n}) inverse`, () => A.inverse());
    add(`Matrix(${n}) determinant`, () => A.determinant());
    add(`Matrix(${n}) rank (elimination)`, () => A.rank({ method: 'elimination' }));
  }
  for (const n of [3, 10, 50]) {
    const A = dominant(n);
    add(`Matrix(${n}) cond1`, () => A.cond1());
  }
  for (const n of [10, 50]) {
    const A = dominant(n);
    const out = A.lu(); // storage to reuse: the allocation free path used by the ODE solvers
    add(`Matrix(${n}) lu(out) reuse`, () => A.lu(out));
    const S = spd(n);
    const b = vec(n);
    add(`Matrix(${n}) cholesky`, () => S.cholesky());
    const L = S.cholesky();
    // `choleskySolve` is the name this method had in older versions.
    const solveCholesky = L.solveCholesky ?? L.choleskySolve;
    add(`Matrix(${n}) solveCholesky`, () => solveCholesky.call(L, b));
  }
  for (const n of [10, 40]) {
    const A = mat(n);
    add(`Matrix(${n}) singularValues`, () => A.singularValues());
    add(`Matrix(${n}) rank (svd)`, () => A.rank());
  }
  const only = process.env.FILTER ? new RegExp(process.env.FILTER) : null;
  return only ? list.filter(([name]) => only.test(name)) : list;
}

/** Calls per round such that a round lasts about `TARGET_MS`. */
function calibrate(fn) {
  let reps = 1;
  for (;;) {
    const t0 = performance.now();
    for (let i = 0; i < reps; i++) fn();
    const dt = performance.now() - t0;
    if (dt >= TARGET_MS / 4 || reps >= 1 << 24)
      return Math.max(1, Math.round((reps * TARGET_MS) / Math.max(dt, 1e-3)));
    reps *= 4;
  }
}

const builds = [];
for (const dist of dists) builds.push({ dist, list: await cases(dist) });
const names = builds[0].list.map(([name]) => name);
for (const b of builds) {
  if (b.list.length !== names.length) throw new Error('the builds do not run the same cases');
}

const best = builds.map(() => new Array(names.length).fill(Infinity));
for (let c = 0; c < names.length; c++) {
  const reps = calibrate(builds[0].list[c][1]);
  for (const b of builds) for (let i = 0; i < Math.min(reps, 200); i++) b.list[c][1](); // warm up
  for (let round = 0; round < ROUNDS; round++) {
    for (let k = 0; k < builds.length; k++) {
      const fn = builds[k].list[c][1];
      const t0 = performance.now();
      for (let i = 0; i < reps; i++) fn();
      const us = ((performance.now() - t0) * 1000) / reps;
      if (us < best[k][c]) best[k][c] = us;
    }
  }
}

const fmt = (us) => (us >= 100 ? us.toFixed(0) : us >= 10 ? us.toFixed(1) : us.toFixed(2));
const w = Math.max(...names.map((s) => s.length));
let header = 'case'.padEnd(w);
const label = (dist) => {
  const abs = path.resolve(dist);
  return path.basename(abs) === 'dist' ? path.basename(path.dirname(abs)) : path.basename(abs);
};
for (const b of builds) header += `  ${label(b.dist).slice(0, 14).padStart(14)}`;
if (builds.length > 1)
  for (let k = 1; k < builds.length; k++) header += `  ${`ratio ${k}/0`.padStart(10)}`;
console.log(`${header}   (µs per call, best of ${ROUNDS} rounds)`);
for (let c = 0; c < names.length; c++) {
  let line = names[c].padEnd(w);
  for (let k = 0; k < builds.length; k++) line += `  ${fmt(best[k][c]).padStart(14)}`;
  for (let k = 1; k < builds.length; k++)
    line += `  ${(best[k][c] / best[0][c]).toFixed(2).padStart(10)}`;
  console.log(line);
}
if (builds.length > 1) {
  for (let k = 1; k < builds.length; k++) {
    const r = names.map((_, c) => best[k][c] / best[0][c]).sort((a, b) => a - b);
    const geo = Math.exp(r.reduce((s, x) => s + Math.log(x), 0) / r.length);
    console.log(
      `build ${k} vs build 0: geometric mean ${geo.toFixed(3)}, median ${r[r.length >> 1].toFixed(3)}, ` +
        `worst ${r[r.length - 1].toFixed(2)}, best ${r[0].toFixed(2)}`
    );
  }
}
