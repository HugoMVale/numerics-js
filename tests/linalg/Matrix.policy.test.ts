import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import { rand } from '../../src/math.js';

// Tests of the linear-algebra error and non-finite policy (see AGENTS.md):
// NaN propagates, infinite entries throw RangeError, singular matrices throw
// Error, and every message reads "Matrix.method: ... .".

const EPS = Number.EPSILON;

const randomMatrix = (m: number, n: number, seed: number): Matrix => {
  const rnd = rand(seed);
  const A = new Matrix(m, n);
  for (let i = 0; i < m * n; i++) A.data[i] = rnd() * 2 - 1;
  return A;
};

/** Well-conditioned square matrix: random plus a dominant diagonal. */
const wellConditioned = (n: number, seed: number): Matrix => {
  const A = randomMatrix(n, n, seed);
  for (let i = 0; i < n; i++) A.set(i, i, A.get(i, i) + n);
  return A;
};

const spd = (n: number, seed: number): Matrix => {
  const B = randomMatrix(n, n, seed);
  return B.transpose().matmul(B).add(Matrix.identity(n));
};

const allNaN = (x: Matrix | Vector): boolean => x.data.every(Number.isNaN);

const NaN2 = (): Matrix => new Matrix(2, 2, [NaN, 1, 1, 1]);
const Inf2 = (): Matrix => new Matrix(2, 2, [1, 2, Infinity, 1]);

/** Every method covered by the non-finite policy, applied to a 2x2 or 2x3 matrix `A`. */
const policyMethods: Record<string, (A: Matrix) => unknown> = {
  determinant: (A) => A.determinant(),
  inverse: (A) => A.inverse(),
  lu: (A) => A.lu(),
  solve: (A) => A.solve([1, 1]),
  rank: (A) => A.rank(),
  rankElimination: (A) => A.rank({ method: 'elimination' }),
  cholesky: (A) => A.cholesky(),
  qr: (A) => A.qr(),
  qrPivot: (A) => A.qrPivot(),
  svd: (A) => A.svd(),
  singularValues: (A) => A.singularValues(),
  pinv: (A) => A.pinv(),
  lstsq: (A) => A.lstsq([1, 1]),
  cond1: (A) => A.cond1(),
  cond2: (A) => A.cond2(),
  eigenvalues: (A) => A.eigenvalues(),
  eig: (A) => A.eig()
};

describe('non-finite input: NaN propagates', () => {
  it('determinant() is NaN, not 0', () => {
    expect(NaN2().determinant()).toBeNaN();
    expect(new Matrix(1, 1, [NaN]).determinant()).toBeNaN();
  });

  it('lu() and solve() do not report a NaN matrix as singular', () => {
    const dec = NaN2().lu();
    expect(allNaN(dec.L)).toBe(true);
    expect(allNaN(dec.U)).toBe(true);
    expect(dec.perm).toEqual([0, 1]);
    expect(dec.sign).toBe(1);
    expect(allNaN(NaN2().solve([1, 1]))).toBe(true);
  });

  it('lu(out) fills the supplied storage, also when factoring in place', () => {
    const out = Matrix.identity(2).lu();
    out.perm[0] = 1;
    out.perm[1] = 0;
    out.sign = -1;
    expect(NaN2().lu(out)).toBe(out);
    expect(out.perm).toEqual([0, 1]);
    expect(out.sign).toBe(1);
    expect(allNaN(out.U)).toBe(true);
    const inPlace = out.U;
    inPlace.data.set([NaN, 1, 1, 1]);
    expect(inPlace.lu(out).U).toBe(inPlace);
    expect(allNaN(inPlace)).toBe(true);
    // The storage is usable again afterwards.
    const ok = new Matrix(2, 2, [1, 2, 3, 4]).lu(out);
    expect(ok.U.get(0, 0)).toBe(3);
    expect(ok.perm).toEqual([1, 0]);
  });

  it('qrPivot(out) fills the supplied storage', () => {
    const out = new Matrix(2, 2, [1, 2, 3, 4]).qrPivot();
    const dec = NaN2().qrPivot(out);
    expect(dec).toBe(out);
    expect(allNaN(dec.QR)).toBe(true);
    expect(allNaN(dec.tau)).toBe(true);
    expect(dec.perm).toEqual([0, 1]);
  });

  it('every method returns NaN results of the documented shape', () => {
    const A = new Matrix(3, 2, [1, NaN, 3, 4, 5, 6]);
    const sq = NaN2();
    expect(allNaN(sq.inverse())).toBe(true);
    expect(sq.rank()).toBeNaN();
    expect(sq.rank({ method: 'elimination' })).toBeNaN();
    expect(sq.cond1()).toBeNaN();
    expect(sq.cond2()).toBeNaN();

    const L = sq.cholesky();
    expect([L.rows, L.cols]).toEqual([2, 2]);
    expect(allNaN(L)).toBe(true);

    const { Q, R } = A.qr();
    expect([Q.rows, Q.cols, R.rows, R.cols]).toEqual([3, 3, 3, 2]);
    expect(allNaN(Q) && allNaN(R)).toBe(true);

    const P = A.pinv();
    expect([P.rows, P.cols]).toEqual([2, 3]);
    expect(allNaN(P)).toBe(true);

    const { U, s, V } = A.svd();
    expect([U.rows, U.cols, s.size, V.rows, V.cols]).toEqual([3, 2, 2, 2, 2]);
    expect(allNaN(U) && allNaN(s) && allNaN(V)).toBe(true);
    expect(A.singularValues().size).toBe(2);

    const wide = new Matrix(2, 3, [1, 2, NaN, 4, 5, 6]);
    const w = wide.svd({ full: true });
    expect([w.U.rows, w.U.cols, w.V.rows, w.V.cols]).toEqual([2, 2, 3, 3]);
    expect(allNaN(w.U) && allNaN(w.V)).toBe(true);

    const ls = A.lstsq([1, 2, 3]);
    expect([ls.x.size, ls.s.size]).toEqual([2, 2]);
    expect(allNaN(ls.x) && allNaN(ls.s)).toBe(true);
    expect(ls.rank).toBeNaN();
    expect(ls.residualNorm).toBeNaN();

    const ev = sq.eigenvalues();
    expect(ev).toHaveLength(2);
    expect(ev.every((e) => Number.isNaN(e.re) && Number.isNaN(e.im))).toBe(true);
    const pairs = sq.eig();
    expect(pairs).toHaveLength(2);
    for (const p of pairs) {
      expect(p.value.re).toBeNaN();
      expect(allNaN(p.vectorRe) && allNaN(p.vectorIm)).toBe(true);
    }
  });

  it('a NaN in the 1x1 case propagates too', () => {
    const A = new Matrix(1, 1, [NaN]);
    expect(allNaN(A.inverse())).toBe(true);
    expect(allNaN(A.solve([1]))).toBe(true);
    expect(allNaN(A.cholesky())).toBe(true);
    expect(allNaN(A.pinv())).toBe(true);
    expect(A.eigenvalues()[0].re).toBeNaN();
  });

  it('a NaN right-hand side propagates through solve() and lstsq()', () => {
    const A = wellConditioned(3, 1);
    expect(allNaN(A.solve([NaN, 1, 1]))).toBe(true);
    expect(allNaN(A.lstsq([NaN, 1, 1]).x)).toBe(true);
  });

  it('does not modify the input', () => {
    for (const name of Object.keys(policyMethods)) {
      const A = new Matrix(2, 2, [NaN, 1, 2, 3]);
      const before = Array.from(A.data);
      policyMethods[name](A);
      expect(Array.from(A.data).map(String), name).toEqual(before.map(String));
    }
  });

  it('cholesky() reads only the lower triangle, so NaN or Infinity above it is ignored', () => {
    const L = new Matrix(2, 2, [4, NaN, 2, 3]).cholesky();
    expect(L.get(0, 0)).toBe(2);
    expect(L.get(1, 0)).toBe(1);
    expect(L.get(1, 1)).toBeCloseTo(Math.sqrt(2), 14);
    expect(() => new Matrix(2, 2, [4, Infinity, 2, 3]).cholesky()).not.toThrow();
    expect(() => new Matrix(2, 2, [4, 0, Infinity, 3]).cholesky()).toThrow(RangeError);
    expect(allNaN(new Matrix(2, 2, [4, 0, NaN, 3]).cholesky())).toBe(true);
  });
});

describe('non-finite input: infinite entries throw RangeError', () => {
  it('inverse() of [[Infinity, 1], [1, 1]] throws instead of returning a wrong matrix', () => {
    expect(() => new Matrix(2, 2, [Infinity, 1, 1, 1]).inverse()).toThrow(RangeError);
    expect(() => new Matrix(2, 2, [Infinity, 1, 1, 1]).inverse()).toThrow(
      'Matrix.inverse: matrix contains an infinite entry at (0, 0), got Infinity.'
    );
  });

  for (const name of Object.keys(policyMethods)) {
    it(`${name}() names itself and the position`, () => {
      const run = () => policyMethods[name](Inf2());
      expect(run).toThrow(RangeError);
      const method = name === 'rankElimination' ? 'rank' : name;
      expect(run).toThrow(
        `Matrix.${method}: matrix contains an infinite entry at (1, 0), got Infinity.`
      );
    });
  }

  it('also rejects -Infinity, and wins over NaN', () => {
    expect(() => new Matrix(2, 2, [1, -Infinity, 1, 1]).determinant()).toThrow(/got -Infinity/);
    expect(() => new Matrix(2, 2, [NaN, 1, 1, Infinity]).determinant()).toThrow(RangeError);
    expect(() => new Matrix(2, 2, [Infinity, 1, 1, NaN]).svd()).toThrow(RangeError);
  });

  it('throws for rectangular matrices too', () => {
    const A = new Matrix(2, 3, [1, 2, 3, 4, Infinity, 6]);
    expect(() => A.svd()).toThrow(/at \(1, 1\)/);
    expect(() => A.qr()).toThrow(RangeError);
    expect(() => A.qrPivot()).toThrow(RangeError);
    expect(() => A.pinv()).toThrow(RangeError);
    expect(() => A.rank()).toThrow(RangeError);
  });

  it('does not modify the input or the lu(out) storage contract on a rejected matrix', () => {
    const A = Inf2();
    const before = Array.from(A.data);
    expect(() => A.lu()).toThrow(RangeError);
    expect(Array.from(A.data)).toEqual(before);
  });

  it('the substitution kernels follow IEEE arithmetic instead of scanning', () => {
    const L = new Matrix(2, 2, [1, 0, 1, 1]);
    expect(L.solveLower([Infinity, 1]).data[0]).toBe(Infinity);
    const U = new Matrix(2, 2, [2, 1, 0, 1]);
    expect(allNaN(U.solveUpper([NaN, NaN]))).toBe(true);
  });
});

describe('singular and rank-deficient input', () => {
  const singular = () => new Matrix(2, 2, [1, 2, 2, 4]);

  it('lu(), solve() and inverse() throw a plain Error naming the method', () => {
    for (const [name, run] of [
      ['lu', () => singular().lu()],
      ['solve', () => singular().solve([1, 1])],
      ['inverse', () => singular().inverse()]
    ] as const) {
      expect(run, name).toThrow(`Matrix.${name}: matrix is singular.`);
      let error: unknown;
      try {
        run();
      } catch (e) {
        error = e;
      }
      expect(error).toBeInstanceOf(Error);
      expect(error, name).not.toBeInstanceOf(RangeError);
      expect(error, name).not.toBeInstanceOf(TypeError);
    }
  });

  it('a zero matrix and a 1x1 zero are singular; the 1x1 case is otherwise ordinary', () => {
    expect(() => new Matrix(3, 3).lu()).toThrow(/singular/);
    expect(() => new Matrix(1, 1).inverse()).toThrow(/singular/);
    expect(() => new Matrix(1, 1).solve([1])).toThrow(/singular/);
    expect(new Matrix(1, 1).determinant()).toBe(0);
    expect(new Matrix(1, 1, [4]).inverse().get(0, 0)).toBe(0.25);
    expect(new Matrix(1, 1, [4]).solve([2]).get(0)).toBe(0.5);
    expect(new Matrix(1, 1, [-4]).determinant()).toBe(-4);
    expect(new Matrix(1, 1, [4]).rank()).toBe(1);
    expect(new Matrix(1, 1).rank()).toBe(0);
  });

  it('a zero pivot appearing late is found, and only an exactly zero pivot counts as singular', () => {
    // Row 1 = row 2 / 2 and all multipliers are powers of two: elimination is exact,
    // so the zero pivot only shows up in the last column.
    const exact = new Matrix(3, 3, [2, 1, 1, 4, 2, 2, 1, 3, 5]);
    expect(() => exact.lu()).toThrow('Matrix.lu: matrix is singular.');
    expect(() => exact.inverse()).toThrow('Matrix.inverse: matrix is singular.');
    expect(exact.determinant()).toBe(0);
    expect(exact.rank()).toBe(2);
    // No tolerance: a pivot of 1e-300 is not "missing".
    const tiny = new Matrix(2, 2, [1, 0, 0, 1e-300]);
    expect(() => tiny.lu()).not.toThrow();
    expect(tiny.determinant()).toBe(1e-300);
  });

  it('determinant() is 0 for singular input and never throws for it', () => {
    expect(singular().determinant()).toBe(0);
    expect(new Matrix(3, 3).determinant()).toBe(0);
  });

  it('rank, svd, pinv, lstsq, cond1, cond2 and cholesky handle rank deficiency without LU errors', () => {
    const A = singular();
    expect(A.rank()).toBe(1);
    expect(A.rank({ method: 'elimination' })).toBe(1);
    const { U, s, V } = A.svd();
    const norm = A.norm();
    expect(s.get(1)).toBeLessThan(10 * EPS * norm);
    const recon = U.matmul(Matrix.diag(s)).matmul(V.transpose());
    expect(recon.sub(A).norm()).toBeLessThan(20 * EPS * norm);
    expect(A.cond1()).toBe(Infinity);
    expect(new Matrix(2, 2).cond2()).toBe(Infinity);
    expect(A.lstsq([1, 1]).rank).toBe(1);
    // Moore-Penrose: A * pinv(A) * A = A
    const P = A.pinv();
    expect(A.matmul(P).matmul(A).sub(A).norm()).toBeLessThan(50 * EPS * norm);
    expect(() => A.cholesky()).toThrow('Matrix.cholesky: matrix is not positive definite');
  });

  it('solveLower() and solveUpper() throw on an exactly zero diagonal entry', () => {
    expect(() => new Matrix(2, 2, [0, 0, 1, 1]).solveLower([1, 1])).toThrow(
      'Matrix.solveLower: zero diagonal entry, matrix is singular.'
    );
    expect(() => new Matrix(2, 2, [1, 1, 0, 0]).solveUpper([1, 1])).toThrow(
      'Matrix.solveUpper: zero diagonal entry, matrix is singular.'
    );
  });

  it('a nearly singular matrix is not rejected by lu(); rank() and cond1() detect it', () => {
    const A = new Matrix(2, 2, [1, 1, 1, 1 + EPS]); // smallest singular value ~ EPS / 2
    expect(() => A.lu()).not.toThrow();
    expect(A.rank()).toBe(1);
    expect(A.cond1()).toBeGreaterThan(1e15);
  });
});

describe('non-square input', () => {
  const wide = () => new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);

  it('methods that need a square matrix throw RangeError "expected a square matrix"', () => {
    const squareOnly: Record<string, (A: Matrix) => unknown> = {
      determinant: (A) => A.determinant(),
      inverse: (A) => A.inverse(),
      lu: (A) => A.lu(),
      solve: (A) => A.solve([1, 1]),
      cholesky: (A) => A.cholesky(),
      cond1: (A) => A.cond1(),
      cond1Upper: (A) => A.cond1Upper(),
      cond1Lower: (A) => A.cond1Lower(),
      solveLower: (A) => A.solveLower([1, 1]),
      solveUpper: (A) => A.solveUpper([1, 1]),
      eigenvalues: (A) => A.eigenvalues(),
      eig: (A) => A.eig()
    };
    for (const [name, run] of Object.entries(squareOnly)) {
      expect(() => run(wide()), name).toThrow(RangeError);
      expect(() => run(wide()), name).toThrow(`Matrix.${name}: expected a square matrix, got 2x3.`);
    }
  });

  it('rank, qr, qrPivot, svd, singularValues, pinv, lstsq and cond2 accept any shape', () => {
    const A = wide();
    const norm = A.norm();
    expect(A.rank()).toBe(2);
    expect(A.transpose().rank()).toBe(2);
    expect(A.rank({ method: 'elimination' })).toBe(2);

    const { Q, R } = A.qr();
    expect(Q.matmul(R).sub(A).norm()).toBeLessThan(20 * EPS * norm);

    const { U, s, V } = A.svd();
    expect(U.matmul(Matrix.diag(s)).matmul(V.transpose()).sub(A).norm()).toBeLessThan(
      20 * EPS * norm
    );
    expect(A.cond2()).toBeCloseTo(s.get(0) / s.get(1), 12);
    expect(A.pinv().rows).toBe(3);
    expect(A.lstsq([1, 2]).x.size).toBe(3);
  });

  it('shape mismatches of the right-hand side say "vector of size"', () => {
    const A = wellConditioned(3, 2);
    expect(() => A.solve([1, 2])).toThrow('Matrix.solve: shape mismatch: 3x3 vs vector of size 2.');
    expect(() => A.lstsq([1, 2])).toThrow('Matrix.lstsq: shape mismatch: 3x3 vs vector of size 2.');
    expect(() => A.solveLower([1])).toThrow(
      'Matrix.solveLower: shape mismatch: 3x3 vs vector of size 1.'
    );
    expect(() => A.cholesky().choleskySolve([1])).toThrow(
      'Matrix.choleskySolve: shape mismatch: 3x3 vs vector of size 1.'
    );
  });
});

describe('identities with tolerances scaled to the matrix norm', () => {
  const sizes = [1, 2, 3, 5, 8, 13];

  it('A * inverse(A) = I and inverse(A) * A = I', () => {
    for (const n of sizes) {
      const A = wellConditioned(n, 10 + n);
      const Ainv = A.inverse();
      const tol = 20 * n * EPS * A.norm1() * Ainv.norm1();
      expect(A.matmul(Ainv).sub(Matrix.identity(n)).norm1()).toBeLessThan(tol);
      expect(Ainv.matmul(A).sub(Matrix.identity(n)).norm1()).toBeLessThan(tol);
    }
  });

  it('solve() has a small residual, scaled by ||A|| ||x||', () => {
    for (const n of sizes) {
      const A = wellConditioned(n, 30 + n);
      const b = Vector.from(Array.from({ length: n }, (_, i) => Math.sin(i + 1)));
      const x = A.solve(b);
      const resid = A.mulVec(x).sub(b).normInf();
      expect(resid).toBeLessThan(20 * n * EPS * (A.normInf() * x.normInf() + b.normInf()));
    }
  });

  it('P A = L U for lu(), with perm and sign consistent with determinant()', () => {
    for (const n of sizes) {
      const A = randomMatrix(n, n, 50 + n);
      const { L, U, perm, sign } = A.lu();
      const PA = new Matrix(n, n);
      for (let i = 0; i < n; i++) PA.setRow(i, A.row(perm[i]));
      expect(L.matmul(U).sub(PA).norm1()).toBeLessThan(20 * n * EPS * A.norm1());
      let det = sign;
      for (let i = 0; i < n; i++) det *= U.get(i, i);
      expect(Math.abs(det - A.determinant())).toBeLessThanOrEqual(
        20 * n * EPS * Math.abs(det) + 1e-300
      );
    }
  });

  it('Q R = A and Q^T Q = I for qr()', () => {
    for (const [m, n] of [
      [1, 1],
      [4, 4],
      [6, 3],
      [3, 6]
    ]) {
      const A = randomMatrix(m, n, 70 + m * n);
      const { Q, R } = A.qr();
      expect(Q.matmul(R).sub(A).norm()).toBeLessThan(20 * Math.max(m, n) * EPS * A.norm());
      expect(Q.transpose().matmul(Q).sub(Matrix.identity(m)).norm()).toBeLessThan(20 * m * EPS);
    }
  });

  it('U S V^T = A and orthonormal factors for svd(), thin and full', () => {
    for (const [m, n] of [
      [1, 1],
      [2, 2],
      [5, 3],
      [3, 5],
      [7, 7]
    ]) {
      const A = randomMatrix(m, n, 90 + m * n);
      for (const full of [false, true]) {
        const { U, s, V } = A.svd({ full });
        const k = Math.min(m, n);
        const S = new Matrix(U.cols, V.cols);
        for (let i = 0; i < k; i++) S.set(i, i, s.get(i));
        expect(U.matmul(S).matmul(V.transpose()).sub(A).norm()).toBeLessThan(
          30 * Math.max(m, n) * EPS * A.norm()
        );
        expect(U.transpose().matmul(U).sub(Matrix.identity(U.cols)).norm()).toBeLessThan(
          30 * Math.max(m, n) * EPS
        );
        expect(V.transpose().matmul(V).sub(Matrix.identity(V.cols)).norm()).toBeLessThan(
          30 * Math.max(m, n) * EPS
        );
      }
    }
  });

  it('L L^T = A for cholesky()', () => {
    for (const n of sizes) {
      const A = spd(n, 110 + n);
      const L = A.cholesky();
      expect(L.matmul(L.transpose()).sub(A).norm1()).toBeLessThan(20 * n * EPS * A.norm1());
    }
  });

  it('lstsq() satisfies the normal equations and pinv(A) b equals lstsq(A, b).x', () => {
    for (const [m, n] of [
      [6, 3],
      [3, 3],
      [3, 6]
    ]) {
      const A = randomMatrix(m, n, 130 + m * n);
      const b = Vector.from(Array.from({ length: m }, (_, i) => Math.cos(i)));
      const { x } = A.lstsq(b);
      const xp = A.pinv().mulVec(b);
      expect(x.sub(xp).norm()).toBeLessThan(1e-10 * (1 + x.norm()));
      const g = A.transpose().mulVec(A.mulVec(x).sub(b));
      expect(g.norm()).toBeLessThan(1e-10 * (1 + A.norm() ** 2 * x.norm()));
    }
  });

  it('rank() agrees between lstsq().rank and svd for a rank-deficient product', () => {
    const A = randomMatrix(6, 2, 7).matmul(randomMatrix(2, 5, 8));
    expect(A.rank()).toBe(2);
    expect(A.lstsq([1, 2, 3, 4, 5, 6]).rank).toBe(2);
  });
});

describe('svd option validation', () => {
  it('rejects non-object options with a TypeError, also for NaN matrices', () => {
    const A = new Matrix(2, 2, [1, 2, 3, 4]);
    expect(() => A.svd('x' as never)).toThrow(TypeError);
    expect(() => A.svd(5 as never)).toThrow('Matrix.svd: options must be an object, got number.');
    expect(() => new Matrix(1, 1, [NaN]).svd('x' as never)).toThrow(TypeError);
    expect(() => A.svd()).not.toThrow();
    expect(() => A.svd({ full: true })).not.toThrow();
  });
});

describe('error types and messages', () => {
  it('every error thrown by linalg methods has the form "Matrix.method: ... ."', () => {
    const sq = wellConditioned(3, 4);
    const lower = new Matrix(2, 2, [1, 0, 1, 1]);
    const dec = new Matrix(2, 2, [1, 2, 3, 4]).lu();
    const qrp = new Matrix(2, 2, [1, 2, 3, 4]).qrPivot();
    const rect = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
    const calls: (() => unknown)[] = [
      () => rect.determinant(),
      () => rect.inverse(),
      () => new Matrix(2, 2, [1, 2, 2, 4]).inverse(),
      () => rect.lu(),
      () => new Matrix(2, 2, [1, 2, 2, 4]).lu(),
      () => sq.lu(Matrix.identity(2).lu()),
      () => sq.lu({} as never),
      () => rect.solve([1, 2]),
      () => sq.solve([1]),
      () => new Matrix(2, 2, [1, 2, 2, 4]).solve([1, 1]),
      () => rect.solveLower([1, 2]),
      () => lower.solveLower([1]),
      () => lower.solveLower([1, 1], false, new Vector(1)),
      () => new Matrix(2, 2, [0, 0, 1, 1]).solveLower([1, 1]),
      () => rect.solveUpper([1, 2]),
      () => lower.solveUpper([1]),
      () => lower.solveUpper([1, 1], new Vector(1)),
      () => new Matrix(2, 2, [1, 1, 0, 0]).solveUpper([1, 1]),
      () => Matrix.luSolve(dec, [1]),
      () => Matrix.luSolve(dec, [1, 1], new Vector(1)),
      () => rect.cond1(),
      () => rect.cond1Upper(),
      () => rect.cond1Lower(),
      () => rect.cholesky(),
      () => new Matrix(2, 2, [1, 2, 2, 1]).cholesky(),
      () => lower.choleskySolve([1]),
      () => rect.choleskySolve([1, 1]),
      () => sq.qrPivot(qrp),
      () => Matrix.qrPivotMulQt(qrp, [1]),
      () => Matrix.qrPivotMulQt(qrp, [1, 1], new Vector(1)),
      () => rect.eigenvalues(),
      () => rect.eig(),
      () => rect.rank({ atol: -1 }),
      () => rect.rank({ rtol: NaN }),
      () => rect.rank({ method: 'x' as 'svd' }),
      () => rect.rank({ method: 'elimination', atol: -1 }),
      () => rect.lstsq([1]),
      () => rect.lstsq([1, 2], { atol: -1 }),
      () => rect.pinv({ rtol: -1 }),
      () => new Matrix(2, 2, [1, Infinity, 1, 1]).svd(),
      () => new Matrix(2, 2, [1, Infinity, 1, 1]).cond2()
    ];
    calls.forEach((call, i) => {
      let message: string | undefined;
      try {
        call();
      } catch (e) {
        message = (e as Error).message;
      }
      expect(message, `call #${i} must throw`).toBeDefined();
      expect(message, `call #${i}`).toMatch(/^Matrix\.[A-Za-z0-9]+: .*\.$/);
    });
  });

  it('uses RangeError for values and shapes, Error for singularity and indefiniteness', () => {
    expect(() => new Matrix(2, 3).inverse()).toThrow(RangeError);
    expect(() => new Matrix(2, 2).rank({ atol: -1 })).toThrow(RangeError);
    expect(() => new Matrix(2, 2).inverse()).toThrow(Error);
    for (const run of [
      () => new Matrix(2, 2).inverse(),
      () => new Matrix(2, 2, [1, 2, 2, 1]).cholesky()
    ]) {
      try {
        run();
        expect.unreachable();
      } catch (e) {
        expect((e as Error).constructor).toBe(Error);
      }
    }
  });
});

describe('constructors reject sizes that cannot be allocated', () => {
  it('new Matrix(1e10, 1e10) throws a "Matrix.constructor:" RangeError', () => {
    expect(() => new Matrix(1e10, 1e10)).toThrow(RangeError);
    expect(() => new Matrix(1e10, 1e10)).toThrow(
      'Matrix.constructor: a 10000000000x10000000000 matrix is too large to allocate.'
    );
    expect(() => new Matrix(2 ** 53, 1)).toThrow(/^Matrix\.constructor: /);
    expect(() => new Matrix(1, 2 ** 40)).toThrow(/^Matrix\.constructor: /);
  });

  it('new Vector(n) and new Vector({ length }) throw a "Vector.constructor:" RangeError', () => {
    expect(() => new Vector(1e20)).toThrow(
      'Vector.constructor: a vector of size 100000000000000000000 is too large to allocate.'
    );
    expect(() => new Vector(2 ** 53)).toThrow(/^Vector\.constructor: /);
    expect(() => new Vector({ length: 2 ** 40 })).toThrow(/^Vector\.constructor: /);
    expect(() => new Vector(2 ** 40)).toThrow(RangeError);
  });

  it('does not swallow other errors from an array-like', () => {
    const bad = {
      length: 1,
      get 0(): number {
        throw new TypeError('boom');
      }
    };
    expect(() => new Vector(bad)).toThrow('boom');
  });

  it('does not relabel a RangeError raised while reading the elements', () => {
    const bad = {
      length: 1,
      get 0(): number {
        throw new RangeError('user range error');
      }
    };
    expect(() => new Vector(bad)).toThrow('user range error');
  });

  it('still accepts iterables without a length', () => {
    expect(new Vector(new Set([1, 2]) as never).data).toEqual(new Float64Array([1, 2]));
  });

  it('ordinary sizes still work', () => {
    expect(new Matrix(3, 4).data.length).toBe(12);
    expect(new Vector(0).size).toBe(0);
    expect(new Vector(5).size).toBe(5);
  });
});

describe('argument validation', () => {
  const A = () => new Matrix(2, 2, [2, 1, 1, 3]);

  it('eig() and eigenvalues() validate their options', () => {
    for (const name of ['eig', 'eigenvalues'] as const) {
      for (const bad of [NaN, -1, 0, 1.5, Infinity]) {
        expect(() => A()[name]({ maxIterations: bad }), `${name} ${bad}`).toThrow(
          new RegExp(`^Matrix\\.${name}: maxIterations must be a positive integer, got `)
        );
      }
      expect(() => A()[name]({ tol: -1 })).toThrow(`Matrix.${name}: tol must be >= 0, got -1.`);
      expect(() => A()[name]({ tol: NaN })).toThrow(RangeError);
      expect(() => A()[name]({ tol: '1' as unknown as number })).toThrow(TypeError);
      expect(() => A()[name](5 as never)).toThrow(
        `Matrix.${name}: options must be an object, got number.`
      );
      expect(() => new Matrix(1, 1, [3])[name]({ tol: -1 })).toThrow(RangeError);
      expect(() => new Matrix(2, 2, [NaN, 0, 0, 1])[name]({ tol: -1 })).toThrow(RangeError);
    }
    expect(() => A().eigenvalues({ tol: 0, maxIterations: 1 })).not.toThrow();
  });

  it('rank() rejects options of the wrong type', () => {
    expect(() => A().rank('x' as never)).toThrow(
      'Matrix.rank: options must be an object, got string.'
    );
    expect(() => A().rank({ atol: '1' as never })).toThrow(TypeError);
    expect(() => A().rank({ method: 'elimination', rtol: '1' as never })).toThrow(TypeError);
  });

  it('svd-family cutoffs reject non-numbers with a TypeError and negatives with a RangeError', () => {
    expect(() => A().pinv({ atol: 'a' as never })).toThrow(
      'Matrix.pinv: atol must be a number, got string.'
    );
    expect(() => A().lstsq([1, 1], { rtol: -1 })).toThrow(
      'Matrix.lstsq: rtol must be >= 0, got -1.'
    );
    expect(() => A().pinv(5 as never)).toThrow(TypeError);
  });

  it('factorization arguments of the wrong type throw a TypeError naming the method', () => {
    const msg = (m: string, what: string, got: string) => `Matrix.${m}: ${what} must be ${got}`;
    expect(() => Matrix.luSolve(null as never, [1, 1])).toThrow(
      msg('luSolve', 'decomposition', 'an LUDecomposition { L, U, perm, sign }, got null.')
    );
    expect(() => Matrix.luSolve({} as never, [1, 1])).toThrow(TypeError);
    expect(() => A().lu({} as never)).toThrow(
      msg('lu', 'out', 'an LUDecomposition { L, U, perm, sign }, got Object.')
    );
    expect(() => A().lu(null as never)).toThrow(TypeError);
    expect(() => A().qrPivot({} as never)).toThrow(TypeError);
    expect(() => Matrix.qrPivotMulQt(null as never, [1, 1])).toThrow(/^Matrix\.qrPivotMulQt: /);
    expect(() => Matrix.qrPivotQ({ QR: A() } as never)).toThrow(/^Matrix\.qrPivotQ: /);
    expect(() => Matrix.qrPivotR(undefined as never)).toThrow(/^Matrix\.qrPivotR: /);
    expect(() => Matrix.qrUpdate(null as never, [1, 1], [1, 1])).toThrow(
      msg('qrUpdate', 'qr', 'a QRDecomposition { Q, R }, got null.')
    );
    expect(() => Matrix.qrUpdateSelf({ Q: A() } as never, [1, 1], [1, 1])).toThrow(TypeError);
  });

  it('substitution errors name the public method that was called', () => {
    const zeroDiag = new Matrix(2, 2, [0, 0, 1, 1]);
    expect(() => zeroDiag.choleskySolve([1, 1])).toThrow(
      'Matrix.choleskySolve: zero diagonal entry, matrix is singular.'
    );
    const dec = new Matrix(2, 2, [1, 2, 3, 4]).lu();
    dec.U.set(1, 1, 0);
    expect(() => Matrix.luSolve(dec, [1, 1])).toThrow(
      'Matrix.luSolve: zero diagonal entry, matrix is singular.'
    );
  });
});

describe('extreme magnitudes', () => {
  it('inverse() of a matrix with a subnormal pivot overflows to Infinity instead of NaN', () => {
    const D = Matrix.diag([1e-320, 1e-320]);
    const inv = D.inverse();
    expect(Array.from(inv.data)).toEqual([Infinity, 0, 0, Infinity]);
    expect(D.cond1()).toBe(Infinity);
  });

  it('inverse() is unchanged for ordinary pivots (reciprocal scaling)', () => {
    expect(Array.from(new Matrix(2, 2, [4, 7, 2, 6]).inverse().data)).toEqual([
      0.6000000000000001, -0.7000000000000001, -0.2, 0.4
    ]);
    const big = Matrix.diag([1e300, 1e300]).inverse();
    expect(big.get(0, 0)).toBe(1e-300);
  });

  it('solve() and svd() keep working at the extremes of the double range', () => {
    expect(Array.from(Matrix.diag([1e-300, 1e-300]).solve([1e-300, 1e-300]).data)).toEqual([1, 1]);
    expect(Array.from(Matrix.diag([1e-320, 1e-320]).svd().s.data)).toEqual([1e-320, 1e-320]);
    const s = new Matrix(2, 2, [1e300, 1e300, 0, 1e300]).svd().s;
    expect(s.get(0)).toBeCloseTo(1.618033988749895e300, -285);
  });

  it('determinant() overflows to Infinity and underflows to 0 without scaling', () => {
    expect(Matrix.diag([1e200, 1e200]).determinant()).toBe(Infinity);
    expect(Matrix.diag([1e-200, 1e-200]).determinant()).toBe(0);
  });
});
