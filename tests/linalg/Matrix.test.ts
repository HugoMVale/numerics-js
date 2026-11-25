import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import {
  Matrix,
  type Eigenvalue,
  type Eigenpair,
  type LUDecomposition
} from '../../src/linalg/Matrix.js';
import { rand } from '../../src/math.js';
import { vectorLikeForms } from '../vectorLikeForms.js';

/** Exact 1-norm condition number via an explicit inverse, used as ground truth. */
function exactCond1(M: Matrix): number {
  return M.norm1() * M.inverse().norm1();
}

/** Builds a random n x n triangular matrix with a well-scaled diagonal (bounded condition number). */
function randomTriangular(n: number, upper: boolean, seed: number): Matrix {
  const rnd = rand(seed);
  const M = new Matrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if ((upper && j >= i) || (!upper && j <= i)) {
        M.set(i, j, i === j ? 1 + rnd() * 2 : (rnd() * 2 - 1) * 2);
      } else {
        M.set(i, j, 0);
      }
    }
  }
  return M;
}

/**
 * Sorts eigenvalues by real part then imaginary part, for
 * order-independent comparisons — `eigenvalues()` returns them in
 * deflation order, not sorted.
 */
function sortEigs(eigs: Eigenvalue[]): Eigenvalue[] {
  return [...eigs].sort((a, b) => a.re - b.re || a.im - b.im);
}

/** Sum of a set of eigenvalues (complex addition); should equal the trace. */
function sumEigs(eigs: Eigenvalue[]): Eigenvalue {
  return eigs.reduce((acc, e) => ({ re: acc.re + e.re, im: acc.im + e.im }), { re: 0, im: 0 });
}

/** Product of a set of eigenvalues (complex multiplication); should equal the determinant. */
function productEigs(eigs: Eigenvalue[]): Eigenvalue {
  return eigs.reduce(
    (acc, e) => ({ re: acc.re * e.re - acc.im * e.im, im: acc.re * e.im + acc.im * e.re }),
    { re: 1, im: 0 }
  );
}

/** Sorts eigenpairs by their eigenvalue's real part then imaginary part. */
function sortPairs(pairs: Eigenpair[]): Eigenpair[] {
  return [...pairs].sort((a, b) => a.value.re - b.value.re || a.value.im - b.value.im);
}

/**
 * Max absolute error, across every entry and both real/imaginary parts, of
 * `A * v - lambda * v` for a single eigenpair — should be ~0 for a correct
 * one, regardless of whether the eigenvalue/eigenvector are real or complex.
 */
function eigResidual(a: Matrix, pair: Eigenpair): number {
  const n = a.rows;
  const vRe = pair.vectorRe.toArray();
  const vIm = pair.vectorIm.toArray();
  let maxErr = 0;
  for (let i = 0; i < n; i++) {
    let sumRe = 0,
      sumIm = 0;
    for (let j = 0; j < n; j++) {
      sumRe += a.get(i, j) * vRe[j];
      sumIm += a.get(i, j) * vIm[j];
    }
    const lamRe = pair.value.re * vRe[i] - pair.value.im * vIm[i];
    const lamIm = pair.value.re * vIm[i] + pair.value.im * vRe[i];
    maxErr = Math.max(maxErr, Math.abs(sumRe - lamRe), Math.abs(sumIm - lamIm));
  }
  return maxErr;
}

/** Euclidean norm of a (possibly complex) eigenvector. */
function eigNorm(pair: Eigenpair): number {
  const vRe = pair.vectorRe.toArray();
  const vIm = pair.vectorIm.toArray();
  let sum = 0;
  for (let i = 0; i < vRe.length; i++) sum += vRe[i] * vRe[i] + vIm[i] * vIm[i];
  return Math.sqrt(sum);
}

describe('Matrix', () => {
  it('enforces 0-based indexing for get and set', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    expect(m.get(0, 0)).toBe(1);
    expect(m.get(0, 1)).toBe(2);
    expect(m.get(1, 0)).toBe(3);
    expect(m.get(1, 1)).toBe(4);

    expect(() => m.get(-1, 0)).toThrowError(RangeError);
    expect(() => m.get(2, 0)).toThrowError(RangeError);
  });

  it('extracts rows and columns accurately', () => {
    const m: Matrix = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
    expect(m.row(1).toArray()).toEqual([4, 5, 6]);
    expect(m.col(1).toArray()).toEqual([2, 5]);
  });

  it('performs matrix multiplication (matmul)', () => {
    const m1: Matrix = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
    const m2: Matrix = new Matrix(3, 2, [7, 8, 9, 10, 11, 12]);
    const result: Matrix = m1.matmul(m2);

    expect(result.rows).toBe(2);
    expect(result.cols).toBe(2);
    expect(result.toArray()).toEqual([
      [58, 64],
      [139, 154]
    ]);
  });

  it('computes determinants and inverses', () => {
    const m: Matrix = new Matrix(2, 2, [4, 7, 2, 6]);
    expect(m.determinant()).toBeCloseTo(10);

    const inv: Matrix = m.inverse();
    const identity: Matrix = m.matmul(inv);
    expect(identity.allClose(Matrix.identity(2))).toBe(true);
  });

  it('solves linear systems efficiently', () => {
    // System: 2x + y = 5, -x + y = 2  => x = 1, y = 3
    const m: Matrix = new Matrix(2, 2, [2, 1, -1, 1]);
    const b: Vector = new Vector([5, 2]);
    const x: Vector = m.solve(b);

    expect(x.allClose(new Vector([1, 3]))).toBe(true);
  });

  describe('lu()', () => {
    it('factors a matrix such that P*A = L*U', () => {
      const a: Matrix = Matrix.from([
        [4, 3, 0],
        [3, 4, -1],
        [0, -1, 4]
      ]);
      const { L, U, perm } = a.lu();

      const pa = new Matrix(3, 3);
      for (let i = 0; i < 3; i++) pa.setRow(i, a.row(perm[i]));

      expect(pa.allClose(L.matmul(U))).toBe(true);
    });

    it('returns a unit-lower-triangular L and an upper-triangular U', () => {
      const a: Matrix = Matrix.from([
        [2, -1, 3],
        [4, 2, 1],
        [-2, 5, 6]
      ]);
      const { L, U } = a.lu();

      for (let i = 0; i < 3; i++) {
        expect(L.get(i, i)).toBe(1);
        for (let j = i + 1; j < 3; j++) {
          expect(L.get(i, j)).toBeCloseTo(0);
          expect(U.get(j, i)).toBeCloseTo(0);
        }
      }
    });

    it('needs no pivoting on an already-triangular matrix: identity perm, sign +1', () => {
      const a: Matrix = Matrix.from([
        [5, 0, 0],
        [1, 3, 0],
        [2, 1, 4]
      ]);
      const { perm, sign } = a.lu();
      expect(perm).toEqual([0, 1, 2]);
      expect(sign).toBe(1);
    });

    it('tracks the row-swap parity in sign, consistent with determinant()', () => {
      const a: Matrix = Matrix.from([
        [0, 1],
        [1, 0]
      ]); // needs exactly one pivot swap
      const { U, sign } = a.lu();
      const detFromLU = sign * U.get(0, 0) * U.get(1, 1);
      expect(detFromLU).toBeCloseTo(a.determinant());
    });

    it('throws on a singular matrix', () => {
      const singular: Matrix = new Matrix(2, 2, [1, 2, 2, 4]); // row1 = 2*row0
      expect(() => singular.lu()).toThrowError();
    });

    it('rejects non-square input', () => {
      expect(() => new Matrix(2, 3).lu()).toThrowError(RangeError);
    });

    it('handles the 1x1 case', () => {
      const a: Matrix = Matrix.from([[7]]);
      const { L, U, perm, sign } = a.lu();
      expect(L.get(0, 0)).toBe(1);
      expect(U.get(0, 0)).toBe(7);
      expect(perm).toEqual([0]);
      expect(sign).toBe(1);
    });
  });

  describe('lu(out), solveLU() and the out argument of the triangular solves', () => {
    const a1: Matrix = Matrix.from([
      [0, 2, 1],
      [4, 1, -1],
      [2, 5, 3]
    ]); // needs pivoting
    const a2: Matrix = Matrix.from([
      [3, 0, 0],
      [0, 2, 0],
      [0, 0, 5]
    ]); // diagonal: no multipliers at all

    const expectSameFactorization = (x: LUDecomposition, y: LUDecomposition) => {
      expect(x.L.data).toEqual(y.L.data);
      expect(x.U.data).toEqual(y.U.data);
      expect(x.perm).toEqual(y.perm);
      expect(x.sign).toBe(y.sign);
    };

    it('lu(out) fills and returns the given storage, giving the same factorization as lu()', () => {
      const out = a2.lu();
      const result = a1.lu(out);
      expect(result).toBe(out);
      expectSameFactorization(result, a1.lu());
    });

    it('lu(out) leaves no stale entries from the previous factorization', () => {
      const out = a1.lu(); // L has non-zero multipliers, perm is not the identity
      const result = a2.lu(out);
      expect(result).toBe(out);
      expectSameFactorization(result, a2.lu());
      expect(result.perm).toEqual([0, 1, 2]);
      expect(result.sign).toBe(1);
    });

    it('lu(out) does not modify the input matrix', () => {
      const a = a1.copy();
      a.lu(a2.lu());
      expect(a.data).toEqual(a1.data);
    });

    it('factors in place when the matrix is out.U', () => {
      const out = a2.lu();
      out.U.data.set(a1.data);
      const result = out.U.lu(out);
      expect(result).toBe(out);
      expectSameFactorization(result, a1.lu());
    });

    it('returns an exactly upper-triangular U and a unit-lower-triangular L', () => {
      const { L, U } = a1.lu();
      for (let i = 0; i < 3; i++) {
        expect(L.get(i, i)).toBe(1);
        for (let j = 0; j < i; j++) expect(U.get(i, j)).toBe(0);
        for (let j = i + 1; j < 3; j++) expect(L.get(i, j)).toBe(0);
      }
    });

    it('lu(out) rejects an out that does not match the matrix, or aliases L', () => {
      expect(() => a1.lu(Matrix.identity(2).lu())).toThrowError(RangeError);
      const bad = a1.lu();
      bad.perm.pop();
      expect(() => a1.lu(bad)).toThrowError(RangeError);
      const out = a2.lu();
      expect(() => out.L.lu(out)).toThrowError(RangeError);
    });

    it('lu(out) still throws on a singular matrix', () => {
      const singular: Matrix = new Matrix(2, 2, [1, 2, 2, 4]);
      expect(() => singular.lu(Matrix.identity(2).lu())).toThrowError(/singular/);
    });

    it('solveLU solves A x = b and agrees with solve()', () => {
      const b = new Vector([1, 2, 3]);
      const x = Matrix.solveLU(a1.lu(), b);
      expect(a1.mulVec(x).allClose(b)).toBe(true);
      expect(x.allClose(a1.solve(b))).toBe(true);
    });

    it('solveLU can be reused for several right-hand sides, with or without out', () => {
      const decomposition = a1.lu();
      const out = new Vector(3);
      for (const b of [new Vector([1, 0, 0]), new Vector([0, 1, 0]), new Vector([5, -2, 7])]) {
        const x = Matrix.solveLU(decomposition, b, out);
        expect(x).toBe(out);
        expect(a1.mulVec(x).allClose(b)).toBe(true);
      }
    });

    it('solveLU leaves b untouched and rejects out === b', () => {
      const decomposition = a1.lu();
      const b = new Vector([1, 2, 3]);
      Matrix.solveLU(decomposition, b, new Vector(3));
      expect(b.data).toEqual(new Float64Array([1, 2, 3]));
      expect(() => Matrix.solveLU(decomposition, b, b)).toThrowError(RangeError);
    });

    it('solveLU rejects mismatched sizes', () => {
      const decomposition = a1.lu();
      expect(() => Matrix.solveLU(decomposition, new Vector(2))).toThrowError(RangeError);
      expect(() => Matrix.solveLU(decomposition, new Vector(3), new Vector(2))).toThrowError(
        RangeError
      );
    });

    it('solveLU throws on a zero diagonal in U', () => {
      const decomposition = a1.lu();
      decomposition.U.set(1, 1, 0);
      expect(() => Matrix.solveLU(decomposition, new Vector([1, 2, 3]))).toThrowError(
        /zero diagonal/
      );
    });

    it('solveLower and solveUpper write into out and may be applied in place (out === b)', () => {
      const l: Matrix = Matrix.from([
        [2, 0, 0],
        [1, 3, 0],
        [-1, 4, 5]
      ]);
      const u: Matrix = Matrix.from([
        [2, 1, -1],
        [0, 3, 2],
        [0, 0, 4]
      ]);
      const b = new Vector([4, 5, 6]);

      const out = new Vector(3);
      expect(l.solveLower(b, false, out)).toBe(out);
      expect(out.allClose(l.solveLower(b))).toBe(true);
      expect(u.solveUpper(b, out)).toBe(out);
      expect(out.allClose(u.solveUpper(b))).toBe(true);

      const inPlaceLower = b.copy();
      expect(l.solveLower(inPlaceLower, false, inPlaceLower)).toBe(inPlaceLower);
      expect(inPlaceLower.allClose(l.solveLower(b))).toBe(true);
      const inPlaceUnit = b.copy();
      l.solveLower(inPlaceUnit, true, inPlaceUnit);
      expect(inPlaceUnit.allClose(l.solveLower(b, true))).toBe(true);
      const inPlaceUpper = b.copy();
      expect(u.solveUpper(inPlaceUpper, inPlaceUpper)).toBe(inPlaceUpper);
      expect(inPlaceUpper.allClose(u.solveUpper(b))).toBe(true);
    });

    it('solveLower and solveUpper reject an out of the wrong size', () => {
      const m = Matrix.identity(3);
      expect(() => m.solveLower(new Vector(3), false, new Vector(2))).toThrowError(RangeError);
      expect(() => m.solveUpper(new Vector(3), new Vector(4))).toThrowError(RangeError);
    });
  });

  describe('solveLower() and solveUpper()', () => {
    it('solveLower solves a unit-lower-triangular system when unitDiagonal is true', () => {
      const l: Matrix = Matrix.from([
        [1, 0, 0],
        [2, 1, 0],
        [-1, 3, 1]
      ]);
      const b: Vector = new Vector([1, 4, 2]);
      const x = l.solveLower(b, true);
      expect(l.mulVec(x).allClose(b)).toBe(true);
    });

    it('solveLower divides by the diagonal when unitDiagonal is false (the default)', () => {
      const l: Matrix = Matrix.from([
        [2, 0],
        [3, 4]
      ]);
      const b: Vector = new Vector([4, 11]);
      const x = l.solveLower(b);
      expect(x.allClose(new Vector([2, 1.25]))).toBe(true);
    });

    it('solveUpper solves an upper-triangular system via back-substitution', () => {
      const u: Matrix = Matrix.from([
        [2, 1, -1],
        [0, 3, 2],
        [0, 0, 4]
      ]);
      const b: Vector = new Vector([3, 11, 8]);
      const x = u.solveUpper(b);
      expect(u.mulVec(x).allClose(b)).toBe(true);
    });

    it('ignores entries on the wrong side of the diagonal rather than validating shape', () => {
      // solveLower only reads j <= i; entries above the diagonal are
      // simply never consulted, even if the matrix isn't actually
      // lower triangular.
      const notLower: Matrix = Matrix.from([
        [2, 99],
        [1, 3]
      ]);
      const b: Vector = new Vector([4, 5]);
      const x = notLower.solveLower(b);
      expect(x.get(0)).toBeCloseTo(2); // 4 / 2, the "99" above the diagonal is ignored
    });

    it('throws on a zero diagonal entry', () => {
      const u: Matrix = Matrix.from([
        [2, 1],
        [0, 0]
      ]);
      expect(() => u.solveUpper(new Vector([1, 1]))).toThrowError();

      const l: Matrix = Matrix.from([
        [0, 0],
        [1, 2]
      ]);
      expect(() => l.solveLower(new Vector([1, 1]))).toThrowError();
    });

    it('rejects non-square input and shape-mismatched vectors', () => {
      expect(() => new Matrix(2, 3).solveLower(new Vector(2))).toThrowError(RangeError);
      expect(() => new Matrix(2, 3).solveUpper(new Vector(2))).toThrowError(RangeError);

      const square: Matrix = new Matrix(2, 2, [1, 0, 1, 1]);
      expect(() => square.solveLower(new Vector(3))).toThrowError(RangeError);
      expect(() => square.solveUpper(new Vector(3))).toThrowError(RangeError);
    });
  });

  it('solve() composes lu(), solveLower(), and solveUpper() consistently', () => {
    // Factoring once and reusing L/U/perm directly (the pattern a Newton
    // solver would use across several right-hand sides) must agree with
    // calling solve() fresh each time.
    const a: Matrix = Matrix.from([
      [4, 3, 0],
      [3, 4, -1],
      [0, -1, 4]
    ]);
    const { L, U, perm } = a.lu();

    for (const raw of [
      [1, 2, 3],
      [0, 1, 0],
      [-1, -2, -3]
    ]) {
      const b = new Vector(raw);
      const pb = new Vector(perm.map((p) => b.get(p)));
      const reused = U.solveUpper(L.solveLower(pb, true));
      expect(reused.allClose(a.solve(b))).toBe(true);
    }
  });

  describe('cholesky()', () => {
    it('factors a symmetric positive-definite matrix such that A = L * L^T', () => {
      // Classic textbook example.
      const a: Matrix = Matrix.from([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98]
      ]);
      const l: Matrix = a.cholesky();
      expect(
        l.allClose(
          Matrix.from([
            [2, 0, 0],
            [6, 1, 0],
            [-8, 5, 3]
          ])
        )
      ).toBe(true);
      expect(l.matmul(l.transpose()).allClose(a)).toBe(true);
    });

    it('returns a lower-triangular factor with a positive diagonal', () => {
      const a: Matrix = Matrix.from([
        [6, 3, 4],
        [3, 6, 5],
        [4, 5, 10]
      ]);
      const l: Matrix = a.cholesky();
      expect(l.isLowerTriangular()).toBe(true);
      for (let i = 0; i < 3; i++) expect(l.get(i, i)).toBeGreaterThan(0);
    });

    it('ignores the upper triangle rather than validating symmetry', () => {
      // cholesky only reads j <= i; entries above the diagonal are
      // simply never consulted, even if the matrix isn't actually
      // symmetric, same convention as solveLower/solveUpper.
      const asymmetric: Matrix = Matrix.from([
        [4, 999],
        [2, 5]
      ]);
      const symmetric: Matrix = Matrix.from([
        [4, 2],
        [2, 5]
      ]);
      expect(asymmetric.cholesky().allClose(symmetric.cholesky())).toBe(true);
    });

    it('throws when a leading minor is not positive definite', () => {
      const notPD: Matrix = Matrix.from([
        [1, 2],
        [2, 1]
      ]); // eigenvalues 3, -1
      expect(() => notPD.cholesky()).toThrowError();
    });

    it('rejects non-square input', () => {
      expect(() => new Matrix(2, 3).cholesky()).toThrowError(RangeError);
    });

    it('handles the 1x1 case', () => {
      const a: Matrix = Matrix.from([[9]]);
      const l: Matrix = a.cholesky();
      expect(l.get(0, 0)).toBeCloseTo(3);
    });
  });

  describe('solveCholesky()', () => {
    it('solves A * x = b given the Cholesky factor of A', () => {
      const a: Matrix = Matrix.from([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98]
      ]);
      const l: Matrix = a.cholesky();
      const b: Vector = new Vector([1, 2, 3]);
      const x = l.solveCholesky(b);
      expect(a.mulVec(x).allClose(b)).toBe(true);
    });

    it('agrees with solve() across several right-hand sides, factoring only once', () => {
      const a: Matrix = Matrix.from([
        [6, 3, 4],
        [3, 6, 5],
        [4, 5, 10]
      ]);
      const l: Matrix = a.cholesky();

      for (const raw of [
        [1, 2, 3],
        [0, 1, 0],
        [-1, -2, -3]
      ]) {
        const b = new Vector(raw);
        expect(l.solveCholesky(b).allClose(a.solve(b))).toBe(true);
      }
    });

    it('rejects non-square input and shape-mismatched vectors', () => {
      expect(() => new Matrix(2, 3).solveCholesky(new Vector(2))).toThrowError(RangeError);

      const square: Matrix = new Matrix(2, 2, [1, 0, 1, 1]);
      expect(() => square.solveCholesky(new Vector(3))).toThrowError(RangeError);
    });
  });

  describe('qr()', () => {
    it('factors a square matrix such that Q*R = A, with Q orthogonal and R upper triangular', () => {
      const a: Matrix = Matrix.from([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
      ]);
      const { Q, R } = a.qr();

      expect(Q.matmul(R).allClose(a, 1e-9)).toBe(true);
      expect(Q.transpose().matmul(Q).allClose(Matrix.identity(3), 1e-9)).toBe(true);
      for (let i = 1; i < 3; i++) {
        for (let j = 0; j < i; j++) expect(R.get(i, j)).toBe(0);
      }
    });

    it('factors a tall (rows > cols) rectangular matrix', () => {
      const a: Matrix = Matrix.from([
        [1, 2],
        [3, 4],
        [5, 6]
      ]);
      const { Q, R } = a.qr();

      expect(Q.rows).toBe(3);
      expect(Q.cols).toBe(3);
      expect(R.rows).toBe(3);
      expect(R.cols).toBe(2);
      expect(Q.matmul(R).allClose(a, 1e-9)).toBe(true);
      expect(Q.transpose().matmul(Q).allClose(Matrix.identity(3), 1e-9)).toBe(true);
      expect(R.get(1, 0)).toBe(0);
      expect(R.get(2, 0)).toBe(0);
      expect(R.get(2, 1)).toBe(0);
    });

    it('handles a matrix with a column already zero below the diagonal', () => {
      // Exercises the Householder step's "column already zero" early-out.
      const a: Matrix = Matrix.from([
        [0, 1],
        [0, 1]
      ]);
      const { Q, R } = a.qr();
      expect(Q.matmul(R).allClose(a, 1e-9)).toBe(true);
    });

    it('reduces to the identity/original matrix on an already-orthogonal input', () => {
      const a: Matrix = Matrix.from([
        [0, 1],
        [1, 0]
      ]);
      const { Q, R } = a.qr();
      expect(Q.matmul(R).allClose(a, 1e-9)).toBe(true);
      expect(Q.transpose().matmul(Q).allClose(Matrix.identity(2), 1e-9)).toBe(true);
    });
  });

  describe('qrPivot() and its helpers', () => {
    /** Random m x n matrix with entries in [-1, 1). */
    const randomMatrix = (m: number, n: number, seed: number): Matrix => {
      const rnd = rand(seed);
      return new Matrix(m, n).map(() => 2 * rnd() - 1);
    };

    /** Columns of `a` taken in the order given by `perm`. */
    const permuteCols = (a: Matrix, perm: number[]): Matrix => {
      const res = new Matrix(a.rows, a.cols);
      perm.forEach((src, dst) => res.setCol(dst, a.col(src)));
      return res;
    };

    it.each([
      [5, 3],
      [3, 3],
      [3, 5],
      [1, 4],
      [4, 1],
      [60, 7]
    ])(
      'factors a %i x %i matrix: A[:, perm] = Q*R, Q orthonormal, |diag(R)| nonincreasing',
      (m, n) => {
        const a = randomMatrix(m, n, 1000 + 10 * m + n);
        const k = Math.min(m, n);
        const dec = a.qrPivot();
        const Q = Matrix.qrPivotQ(dec);
        const R = Matrix.qrPivotR(dec);

        expect(Q.rows).toBe(m);
        expect(Q.cols).toBe(k);
        expect(R.rows).toBe(k);
        expect(R.cols).toBe(n);
        expect(dec.QR.rows).toBe(m);
        expect(dec.QR.cols).toBe(n);
        expect(dec.tau.size).toBe(k);
        expect([...dec.perm].sort((x, y) => x - y)).toEqual(Array.from({ length: n }, (_, i) => i));

        expect(Q.matmul(R).allClose(permuteCols(a, dec.perm), 1e-12, 1e-12)).toBe(true);
        expect(Q.transpose().matmul(Q).allClose(Matrix.identity(k), 1e-12, 1e-12)).toBe(true);
        for (let i = 0; i < k; i++) {
          for (let j = 0; j < Math.min(i, n); j++) expect(R.get(i, j)).toBe(0);
          if (i > 0)
            expect(Math.abs(R.get(i, i))).toBeLessThanOrEqual(
              Math.abs(R.get(i - 1, i - 1)) * (1 + 1e-12)
            );
        }
      }
    );

    it('moves the column of largest norm to the front, and keeps ties in original order', () => {
      const a = Matrix.from([
        [1, 0, 10],
        [1, 1, 0],
        [1, 1, 0]
      ]);
      expect(a.qrPivot().perm[0]).toBe(2);

      const tie = Matrix.from([
        [1, 1],
        [1, 1],
        [1, 1]
      ]);
      expect(tie.qrPivot().perm).toEqual([0, 1]);
    });

    it('reveals the rank of a matrix with dependent columns', () => {
      // Column 1 = 2 * column 0, so rank is 2.
      const a = Matrix.from([
        [1, 2, 0],
        [2, 4, 1],
        [3, 6, 0],
        [4, 8, 1]
      ]);
      const dec = a.qrPivot();
      const R = Matrix.qrPivotR(dec);

      expect(Math.abs(R.get(2, 2))).toBeLessThan(1e-12 * Math.abs(R.get(0, 0)));
      expect(Math.abs(R.get(1, 1))).toBeGreaterThan(1e-3 * Math.abs(R.get(0, 0)));
      expect(dec.perm[2]).not.toBe(2); // one of the dependent columns is pushed last
      expect(Matrix.qrPivotQ(dec).matmul(R).allClose(permuteCols(a, dec.perm), 1e-12, 1e-12)).toBe(
        true
      );
    });

    it('handles zero columns and the zero matrix without producing NaN', () => {
      const a = Matrix.from([
        [0, 3, 0],
        [0, 4, 0]
      ]);
      const dec = a.qrPivot();
      expect(dec.QR.allFinite()).toBe(true);
      expect(dec.tau.allFinite()).toBe(true);
      expect(
        Matrix.qrPivotQ(dec)
          .matmul(Matrix.qrPivotR(dec))
          .allClose(permuteCols(a, dec.perm), 1e-12, 1e-12)
      ).toBe(true);

      const z = new Matrix(3, 2).qrPivot();
      expect(z.QR.allFinite()).toBe(true);
      expect(Matrix.qrPivotR(z).allClose(new Matrix(2, 2))).toBe(true);
    });

    it('stays accurate on an ill-conditioned (Hilbert) tall matrix', () => {
      const a = new Matrix(8, 5).map((_, i, j) => 1 / (i + j + 1));
      const dec = a.qrPivot();
      const err =
        Matrix.qrPivotQ(dec).matmul(Matrix.qrPivotR(dec)).sub(permuteCols(a, dec.perm)).norm() /
        a.norm();
      expect(err).toBeLessThan(1e-14);
    });

    it('does not modify the input matrix', () => {
      const a = randomMatrix(6, 3, 7);
      const before = a.copy();
      a.qrPivot();
      expect(a.allClose(before, 0, 0)).toBe(true);
    });

    it('reuses the storage passed as `out`, and can factor in place', () => {
      const a = randomMatrix(6, 3, 11);
      const reference = a.qrPivot();

      const out = randomMatrix(6, 3, 12).qrPivot();
      const res = a.qrPivot(out);
      expect(res).toBe(out);
      expect(res.QR.allClose(reference.QR, 0, 0)).toBe(true);
      expect(res.tau.allClose(reference.tau, 0, 0)).toBe(true);
      expect(res.perm).toEqual(reference.perm);

      const inPlace = a.copy();
      const decInPlace = inPlace.qrPivot({
        QR: inPlace,
        tau: new Vector(3),
        perm: [0, 0, 0]
      });
      expect(decInPlace.QR).toBe(inPlace);
      expect(decInPlace.QR.allClose(reference.QR, 0, 0)).toBe(true);
      expect(decInPlace.perm).toEqual(reference.perm);
    });

    it('throws a RangeError if `out` does not match the shape of the matrix', () => {
      const a = randomMatrix(5, 3, 3);
      expect(() => a.qrPivot(randomMatrix(4, 3, 3).qrPivot())).toThrowError(RangeError);
      expect(() =>
        a.qrPivot({ QR: new Matrix(5, 3), tau: new Vector(2), perm: [0, 1, 2] })
      ).toThrowError(RangeError);
      expect(() =>
        a.qrPivot({ QR: new Matrix(5, 3), tau: new Vector(3), perm: [0, 1] })
      ).toThrowError(RangeError);
    });

    describe('qrPivotMulQt()', () => {
      it('matches Q^T*b computed from the explicit thin Q, and preserves the norm of b', () => {
        const a = randomMatrix(9, 4, 21);
        const b = Vector.from(Array.from({ length: 9 }, (_, i) => Math.sin(i + 1)));
        const dec = a.qrPivot();
        const qtb = Matrix.qrPivotMulQt(dec, b);

        const thin = Matrix.qrPivotQ(dec).transpose().mulVec(b);
        expect(qtb.size).toBe(9);
        expect(qtb.slice(0, 4).allClose(thin, 1e-12, 1e-12)).toBe(true);
        expect(qtb.norm()).toBeCloseTo(b.norm(), 12);
      });

      it('gives the least-squares solution and residual norm of a tall problem', () => {
        const a = randomMatrix(12, 3, 31);
        const xTrue = Vector.from([1, -2, 0.5]);
        const noise = Vector.from(Array.from({ length: 12 }, (_, i) => 0.01 * Math.cos(3 * i)));
        const b = a.mulVec(xTrue).add(noise);

        const dec = a.qrPivot();
        const qtb = Matrix.qrPivotMulQt(dec, b);
        const z = Matrix.qrPivotR(dec).solveUpper(qtb.slice(0, 3));
        const x = new Vector(3);
        dec.perm.forEach((src, j) => x.set(src, z.get(j)));

        // Normal equations as the independent reference.
        const xNormal = a.transpose().matmul(a).solve(a.transpose().mulVec(b));
        expect(x.allClose(xNormal, 1e-10, 1e-10)).toBe(true);
        expect(qtb.slice(3).norm()).toBeCloseTo(a.mulVec(x).sub(b).norm(), 10);
      });

      it('supports `out`, including `out === b`, and leaves b untouched otherwise', () => {
        const a = randomMatrix(5, 2, 41);
        const b = Vector.from([1, 2, 3, 4, 5]);
        const dec = a.qrPivot();
        const expected = Matrix.qrPivotMulQt(dec, b);
        expect(b.toArray()).toEqual([1, 2, 3, 4, 5]);

        const out = new Vector(5);
        expect(Matrix.qrPivotMulQt(dec, b, out)).toBe(out);
        expect(out.allClose(expected, 0, 0)).toBe(true);

        const inPlace = b.copy();
        expect(Matrix.qrPivotMulQt(dec, inPlace, inPlace)).toBe(inPlace);
        expect(inPlace.allClose(expected, 0, 0)).toBe(true);
      });

      it('throws a RangeError for mismatched sizes', () => {
        const dec = randomMatrix(5, 2, 51).qrPivot();
        expect(() => Matrix.qrPivotMulQt(dec, new Vector(4))).toThrowError(RangeError);
        expect(() => Matrix.qrPivotMulQt(dec, new Vector(5), new Vector(4))).toThrowError(
          RangeError
        );
      });
    });
  });

  describe('qrUpdate() and qrUpdateSelf()', () => {
    it('updates a square QR decomposition to match a from-scratch refactor of A + u*v^T', () => {
      const a: Matrix = Matrix.from([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
      ]);
      const u = new Vector([1, 2, 3]);
      const v = new Vector([-1, 0.5, 2]);
      const expected = a.add(u.outer(v));

      const { Q, R } = Matrix.qrUpdate(a.qr(), u, v);

      expect(Q.matmul(R).allClose(expected, 1e-9)).toBe(true);
      expect(Q.transpose().matmul(Q).allClose(Matrix.identity(3), 1e-9)).toBe(true);
      expect(R.isUpperTriangular(1e-9)).toBe(true);
    });

    it('updates a tall (rows > cols) QR decomposition, keeping Q square and R trapezoidal', () => {
      const a: Matrix = Matrix.from([
        [1, 2],
        [3, 4],
        [5, 6]
      ]);
      const u = new Vector([1, -1, 2]);
      const v = new Vector([0.5, -2]);
      const expected = a.add(u.outer(v));

      const { Q, R } = Matrix.qrUpdate(a.qr(), u, v);

      expect(Q.rows).toBe(3);
      expect(Q.cols).toBe(3);
      expect(R.rows).toBe(3);
      expect(R.cols).toBe(2);
      expect(Q.matmul(R).allClose(expected, 1e-9)).toBe(true);
      expect(Q.transpose().matmul(Q).allClose(Matrix.identity(3), 1e-9)).toBe(true);
      expect(R.isUpperTriangular(1e-9)).toBe(true);
    });

    it('updates a wide (rows < cols) QR decomposition', () => {
      const a: Matrix = Matrix.from([
        [1, 2, 3],
        [4, 5, 6]
      ]);
      const u = new Vector([2, -1]);
      const v = new Vector([1, 0, -2]);
      const expected = a.add(u.outer(v));

      const { Q, R } = Matrix.qrUpdate(a.qr(), u, v);

      expect(Q.rows).toBe(2);
      expect(Q.cols).toBe(2);
      expect(R.rows).toBe(2);
      expect(R.cols).toBe(3);
      expect(Q.matmul(R).allClose(expected, 1e-9)).toBe(true);
      expect(Q.transpose().matmul(Q).allClose(Matrix.identity(2), 1e-9)).toBe(true);
      expect(R.isUpperTriangular(1e-9)).toBe(true);
    });

    it('qrUpdateSelf mutates Q and R in place and returns the same object, for chaining', () => {
      const a: Matrix = Matrix.from([
        [4, 1],
        [2, 3]
      ]);
      const qr = a.qr();
      const { Q: origQ, R: origR } = qr;
      const u = new Vector([1, 1]);
      const v = new Vector([2, -1]);

      const result = Matrix.qrUpdateSelf(qr, u, v);

      expect(result).toBe(qr);
      expect(result.Q).toBe(origQ);
      expect(result.R).toBe(origR);
      expect(result.Q.matmul(result.R).allClose(a.add(u.outer(v)), 1e-9)).toBe(true);
    });

    it('qrUpdate leaves the original factorization untouched (unlike qrUpdateSelf)', () => {
      const a: Matrix = Matrix.from([
        [4, 1],
        [2, 3]
      ]);
      const qr = a.qr();
      const qBefore = qr.Q.copy();
      const rBefore = qr.R.copy();
      const u = new Vector([1, 1]);
      const v = new Vector([2, -1]);

      const updated = Matrix.qrUpdate(qr, u, v);

      expect(qr.Q.allClose(qBefore)).toBe(true);
      expect(qr.R.allClose(rBefore)).toBe(true);
      expect(updated).not.toBe(qr);
      expect(updated.Q.matmul(updated.R).allClose(a.add(u.outer(v)), 1e-9)).toBe(true);
    });

    it('applies a rank-2 change as two sequential rank-1 updates', () => {
      // A composite update A + U*V^T for 2-column U, V, applied as
      // two rank-1 calls in a row — the strategy the docs recommend
      // for rank-k updates.
      const a: Matrix = Matrix.from([
        [2, 0, 1],
        [0, 3, 0],
        [1, 0, 4]
      ]);
      const u1 = new Vector([1, 0, -1]);
      const v1 = new Vector([1, 1, 1]);
      const u2 = new Vector([0, 2, 1]);
      const v2 = new Vector([-1, 0, 2]);
      const expected = a.add(u1.outer(v1)).add(u2.outer(v2));

      const qr = a.qr();
      Matrix.qrUpdateSelf(qr, u1, v1);
      Matrix.qrUpdateSelf(qr, u2, v2);

      expect(qr.Q.matmul(qr.R).allClose(expected, 1e-9)).toBe(true);
      expect(qr.Q.transpose().matmul(qr.Q).allClose(Matrix.identity(3), 1e-9)).toBe(true);
      expect(qr.R.isUpperTriangular(1e-9)).toBe(true);
    });

    it('leaves the factorization unchanged when updating with a zero vector', () => {
      const a: Matrix = Matrix.from([
        [4, 1],
        [2, 3]
      ]);
      const qr = a.qr();
      const qBefore = qr.Q.copy();
      const rBefore = qr.R.copy();

      Matrix.qrUpdateSelf(qr, new Vector([0, 0]), new Vector([5, -3]));

      expect(qr.Q.allClose(qBefore, 1e-9)).toBe(true);
      expect(qr.R.allClose(rBefore, 1e-9)).toBe(true);
    });

    it('throws if Q is not square', () => {
      const notSquareQ = new Matrix(2, 3);
      const r = new Matrix(2, 2);
      expect(() =>
        Matrix.qrUpdateSelf({ Q: notSquareQ, R: r }, new Vector(2), new Vector(2))
      ).toThrowError(RangeError);
    });

    it('throws on a Q/R row-count mismatch', () => {
      const q = Matrix.identity(3);
      const r = new Matrix(2, 2);
      expect(() => Matrix.qrUpdateSelf({ Q: q, R: r }, new Vector(3), new Vector(2))).toThrowError(
        RangeError
      );
    });

    it('throws when u or v has the wrong size', () => {
      const a: Matrix = Matrix.from([
        [4, 1],
        [2, 3]
      ]);
      const qr = a.qr();
      expect(() => Matrix.qrUpdateSelf(qr, new Vector(3), new Vector([1, 1]))).toThrowError(
        RangeError
      );
      expect(() => Matrix.qrUpdateSelf(qr, new Vector([1, 1]), new Vector(3))).toThrowError(
        RangeError
      );
    });
  });

  describe('eigenvalues()', () => {
    it('returns the diagonal entries of a diagonal matrix', () => {
      const m: Matrix = Matrix.from([
        [3, 0, 0],
        [0, 5, 0],
        [0, 0, -2]
      ]);
      const eigs = sortEigs(m.eigenvalues());
      expect(eigs.map((e) => e.re)).toEqual([-2, 3, 5]);
      expect(eigs.every((e) => e.im === 0)).toBe(true);
    });

    it('returns the diagonal entries of a triangular matrix', () => {
      const m: Matrix = Matrix.from([
        [1, 4, 5],
        [0, 2, 6],
        [0, 0, 3]
      ]);
      const eigs = sortEigs(m.eigenvalues()).map((e) => e.re);
      expect(eigs[0]).toBeCloseTo(1);
      expect(eigs[1]).toBeCloseTo(2);
      expect(eigs[2]).toBeCloseTo(3);
    });

    it('computes the real eigenvalues of a symmetric matrix', () => {
      const m: Matrix = Matrix.from([
        [2, 1],
        [1, 2]
      ]);
      const eigs = sortEigs(m.eigenvalues());
      expect(eigs[0].re).toBeCloseTo(1);
      expect(eigs[1].re).toBeCloseTo(3);
      expect(eigs.every((e) => e.im === 0)).toBe(true);
    });

    it('computes the real eigenvalues of a non-symmetric matrix (known closed form)', () => {
      const m: Matrix = Matrix.from([
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
      ]);
      const eigs = sortEigs(m.eigenvalues()).map((e) => e.re);
      expect(eigs[0]).toBeCloseTo(2 - Math.SQRT2);
      expect(eigs[1]).toBeCloseTo(2);
      expect(eigs[2]).toBeCloseTo(2 + Math.SQRT2);
    });

    it('extracts a complex-conjugate eigenvalue pair from a real 2x2 matrix', () => {
      const rotation: Matrix = Matrix.from([
        [0, -1],
        [1, 0]
      ]);
      const eigs = rotation.eigenvalues();
      const ims = eigs.map((e) => e.im).sort((a, b) => a - b);
      expect(eigs[0].re).toBeCloseTo(0);
      expect(eigs[1].re).toBeCloseTo(0);
      expect(ims[0]).toBeCloseTo(-1);
      expect(ims[1]).toBeCloseTo(1);
    });

    it('extracts two independent complex-conjugate pairs from a 4x4 block-diagonal matrix', () => {
      const m: Matrix = Matrix.from([
        [0, -2, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, -5],
        [0, 0, 5, 0]
      ]);
      const ims = sortEigs(m.eigenvalues()).map((e) => e.im);
      expect(m.eigenvalues().every((e) => Math.abs(e.re) < 1e-9)).toBe(true);
      expect(ims).toEqual([
        expect.closeTo(-5, 5),
        expect.closeTo(-2, 5),
        expect.closeTo(2, 5),
        expect.closeTo(5, 5)
      ]);
    });

    it('agrees with trace() (sum) and determinant() (product) on a general non-symmetric matrix', () => {
      const m: Matrix = Matrix.from([
        [4, 1, 2, 0],
        [1, 3, 0, 1],
        [2, 0, 5, 2],
        [0, 1, 2, 6]
      ]);
      const eigs = m.eigenvalues();
      const sum = sumEigs(eigs);
      const product = productEigs(eigs);

      expect(sum.re).toBeCloseTo(m.trace());
      expect(sum.im).toBeCloseTo(0);
      expect(product.re).toBeCloseTo(m.determinant());
      expect(product.im).toBeCloseTo(0);
    });

    it('each real eigenvalue makes (A - lambda*I) singular', () => {
      const m: Matrix = Matrix.from([
        [2, 0, 0],
        [1, 3, -1],
        [1, 1, 1]
      ]);
      for (const e of m.eigenvalues()) {
        expect(e.im).toBeCloseTo(0);
        const shifted = m.sub(Matrix.identity(3).mult(e.re));
        expect(shifted.determinant()).toBeCloseTo(0);
      }
    });

    it('handles the 1x1 case directly', () => {
      expect(Matrix.from([[7]]).eigenvalues()).toEqual([{ re: 7, im: 0 }]);
    });

    it('handles the 2x2 case directly, without needing any QR iteration', () => {
      const eigs = sortEigs(
        Matrix.from([
          [4, 1],
          [2, 3]
        ]).eigenvalues()
      );
      expect(eigs[0].re).toBeCloseTo(2);
      expect(eigs[1].re).toBeCloseTo(5);
    });

    it('returns a repeated real eigenvalue for a defective (non-diagonalizable) matrix', () => {
      const jordan: Matrix = Matrix.from([
        [2, 1],
        [0, 2]
      ]);
      const eigs = jordan.eigenvalues();
      expect(eigs[0].re).toBeCloseTo(2);
      expect(eigs[1].re).toBeCloseTo(2);
      expect(eigs[0].im).toBeCloseTo(0);
      expect(eigs[1].im).toBeCloseTo(0);
    });

    it('returns a zero eigenvalue for a singular matrix', () => {
      const singular: Matrix = Matrix.from([
        [1, 2],
        [2, 4]
      ]);
      const eigs = sortEigs(singular.eigenvalues());
      expect(eigs[0].re).toBeCloseTo(0);
      expect(eigs[1].re).toBeCloseTo(5);
    });

    it('rejects non-square input', () => {
      expect(() =>
        Matrix.from([
          [1, 2, 3],
          [4, 5, 6]
        ]).eigenvalues()
      ).toThrowError(RangeError);
    });

    it('throws once the iteration budget is exhausted', () => {
      const m: Matrix = Matrix.from([
        [4, 1, 2, 0],
        [1, 3, 0, 1],
        [2, 0, 5, 2],
        [0, 1, 2, 6]
      ]);
      expect(() => m.eigenvalues({ maxIterations: 0 })).toThrowError();
    });

    it('accepts a custom rtol without breaking correctness on a well-behaved matrix', () => {
      const m: Matrix = Matrix.from([
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
      ]);
      const eigs = sortEigs(m.eigenvalues({ rtol: 1e-10 })).map((e) => e.re);
      expect(eigs[0]).toBeCloseTo(2 - Math.SQRT2);
      expect(eigs[1]).toBeCloseTo(2);
      expect(eigs[2]).toBeCloseTo(2 + Math.SQRT2);
    });

    it('rtol is the relative deflation tolerance: a loose value measurably degrades the result', () => {
      const m = Matrix.from([
        [4, 1, 2, 0.5, 1, 0],
        [0.3, 3, 1, 2, 0, 1],
        [1, 0.2, 5, 1, 1, 0.5],
        [0, 1, 0.4, 2, 1, 1],
        [2, 0, 1, 0.1, 6, 1],
        [1, 1, 0, 1, 0.3, 1]
      ]);
      const re = (e: { re: number }[]) => e.map((x) => x.re).sort((a, b) => a - b);
      const maxDiff = (a: number[], b: number[]) =>
        Math.max(...a.map((x, i) => Math.abs(x - b[i])));
      const reference = re(m.eigenvalues());
      // Measured: about 6e-9 for rtol 1e-8 and about 1e-1 for rtol 0.1.
      expect(maxDiff(re(m.eigenvalues({ rtol: 1e-8 })), reference)).toBeLessThan(1e-6);
      expect(maxDiff(re(m.eigenvalues({ rtol: 0.1 })), reference)).toBeGreaterThan(1e-2);
      const pairs = m.eig({ rtol: 0.1 }).map((p) => p.value);
      expect(maxDiff(re(pairs), reference)).toBeGreaterThan(1e-2);
    });
  });

  describe('eig()', () => {
    it('returns eigenpairs satisfying A*v = lambda*v, each with unit norm', () => {
      const m: Matrix = Matrix.from([
        [4, 1, 2, 0],
        [1, 3, 0, 1],
        [2, 0, 5, 2],
        [0, 1, 2, 6]
      ]);
      const pairs = m.eig();
      expect(pairs).toHaveLength(4);
      for (const pair of pairs) {
        expect(eigResidual(m, pair)).toBeCloseTo(0, 9);
        expect(eigNorm(pair)).toBeCloseTo(1, 9);
      }
    });

    it('agrees with eigenvalues() on which eigenvalues are found', () => {
      const m: Matrix = Matrix.from([
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
      ]);
      const values = sortEigs(m.eigenvalues());
      const pairValues = sortEigs(m.eig().map((p) => p.value));
      for (let i = 0; i < values.length; i++) {
        expect(pairValues[i].re).toBeCloseTo(values[i].re);
        expect(pairValues[i].im).toBeCloseTo(values[i].im);
      }
    });

    it('returns a real eigenvector for a real eigenvalue, with an all-zero imaginary part', () => {
      const m: Matrix = Matrix.from([
        [2, 0, 0],
        [1, 3, -1],
        [1, 1, 1]
      ]);
      for (const pair of m.eig()) {
        expect(pair.value.im).toBe(0);
        expect(pair.vectorIm.toArray().every((x) => x === 0)).toBe(true);
        expect(eigResidual(m, pair)).toBeCloseTo(0, 9);
      }
    });

    it('computes real eigenvectors for a symmetric matrix, orthogonal to one another', () => {
      const m: Matrix = Matrix.from([
        [2, 1],
        [1, 2]
      ]);
      const pairs = sortPairs(m.eig());
      expect(pairs[0].value.re).toBeCloseTo(1);
      expect(pairs[1].value.re).toBeCloseTo(3);

      const v0 = pairs[0].vectorRe,
        v1 = pairs[1].vectorRe;
      const dot = v0.get(0) * v1.get(0) + v0.get(1) * v1.get(1);
      expect(dot).toBeCloseTo(0);
    });

    it('extracts a complex-conjugate eigenvector pair from a rotation matrix', () => {
      const rotation: Matrix = Matrix.from([
        [0, -1],
        [1, 0]
      ]);
      const pairs = [...rotation.eig()].sort((a, b) => a.value.im - b.value.im);
      const [pMinus, pPlus] = pairs;
      expect(pMinus.value.im).toBeCloseTo(-1);
      expect(pPlus.value.im).toBeCloseTo(1);

      // The two eigenvectors of a conjugate pair should themselves be
      // exact complex conjugates of one another.
      expect(pPlus.vectorRe.allClose(pMinus.vectorRe)).toBe(true);
      const negatedIm = new Vector(pMinus.vectorIm.toArray().map((x) => -x));
      expect(pPlus.vectorIm.allClose(negatedIm)).toBe(true);

      expect(eigResidual(rotation, pPlus)).toBeCloseTo(0, 9);
      expect(eigResidual(rotation, pMinus)).toBeCloseTo(0, 9);
    });

    it('correctly back-substitutes through an earlier complex-conjugate block (two stacked complex pairs)', () => {
      // Regression case: T ends up quasi-upper-triangular with TWO
      // irreducible 2x2 blocks, so computing the eigenvector for the
      // second (lower) pair requires back-substituting *through* the
      // first pair's own 2x2 block above it — that block has its own
      // nonzero subdiagonal entry, so it must be solved as a coupled
      // 2x2 system rather than one row at a time.
      const m: Matrix = Matrix.from([
        [0, -6, -1, 0],
        [6, 0, 0, -1],
        [0, 0, 0, -6],
        [0, 0, 6, 0]
      ]);
      const pairs = m.eig();
      expect(pairs).toHaveLength(4);
      for (const pair of pairs) {
        expect(eigResidual(m, pair)).toBeCloseTo(0, 9);
        expect(eigNorm(pair)).toBeCloseTo(1, 9);
      }
    });

    it('normalizes a complex eigenvector so its largest-magnitude component is real and positive', () => {
      const m: Matrix = Matrix.from([
        [0, -6, -1, 0],
        [6, 0, 0, -1],
        [0, 0, 0, -6],
        [0, 0, 6, 0]
      ]);
      for (const pair of m.eig()) {
        if (pair.value.im === 0) continue;
        const vRe = pair.vectorRe.toArray(),
          vIm = pair.vectorIm.toArray();
        let maxIdx = 0,
          maxMag = -1;
        for (let i = 0; i < vRe.length; i++) {
          const mag = vRe[i] * vRe[i] + vIm[i] * vIm[i];
          if (mag > maxMag) {
            maxMag = mag;
            maxIdx = i;
          }
        }
        expect(vIm[maxIdx]).toBeCloseTo(0);
        expect(vRe[maxIdx]).toBeGreaterThan(0);
      }
    });

    it('returns one eigenvector for a defective (non-diagonalizable) matrix without throwing', () => {
      const jordan: Matrix = Matrix.from([
        [2, 1],
        [0, 2]
      ]);
      const pairs = jordan.eig();
      expect(pairs).toHaveLength(2);
      for (const pair of pairs) {
        expect(pair.value.re).toBeCloseTo(2);
        expect(eigResidual(jordan, pair)).toBeCloseTo(0, 6);
        expect(eigNorm(pair)).toBeCloseTo(1, 9);
      }
    });

    it('handles a singular matrix (zero eigenvalue)', () => {
      const singular: Matrix = Matrix.from([
        [1, 2],
        [2, 4]
      ]);
      const pairs = sortPairs(singular.eig());
      expect(pairs[0].value.re).toBeCloseTo(0);
      expect(pairs[1].value.re).toBeCloseTo(5);
      for (const pair of pairs) expect(eigResidual(singular, pair)).toBeCloseTo(0, 9);
    });

    it('handles the 1x1 case directly', () => {
      const pairs = Matrix.from([[7]]).eig();
      expect(pairs).toEqual([
        { value: { re: 7, im: 0 }, vectorRe: new Vector([1]), vectorIm: new Vector([0]) }
      ]);
    });

    it('rejects non-square input', () => {
      expect(() =>
        Matrix.from([
          [1, 2, 3],
          [4, 5, 6]
        ]).eig()
      ).toThrowError(RangeError);
    });

    it('throws once the iteration budget is exhausted', () => {
      const m: Matrix = Matrix.from([
        [4, 1, 2, 0],
        [1, 3, 0, 1],
        [2, 0, 5, 2],
        [0, 1, 2, 6]
      ]);
      expect(() => m.eig({ maxIterations: 0 })).toThrowError();
    });
  });

  it('rejects invalid shapes and mismatched input length in the constructor', () => {
    expect(() => new Matrix(0, 2)).toThrowError(RangeError);
    expect(() => new Matrix(2, 0)).toThrowError(RangeError);
    expect(() => new Matrix(1.5, 2)).toThrowError(RangeError);
    expect(() => new Matrix(2, 2, [1, 2, 3])).toThrowError(RangeError);
  });

  it('row() returns an independent copy, not a view into the matrix', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    const r = m.row(0);
    r.set(0, 999);
    expect(m.get(0, 0)).toBe(1); // original matrix must be untouched
    expect(r.get(0)).toBe(999);
  });

  it('throws when accessing an out-of-bounds row or column', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    expect(() => m.row(5)).toThrowError(RangeError);
    expect(() => m.col(5)).toThrowError(RangeError);
  });

  it('validates length when overwriting a row via setRow()', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    expect(() => m.setRow(0, [1, 2, 3])).toThrowError(RangeError); // too long
    expect(() => m.setRow(0, [1])).toThrowError(RangeError); // too short
    expect(() => m.setRow(5, [1, 2])).toThrowError(RangeError); // out of bounds
    m.setRow(0, [9, 9]);
    expect(m.row(0).toArray()).toEqual([9, 9]);
  });

  it('validates length when overwriting a column via setCol()', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    expect(() => m.setCol(0, [1, 2, 3])).toThrowError(RangeError); // too long
    expect(() => m.setCol(0, [1])).toThrowError(RangeError); // too short
    expect(() => m.setCol(5, [1, 2])).toThrowError(RangeError); // out of bounds
    m.setCol(1, [7, 8]);
    expect(m.col(1).toArray()).toEqual([7, 8]);
  });

  it('swaps rows in place and validates both indices', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    m.swapRowsSelf(0, 1);
    expect(m.toArray()).toEqual([
      [3, 4],
      [1, 2]
    ]);

    expect(() => m.swapRowsSelf(5, 0)).toThrowError(RangeError);
    expect(() => m.swapRowsSelf(0, 5)).toThrowError(RangeError);
    expect(() => m.swapRowsSelf(5, 5)).toThrowError(RangeError); // equal, out-of-bounds indices must still throw

    const unchanged: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    unchanged.swapRowsSelf(1, 1); // same in-bounds index is a no-op
    expect(unchanged.toArray()).toEqual([
      [1, 2],
      [3, 4]
    ]);
  });

  it('scales rows and columns and adds a scaled row in place', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    m.scaleRowSelf(0, 10);
    expect(m.row(0).toArray()).toEqual([10, 20]);

    expect(m.scaleColSelf(1, 3)).toBe(m);
    expect(m.col(1).toArray()).toEqual([60, 12]);

    m.addScaledRowSelf(1, 0, 2); // row1 += row0 * 2
    expect(m.row(1).toArray()).toEqual([23, 132]);

    expect(() => m.scaleRowSelf(5, 1)).toThrowError(RangeError);
    expect(() => m.scaleColSelf(5, 1)).toThrowError(RangeError);
    expect(() => m.addScaledRowSelf(5, 0, 1)).toThrowError(RangeError);
    expect(() => m.addScaledRowSelf(0, 5, 1)).toThrowError(RangeError);
  });

  it('rejects shape mismatches in elementwise and product operations', () => {
    const a: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    const b: Matrix = new Matrix(3, 3);
    expect(() => a.add(b)).toThrowError(RangeError);
    expect(() => a.sub(b)).toThrowError(RangeError);
    expect(() => a.matmul(b)).toThrowError(RangeError);
    expect(() => a.mulVec(new Vector(3))).toThrowError(RangeError);
  });

  it('multiplies a matrix by a column vector with mulVec', () => {
    const m: Matrix = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
    const v: Vector = new Vector([1, 0, 1]);
    expect(m.mulVec(v).toArray()).toEqual([4, 10]);
  });

  it('transposes matrices, including in place for square matrices', () => {
    const m: Matrix = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
    expect(m.transpose().toArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6]
    ]);
    expect(() => m.transposeSelf()).toThrowError(RangeError); // not square

    const sq: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    sq.transposeSelf();
    expect(sq.toArray()).toEqual([
      [1, 3],
      [2, 4]
    ]);
  });

  it('computes the trace of a square matrix and rejects non-square input', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    expect(m.trace()).toBe(5);
    expect(() => new Matrix(2, 3).trace()).toThrowError(RangeError);
  });

  it('treats a singular matrix as rank-deficient with zero determinant', () => {
    const singular: Matrix = new Matrix(2, 2, [1, 2, 2, 4]); // row1 = 2*row0
    expect(singular.determinant()).toBeCloseTo(0);
    expect(singular.rank()).toBe(1);
    expect(() => singular.inverse()).toThrowError();
    expect(() => singular.solve(new Vector([1, 1]))).toThrowError();
  });

  it("rank({ method: 'elimination' })'s default tolerance treats tiny floating-point noise as rank-deficient, but an explicit atol=0 does not", () => {
    // row1 is row0*2 plus a perturbation far below float64 precision at
    // this matrix's scale: mathematically singular, numerically not
    // quite. determinant() (no tolerance, matching LAPACK/numpy) still
    // reports the tiny nonzero value rather than exactly 0.
    const nearSingular: Matrix = new Matrix(2, 2, [1, 2, 2, 4 + 1e-14]);
    expect(nearSingular.rank({ method: 'elimination' })).toBe(1);
    expect(nearSingular.rank({ method: 'elimination', atol: 0, rtol: 0 })).toBe(2);
    expect(nearSingular.determinant()).not.toBe(0);
    expect(nearSingular.determinant()).toBeCloseTo(0);
  });

  it("rank({ method: 'elimination' })'s default tolerance scales with the matrix's own magnitude, unlike a fixed threshold", () => {
    // Same relative near-singularity as above, but every entry scaled
    // up by 1e8. A fixed absolute tolerance (e.g. the old default of
    // 1e-10) would be swamped by the larger elimination noise at this
    // scale and misreport this as full rank; the auto-scaled default
    // (proportional to the matrix's own infinity norm) still catches it.
    const scale = 1e8;
    const scaledNearSingular: Matrix = new Matrix(2, 2, [
      scale,
      2 * scale,
      2 * scale,
      4 * scale + scale * 1e-14
    ]);
    expect(scaledNearSingular.rank({ method: 'elimination' })).toBe(1);
    expect(scaledNearSingular.rank({ method: 'elimination', atol: 1e-10, rtol: 0 })).toBe(2);
  });

  it('rejects non-square input and shape-mismatched vectors for determinant/inverse/solve', () => {
    const nonSquare: Matrix = new Matrix(2, 3);
    expect(() => nonSquare.determinant()).toThrowError(RangeError);
    expect(() => nonSquare.inverse()).toThrowError(RangeError);
    expect(() => nonSquare.solve(new Vector(2))).toThrowError(RangeError);

    const square: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    expect(() => square.solve(new Vector(3))).toThrowError(RangeError);
  });

  it('copy() produces an independent matrix', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    const c = m.copy();
    c.set(0, 0, 999);
    expect(m.get(0, 0)).toBe(1);
    expect(c.get(0, 0)).toBe(999);
  });

  it('returns a per-element boolean array from isClose, in row-major order', () => {
    const a: Matrix = new Matrix(2, 2, [1, 2, 100, 4]);
    const b: Matrix = new Matrix(2, 2, [1.00001, 2.5, 100, 4]);
    expect(a.isClose(b)).toEqual([true, false, true, true]);
  });

  it('tests elements for NaN and finiteness in row-major order', () => {
    const m = new Matrix(2, 2, [1, NaN, Infinity, -2]);
    expect(m.isNaN()).toEqual([false, true, false, false]);
    expect(m.isFinite()).toEqual([true, false, false, true]);
    expect(m.allFinite()).toBe(false);
  });

  it('isClose: rtol scales the argument, not `this` (asymmetric)', () => {
    const smaller: Matrix = new Matrix(1, 1, [0.5]);
    const larger: Matrix = new Matrix(1, 1, [1]);
    expect(smaller.isClose(larger, 0.5, 0)).toEqual([true]);
    expect(larger.isClose(smaller, 0.5, 0)).toEqual([false]);
  });

  it('throws on shape mismatch in isClose/allClose, rather than returning false', () => {
    const a: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    expect(() => a.isClose(new Matrix(3, 3))).toThrowError(RangeError);
    expect(() => a.allClose(new Matrix(3, 3))).toThrowError(RangeError);
  });

  it('never considers NaN close to anything, including another NaN', () => {
    const a: Matrix = new Matrix(1, 1, [NaN]);
    const b: Matrix = new Matrix(1, 1, [NaN]);
    expect(a.isClose(b)).toEqual([false]);
    expect(a.allClose(b)).toBe(false);
  });

  it('allClose is true only when every element is close', () => {
    const a: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    const b: Matrix = new Matrix(2, 2, [1, 2, 3, 4.0000001]);
    const c: Matrix = new Matrix(2, 2, [1, 2, 3, 5]);
    expect(a.allClose(a.copy())).toBe(true);
    expect(a.allClose(b)).toBe(true);
    expect(a.allClose(c)).toBe(false);
  });

  it('computes sum/mean/min/max/variance/std over all elements when axis is omitted', () => {
    const m: Matrix = new Matrix(2, 2, [1, -2, 3, 4]);
    expect(m.sum()).toBe(6);
    expect(m.mean()).toBe(1.5);
    expect(m.min()).toBe(-2);
    expect(m.max()).toBe(4);
    expect(m.variance()).toBeCloseTo(5.25);
    expect(m.std()).toBeCloseTo(Math.sqrt(5.25));
  });

  it('propagates NaN in min()/max(), matching Math.min/Math.max semantics', () => {
    const m: Matrix = new Matrix(2, 2, [1, NaN, 3, 4]);
    expect(Number.isNaN(m.min())).toBe(true);
    expect(Number.isNaN(m.max())).toBe(true);
  });

  it('reduces along axis 0 (down each column) and axis 1 (across each row)', () => {
    const m: Matrix = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
    expect(m.sum(0).toArray()).toEqual([5, 7, 9]);
    expect(m.sum(1).toArray()).toEqual([6, 15]);

    expect(m.mean(0).toArray()).toEqual([2.5, 3.5, 4.5]);
    expect(m.mean(1).toArray()).toEqual([2, 5]);

    expect(m.min(0).toArray()).toEqual([1, 2, 3]);
    expect(m.min(1).toArray()).toEqual([1, 4]);

    expect(m.max(0).toArray()).toEqual([4, 5, 6]);
    expect(m.max(1).toArray()).toEqual([3, 6]);
  });

  it('computes variance/std per axis, with ddof forwarded correctly', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 5]);
    // Column 0: [1, 3], column 1: [2, 5]
    expect(m.variance(0).toArray()).toEqual([1, 2.25]);
    expect(m.variance(0, 1).toArray()).toEqual([2, 4.5]);
    expect(m.std(0).toArray()).toEqual([1, 1.5]);
  });

  it('cumsum flattens in row-major order when axis is omitted', () => {
    const m: Matrix = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
    expect(m.cumsum().toArray()).toEqual([1, 3, 6, 10, 15, 21]);
  });

  it('cumsum accumulates independently down each column (axis 0) or across each row (axis 1)', () => {
    const m: Matrix = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
    expect(m.cumsum(0).toArray()).toEqual([
      [1, 2, 3],
      [5, 7, 9]
    ]);
    expect(m.cumsum(1).toArray()).toEqual([
      [1, 3, 6],
      [4, 9, 15]
    ]);
  });

  it('argmin/argmax return a flat index when axis is omitted', () => {
    const m: Matrix = new Matrix(2, 3, [5, 1, 4, 2, 8, 0]);
    expect(m.argmin()).toBe(5); // value 0 at flat index 5
    expect(m.argmax()).toBe(4); // value 8 at flat index 4
  });

  it('argmin/argmax return the row index per column (axis 0) or column index per row (axis 1)', () => {
    const m: Matrix = new Matrix(2, 3, [5, 1, 4, 2, 8, 0]);
    // Columns: [5,2], [1,8], [4,0]
    expect(m.argmin(0).toArray()).toEqual([1, 0, 1]);
    expect(m.argmax(0).toArray()).toEqual([0, 1, 0]);
    // Rows: [5,1,4], [2,8,0]
    expect(m.argmin(1).toArray()).toEqual([1, 2]);
    expect(m.argmax(1).toArray()).toEqual([0, 1]);
  });

  it('argmin/argmax return the first index on ties, and prioritize NaN, in every axis mode', () => {
    const ties: Matrix = new Matrix(2, 2, [3, 3, 3, 3]);
    expect(ties.argmin()).toBe(0);
    expect(ties.argmin(0).toArray()).toEqual([0, 0]);
    expect(ties.argmin(1).toArray()).toEqual([0, 0]);

    const withNaN: Matrix = new Matrix(2, 2, [1, NaN, 3, 4]);
    expect(withNaN.argmin()).toBe(1);
    expect(withNaN.argmin(0).toArray()).toEqual([0, 0]); // col 0 = [1, 3] -> index 0; col 1 = [NaN, 4] -> index 0
    expect(withNaN.argmin(1).toArray()).toEqual([1, 0]); // row 0 = [1, NaN] -> index 1; row 1 = [3, 4] -> index 0
  });

  it('sort() defaults to axis 1, sorting each row independently', () => {
    const m: Matrix = new Matrix(2, 3, [5, 6, 4, 1, 2, 3]);
    expect(m.sort().toArray()).toEqual([
      [4, 5, 6],
      [1, 2, 3]
    ]);
  });

  it('sort(0) sorts each column independently', () => {
    const m: Matrix = new Matrix(2, 3, [5, 6, 4, 1, 2, 3]);
    expect(m.sort(0).toArray()).toEqual([
      [1, 2, 3],
      [5, 6, 4]
    ]);
  });

  it('sort() does not mutate the original matrix', () => {
    const m: Matrix = new Matrix(2, 2, [4, 3, 2, 1]);
    const sorted = m.sort();
    expect(m.toArray()).toEqual([
      [4, 3],
      [2, 1]
    ]);
    expect(sorted.toArray()).toEqual([
      [3, 4],
      [1, 2]
    ]);
  });

  it('sort() accepts a custom comparator', () => {
    const m: Matrix = new Matrix(2, 3, [5, 6, 4, 1, 2, 3]);
    expect(m.sort(1, (a, b) => b - a).toArray()).toEqual([
      [6, 5, 4],
      [3, 2, 1]
    ]);
  });

  it('slices a sub-matrix with Array.prototype.slice semantics, per axis', () => {
    const m: Matrix = new Matrix(3, 4, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    expect(m.slice(1, 3, 1, 3).toArray()).toEqual([
      [5, 6],
      [9, 10]
    ]);
    expect(m.slice(1).toArray()).toEqual([
      [4, 5, 6, 7],
      [8, 9, 10, 11]
    ]);
    expect(m.slice(undefined, undefined, 2).toArray()).toEqual([
      [2, 3],
      [6, 7],
      [10, 11]
    ]);
    expect(m.slice().toArray()).toEqual(m.toArray());
  });

  it('slice() supports negative indices, counting back from the end, per axis', () => {
    const m: Matrix = new Matrix(3, 3, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect(m.slice(-2).toArray()).toEqual([
      [3, 4, 5],
      [6, 7, 8]
    ]);
    expect(m.slice(undefined, undefined, -2).toArray()).toEqual([
      [1, 2],
      [4, 5],
      [7, 8]
    ]);
  });

  it('slice() returns an independent copy', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    const s = m.slice(0, 1, 0, 1);
    s.set(0, 0, 999);
    expect(m.get(0, 0)).toBe(1);
  });

  it('slice() throws if the resolved range is empty on either axis', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    expect(() => m.slice(1, 1)).toThrowError(RangeError);
    expect(() => m.slice(undefined, undefined, 1, 1)).toThrowError(RangeError);
    expect(() => m.slice(5, 6)).toThrowError(RangeError);
  });

  it('multiplies and divides elementwise via mult/div with a matrix argument (Hadamard)', () => {
    const a: Matrix = new Matrix(2, 2, [2, 3, 4, 6]);
    const b: Matrix = new Matrix(2, 2, [1, 2, 4, 3]);
    expect(a.mult(b).toArray()).toEqual([
      [2, 6],
      [16, 18]
    ]);
    expect(a.div(b).toArray()).toEqual([
      [2, 1.5],
      [1, 2]
    ]);
    expect(() => a.mult(new Matrix(3, 3))).toThrowError(RangeError);
    expect(() => a.div(new Matrix(3, 3))).toThrowError(RangeError);
  });

  it('accepts a scalar in add/sub, matching Vector', () => {
    const a: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    expect(a.add(10).toArray()).toEqual([
      [11, 12],
      [13, 14]
    ]);
    expect(a.sub(1).toArray()).toEqual([
      [0, 1],
      [2, 3]
    ]);
  });

  describe('row-vector broadcasting in add/sub/mult/div (and the Self variants)', () => {
    type Op = 'add' | 'sub' | 'mult' | 'div';
    type SelfOp = 'addSelf' | 'subSelf' | 'multSelf' | 'divSelf';
    const ops: Op[] = ['add', 'sub', 'mult', 'div'];
    const selfOf = (op: Op): SelfOp => `${op}Self` as SelfOp;
    const scalarFns: Record<Op, (a: number, b: number) => number> = {
      add: (a, b) => a + b,
      sub: (a, b) => a - b,
      mult: (a, b) => a * b,
      div: (a, b) => a / b
    };

    // Golden values below were generated with numpy 2.4 (np.add/subtract/multiply/divide
    // of a 2-D array with a 1-D array), not derived from this implementation.
    const A23 = (): Matrix => new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
    const v3 = (): Vector => new Vector([10, 20, 30]);

    it('matches numpy for a non-square matrix, for all four operations', () => {
      expect(A23().add(v3()).toArray()).toEqual([
        [11, 22, 33],
        [14, 25, 36]
      ]);
      expect(A23().sub(v3()).toArray()).toEqual([
        [-9, -18, -27],
        [-6, -15, -24]
      ]);
      expect(A23().mult(v3()).toArray()).toEqual([
        [10, 40, 90],
        [40, 100, 180]
      ]);
      expect(A23().div(v3()).toArray()).toEqual([
        [1 / 10, 2 / 20, 3 / 30],
        [4 / 10, 5 / 20, 6 / 30]
      ]);
      // numpy prints these as [[0.1, 0.1, 0.1], [0.4, 0.25, 0.2]]
      const q = A23().div(v3());
      expect(q.get(0, 0)).toBeCloseTo(0.1, 15);
      expect(q.get(1, 1)).toBe(0.25);
      expect(q.get(1, 2)).toBeCloseTo(0.2, 15);
    });

    it('treats the vector as a ROW on a square matrix (numpy default), not as a column', () => {
      // On a square matrix both orientations are shape-valid, so only the
      // values can tell them apart: v[j] must go to column j.
      const B = (): Matrix => new Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 10]);
      const w = (): Vector => new Vector([1, 10, 100]);
      expect(B().add(w()).toArray()).toEqual([
        [2, 12, 103],
        [5, 15, 106],
        [8, 18, 110]
      ]);
      expect(B().sub(w()).toArray()).toEqual([
        [0, -8, -97],
        [3, -5, -94],
        [6, -2, -90]
      ]);
      expect(B().mult(w()).toArray()).toEqual([
        [1, 20, 300],
        [4, 50, 600],
        [7, 80, 1000]
      ]);
      const q = B().div(w()).toArray();
      expect(q[0]).toEqual([1, 0.2, 0.03]);
      expect(q[1]).toEqual([4, 0.5, 0.06]);
      expect(q[2]).toEqual([7, 0.8, 0.1]);
    });

    it('keeps operand order: the vector is always the right-hand side of sub/div', () => {
      // numpy: [[10,20],[30,40]] - [1,2] = [[9,18],[29,38]]; / [1,2] = [[10,10],[30,20]]
      const C = (): Matrix => new Matrix(2, 2, [10, 20, 30, 40]);
      const u = (): Vector => new Vector([1, 2]);
      expect(C().sub(u()).toArray()).toEqual([
        [9, 18],
        [29, 38]
      ]);
      expect(C().div(u()).toArray()).toEqual([
        [10, 10],
        [30, 20]
      ]);
    });

    it('acts as a scalar when the vector has size 1, for every operation (numpy)', () => {
      const s = new Vector([5]);
      expect(A23().add(s).toArray()).toEqual([
        [6, 7, 8],
        [9, 10, 11]
      ]);
      expect(A23().sub(s).toArray()).toEqual([
        [-4, -3, -2],
        [-1, 0, 1]
      ]);
      expect(A23().mult(s).toArray()).toEqual([
        [5, 10, 15],
        [20, 25, 30]
      ]);
      expect(A23().div(s).toArray()).toEqual([
        [0.2, 0.4, 0.6],
        [0.8, 1, 1.2]
      ]);
    });

    it('acts as a scalar for size-1 vectors on 1x1 and n x 1 matrices too (result shape unchanged)', () => {
      const s = new Vector([3]);
      const one = new Matrix(1, 1, [4]);
      expect(one.add(s).toArray()).toEqual([[7]]);
      const col = new Matrix(3, 1, [1, 2, 3]);
      const r = col.mult(s);
      expect(r.rows).toBe(3);
      expect(r.cols).toBe(1);
      expect(r.toArray()).toEqual([[3], [6], [9]]);
      expect(col.copy().subSelf(s).toArray()).toEqual([[-2], [-1], [0]]);
    });

    it('broadcasts against a single-row matrix', () => {
      // numpy: [[1,2,3]] + [10,20,30] = [[11,22,33]]
      const row = new Matrix(1, 3, [1, 2, 3]);
      expect(row.add(v3()).toArray()).toEqual([[11, 22, 33]]);
    });

    it("returns a new Matrix of this matrix's shape and leaves both operands untouched", () => {
      for (const op of ops) {
        const a = A23();
        const v = v3();
        const r = a[op](v);
        expect(r).toBeInstanceOf(Matrix);
        expect(r).not.toBe(a);
        expect(r.data).not.toBe(a.data);
        expect(r.rows).toBe(2);
        expect(r.cols).toBe(3);
        expect(a.toArray()).toEqual([
          [1, 2, 3],
          [4, 5, 6]
        ]);
        expect(v.toArray()).toEqual([10, 20, 30]);
      }
    });

    it('Self variants mutate in place, return `this`, leave the vector untouched, and equal the immutable result', () => {
      for (const op of ops) {
        const expected = A23()[op](v3());
        const a = A23();
        const buffer = a.data;
        const v = v3();
        const r = a[selfOf(op)](v);
        expect(r).toBe(a);
        expect(a.data).toBe(buffer);
        expect(a.toArray()).toEqual(expected.toArray());
        expect(v.toArray()).toEqual([10, 20, 30]);
      }
    });

    it('chains with other operations', () => {
      const r = A23()
        .subSelf(new Vector([1, 2, 3]))
        .multSelf(new Vector([2, 2, 2]))
        .add(1);
      // ([[1,2,3],[4,5,6]] - [1,2,3]) * 2 + 1
      expect(r.toArray()).toEqual([
        [1, 1, 1],
        [7, 7, 7]
      ]);
    });

    it('does not change the existing matrix/scalar paths', () => {
      const a = new Matrix(2, 2, [1, 2, 3, 4]);
      const b = new Matrix(2, 2, [10, 20, 30, 40]);
      expect(a.add(b).toArray()).toEqual([
        [11, 22],
        [33, 44]
      ]);
      expect(a.copy().addSelf(b).toArray()).toEqual([
        [11, 22],
        [33, 44]
      ]);
      expect(a.mult(2).toArray()).toEqual([
        [2, 4],
        [6, 8]
      ]);
      expect(a.copy().divSelf(2).toArray()).toEqual([
        [0.5, 1],
        [1.5, 2]
      ]);
    });

    it('still rejects matrix-with-matrix shape mismatches, including numpy-broadcastable ones', () => {
      // Broadcasting is limited to a Vector operand: a 1x3 *Matrix* is not broadcast.
      const a = new Matrix(2, 3);
      for (const op of ops) {
        expect(() => a[op](new Matrix(1, 3))).toThrowError(RangeError);
        expect(() => a[op](new Matrix(2, 1))).toThrowError(RangeError);
        expect(() => a[selfOf(op)](new Matrix(1, 3))).toThrowError(RangeError);
      }
    });

    it('propagates NaN/Infinity and divides by zero following float semantics (numpy)', () => {
      // numpy: [[1,-1,0],[2,0,-3]] / [0,0,0] = [[inf,-inf,nan],[inf,nan,-inf]]
      const d = new Matrix(2, 3, [1, -1, 0, 2, 0, -3]).div(new Vector(3));
      expect(d.get(0, 0)).toBe(Infinity);
      expect(d.get(0, 1)).toBe(-Infinity);
      expect(Number.isNaN(d.get(0, 2))).toBe(true);
      expect(d.get(1, 0)).toBe(Infinity);
      expect(Number.isNaN(d.get(1, 1))).toBe(true);
      expect(d.get(1, 2)).toBe(-Infinity);

      const n = new Matrix(2, 2, [1, 2, 3, 4]).add(new Vector([NaN, Infinity]));
      expect(Number.isNaN(n.get(0, 0))).toBe(true);
      expect(Number.isNaN(n.get(1, 0))).toBe(true);
      expect(n.get(0, 1)).toBe(Infinity);
      expect(n.get(1, 1)).toBe(Infinity);
    });

    it('throws RangeError, with both shapes in the message, when the vector size is neither 1 nor cols', () => {
      for (const op of ops) {
        expect(() => A23()[op](new Vector(2))).toThrowError(RangeError);
        expect(() => A23()[op](new Vector(2))).toThrowError(
          new RegExp(`Matrix\\.${op}:.*\\(2,3\\) \\(2\\)`)
        );
        expect(() => A23()[op](new Vector(4))).toThrowError(RangeError);
        expect(() => A23()[selfOf(op)](new Vector(2))).toThrowError(
          new RegExp(`Matrix\\.${op}Self:.*\\(2,3\\) \\(2\\)`)
        );
      }
    });

    it('rejects an empty vector', () => {
      for (const op of ops) {
        expect(() => A23()[op](new Vector(0))).toThrowError(RangeError);
        expect(() => A23()[selfOf(op)](new Vector(0))).toThrowError(RangeError);
        // Also for an n x 1 matrix (numpy would return an empty n x 0 result; Matrix can't hold that).
        expect(() => new Matrix(3, 1)[op](new Vector(0))).toThrowError(RangeError);
      }
    });

    it('rejects the numpy-valid case that would change the result shape (n x 1 matrix, vector of size k > 1)', () => {
      // numpy: ones((3,1)) + [1,2] has shape (3,2). Matrix.add would have to return a
      // different shape than `this`, so it is rejected instead of silently diverging.
      for (const op of ops) {
        const col = (): Matrix => new Matrix(3, 1, [1, 2, 3]);
        expect(() => col()[op](new Vector([1, 2]))).toThrowError(RangeError);
        expect(() => col()[op](new Vector([1, 2]))).toThrowError(
          /would change the result shape to 3x2/
        );
        expect(() => col()[selfOf(op)](new Vector([1, 2]))).toThrowError(RangeError);
      }
    });

    it('leaves the matrix unmodified when a Self variant throws', () => {
      for (const op of ops) {
        const a = A23();
        expect(() => a[selfOf(op)](new Vector(2))).toThrowError(RangeError);
        expect(a.toArray()).toEqual([
          [1, 2, 3],
          [4, 5, 6]
        ]);
      }
    });

    it('agrees bit-for-bit with the same-shape matrix path on tiled vectors (randomized cross-check)', () => {
      // Independent reference: tile the vector into a full matrix and use the
      // long-standing same-shape elementwise path. Each element undergoes the
      // exact same single floating-point operation, so results must be identical.
      const rnd = rand(12345);
      const shapes: Array<[number, number]> = [
        [1, 1],
        [1, 5],
        [4, 1],
        [3, 3],
        [2, 7],
        [6, 4],
        [17, 13]
      ];
      for (const [m, n] of shapes) {
        const a = new Matrix(
          m,
          n,
          Array.from({ length: m * n }, () => rnd() * 20 - 10)
        );
        const vec = new Vector(Array.from({ length: n }, () => rnd() * 20 - 10));
        const tiled = new Matrix(m, n);
        for (let i = 0; i < m; i++) tiled.setRow(i, vec);
        for (const op of ops) {
          expect(a[op](vec).toArray()).toEqual(a[op](tiled).toArray());
          expect(a.copy()[selfOf(op)](vec).toArray()).toEqual(
            a.copy()[selfOf(op)](tiled).toArray()
          );
          // ...and against a plain scalar loop, guarding against the two paths sharing a bug
          const r = a[op](vec);
          for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
              expect(r.get(i, j)).toBe(scalarFns[op](a.get(i, j), vec.get(j)));
            }
          }
        }
      }
    });
  });

  it('divides by zero following standard float semantics', () => {
    const a: Matrix = new Matrix(1, 3, [1, -1, 0]);
    const zero: Matrix = new Matrix(1, 3);
    const result = a.div(zero).toArray()[0];
    expect(result[0]).toBe(Infinity);
    expect(result[1]).toBe(-Infinity);
    expect(Number.isNaN(result[2])).toBe(true);
  });

  it('performs addSelf/subSelf/multSelf/divSelf/fill in place', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    m.addSelf(new Matrix(2, 2, [1, 1, 1, 1]));
    expect(m.toArray()).toEqual([
      [2, 3],
      [4, 5]
    ]);

    m.subSelf(new Matrix(2, 2, [1, 1, 1, 1]));
    m.multSelf(10);
    expect(m.toArray()).toEqual([
      [10, 20],
      [30, 40]
    ]);

    m.divSelf(new Matrix(2, 2, [2, 2, 2, 2]));
    expect(m.toArray()).toEqual([
      [5, 10],
      [15, 20]
    ]);

    m.fill(0);
    expect(m.toArray()).toEqual([
      [0, 0],
      [0, 0]
    ]);
  });

  it('computes the Frobenius norm, inner product, and distance (inherited from ArrayND)', () => {
    const a: Matrix = new Matrix(2, 2, [3, 0, 0, 4]);
    expect(a.norm()).toBe(5);
    expect(a.normSq()).toBe(25);

    const b: Matrix = new Matrix(2, 2, [1, 0, 0, 1]);
    expect(a.dot(b)).toBe(7); // 3*1 + 0*0 + 0*0 + 4*1
    expect(() => a.dot(new Matrix(3, 3))).toThrowError(RangeError);

    const zero: Matrix = new Matrix(2, 2);
    expect(zero.dist(a)).toBe(5);
  });

  it('builds constant matrices via static ones() and full(), like Vector.ones() and Vector.full()', () => {
    expect(Matrix.ones(2, 3).toArray()).toEqual([
      [1, 1, 1],
      [1, 1, 1]
    ]);
    expect(Matrix.full(2, 2, 2.5).toArray()).toEqual([
      [2.5, 2.5],
      [2.5, 2.5]
    ]);
    expect(Matrix.full(1, 3, 0).toArray()).toEqual(Matrix.zero(1, 3).toArray());
    // `full` stores the value as given, as `Vector.full` does.
    expect(Matrix.full(2, 2, NaN).data.every(Number.isNaN)).toBe(true);
    expect(Matrix.full(1, 2, -Infinity).toArray()).toEqual([[-Infinity, -Infinity]]);
    // Every call returns a new matrix with its own storage.
    const a = Matrix.ones(2, 2);
    const b = Matrix.ones(2, 2);
    a.set(0, 0, 9);
    expect(b.get(0, 0)).toBe(1);
    // Same shape checks as zero() and the constructor.
    expect(() => Matrix.ones(0, 2)).toThrowError(RangeError);
    expect(() => Matrix.ones(2, 1.5)).toThrowError(RangeError);
    expect(() => Matrix.full(-1, 2, 3)).toThrowError(RangeError);
    expect(
      Matrix.ones(3, 3)
        .sub(Matrix.diag(Vector.ones(3)))
        .norm1()
    ).toBe(2);
  });

  it('builds matrices via static zero(), identity(), and from()', () => {
    expect(Matrix.zero(2, 2).toArray()).toEqual([
      [0, 0],
      [0, 0]
    ]);
    expect(Matrix.identity(2).toArray()).toEqual([
      [1, 0],
      [0, 1]
    ]);
    expect(
      Matrix.from([
        [1, 2],
        [3, 4]
      ]).toArray()
    ).toEqual([
      [1, 2],
      [3, 4]
    ]);
  });

  it('rejects empty input and ragged rows in Matrix.from()', () => {
    expect(() => Matrix.from([])).toThrowError(RangeError);
    expect(() => Matrix.from([[]])).toThrowError(RangeError);
    expect(() => Matrix.from([[1, 2], [3]])).toThrowError(RangeError); // ragged row
  });

  it('is iterable over its rows', () => {
    const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
    const rows = [...m].map((r) => r.toArray());
    expect(rows).toEqual([
      [1, 2],
      [3, 4]
    ]);
  });

  describe('map()', () => {
    it('applies a function elementwise, passing the value and (row, col) indices', () => {
      const m: Matrix = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
      const seen: Array<[number, number, number]> = [];
      const doubled = m.map((v, i, j) => {
        seen.push([v, i, j]);
        return v * 2;
      });
      expect(doubled.toArray()).toEqual([
        [2, 4, 6],
        [8, 10, 12]
      ]);
      expect(seen).toEqual([
        [1, 0, 0],
        [2, 0, 1],
        [3, 0, 2],
        [4, 1, 0],
        [5, 1, 1],
        [6, 1, 2]
      ]);
    });

    it('returns a new matrix, leaving the original untouched', () => {
      const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
      const mapped = m.map((v) => v + 100);
      expect(m.toArray()).toEqual([
        [1, 2],
        [3, 4]
      ]);
      expect(mapped.toArray()).toEqual([
        [101, 102],
        [103, 104]
      ]);
      expect(mapped).not.toBe(m);
    });

    it('can use the row/col indices to build position-dependent values', () => {
      const m: Matrix = new Matrix(3, 3);
      const idx = m.map((_v, i, j) => i * 10 + j);
      expect(idx.toArray()).toEqual([
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22]
      ]);
    });
  });

  describe('diag()', () => {
    it('extracts the main diagonal of a square matrix as a vector', () => {
      const m: Matrix = new Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      expect(m.diag().toArray()).toEqual([1, 5, 9]);
    });

    it('extracts min(rows, cols) entries from a non-square matrix', () => {
      const tall: Matrix = new Matrix(3, 2, [1, 2, 3, 4, 5, 6]);
      expect(tall.diag().toArray()).toEqual([1, 4]);

      const wide: Matrix = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
      expect(wide.diag().toArray()).toEqual([1, 5]);
    });

    it('returns an independent copy, not a view into the matrix', () => {
      const m: Matrix = new Matrix(2, 2, [1, 2, 3, 4]);
      const d = m.diag();
      d.set(0, 999);
      expect(m.get(0, 0)).toBe(1);
    });
  });

  describe('Matrix.diag()', () => {
    it('builds a diagonal matrix from a vector', () => {
      const v = Vector.from([1, 2, 3]);
      expect(Matrix.diag(v).toArray()).toEqual([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
      ]);
    });

    it('is the inverse of the instance method diag() for a square matrix', () => {
      const m: Matrix = new Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      expect(Matrix.diag(m.diag()).toArray()).toEqual([
        [1, 0, 0],
        [0, 5, 0],
        [0, 0, 9]
      ]);
    });

    it('matches identity(n) when built from a vector of ones', () => {
      expect(Matrix.diag(Vector.ones(3)).toArray()).toEqual(Matrix.identity(3).toArray());
    });

    it('throws when given an empty vector', () => {
      expect(() => Matrix.diag(new Vector(0))).toThrowError(RangeError);
    });
  });

  describe('Matrix.hstack()', () => {
    it('glues matrices side by side, growing the column count', () => {
      const a = Matrix.from([
        [1, 2],
        [3, 4]
      ]);
      const b = Matrix.from([[5], [6]]);
      expect(Matrix.hstack([a, b]).toArray()).toEqual([
        [1, 2, 5],
        [3, 4, 6]
      ]);
    });

    it('concatenates more than two matrices, in order', () => {
      const a = Matrix.from([[1], [2]]);
      const b = Matrix.from([[3], [4]]);
      const c = Matrix.from([[5], [6]]);
      expect(Matrix.hstack([a, b, c]).toArray()).toEqual([
        [1, 3, 5],
        [2, 4, 6]
      ]);
    });

    it('is the inverse of slicing out column ranges', () => {
      const m = Matrix.from([
        [1, 2, 3],
        [4, 5, 6]
      ]);
      const left = m.slice(undefined, undefined, 0, 1);
      const right = m.slice(undefined, undefined, 1, 3);
      expect(Matrix.hstack([left, right]).toArray()).toEqual(m.toArray());
    });

    it('throws on an empty input or a row-count mismatch', () => {
      expect(() => Matrix.hstack([])).toThrowError(RangeError);
      expect(() => Matrix.hstack([new Matrix(2, 2), new Matrix(3, 2)])).toThrowError(RangeError);
    });
  });

  describe('Matrix.vstack()', () => {
    it('stacks matrices top to bottom, growing the row count', () => {
      const a = Matrix.from([
        [1, 2],
        [3, 4]
      ]);
      const b = Matrix.from([[5, 6]]);
      expect(Matrix.vstack([a, b]).toArray()).toEqual([
        [1, 2],
        [3, 4],
        [5, 6]
      ]);
    });

    it('concatenates more than two matrices, in order', () => {
      const a = Matrix.from([[1, 2]]);
      const b = Matrix.from([[3, 4]]);
      const c = Matrix.from([[5, 6]]);
      expect(Matrix.vstack([a, b, c]).toArray()).toEqual([
        [1, 2],
        [3, 4],
        [5, 6]
      ]);
    });

    it('is the inverse of slicing out row ranges', () => {
      const m = Matrix.from([
        [1, 2],
        [3, 4],
        [5, 6]
      ]);
      const top = m.slice(0, 1);
      const bottom = m.slice(1, 3);
      expect(Matrix.vstack([top, bottom]).toArray()).toEqual(m.toArray());
    });

    it('throws on an empty input or a column-count mismatch', () => {
      expect(() => Matrix.vstack([])).toThrowError(RangeError);
      expect(() => Matrix.vstack([new Matrix(2, 2), new Matrix(2, 3)])).toThrowError(RangeError);
    });
  });

  describe('normInf() and norm1()', () => {
    it('normInf() computes the largest absolute row sum', () => {
      const m = Matrix.from([
        [1, -2, 3],
        [-4, 5, -6],
        [0, 0, 1]
      ]);
      // row sums: 6, 15, 1
      expect(m.normInf()).toBe(15);
    });

    it('norm1() computes the largest absolute column sum', () => {
      const m = Matrix.from([
        [1, -2, 3],
        [-4, 5, -6],
        [0, 0, 1]
      ]);
      // col sums: 5, 7, 10
      expect(m.norm1()).toBe(10);
    });

    it('normInf() and norm1() propagate NaN instead of ignoring it', () => {
      const withNaN = Matrix.from([
        [NaN, 1],
        [1, 1]
      ]);
      expect(withNaN.normInf()).toBeNaN();
      expect(withNaN.norm1()).toBeNaN();
      // NaN in the last row / column, after larger finite sums.
      const late = Matrix.from([
        [5, 5],
        [1, NaN]
      ]);
      expect(late.normInf()).toBeNaN();
      expect(late.norm1()).toBeNaN();
    });

    it('norm1() of a matrix equals normInf() of its transpose, and vice versa', () => {
      const m = Matrix.from([
        [1, 2, 3],
        [4, 5, 6]
      ]);
      expect(m.norm1()).toBe(m.transpose().normInf());
      expect(m.normInf()).toBe(m.transpose().norm1());
    });

    it('returns 0 for a matrix of all zeros', () => {
      const m = new Matrix(3, 4);
      expect(m.normInf()).toBe(0);
      expect(m.norm1()).toBe(0);
    });

    it('handles non-square matrices', () => {
      const tall = Matrix.from([
        [1, 2],
        [3, 4],
        [5, 6]
      ]);
      expect(tall.normInf()).toBe(11); // row sums: 3, 7, 11
      expect(tall.norm1()).toBe(12); // col sums: 9, 12
    });
  });

  describe('isSquare()', () => {
    it('is true for a square matrix and false for a rectangular one', () => {
      expect(new Matrix(3, 3).isSquare()).toBe(true);
      expect(new Matrix(3, 4).isSquare()).toBe(false);
      expect(new Matrix(4, 3).isSquare()).toBe(false);
    });

    it('is true for the trivial 1x1 case', () => {
      expect(new Matrix(1, 1).isSquare()).toBe(true);
    });
  });

  describe('isSymmetric()', () => {
    it('is true for a symmetric matrix', () => {
      const m = Matrix.from([
        [1, 2, 3],
        [2, 5, 6],
        [3, 6, 9]
      ]);
      expect(m.isSymmetric()).toBe(true);
    });

    it('is false when any off-diagonal pair differs', () => {
      const m = Matrix.from([
        [1, 2, 3],
        [2, 5, 6],
        [3, 7, 9] // (1,2)=6 vs (2,1)=7
      ]);
      expect(m.isSymmetric()).toBe(false);
    });

    it('is false for a non-square matrix, regardless of atol', () => {
      const m = new Matrix(2, 3);
      expect(m.isSymmetric()).toBe(false);
      expect(m.isSymmetric(1e9)).toBe(false);
    });

    it('is true for the trivial 1x1 case', () => {
      expect(Matrix.from([[5]]).isSymmetric()).toBe(true);
    });

    it('defaults to exact equality, but accepts a tolerance for near-symmetric matrices', () => {
      const m = Matrix.from([
        [1, 2],
        [2 + 1e-6, 1]
      ]);
      expect(m.isSymmetric()).toBe(false);
      expect(m.isSymmetric(1e-5)).toBe(true);
    });
  });

  describe('isLowerTriangular() and isUpperTriangular()', () => {
    it('isLowerTriangular() is true when every entry above the diagonal is zero', () => {
      const m = Matrix.from([
        [1, 0, 0],
        [2, 3, 0],
        [4, 5, 6]
      ]);
      expect(m.isLowerTriangular()).toBe(true);
      expect(m.isUpperTriangular()).toBe(false);
    });

    it('isUpperTriangular() is true when every entry below the diagonal is zero', () => {
      const m = Matrix.from([
        [1, 2, 3],
        [0, 4, 5],
        [0, 0, 6]
      ]);
      expect(m.isUpperTriangular()).toBe(true);
      expect(m.isLowerTriangular()).toBe(false);
    });

    it('a diagonal matrix is both lower and upper triangular', () => {
      const m = Matrix.diag(Vector.from([1, 2, 3]));
      expect(m.isLowerTriangular()).toBe(true);
      expect(m.isUpperTriangular()).toBe(true);
    });

    it('generalizes to trapezoidal shapes on non-square matrices', () => {
      const tall = Matrix.from([
        [1, 2],
        [0, 3],
        [0, 0]
      ]);
      expect(tall.isUpperTriangular()).toBe(true);

      const wide = Matrix.from([
        [1, 0, 0],
        [2, 3, 0]
      ]);
      expect(wide.isLowerTriangular()).toBe(true);
    });

    it('rejects a single nonzero entry on the wrong side of the diagonal', () => {
      const m = Matrix.from([
        [1, 0],
        [0.001, 1]
      ]);
      expect(m.isUpperTriangular()).toBe(false);
      expect(m.isUpperTriangular(0.001)).toBe(true);
      expect(m.isUpperTriangular(0.0001)).toBe(false);
    });

    it('is upper/lower triangular for QR results, within floating-point tolerance', () => {
      const a = Matrix.from([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
      ]);
      const { R } = a.qr();
      expect(R.isUpperTriangular(1e-9)).toBe(true);
    });
  });

  describe('normInf()', () => {
    it('overrides the base entrywise meaning with the induced (max row-sum) operator norm', () => {
      // [[1, -2], [3, 4]]: row sums of abs values are 3 and 7.
      const m = Matrix.from([
        [1, -2],
        [3, 4]
      ]);
      expect(m.normInf()).toBe(7); // NOT 4, which would be the entrywise max.
    });

    it('returns 0 for the zero matrix', () => {
      expect(Matrix.zero(3, 3).normInf()).toBe(0);
    });

    it('equals the single entry for a 1x1 matrix', () => {
      const m = Matrix.from([[-9]]);
      expect(m.normInf()).toBe(9);
    });

    it('satisfies norm1(A) === normInf(A^T) for a non-symmetric matrix', () => {
      const m = Matrix.from([
        [1, 2, 3],
        [-4, 5, -6]
      ]);
      expect(m.norm1()).toBeCloseTo(m.transpose().normInf(), 12);
    });
  });

  describe('cond1() (general, exact)', () => {
    it('is 1 for the identity matrix', () => {
      expect(Matrix.identity(5).cond1()).toBeCloseTo(1, 10);
    });

    it('equals max(|d_i|) / min(|d_i|) for a diagonal matrix (exact, closed form)', () => {
      const M = Matrix.diag(Vector.from([4, -1, 2, 8]));
      expect(M.cond1()).toBeCloseTo(8 / 1, 10);
    });

    it('is Infinity for a singular matrix', () => {
      const M = Matrix.from([
        [1, 2],
        [2, 4]
      ]); // rank-deficient
      expect(M.cond1()).toBe(Infinity);
    });

    it('is Infinity for a 1x1 zero matrix', () => {
      expect(Matrix.zero(1, 1).cond1()).toBe(Infinity);
    });

    it('throws RangeError for a non-square matrix', () => {
      expect(() => new Matrix(2, 3).cond1()).toThrow(RangeError);
    });
  });

  describe('cond1Upper() / cond1Lower() (O(n^2) estimate)', () => {
    it('is exactly 1 for the identity matrix (upper and lower)', () => {
      expect(Matrix.identity(6).cond1Upper()).toBeCloseTo(1, 10);
      expect(Matrix.identity(6).cond1Lower()).toBeCloseTo(1, 10);
    });

    it('is a valid (non-tight) lower bound for a diagonal matrix', () => {
      // The classic CMSW/LINPACK estimator is a heuristic: even for a
      // diagonal matrix it doesn't always find the worst-conditioned
      // direction (the adaptive sign choice never gets data to react to,
      // since off-diagonal coupling is exactly zero). For diag(4,-1,2),
      // the true condition number is 4/1 = 4; hand-tracing the algorithm
      // gives exactly 3 — a valid but non-tight lower bound. This is
      // expected estimator behavior, not a bug (see the lower-bound
      // property test below for the general guarantee).
      const M = Matrix.from([
        [4, 0, 0],
        [0, -1, 0],
        [0, 0, 2]
      ]);
      expect(M.cond1Upper()).toBeCloseTo(3, 10);
      expect(M.cond1Lower()).toBeCloseTo(3, 10);
    });

    it('returns Infinity when the triangular matrix has a zero diagonal entry (exactly singular)', () => {
      const U = Matrix.from([
        [1, 2, 3],
        [0, 0, 5],
        [0, 0, 6]
      ]);
      expect(U.cond1Upper()).toBe(Infinity);

      const L = Matrix.from([
        [1, 0, 0],
        [2, 0, 0],
        [3, 5, 6]
      ]);
      expect(L.cond1Lower()).toBe(Infinity);
    });

    it('returns Infinity for a zero matrix', () => {
      expect(Matrix.zero(3, 3).cond1Upper()).toBe(Infinity);
      expect(Matrix.zero(3, 3).cond1Lower()).toBe(Infinity);
    });

    it('handles the 1x1 case', () => {
      expect(Matrix.from([[7]]).cond1Upper()).toBeCloseTo(1, 10);
      expect(Matrix.from([[0]]).cond1Upper()).toBe(Infinity);
    });

    it('respects unitDiagonal on cond1Lower (stored diagonal values are ignored, not just the divide skipped)', () => {
      // Same off-diagonal structure, deliberately different (garbage)
      // stored diagonals. With unitDiagonal=true, both must produce the
      // exact same estimate, since the diagonal must never actually be
      // read (matching solveLower's documented contract).
      const L1 = Matrix.from([
        [999, 0, 0],
        [2, -777, 0],
        [3, 5, 42]
      ]);
      const L2 = Matrix.from([
        [-3, 0, 0],
        [2, 0.5, 0],
        [3, 5, 1e6]
      ]);
      expect(L1.cond1Lower(true)).toBe(L2.cond1Lower(true));

      // And, for the realistic case where the diagonal genuinely is all
      // 1s (as lu()'s L always is), the estimate should still be a valid
      // lower bound on the true condition number.
      const Lunit = Matrix.from([
        [1, 0, 0],
        [2, 1, 0],
        [3, 5, 1]
      ]);
      expect(L1.cond1Lower(true)).toBeLessThanOrEqual(exactCond1(Lunit) * (1 + 1e-9));
    });

    it('exactly matches the true condition number for a clearly ill-conditioned (near-singular) matrix', () => {
      const U = randomTriangular(6, true, 999);
      U.set(5, 5, 1e-12); // force a near-singular direction that dominates the estimate
      const exact = exactCond1(U);
      const est = U.cond1Upper();
      expect(est / exact).toBeCloseTo(1, 2);
    });

    it('never overestimates the true condition number (lower-bound property), across many random triangular matrices', () => {
      for (let n = 1; n <= 10; n++) {
        for (let trial = 0; trial < 20; trial++) {
          const seedBase = n * 1000 + trial;
          const U = randomTriangular(n, true, seedBase);
          const L = randomTriangular(n, false, seedBase + 500);

          const exactU = exactCond1(U);
          const estU = U.cond1Upper();
          expect(estU).toBeLessThanOrEqual(exactU * (1 + 1e-9));

          const exactL = exactCond1(L);
          const estL = L.cond1Lower();
          expect(estL).toBeLessThanOrEqual(exactL * (1 + 1e-9));
        }
      }
    });

    it('stays within a reasonable factor of the true condition number for well-conditioned random matrices', () => {
      for (let n = 2; n <= 8; n++) {
        const U = randomTriangular(n, true, n * 37 + 1);
        const exact = exactCond1(U);
        const est = U.cond1Upper();
        // The classic CMSW/LINPACK estimator is a heuristic, not exact, but
        // should stay within an order of magnitude for well-conditioned matrices.
        expect(est / exact).toBeGreaterThan(0.1);
        expect(est / exact).toBeLessThanOrEqual(1 + 1e-9);
      }
    });

    it('throws RangeError for a non-square matrix', () => {
      expect(() => new Matrix(2, 3).cond1Upper()).toThrow(RangeError);
      expect(() => new Matrix(3, 2).cond1Lower()).toThrow(RangeError);
    });
  });

  describe('repeat()', () => {
    // Golden values below follow numpy's np.repeat on [[1, 2, 3], [4, 5, 6]].
    const A23 = (): Matrix => new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);

    it('with no axis, flattens in row-major order and returns a Vector (numpy)', () => {
      const r = A23().repeat(2);
      expect(r).toBeInstanceOf(Vector);
      expect(r.toArray()).toEqual([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]);
    });

    it('with no axis, repeat(0) returns an empty Vector', () => {
      expect(A23().repeat(0).size).toBe(0);
    });

    it('axis 0 repeats each row in place', () => {
      const r = A23().repeat(2, 0);
      expect(r).toBeInstanceOf(Matrix);
      expect(r.rows).toBe(4);
      expect(r.cols).toBe(3);
      expect(r.toArray()).toEqual([
        [1, 2, 3],
        [1, 2, 3],
        [4, 5, 6],
        [4, 5, 6]
      ]);
    });

    it('axis 1 repeats each column in place', () => {
      const r = A23().repeat(2, 1);
      expect(r.rows).toBe(2);
      expect(r.cols).toBe(6);
      expect(r.toArray()).toEqual([
        [1, 1, 2, 2, 3, 3],
        [4, 4, 5, 5, 6, 6]
      ]);
    });

    it('supports counts above 2 on a non-square matrix, on both axes', () => {
      expect(A23().repeat(3, 0).toArray()).toEqual([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [4, 5, 6],
        [4, 5, 6],
        [4, 5, 6]
      ]);
      expect(A23().repeat(3, 1).toArray()).toEqual([
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]
      ]);
    });

    it('repeat(1, axis) equals the original, as an independent copy', () => {
      for (const axis of [0, 1] as const) {
        const a = A23();
        const r = a.repeat(1, axis);
        expect(r.toArray()).toEqual(a.toArray());
        expect(r.data).not.toBe(a.data);
      }
    });

    it('handles single-row, single-column and 1x1 matrices', () => {
      expect(new Matrix(1, 2, [1, 2]).repeat(2, 0).toArray()).toEqual([
        [1, 2],
        [1, 2]
      ]);
      expect(new Matrix(1, 2, [1, 2]).repeat(2, 1).toArray()).toEqual([[1, 1, 2, 2]]);
      expect(new Matrix(2, 1, [1, 2]).repeat(2, 0).toArray()).toEqual([[1], [1], [2], [2]]);
      expect(new Matrix(2, 1, [1, 2]).repeat(2, 1).toArray()).toEqual([
        [1, 1],
        [2, 2]
      ]);
      expect(new Matrix(1, 1, [5]).repeat(3, 0).toArray()).toEqual([[5], [5], [5]]);
      expect(new Matrix(1, 1, [5]).repeat(3, 1).toArray()).toEqual([[5, 5, 5]]);
    });

    it('agrees with row() / Vector.repeat() on both axes (cross-check)', () => {
      // Axis 0: each source row must appear `repeats` times in a row.
      const a = new Matrix(3, 2, [1, 2, 3, 4, 5, 6]);
      const r = a.repeat(3, 0);
      for (let i = 0; i < 3; i++) {
        for (let k = 0; k < 3; k++) expect(r.row(i * 3 + k).toArray()).toEqual(a.row(i).toArray());
      }
      // And axis 1 against Vector.repeat applied per row.
      const c = a.repeat(3, 1);
      for (let i = 0; i < 3; i++) expect(c.row(i).toArray()).toEqual(a.row(i).repeat(3).toArray());
    });

    it('does not mutate the original and does not alias its buffer', () => {
      const a = A23();
      for (const axis of [undefined, 0, 1] as const) {
        const r = axis === undefined ? a.repeat(2) : a.repeat(2, axis);
        r.data[0] = 999;
        expect(a.toArray()).toEqual([
          [1, 2, 3],
          [4, 5, 6]
        ]);
      }
    });

    it('throws RangeError for repeats=0 along an axis, since Matrix cannot be empty', () => {
      expect(() => A23().repeat(0, 0)).toThrowError(RangeError);
      expect(() => A23().repeat(0, 1)).toThrowError(RangeError);
      expect(() => A23().repeat(0, 0)).toThrowError(/Matrix\.repeat/);
    });

    it('rejects counts that are negative, fractional, NaN or infinite, with and without axis', () => {
      for (const bad of [-1, 1.5, NaN, Infinity]) {
        expect(() => A23().repeat(bad)).toThrowError(RangeError);
        expect(() => A23().repeat(bad, 0)).toThrowError(RangeError);
        expect(() => A23().repeat(bad, 1)).toThrowError(RangeError);
      }
    });

    it('rejects an invalid axis', () => {
      expect(() => A23().repeat(2, 2 as never)).toThrowError(RangeError);
      expect(() => A23().repeat(2, -1 as never)).toThrowError(RangeError);
    });
  });

  describe('tile()', () => {
    // Golden values below follow numpy's np.tile on [[1, 2, 3], [4, 5, 6]].
    const A23 = (): Matrix => new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);

    it('a scalar tiles along the columns only (numpy promotes it to (1, reps))', () => {
      const t = A23().tile(2);
      expect(t.rows).toBe(2);
      expect(t.cols).toBe(6);
      expect(t.toArray()).toEqual([
        [1, 2, 3, 1, 2, 3],
        [4, 5, 6, 4, 5, 6]
      ]);
    });

    it('a [rowReps, colReps] pair tiles along both axes', () => {
      expect(A23().tile([2, 2]).toArray()).toEqual([
        [1, 2, 3, 1, 2, 3],
        [4, 5, 6, 4, 5, 6],
        [1, 2, 3, 1, 2, 3],
        [4, 5, 6, 4, 5, 6]
      ]);
      expect(A23().tile([2, 1]).toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6]
      ]);
    });

    it('produces the right shape for unequal counts', () => {
      const t = A23().tile([3, 2]);
      expect(t.rows).toBe(6);
      expect(t.cols).toBe(6);
      // Every block must equal the original.
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 2; b++) {
          expect(t.slice(a * 2, a * 2 + 2, b * 3, b * 3 + 3).toArray()).toEqual(A23().toArray());
        }
      }
    });

    it('tile(1) and tile([1, 1]) equal the original, as independent copies', () => {
      for (const reps of [1, [1, 1] as [number, number]]) {
        const a = A23();
        const t = a.tile(reps);
        expect(t.toArray()).toEqual(a.toArray());
        expect(t.data).not.toBe(a.data);
      }
    });

    it('handles single-row, single-column and 1x1 matrices', () => {
      expect(new Matrix(1, 2, [1, 2]).tile([2, 2]).toArray()).toEqual([
        [1, 2, 1, 2],
        [1, 2, 1, 2]
      ]);
      expect(new Matrix(2, 1, [1, 2]).tile([2, 3]).toArray()).toEqual([
        [1, 1, 1],
        [2, 2, 2],
        [1, 1, 1],
        [2, 2, 2]
      ]);
      expect(new Matrix(1, 1, [5]).tile([2, 2]).toArray()).toEqual([
        [5, 5],
        [5, 5]
      ]);
    });

    it('agrees with hstack/vstack of copies (independent cross-check)', () => {
      const a = A23();
      const viaStack = Matrix.vstack([Matrix.hstack([a, a, a]), Matrix.hstack([a, a, a])]);
      expect(a.tile([2, 3]).toArray()).toEqual(viaStack.toArray());
    });

    it('agrees with Vector.tile on each row for a scalar count', () => {
      const a = A23();
      const t = a.tile(3);
      for (let i = 0; i < a.rows; i++)
        expect(t.row(i).toArray()).toEqual(a.row(i).tile(3).toArray());
    });

    it('does not mutate the original and does not alias its buffer', () => {
      const a = A23();
      const t = a.tile([2, 2]);
      t.set(0, 0, 999);
      expect(a.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6]
      ]);
      // The replicated copy of (0, 0) must be a separate element, too.
      expect(t.get(2, 0)).toBe(1);
      expect(t.get(0, 3)).toBe(1);
    });

    it('throws RangeError when either count is 0, since Matrix cannot be empty', () => {
      expect(() => A23().tile(0)).toThrowError(RangeError);
      expect(() => A23().tile([0, 2])).toThrowError(RangeError);
      expect(() => A23().tile([2, 0])).toThrowError(RangeError);
      expect(() => A23().tile(0)).toThrowError(/Matrix\.tile/);
    });

    it('rejects counts that are negative, fractional, NaN or infinite', () => {
      for (const bad of [-1, 1.5, NaN, Infinity]) {
        expect(() => A23().tile(bad)).toThrowError(RangeError);
        expect(() => A23().tile([bad, 1])).toThrowError(RangeError);
        expect(() => A23().tile([1, bad])).toThrowError(RangeError);
      }
    });
  });
});

describe('Matrix: validation and edge cases', () => {
  const A22 = () => new Matrix(2, 2, [1, 2, 3, 4]);
  const A33 = () => new Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);

  describe('indices must be integers in range', () => {
    // NaN and fractional indices used to address a wrong element (or none) silently, and
    // `swapRowsSelf(NaN, 0)` overwrote row 0 with NaN.
    const bad = [NaN, 0.5, 1.5, Infinity, -Infinity, -1, 2];

    it('get, set, row, col throw RangeError and leave the matrix untouched', () => {
      for (const b of bad) {
        const m = A22();
        expect(() => m.get(b, 0)).toThrowError(RangeError);
        expect(() => m.get(0, b)).toThrowError(RangeError);
        expect(() => m.set(b, 0, 9)).toThrowError(RangeError);
        expect(() => m.set(0, b, 9)).toThrowError(RangeError);
        expect(() => m.row(b)).toThrowError(RangeError);
        expect(() => m.col(b)).toThrowError(RangeError);
        expect(m.toArray()).toEqual(A22().toArray());
      }
    });

    it('setRow, setCol, swapRowsSelf, scaleRowSelf, scaleColSelf, addScaledRowSelf throw and do not mutate', () => {
      for (const b of bad) {
        const m = A22();
        expect(() => m.setRow(b, [7, 8])).toThrowError(RangeError);
        expect(() => m.setCol(b, [7, 8])).toThrowError(RangeError);
        expect(() => m.swapRowsSelf(b, 0)).toThrowError(RangeError);
        expect(() => m.swapRowsSelf(0, b)).toThrowError(RangeError);
        expect(() => m.scaleRowSelf(b, 2)).toThrowError(RangeError);
        expect(() => m.scaleColSelf(b, 2)).toThrowError(RangeError);
        expect(() => m.addScaledRowSelf(b, 0, 2)).toThrowError(RangeError);
        expect(() => m.addScaledRowSelf(0, b, 2)).toThrowError(RangeError);
        expect(m.toArray()).toEqual(A22().toArray());
      }
    });

    it('accepts -0 as index 0', () => {
      expect(A22().get(-0, -0)).toBe(1);
    });

    it('uses the "Matrix.method: ... ." message format', () => {
      expect(() => A22().get(NaN, 0)).toThrowError(
        'Matrix.get: index (NaN, 0) out of bounds for a 2x2 matrix.'
      );
      expect(() => A22().row(2)).toThrowError('Matrix.row: row index 2 out of bounds for 2 rows.');
      expect(() => A22().col(0.5)).toThrowError(
        'Matrix.col: column index 0.5 out of bounds for 2 columns.'
      );
    });
  });

  describe('matmul propagates non-finite values like IEEE multiplication', () => {
    // Zero entries used to be skipped, so 0 * NaN and 0 * Infinity were lost.
    it('0 * NaN is NaN', () => {
      expect(new Matrix(1, 2, [0, 1]).matmul(new Matrix(2, 1, [NaN, 1])).get(0, 0)).toBeNaN();
    });

    it('0 * Infinity is NaN', () => {
      expect(new Matrix(1, 2, [0, 1]).matmul(new Matrix(2, 1, [Infinity, 1])).get(0, 0)).toBeNaN();
    });

    it('agrees with mulVec on non-finite input', () => {
      const m = new Matrix(1, 2, [0, 1]);
      const v = new Vector([Infinity, 1]);
      expect(m.matmul(new Matrix(2, 1, [Infinity, 1])).get(0, 0)).toBe(m.mulVec(v).get(0));
    });

    it('is unchanged for finite input, including zero entries, and A.matmul(A) works', () => {
      const m = new Matrix(2, 2, [0, 2, 3, 0]);
      expect(m.matmul(m).toArray()).toEqual([
        [6, 0],
        [0, 6]
      ]);
      expect(new Matrix(2, 2, [0, 0, 0, 0]).matmul(A22()).toArray()).toEqual([
        [0, 0],
        [0, 0]
      ]);
    });
  });

  describe('predicates treat NaN as neither equal nor zero and validate atol', () => {
    it('isSymmetric is false when a mirrored pair contains NaN', () => {
      expect(new Matrix(2, 2, [1, NaN, NaN, 1]).isSymmetric()).toBe(false);
      expect(new Matrix(2, 2, [1, NaN, 2, 1]).isSymmetric()).toBe(false);
      expect(new Matrix(2, 2, [1, NaN, 2, 1]).isSymmetric(10)).toBe(false);
    });

    it('isSymmetric accepts equal infinities and rejects opposite ones', () => {
      expect(new Matrix(2, 2, [1, Infinity, Infinity, 1]).isSymmetric()).toBe(true);
      expect(new Matrix(2, 2, [1, Infinity, -Infinity, 1]).isSymmetric()).toBe(false);
    });

    it('a NaN diagonal entry does not affect symmetry (only off-diagonal pairs are compared)', () => {
      expect(new Matrix(2, 2, [NaN, 1, 1, NaN]).isSymmetric()).toBe(true);
    });

    it('isLowerTriangular / isUpperTriangular are false for a NaN in the zero part', () => {
      expect(new Matrix(2, 2, [1, NaN, 0, 1]).isLowerTriangular()).toBe(false);
      expect(new Matrix(2, 2, [1, 0, NaN, 1]).isUpperTriangular()).toBe(false);
      expect(new Matrix(2, 2, [1, Infinity, 0, 1]).isLowerTriangular()).toBe(false);
      // NaN on the kept side is irrelevant.
      expect(new Matrix(2, 2, [NaN, 0, NaN, 1]).isLowerTriangular()).toBe(true);
    });

    it('reject a NaN or negative atol with RangeError, even for non-square matrices', () => {
      for (const atol of [NaN, -1]) {
        expect(() => A22().isSymmetric(atol)).toThrowError(RangeError);
        expect(() => A22().isLowerTriangular(atol)).toThrowError(RangeError);
        expect(() => A22().isUpperTriangular(atol)).toThrowError(RangeError);
        expect(() => new Matrix(2, 3).isSymmetric(atol)).toThrowError(RangeError);
      }
      expect(() => A22().isSymmetric(NaN)).toThrowError(/^Matrix\.isSymmetric: atol must be >= 0/);
    });

    it('still honour a positive atol', () => {
      expect(new Matrix(2, 2, [1, 2, 2.05, 1]).isSymmetric(0.1)).toBe(true);
      expect(new Matrix(2, 2, [1, 0.05, 0, 1]).isUpperTriangular(0.1)).toBe(true);
      expect(new Matrix(2, 2, [1, 0.05, 0, 1]).isLowerTriangular(0.1)).toBe(true);
      expect(new Matrix(2, 2, [1, 0.5, 0, 1]).isLowerTriangular(0.1)).toBe(false);
    });
  });

  describe('axis must be 0 or 1', () => {
    // Any other value (2, -1, null, '0') used to fall through to the row-wise path.
    const A = () => new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
    const badAxes = [2, -1, null, '0', NaN] as unknown as (0 | 1)[];

    it('every axis-aware method throws RangeError on an invalid axis', () => {
      for (const axis of badAxes) {
        expect(() => A().sum(axis)).toThrowError(RangeError);
        expect(() => A().mean(axis)).toThrowError(RangeError);
        expect(() => A().min(axis)).toThrowError(RangeError);
        expect(() => A().max(axis)).toThrowError(RangeError);
        expect(() => A().variance(axis)).toThrowError(RangeError);
        expect(() => A().std(axis)).toThrowError(RangeError);
        expect(() => A().argmin(axis)).toThrowError(RangeError);
        expect(() => A().argmax(axis)).toThrowError(RangeError);
        expect(() => A().cumsum(axis)).toThrowError(RangeError);
        expect(() => A().sort(axis)).toThrowError(RangeError);
        expect(() => A().repeat(1, axis)).toThrowError(RangeError);
      }
    });

    it('names the method and quotes string axes', () => {
      expect(() => A().sum(2 as 0)).toThrowError(
        'Matrix.sum: axis must be 0, 1 or undefined, got 2.'
      );
      expect(() => A().sort('0' as unknown as 0)).toThrowError(
        'Matrix.sort: axis must be 0 or 1, got "0".'
      );
    });

    it('undefined still selects the whole-matrix variant and valid axes still work', () => {
      expect(A().sum()).toBe(21);
      expect(A().sum(0).toArray()).toEqual([5, 7, 9]);
      expect(A().sum(1).toArray()).toEqual([6, 15]);
      expect(A().sort(undefined).toArray()).toEqual(A().toArray());
    });
  });

  describe('slice() truncates bounds like Array.prototype.slice and Vector.slice', () => {
    // Fractional bounds used to produce a misaligned matrix (e.g. rows 2,3,4 / 5,6,7).
    it('truncates fractional bounds toward zero', () => {
      expect(A33().slice(0.5, 2.5).toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6]
      ]);
      expect(A33().slice(1.5).toArray()).toEqual([
        [4, 5, 6],
        [7, 8, 9]
      ]);
      expect(A33().slice(0, 2, 0, 1.5).toArray()).toEqual([[1], [4]]);
      expect(A33().slice(-1.5).toArray()).toEqual([[7, 8, 9]]); // -1.5 -> -1
    });

    it('treats NaN as 0 and infinite bounds as clamped', () => {
      expect(A33().slice(NaN).toArray()).toEqual(A33().toArray());
      expect(A33().slice(-Infinity, Infinity).toArray()).toEqual(A33().toArray());
      expect(() => A33().slice(0, NaN)).toThrowError(RangeError); // resolves to an empty range
    });

    it('agrees with Vector.slice for the same bounds', () => {
      const v = new Vector([1, 2, 3]);
      const row = new Matrix(1, 3, [1, 2, 3]);
      for (const [s, e] of [
        [0.5, 2.5],
        [NaN, 2],
        [-1.5, undefined],
        [1, 99]
      ] as [number, number | undefined][]) {
        expect(row.slice(0, 1, s, e).toArray()[0]).toEqual(v.slice(s, e).toArray());
      }
    });

    it('returns an independent copy', () => {
      const a = A33();
      const s = a.slice();
      s.set(0, 0, 99);
      expect(a.get(0, 0)).toBe(1);
    });
  });

  describe('tile() argument validation', () => {
    const A = () => new Matrix(1, 2, [1, 2]);

    it('rejects arrays that are not a [rowReps, colReps] pair with TypeError', () => {
      for (const bad of [[], [1], [1, 2, 3]]) {
        expect(() => A().tile(bad as [number, number])).toThrowError(TypeError);
      }
      expect(() => A().tile([1, 2, 3] as unknown as [number, number])).toThrowError(
        'Matrix.tile: reps must be a number or a [rowReps, colReps] pair, got an array of length 3.'
      );
    });

    it('rejects non-number, non-array arguments with TypeError', () => {
      for (const bad of [undefined, null, '2', {}]) {
        expect(() => A().tile(bad as unknown as number)).toThrowError(TypeError);
      }
    });

    it('still accepts a number and a pair', () => {
      expect(A().tile(2).toArray()).toEqual([[1, 2, 1, 2]]);
      expect(A().tile([2, 1]).toArray()).toEqual([
        [1, 2],
        [1, 2]
      ]);
    });
  });

  describe('Matrix.from() validation', () => {
    it('reports a ragged row as such, not as a setRow error', () => {
      expect(() => Matrix.from([[1, 2], [3]])).toThrowError(
        'Matrix.from: row 1 has 1 elements, expected 2 (as in row 0).'
      );
      expect(() => Matrix.from([[1], [2, 3]])).toThrowError(RangeError);
    });

    it('rejects a non-array argument and non-array rows with TypeError', () => {
      expect(() => Matrix.from(null as unknown as number[][])).toThrowError(TypeError);
      expect(() => Matrix.from(A22() as unknown as number[][])).toThrowError(TypeError);
      expect(() => Matrix.from([1, 2, 3] as unknown as number[][])).toThrowError(
        'Matrix.from: row 0 is not an array of numbers or a Vector, got number.'
      );
      expect(() => Matrix.from([[1], null] as unknown as number[][])).toThrowError(TypeError);
    });

    it('copies the row data', () => {
      const r = [1, 2];
      const m = Matrix.from([r, [3, 4]]);
      r[0] = 99;
      expect(m.toArray()).toEqual([
        [1, 2],
        [3, 4]
      ]);
    });
  });

  describe('Matrix.hstack() / Matrix.vstack() argument validation', () => {
    it('rejects a non-array and non-Matrix elements with TypeError', () => {
      for (const stack of [Matrix.hstack, Matrix.vstack]) {
        expect(() => stack(null as unknown as Matrix[])).toThrowError(TypeError);
        expect(() => stack([A22(), undefined] as unknown as Matrix[])).toThrowError(TypeError);
        expect(() => stack([A22(), new Vector([1, 2])] as unknown as Matrix[])).toThrowError(
          /element 1 is not a Matrix, got Vector/
        );
        expect(() => stack([])).toThrowError(RangeError);
      }
    });

    it('results never share storage with the inputs', () => {
      const a = A22();
      for (const stacked of [Matrix.hstack([a]), Matrix.vstack([a]), Matrix.hstack([a, a])]) {
        stacked.set(0, 0, 99);
        expect(a.get(0, 0)).toBe(1);
      }
    });
  });

  describe('operands that are not matrices', () => {
    it('elementwise operations and comparisons throw TypeError instead of a garbled shape error', () => {
      const m = A22();
      const bad = [[1, 2], null, 'x', {}] as unknown as Matrix[];
      for (const x of bad) {
        expect(() => m.add(x)).toThrowError(TypeError);
        expect(() => m.subSelf(x)).toThrowError(TypeError);
        expect(() => m.isClose(x)).toThrowError(TypeError);
        expect(() => m.dot(x)).toThrowError(TypeError);
      }
      expect(() => m.add([1, 2] as unknown as Matrix)).toThrowError(
        'Matrix.add: expected a Matrix, a Vector or a number, got Array.'
      );
    });
  });

  describe('aliasing', () => {
    it('addSelf/multSelf with a Vector viewing the matrix own first row use the original values', () => {
      // The row-0 view used to be read after row 0 had already been overwritten.
      const mk = () => new Matrix(3, 2, [1, 2, 3, 4, 5, 6]);
      const a = mk();
      a.addSelf(Vector.wrapUnchecked(a.data.subarray(0, 2)));
      expect(a.toArray()).toEqual(
        mk()
          .add(new Vector([1, 2]))
          .toArray()
      );

      const b = mk();
      b.multSelf(Vector.wrapUnchecked(b.data.subarray(0, 2)));
      expect(b.toArray()).toEqual(
        mk()
          .mult(new Vector([1, 2]))
          .toArray()
      );
    });

    it('setCol with a source that overlaps the matrix buffer reads the original values', () => {
      const a = A33();
      a.setCol(1, a.data.subarray(0, 3)); // column 1 <- (1, 2, 3)
      expect(a.toArray()).toEqual([
        [1, 1, 3],
        [4, 2, 6],
        [7, 3, 9]
      ]);
      const b = A33();
      b.setCol(0, b.data.subarray(1, 4)); // column 0 <- (2, 3, 4)
      expect(b.toArray()).toEqual([
        [2, 2, 3],
        [3, 5, 6],
        [4, 8, 9]
      ]);
    });

    it('setRow with an overlapping source reads the original values', () => {
      const a = A22();
      a.setRow(1, a.data.subarray(1, 3));
      expect(a.toArray()).toEqual([
        [1, 2],
        [2, 3]
      ]);
    });

    it('A.addSelf(A), A.subSelf(A), A.multSelf(A), A.divSelf(A) are elementwise', () => {
      const a = A22();
      expect(a.addSelf(a).toArray()).toEqual([
        [2, 4],
        [6, 8]
      ]);
      const b = A22();
      expect(b.subSelf(b).toArray()).toEqual([
        [0, 0],
        [0, 0]
      ]);
      const c = A22();
      expect(c.multSelf(c).toArray()).toEqual([
        [1, 4],
        [9, 16]
      ]);
      const d = A22();
      expect(d.divSelf(d).toArray()).toEqual([
        [1, 1],
        [1, 1]
      ]);
    });

    it('A.matmul(A) returns a fresh matrix and leaves A unchanged', () => {
      const a = A22();
      const p = a.matmul(a);
      expect(p).not.toBe(a);
      expect(p.toArray()).toEqual([
        [7, 10],
        [15, 22]
      ]);
      expect(a.toArray()).toEqual(A22().toArray());
    });
  });
});

describe('Matrix: operand and argument types', () => {
  const I2 = () => Matrix.identity(2);

  it('matmul throws TypeError for a non-Matrix operand', () => {
    expect(() => I2().matmul(null as unknown as Matrix)).toThrowError(
      'Matrix.matmul: expected a Matrix, got null.'
    );
    expect(() =>
      I2().matmul([
        [1, 0],
        [0, 1]
      ] as unknown as Matrix)
    ).toThrowError('Matrix.matmul: expected a Matrix, got Array.');
    expect(() => I2().matmul(new Vector([1, 2]) as unknown as Matrix)).toThrowError(TypeError);
  });

  it('setRow and setCol throw TypeError (and leave the matrix unchanged) for non-array-like values', () => {
    for (const bad of [null, undefined, 5, '12', {}, () => 0] as unknown as number[][]) {
      const m = I2();
      expect(() => m.setRow(0, bad)).toThrowError(TypeError);
      expect(() => m.setCol(0, bad)).toThrowError(TypeError);
      expect(m.toArray()).toEqual(I2().toArray());
    }
    expect(() => I2().setRow(0, null as unknown as number[])).toThrowError(
      'Matrix.setRow: values must be an array-like of numbers or a Vector, got null.'
    );
  });

  it('setRow and setCol still accept arrays, typed arrays and vectors', () => {
    const m = new Matrix(2, 2);
    m.setRow(0, [1, 2]).setRow(1, new Float64Array([3, 4]));
    expect(m.toArray()).toEqual([
      [1, 2],
      [3, 4]
    ]);
    m.setCol(0, new Vector([9, 8]));
    expect(m.toArray()).toEqual([
      [9, 2],
      [8, 4]
    ]);
  });

  it('tolerances must be numbers: null and strings are a TypeError, not 0 or a coerced value', () => {
    for (const atol of [null, '1', {}] as unknown as number[]) {
      expect(() => I2().isSymmetric(atol)).toThrowError(TypeError);
      expect(() => I2().isLowerTriangular(atol)).toThrowError(TypeError);
      expect(() => I2().isUpperTriangular(atol)).toThrowError(TypeError);
    }
    expect(() => I2().isSymmetric(null as unknown as number)).toThrowError(
      'Matrix.isSymmetric: atol must be a number, got null.'
    );
    // undefined still selects the default tolerance of 0.
    expect(I2().isSymmetric(undefined)).toBe(true);
  });
});

describe('Matrix: non-finite values in axis-wise reductions, cumsum and sort', () => {
  const N = () => new Matrix(2, 3, [1, NaN, 3, Infinity, -Infinity, 2]);

  it('sum and mean propagate NaN and cancel opposite infinities to NaN', () => {
    expect(N().sum(0).toArray()).toEqual([Infinity, NaN, 5]);
    expect(N().sum(1).toArray()).toEqual([NaN, NaN]);
    expect(N().mean(0).toArray()).toEqual([Infinity, NaN, 2.5]);
  });

  it('min and max return NaN for a slice containing NaN, else the extreme (infinities included)', () => {
    expect(N().min(0).toArray()).toEqual([1, NaN, 2]);
    expect(N().min(1).toArray()).toEqual([NaN, -Infinity]);
    expect(N().max(0).toArray()).toEqual([Infinity, NaN, 3]);
    expect(N().max(1).toArray()).toEqual([NaN, Infinity]);
  });

  it('argmin and argmax give NaN priority (the index of the first NaN)', () => {
    expect(N().argmin(1).toArray()).toEqual([1, 1]);
    expect(N().argmax(1).toArray()).toEqual([1, 0]);
    expect(N().argmax(0).toArray()).toEqual([1, 0, 0]);
  });

  it('cumsum propagates NaN along the accumulation direction only', () => {
    expect(N().cumsum(0).toArray()).toEqual([
      [1, NaN, 3],
      [Infinity, NaN, 5]
    ]);
    expect(N().cumsum(1).toArray()).toEqual([
      [1, NaN, NaN],
      [Infinity, NaN, NaN]
    ]);
  });

  it('sort orders -Infinity < finite < Infinity < NaN in both directions of axis', () => {
    expect(new Matrix(1, 5, [3, NaN, -Infinity, Infinity, 0]).sort().toArray()).toEqual([
      [-Infinity, 0, 3, Infinity, NaN]
    ]);
    expect(new Matrix(3, 2, [3, 1, NaN, 0, -Infinity, 2]).sort(0).toArray()).toEqual([
      [-Infinity, 0],
      [3, 1],
      [NaN, 2]
    ]);
  });
});

describe('Matrix: input types, naming and error format', () => {
  const A22 = () => new Matrix(2, 2, [2, 1, 1, 3]);

  describe('vector-shaped arguments accept every VectorLike form', () => {
    for (const [form, make] of vectorLikeForms) {
      describe(form, () => {
        it('mulVec', () => {
          expect(new Matrix(2, 3, [1, 2, 3, 4, 5, 6]).mulVec(make([1, 0, 1])).toArray()).toEqual([
            4, 10
          ]);
        });

        it('Matrix.diag', () => {
          expect(Matrix.diag(make([1, 2])).toArray()).toEqual([
            [1, 0],
            [0, 2]
          ]);
        });

        it('solve, solveLower, solveUpper, solveCholesky, solveLU', () => {
          const expected = A22()
            .solve(new Vector([5, 10]))
            .toArray();
          expect(
            A22()
              .solve(make([5, 10]))
              .toArray()
          ).toEqual(expected);
          expect(
            A22()
              .solveCholesky(make([5, 10]))
              .toArray()
          ).toEqual(
            A22()
              .solveCholesky(new Vector([5, 10]))
              .toArray()
          );
          const lower = new Matrix(2, 2, [2, 0, 1, 3]);
          expect(lower.solveLower(make([4, 5])).toArray()).toEqual(
            lower.solveLower(new Vector([4, 5])).toArray()
          );
          const upper = new Matrix(2, 2, [2, 1, 0, 3]);
          expect(upper.solveUpper(make([4, 6])).toArray()).toEqual(
            upper.solveUpper(new Vector([4, 6])).toArray()
          );
          expect(Matrix.solveLU(A22().lu(), make([5, 10])).toArray()).toEqual(expected);
        });

        it('qrUpdate and qrUpdateSelf', () => {
          const mk = () => A22().qr();
          const ref = Matrix.qrUpdate(mk(), new Vector([1, 0]), new Vector([0, 1]));
          const got = Matrix.qrUpdate(mk(), make([1, 0]), make([0, 1]));
          expect(got.R.toArray()).toEqual(ref.R.toArray());
          const self = Matrix.qrUpdateSelf(mk(), make([1, 0]), make([0, 1]));
          expect(self.Q.toArray()).toEqual(ref.Q.toArray());
        });

        it('qrPivotMulQt', () => {
          const dec = A22().qrPivot();
          expect(Matrix.qrPivotMulQt(dec, make([1, 2])).toArray()).toEqual(
            Matrix.qrPivotMulQt(dec, new Vector([1, 2])).toArray()
          );
        });
      });
    }

    it('a plain array operand is not modified and out === b still aliases for a Vector', () => {
      const b = [4, 6];
      const upper = new Matrix(2, 2, [2, 1, 0, 3]);
      upper.solveUpper(b);
      expect(b).toEqual([4, 6]);
      const v = new Vector([4, 6]);
      expect(upper.solveUpper(v, v)).toBe(v);
    });

    it('wrong container types throw TypeError with the Vector.asVector message', () => {
      expect(() => A22().mulVec('ab' as unknown as number[])).toThrowError(TypeError);
      expect(() => A22().solve(null as unknown as number[])).toThrowError(/Vector\.asVector/);
      expect(() => Matrix.diag(undefined as unknown as number[])).toThrowError(TypeError);
    });
  });

  describe('constructor input', () => {
    it('accepts a Vector, Float64Array and plain array and copies the data', () => {
      for (const [, make] of vectorLikeForms) {
        const src = make([1, 2, 3, 4]);
        const m = new Matrix(2, 2, src as Vector | number[]);
        expect(m.toArray()).toEqual([
          [1, 2],
          [3, 4]
        ]);
        m.set(0, 0, 99);
        expect(src instanceof Vector ? src.get(0) : (src as number[])[0]).toBe(1);
      }
    });

    it('rejects null and non-array-like input with TypeError, wrong length with RangeError', () => {
      expect(() => new Matrix(2, 2, null as unknown as number[])).toThrowError(TypeError);
      expect(() => new Matrix(2, 2, 5 as unknown as number[])).toThrowError(TypeError);
      expect(() => new Matrix(2, 2, new Vector(3))).toThrowError(RangeError);
    });

    it('the constructor rejects a string input (it would be coerced digit by digit)', () => {
      expect(() => new Matrix(1, 2, '12' as unknown as number[])).toThrowError(
        'Matrix.constructor: input must be an array-like of numbers or a Vector, got string.'
      );
    });

    it('uses the "Matrix.constructor: ... ." message format', () => {
      expect(() => new Matrix(0, 2)).toThrowError(
        'Matrix.constructor: rows must be a positive integer, got 0.'
      );
      expect(() => new Matrix(2, 1.5)).toThrowError(
        'Matrix.constructor: cols must be a positive integer, got 1.5.'
      );
      expect(() => new Matrix(2, 2, [1])).toThrowError(
        'Matrix.constructor: expected 4 values for a 2x2 matrix, got 1.'
      );
    });
  });

  describe('Matrix.from', () => {
    it('accepts rows given as number[], Float64Array and Vector, mixed', () => {
      const m = Matrix.from([[1, 2], new Float64Array([3, 4]), new Vector([5, 6])]);
      expect(m.toArray()).toEqual([
        [1, 2],
        [3, 4],
        [5, 6]
      ]);
    });

    it('accepts a frozen (readonly) array of rows', () => {
      expect(Matrix.from(Object.freeze([Object.freeze([1, 2])])).toArray()).toEqual([[1, 2]]);
    });

    it('rejects a ragged mix and non-row elements', () => {
      expect(() => Matrix.from([new Vector([1, 2]), [3]])).toThrowError(RangeError);
      expect(() => Matrix.from([new Vector([1, 2]), 5 as unknown as number[]])).toThrowError(
        /row 1 is not an array of numbers or a Vector, got number/
      );
    });
  });

  describe('error messages follow "Matrix.method: ... ."', () => {
    it('shape and square-matrix errors', () => {
      expect(() => new Matrix(2, 3).matmul(new Matrix(2, 3))).toThrowError(
        'Matrix.matmul: shape mismatch: 2x3 * 2x3.'
      );
      expect(() => new Matrix(2, 3).mulVec(new Vector(2))).toThrowError(
        'Matrix.mulVec: shape mismatch: 2x3 * vector of size 2.'
      );
      expect(() => new Matrix(2, 3).trace()).toThrowError(
        'Matrix.trace: expected a square matrix, got 2x3.'
      );
      expect(() => new Matrix(2, 3).transposeSelf()).toThrowError(
        'Matrix.transposeSelf: expected a square matrix, got 2x3.'
      );
    });

    it('linalg and svd errors follow the same format and name the method that was called', () => {
      const sq = Matrix.from([
        [2, 1],
        [1, 3]
      ]);
      const rect = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
      const singular = new Matrix(2, 2, [1, 2, 2, 4]);
      const dec = sq.lu();
      const cases: [string, () => unknown][] = [
        ['determinant', () => rect.determinant()],
        ['inverse', () => rect.inverse()],
        ['inverse', () => singular.inverse()],
        ['lu', () => rect.lu()],
        ['lu', () => singular.lu()],
        ['solve', () => rect.solve([1, 2])],
        ['solve', () => sq.solve([1])],
        ['solveLower', () => sq.solveLower([1])],
        ['solveUpper', () => sq.solveUpper([1, 1], new Vector(1))],
        ['solveLU', () => Matrix.solveLU(dec, [1])],
        ['cond1', () => rect.cond1()],
        ['cholesky', () => rect.cholesky()],
        ['cholesky', () => singular.cholesky()],
        ['solveCholesky', () => sq.solveCholesky([1])],
        ['qrPivot', () => sq.qrPivot(rect.qrPivot())],
        ['eigenvalues', () => rect.eigenvalues()],
        ['eig', () => rect.eig()],
        ['rank', () => rect.rank({ atol: -1 })],
        ['rank', () => rect.rank({ method: 'x' as 'svd' })],
        ['lstsq', () => rect.lstsq([1])],
        ['pinv', () => rect.pinv({ rtol: -1 })],
        ['svd', () => new Matrix(1, 1, [Infinity]).svd()],
        ['cond2', () => new Matrix(1, 1, [Infinity]).cond2()]
      ];
      for (const [name, call] of cases) {
        expect(call, name).toThrowError(new RegExp(`^Matrix\\.${name}: .*\\.$`));
      }
      expect(() => rect.solve([1, 2])).toThrowError(
        'Matrix.solve: expected a square matrix, got 2x3.'
      );
      expect(() => sq.solve([1])).toThrowError(
        'Matrix.solve: shape mismatch: 2x2 vs vector of size 1.'
      );
    });

    it('every message thrown by a Matrix.ts check starts with "Matrix.<name>: " and ends with a period', () => {
      const m = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
      const calls: (() => unknown)[] = [
        () => m.get(9, 0),
        () => m.row(9),
        () => m.col(9),
        () => m.setRow(0, [1]),
        () => m.setCol(0, [1]),
        () => m.swapRowsSelf(9, 0),
        () => m.scaleRowSelf(9, 1),
        () => m.scaleColSelf(9, 1),
        () => m.addScaledRowSelf(9, 0, 1),
        () => m.add(new Matrix(1, 1)),
        () => m.add(new Vector(2)),
        () => m.sum(2 as 0),
        () => m.slice(2, 1),
        () => m.repeat(0, 0),
        () => m.tile([0, 1]),
        () => m.isSymmetric(-1),
        () => Matrix.hstack([]),
        () => Matrix.vstack([m, new Matrix(1, 2)]),
        () => Matrix.from([]),
        () => Matrix.diag([])
      ];
      for (const call of calls) {
        let message = '';
        try {
          call();
        } catch (e) {
          message = (e as Error).message;
        }
        expect(message).toMatch(/^Matrix\.[A-Za-z]+: .*\.$/);
      }
    });
  });

  describe('in-place row/column operations carry the Self suffix', () => {
    it('swapRowsSelf, scaleRowSelf, scaleColSelf and addScaledRowSelf mutate and return this', () => {
      const m = new Matrix(2, 2, [1, 2, 3, 4]);
      expect(m.swapRowsSelf(0, 1)).toBe(m);
      expect(m.scaleRowSelf(0, 2)).toBe(m);
      expect(m.scaleColSelf(1, 10)).toBe(m);
      expect(m.addScaledRowSelf(1, 0, 1)).toBe(m);
      expect(m.toArray()).toEqual([
        [6, 80],
        [7, 100]
      ]);
    });

    it('the old names no longer exist', () => {
      const m = new Matrix(1, 1) as unknown as Record<string, unknown>;
      for (const name of ['swapRows', 'scaleRow', 'scaleCol', 'addScaledRow']) {
        expect(m[name]).toBeUndefined();
      }
    });
  });
});
