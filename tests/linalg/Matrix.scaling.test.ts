import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';

/**
 * Scale invariance of the factorizations. The squared norms inside a Householder step
 * overflow beyond ~1e154 and underflow below ~1e-154, so `qr` and `qrPivot` used to return
 * non-finite factors for entries around 1e+200 and, worse, a silently wrong factorization
 * (the reflector was skipped because the norm came out as 0) around 1e-200.
 */

/** Deterministic pseudo-random entries in (-1, 1). */
function lcg(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return (state / 4294967296) * 2 - 1;
  };
}

function randomMatrix(rows: number, cols: number, seed: number): Matrix {
  const next = lcg(seed);
  return Matrix.from(Array.from({ length: rows }, () => Array.from({ length: cols }, next)));
}

const finite = (m: Matrix) => m.data.every(Number.isFinite);
const relDiff = (a: Matrix, b: Matrix) => a.sub(b).norm() / b.norm();
const orthogonality = (Q: Matrix) => Q.transpose().matmul(Q).sub(Matrix.identity(Q.cols)).norm();

const SCALES = [1e-200, 1e-150, 1e150, 1e200];
const SHAPES: [string, number, number][] = [
  ['tall', 6, 4],
  ['wide', 4, 6],
  ['square', 5, 5]
];

describe('Householder QR is invariant to the scale of the matrix', () => {
  for (const [name, rows, cols] of SHAPES) {
    for (const s of SCALES) {
      it(`qr, ${name} ${rows}x${cols}, scale ${s}`, () => {
        const A = randomMatrix(rows, cols, 1000 + rows * 10 + cols);
        const ref = A.qr();
        const { Q, R } = A.mult(s).qr();
        expect(finite(Q) && finite(R)).toBe(true);
        expect(orthogonality(Q)).toBeLessThan(1e-13);
        // Q * R reproduces the scaled matrix, and R/s is the factor of the unscaled one
        // (Householder reflectors do not depend on the scale, so even the signs agree).
        expect(relDiff(Q.matmul(R).mult(1 / s), A)).toBeLessThan(1e-14);
        expect(relDiff(R.mult(1 / s), ref.R)).toBeLessThan(1e-13);
      });

      it(`qrPivot, ${name} ${rows}x${cols}, scale ${s}`, () => {
        const A = randomMatrix(rows, cols, 2000 + rows * 10 + cols);
        const ref = A.qrPivot();
        const dec = A.mult(s).qrPivot();
        const Q = Matrix.qrPivotQ(dec);
        const R = Matrix.qrPivotR(dec);
        expect(finite(Q) && finite(R) && dec.tau.data.every(Number.isFinite)).toBe(true);
        // The pivot order is decided by the column norms, which must not saturate.
        expect(dec.perm).toEqual(ref.perm);
        expect(orthogonality(Q)).toBeLessThan(1e-13);
        const permuted = Matrix.from(
          Array.from({ length: rows }, (_, i) => dec.perm.map((j) => A.get(i, j)))
        );
        expect(relDiff(Q.matmul(R).mult(1 / s), permuted)).toBeLessThan(1e-14);
        expect(relDiff(R.mult(1 / s), Matrix.qrPivotR(ref))).toBeLessThan(1e-13);
      });
    }
  }

  it('qr and qrPivot of a rank-deficient matrix reconstruct it at extreme scales', () => {
    const A = randomMatrix(6, 4, 3000);
    for (let i = 0; i < 6; i++) A.set(i, 3, A.get(i, 0) + A.get(i, 1));
    for (const s of SCALES) {
      const { Q, R } = A.mult(s).qr();
      expect(relDiff(Q.matmul(R).mult(1 / s), A)).toBeLessThan(1e-14);
      const dec = A.mult(s).qrPivot();
      const Rp = Matrix.qrPivotR(dec);
      const permuted = Matrix.from(
        Array.from({ length: 6 }, (_, i) => dec.perm.map((j) => A.get(i, j)))
      );
      expect(
        relDiff(
          Matrix.qrPivotQ(dec)
            .matmul(Rp)
            .mult(1 / s),
          permuted
        )
      ).toBeLessThan(1e-14);
      // Pivoting still puts the (numerically) dependent column last.
      expect(Math.abs(Rp.get(3, 3))).toBeLessThan(1e-12 * Math.abs(Rp.get(0, 0)));
    }
  });

  it('handles columns of very different magnitude, each relative to its own size', () => {
    // Column 0 ~ 1e+200, column 1 ~ 1e-200, column 2 ~ 1: no single global scale fits them.
    const base = randomMatrix(6, 3, 4000);
    const magnitudes = [1e200, 1e-200, 1];
    const A = Matrix.from(
      Array.from({ length: 6 }, (_, i) => magnitudes.map((g, j) => base.get(i, j) * g))
    );
    const { Q, R } = A.qr();
    expect(finite(Q) && finite(R)).toBe(true);
    expect(orthogonality(Q)).toBeLessThan(1e-13);
    const QR = Q.matmul(R);
    for (let j = 0; j < 3; j++) {
      const column = (M: Matrix) => Vector.from(Array.from({ length: 6 }, (_, i) => M.get(i, j)));
      expect(column(QR).sub(column(A)).norm() / column(A).norm()).toBeLessThan(1e-14);
    }
    const pivoted = A.qrPivot();
    expect(pivoted.perm).toEqual([0, 2, 1]);
    expect(finite(Matrix.qrPivotR(pivoted))).toBe(true);
  });

  it('still treats exactly-zero columns as zero', () => {
    const A = randomMatrix(5, 4, 5000);
    for (let i = 0; i < 5; i++) A.set(i, 1, 0);
    for (const s of [1, 1e-200, 1e200]) {
      const { Q, R } = A.mult(s).qr();
      expect(relDiff(Q.matmul(R).mult(1 / s), A)).toBeLessThan(1e-14);
      const dec = A.mult(s).qrPivot();
      expect(dec.perm[3]).toBe(1);
      expect(finite(Matrix.qrPivotR(dec))).toBe(true);
    }
  });
});

describe('SVD-based solvers hold up at extreme scales', () => {
  // These already worked (the SVD scales the matrix into a safe range, as LAPACK's dgesvd does);
  // pinned here so that stays true.
  for (const s of SCALES) {
    it(`svd, pinv and lstsq, scale ${s}`, () => {
      const A = randomMatrix(6, 4, 6000);
      const b = Vector.from([0.3, -0.7, 0.2, 0.9, -0.1, 0.5]);
      const ref = { svd: A.svd(), pinv: A.pinv(), ls: A.lstsq(b) };
      const As = A.mult(s);
      const { s: sv } = As.svd();
      sv.data.forEach((x, i) => expect(Math.abs(x / s - ref.svd.s.data[i])).toBeLessThan(1e-14));
      expect(relDiff(As.pinv().mult(s), ref.pinv)).toBeLessThan(1e-13);
      const ls = As.lstsq(b.mult(s));
      expect(ls.x.sub(ref.ls.x).norm() / ref.ls.x.norm()).toBeLessThan(1e-13);
      expect(ls.rank).toBe(ref.ls.rank);
    });
  }
});
