import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import { rand } from '../../src/math.js';

const EPS = Number.EPSILON;

/** Random m x n matrix with entries in [-1, 1). */
const randomMatrix = (m: number, n: number, seed: number): Matrix => {
  const rnd = rand(seed);
  return new Matrix(m, n).map(() => 2 * rnd() - 1);
};

/** Random m x n matrix of rank `r`. */
const lowRank = (m: number, n: number, r: number, seed: number): Matrix =>
  randomMatrix(m, r, seed).matmul(randomMatrix(r, n, seed + 1));

/** Frobenius norm of `U^T U - I`. */
const orthogonalityError = (U: Matrix): number =>
  U.transpose().matmul(U).sub(Matrix.identity(U.cols)).norm();

describe('Matrix SVD family', () => {
  describe('svd()', () => {
    it.each([
      [5, 3],
      [3, 3],
      [3, 5],
      [1, 4],
      [4, 1],
      [1, 1],
      [60, 7],
      [7, 60],
      [40, 40]
    ])('factors a %i x %i matrix: A = U*diag(s)*V^T, orthonormal factors, sorted s', (m, n) => {
      const A = randomMatrix(m, n, 100 + 10 * m + n);
      const k = Math.min(m, n);
      const { U, s, V } = A.svd();

      expect(U.rows).toBe(m);
      expect(U.cols).toBe(k);
      expect(V.rows).toBe(n);
      expect(V.cols).toBe(k);
      expect(s.size).toBe(k);
      for (let i = 0; i < k; i++) {
        expect(s.get(i)).toBeGreaterThanOrEqual(0);
        if (i > 0) expect(s.get(i)).toBeLessThanOrEqual(s.get(i - 1));
      }
      const scale = Math.max(m, n) * EPS * A.norm();
      expect(U.matmul(Matrix.diag(s)).matmul(V.transpose()).sub(A).norm()).toBeLessThan(20 * scale);
      expect(orthogonalityError(U)).toBeLessThan(50 * Math.max(m, n) * EPS);
      expect(orthogonalityError(V)).toBeLessThan(50 * Math.max(m, n) * EPS);
    });

    it.each([
      [6, 3],
      [3, 6],
      [4, 4]
    ])('returns square orthogonal factors for a %i x %i matrix with { full: true }', (m, n) => {
      const A = randomMatrix(m, n, 7 + m + n);
      const k = Math.min(m, n);
      const { U, s, V } = A.svd({ full: true });

      expect(U.rows).toBe(m);
      expect(U.cols).toBe(m);
      expect(V.rows).toBe(n);
      expect(V.cols).toBe(n);
      expect(s.size).toBe(k);
      expect(orthogonalityError(U)).toBeLessThan(50 * m * EPS);
      expect(orthogonalityError(V)).toBeLessThan(50 * n * EPS);
      const Uk = U.slice(0, m, 0, k);
      const Vk = V.slice(0, n, 0, k);
      expect(Uk.matmul(Matrix.diag(s)).matmul(Vk.transpose()).allClose(A, 1e-13, 1e-13)).toBe(true);
    });

    it('thin and full decompositions share the same singular values and leading vectors', () => {
      const A = randomMatrix(7, 4, 3);
      const thin = A.svd();
      const full = A.svd({ full: true });
      expect(thin.s.data).toEqual(full.s.data);
      expect(thin.U.allClose(full.U.slice(0, 7, 0, 4), 1e-12, 1e-12)).toBe(true);
    });

    it('matches closed-form singular values', () => {
      // [[3, 0], [4, 5]] has s = sqrt(45), sqrt(5).
      const s = Matrix.from([
        [3, 0],
        [4, 5]
      ]).svd().s;
      expect(s.get(0)).toBeCloseTo(Math.sqrt(45), 14);
      expect(s.get(1)).toBeCloseTo(Math.sqrt(5), 14);
      // A rotation has unit singular values; a diagonal matrix has |d_i| in sorted order.
      const c = Math.cos(0.7);
      const sn = Math.sin(0.7);
      const rot = Matrix.from([
        [c, -sn],
        [sn, c]
      ]).svd().s;
      expect(rot.get(0)).toBeCloseTo(1, 14);
      expect(rot.get(1)).toBeCloseTo(1, 14);
      expect([...Matrix.diag(new Vector([3, -2, 0, 5])).svd().s.data]).toEqual([5, 3, 2, 0]);
    });

    it('handles the zero matrix, identity, scalars and vectors', () => {
      const zero = new Matrix(4, 3).svd();
      expect([...zero.s.data]).toEqual([0, 0, 0]);
      expect(orthogonalityError(zero.U)).toBeLessThan(1e-14);
      expect(orthogonalityError(zero.V)).toBeLessThan(1e-14);

      const id = Matrix.identity(5).svd();
      expect([...id.s.data]).toEqual([1, 1, 1, 1, 1]);

      const neg = Matrix.from([[-3]]).svd();
      expect(neg.s.get(0)).toBe(3);
      expect(neg.U.get(0, 0) * neg.V.get(0, 0)).toBe(-1);

      const col = Matrix.from([[1], [2], [2]]).svd();
      expect(col.s.get(0)).toBeCloseTo(3, 14);
      const row = Matrix.from([[1, 2, 2]]).svd();
      expect(row.s.get(0)).toBeCloseTo(3, 14);
    });

    it('handles repeated singular values', () => {
      const { U, s, V } = Matrix.identity(4).mult(2).svd();
      expect([...s.data]).toEqual([2, 2, 2, 2]);
      expect(
        U.matmul(Matrix.diag(s)).matmul(V.transpose()).allClose(Matrix.identity(4).mult(2))
      ).toBe(true);
    });

    it('reveals the rank of a rank-deficient matrix, tall, wide and square', () => {
      for (const [m, n, r] of [
        [10, 10, 3],
        [12, 5, 2],
        [5, 12, 2],
        [8, 8, 1]
      ]) {
        const A = lowRank(m, n, r, 40 + m);
        const { U, s, V } = A.svd();
        for (let i = 0; i < r; i++) expect(s.get(i)).toBeGreaterThan(1e-3);
        for (let i = r; i < s.size; i++) expect(s.get(i)).toBeLessThan(1e-13);
        expect(U.matmul(Matrix.diag(s)).matmul(V.transpose()).allClose(A, 1e-12, 1e-12)).toBe(true);
        expect(orthogonalityError(U)).toBeLessThan(1e-13);
        expect(orthogonalityError(V)).toBeLessThan(1e-13);
      }
    });

    describe('graded upper bidiagonal matrices (relative accuracy of every singular value)', () => {
      // Reference singular values computed offline with 60-digit arithmetic (mpmath), then rounded
      // to the nearest double. The matrices are upper bidiagonal and pass through the Householder
      // stage unchanged, so they drive the bidiagonal QR iteration directly: the first of each
      // pair takes the zero-shift sweep chasing the bulge top to bottom, the reversed one (same
      // singular values: `J * B^T * J`, `J` the exchange matrix) the bottom-to-top sweep.
      const cases: { name: string; d: number[]; e: number[]; ref: number[] }[] = [
        {
          name: 'five decades per step',
          d: [1, 1e-4, 1e-8, 1e-12, 1e-16],
          e: [1e-2, 1e-6, 1e-10, 1e-14],
          ref: [
            1.0000499987505624, 0.00010000000049995001, 1.0000000000005e-8, 1e-12,
            9.999499987494375e-17
          ]
        },
        {
          name: 'smallest singular value 1e-37',
          d: [1e-16, 1e-8, 1e-12, 1e-4],
          e: [1e-1, 1, 1e-3],
          ref: [1, 0.09999999999999999, 0.001004987562112089, 9.950371902099892e-37]
        },
        {
          name: 'sixteen orders of magnitude between neighbours',
          d: [1e8, 1, 1e-8, 1e-16, 1e-24, 1e-32],
          e: [1e4, 1e-4, 1e-12, 1e-20, 1e-28],
          ref: [100000000.5, 1, 1e-8, 1e-16, 1e-24, 9.99999995e-33]
        }
      ];
      const bidiagonal = (d: number[], e: number[]): Matrix =>
        new Matrix(d.length, d.length).map((_, i, j) => (i === j ? d[i] : j === i + 1 ? e[i] : 0));

      for (const { name, d, e, ref } of cases) {
        for (const [orientation, dd, ee] of [
          ['as given', d, e],
          ['reversed', [...d].reverse(), [...e].reverse()]
        ] as const) {
          it(`${name}, ${orientation}`, () => {
            const A = bidiagonal([...dd], [...ee]);
            const { U, s, V } = A.svd();
            for (let i = 0; i < ref.length; i++) {
              expect(Math.abs(s.get(i) - ref[i]) / ref[i]).toBeLessThan(20 * EPS);
            }
            expect(A.singularValues().data).toEqual(s.data);
            expect(orthogonalityError(U)).toBeLessThan(50 * EPS);
            expect(orthogonalityError(V)).toBeLessThan(50 * EPS);
            // The factorization holds to the usual backward-error level, whatever the grading.
            const rec = U.matmul(Matrix.diag(s)).matmul(V.transpose());
            expect(rec.sub(A).norm()).toBeLessThanOrEqual(50 * EPS * A.norm());
          });
        }
      }
    });

    it.each([
      [1, 1e30, 2],
      [1e-30, 1, 1],
      [1e100, 1e-80, 1e-100],
      [1e-60, 1e60, 1e-70],
      [0, 1, 0],
      [3, 0, 2],
      [-2, 5, -7]
    ])('handles the 2x2 triangular block [[%d, %d], [0, %d]] with extreme ratios', (f, g, h) => {
      // Exercises the closed-form 2x2 SVD kernel (LAPACK dlasv2) on its special branches.
      const A = Matrix.from([
        [f, g],
        [0, h]
      ]);
      const { U, s, V } = A.svd();
      expect(orthogonalityError(U)).toBeLessThan(50 * EPS);
      expect(orthogonalityError(V)).toBeLessThan(50 * EPS);
      expect(U.matmul(Matrix.diag(s)).matmul(V.transpose()).sub(A).norm()).toBeLessThanOrEqual(
        50 * EPS * A.norm()
      );
      // det identity: s0 * s1 = |f * h|, to full relative accuracy.
      const product = s.get(0) * s.get(1);
      const det = Math.abs(f * h);
      if (det === 0) expect(product).toBe(0);
      else expect(Math.abs(product - det) / det).toBeLessThan(20 * EPS);
      expect(s.get(0)).toBeGreaterThanOrEqual(s.get(1));
    });

    it('is accurate for matrices with extreme magnitudes, via internal scaling', () => {
      const B = randomMatrix(5, 4, 11);
      const sB = B.singularValues();
      for (const scale of [1e-300, 1e-150, 1e150, 1e300]) {
        const { U, s, V } = B.mult(scale).svd();
        for (let i = 0; i < 4; i++) {
          expect(Math.abs(s.get(i) / scale - sB.get(i)) / sB.get(0)).toBeLessThan(50 * EPS);
        }
        // Divide by the scale first, so that the check itself cannot overflow.
        const rec = U.matmul(Matrix.diag(s.div(scale))).matmul(V.transpose());
        expect(rec.sub(B).norm()).toBeLessThan(50 * EPS * B.norm());
      }
    });

    it('handles the badly scaled Lauchli matrix', () => {
      const d = 1e-8;
      const s = Matrix.from([
        [1, 1, 1],
        [d, 0, 0],
        [0, d, 0],
        [0, 0, d]
      ]).singularValues();
      expect(Math.abs(s.get(0) - Math.sqrt(3 + d * d)) / s.get(0)).toBeLessThan(10 * EPS);
      expect(Math.abs(s.get(1) - d) / d).toBeLessThan(10 * EPS);
      expect(Math.abs(s.get(2) - d) / d).toBeLessThan(10 * EPS);
    });

    it('is deterministic and does not modify its input', () => {
      const A = randomMatrix(6, 4, 5);
      const copy = A.copy();
      const first = A.svd();
      const second = A.svd();
      expect(A.data).toEqual(copy.data);
      expect(first.U.data).toEqual(second.U.data);
      expect(first.s.data).toEqual(second.s.data);
      expect(first.V.data).toEqual(second.V.data);
    });

    it('handles matrices of random 2x2 blocks with a wide dynamic range', () => {
      const rnd = rand(77);
      const entry = (): number => (2 * rnd() - 1) * 10 ** Math.round(12 * (rnd() - 0.5));
      for (let t = 0; t < 300; t++) {
        const A = Matrix.from([
          [entry(), entry()],
          [entry(), entry()]
        ]);
        const { U, s, V } = A.svd();
        expect(U.matmul(Matrix.diag(s)).matmul(V.transpose()).sub(A).norm()).toBeLessThanOrEqual(
          50 * EPS * A.norm()
        );
        expect(orthogonalityError(U)).toBeLessThan(50 * EPS);
        expect(orthogonalityError(V)).toBeLessThan(50 * EPS);
      }
    });

    it('throws for non-finite entries', () => {
      expect(() =>
        Matrix.from([
          [1, NaN],
          [3, 4]
        ]).svd()
      ).toThrow(RangeError);
      expect(() =>
        Matrix.from([
          [1, Infinity],
          [3, 4]
        ]).svd()
      ).toThrow(/non-finite/);
      expect(() =>
        Matrix.from([
          [1, -Infinity],
          [3, 4]
        ]).singularValues()
      ).toThrow(RangeError);
    });
  });

  describe('singularValues()', () => {
    it('returns exactly the singular values of svd()', () => {
      for (const [m, n] of [
        [5, 3],
        [3, 5],
        [6, 6],
        [1, 3]
      ]) {
        const A = randomMatrix(m, n, 21 + m * n);
        expect(A.singularValues().data).toEqual(A.svd().s.data);
      }
    });

    it('agrees with the square roots of the eigenvalues of A^T A', () => {
      const A = randomMatrix(6, 4, 9);
      const eig = A.transpose()
        .matmul(A)
        .eigenvalues()
        .map((e) => e.re)
        .sort((x, y) => y - x);
      const s = A.singularValues();
      for (let i = 0; i < 4; i++) expect(s.get(i)).toBeCloseTo(Math.sqrt(eig[i]), 12);
    });

    it('does not depend on the orientation of the matrix', () => {
      const A = randomMatrix(8, 3, 31);
      expect(A.transpose().singularValues().allClose(A.singularValues(), 1e-14, 1e-14)).toBe(true);
    });
  });
});
