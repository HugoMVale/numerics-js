import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';

/**
 * `eigenvalues` and `eig` for matrices far from unit scale. The Hessenberg/QR iteration used to fail to
 * converge outside roughly `1e-80`...`1e60`, and the eigenvectors were wrong below about `1e-8`, because an
 * absolute floor was used in the back-substitution.
 */

// Real eigenvalue pair, a complex pair and a coupling between the two blocks.
const BASE = [
  [2, 1, 0, 0],
  [-1, 3, 1, 0.5],
  [0.3, 0, 1, -2],
  [0, 1, 2, 4]
];

const sortKey = (v: { re: number; im: number }[]) =>
  v.map(({ re, im }) => [re, im]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);

const SCALES = [
  1e-310, 1e-300, 1e-100, 1e-50, 1e-30, 1e-20, 1e-10, 1e-3, 1e3, 1e30, 1e50, 1e100, 1e300, 1e307
];

describe('eigenvalues and eig at extreme scales', () => {
  const ref = sortKey(Matrix.from(BASE).eigenvalues());

  it.each(SCALES)('eigenvalues scale with the matrix (%s)', (s) => {
    const A = Matrix.from(BASE.map((r) => r.map((x) => x * s)));
    const got = sortKey(A.eigenvalues());
    for (let i = 0; i < got.length; i++) {
      expect(got[i][0] / s).toBeCloseTo(ref[i][0], 11);
      expect(got[i][1] / s).toBeCloseTo(ref[i][1], 11);
    }
  });

  it.each(SCALES)('eig satisfies A v = lambda v (%s)', (s) => {
    const A = Matrix.from(BASE.map((r) => r.map((x) => x * s)));
    for (const { value, vectorRe: x, vectorIm: y } of A.eig()) {
      const Ax = A.mulVec(x);
      const Ay = A.mulVec(y);
      expect(x.norm() ** 2 + y.norm() ** 2).toBeCloseTo(1, 12);
      for (let i = 0; i < 4; i++) {
        const re = Ax.get(i) - (value.re * x.get(i) - value.im * y.get(i));
        const im = Ay.get(i) - (value.re * y.get(i) + value.im * x.get(i));
        expect(Math.hypot(re, im) / s).toBeLessThan(1e-12);
      }
    }
  });

  it('is bitwise unchanged by an exact power-of-two scaling', () => {
    const base = Matrix.from(BASE.map((r) => r.map((x) => x * 2 ** 150)));
    const scaled = Matrix.from(BASE.map((r) => r.map((x) => x * 2 ** 450)));
    const a = base.eig();
    const b = scaled.eig();
    for (let i = 0; i < a.length; i++) {
      expect(b[i].value.re).toBe(a[i].value.re * 2 ** 300);
      expect(b[i].value.im).toBe(a[i].value.im * 2 ** 300);
      expect(b[i].vectorRe.toArray()).toEqual(a[i].vectorRe.toArray());
      expect(b[i].vectorIm.toArray()).toEqual(a[i].vectorIm.toArray());
    }
  });

  it('handles the zero matrix and a 1x1 matrix of extreme size', () => {
    expect(new Matrix(3, 3).eigenvalues().map((e) => e.re)).toEqual([0, 0, 0]);
    expect(Matrix.from([[1.5e308]]).eig()[0].value.re).toBe(1.5e308);
    expect(Matrix.from([[5e-324]]).eigenvalues()[0].re).toBe(5e-324);
  });

  it('reports NaN input as before and rejects infinite entries', () => {
    expect(
      Matrix.from([
        [NaN, 1],
        [1, 2]
      ])
        .eigenvalues()
        .every((e) => Number.isNaN(e.re))
    ).toBe(true);
    expect(() =>
      Matrix.from([
        [Infinity, 1],
        [1, 2]
      ]).eig()
    ).toThrow(/infinite entry at \(0, 0\)/);
  });
});

describe('qrPivot for subnormal matrices', () => {
  const rows = [
    [1, 2, 3],
    [4, 5, 6.5],
    [7, 8.5, 9],
    [1, 0, 2]
  ];
  const ref = Matrix.from(rows).qrPivot();

  // `1 / (alpha - beta)` overflowed there and the factorization came back with NaN and Infinity.
  it.each([1e-300, 1e-308, 1e-310, 3e-312])(
    'matches the factorization of the unscaled matrix (%s)',
    (s) => {
      const dec = Matrix.from(rows.map((r) => r.map((x) => x * s))).qrPivot();
      expect(dec.perm).toEqual(ref.perm);
      // Entries of a subnormal matrix have only 10-16 significant digits left.
      const tol = 1e-3 * (s < 1e-308 ? 1 : 1e-10);
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 3; j++) {
          const got = dec.QR.get(i, j);
          const want = ref.QR.get(i, j) * (i <= j ? s : 1); // R part scales, reflector part does not
          expect(Number.isFinite(got)).toBe(true);
          expect(Math.abs(got - want)).toBeLessThanOrEqual(tol * Math.abs(want) + 1e-30 * s);
        }
      }
      for (let j = 0; j < 3; j++) expect(dec.tau.get(j)).toBeCloseTo(ref.tau.get(j), 6);
    }
  );
});
