import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';

/**
 * The LU family (`lu`, `solve`, `inverse`, `determinant`, `rank`, `cond1`) for matrices
 * whose entries are close to the limits of the double range. The elimination used to overflow
 * (`solve` returned `NaN` or a wrong solution, `inverse` of a singular matrix returned garbage,
 * `rank` returned 0, `cond1` returned `Infinity`) or lose its entries to underflow.
 *
 * Residuals are checked on the matrix rescaled by an exact power of two, so that the test itself
 * cannot overflow.
 */

/** `x * 2^e` for every entry, in two steps so that `2^e` itself may be out of range. */
function ldexpMatrix(m: Matrix, e: number): Matrix {
  const half = Math.trunc(e / 2);
  return m.mult(2 ** half).mult(2 ** (e - half));
}

/** The exponent `e` such that `2^e` is a power of two just below the largest entry of `m`. */
function exponentOf(m: Matrix): number {
  let max = 0;
  for (const x of m.data) max = Math.max(max, Math.abs(x));
  return Math.floor(Math.log2(max));
}

const BASES: Record<string, number[][]> = {
  identity: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ],
  wellConditioned: [
    [4, 1, 2],
    [1, 5, 1],
    [2, 1, 6]
  ].map((r) => r.map((x) => x / 6)),
  // Six orders of magnitude between the entries.
  mixed: [
    [1, 2e-3, 0],
    [1e-3, 1e-6, 5e-7],
    [0, 4e-6, 1e-3]
  ]
};
const SCALES = [1e300, 1e308, 1e-300, 1e-308];

/** Base matrix times `s`; for the largest entries of the bases being 1, `s` is the largest entry. */
function scaledBase(name: string, s: number): Matrix {
  return Matrix.from(BASES[name]).mult(s);
}

describe('LU family at the limits of the double range', () => {
  for (const name of Object.keys(BASES)) {
    for (const s of SCALES) {
      const label = `${name}, scale ${s}`;

      it(`lu: P*A = L*U, ${label}`, () => {
        const A = scaledBase(name, s);
        const { L, U, perm } = A.lu();
        expect(U.data.every(Number.isFinite)).toBe(true);
        // Compare in the scaled units, where nothing can overflow.
        const e = -exponentOf(A);
        const PA = Matrix.from(perm.map((p) => A.row(p)));
        const diff = ldexpMatrix(PA, e).sub(L.matmul(ldexpMatrix(U, e)));
        expect(diff.normInf()).toBeLessThan(1e-14 * ldexpMatrix(PA, e).normInf());
      });

      it(`solve: small residual and the expected solution, ${label}`, () => {
        const A = scaledBase(name, s);
        const xTrue = Vector.from([0.25, -0.5, 0.125]);
        const b = A.mulVec(xTrue);
        const x = A.solve(b);
        expect(x.data.every(Number.isFinite)).toBe(true);
        const e = -exponentOf(A);
        const bScaled = ldexpMatrix(Matrix.from([b.toArray()]), e).row(0);
        const r = ldexpMatrix(A, e).mulVec(x).sub(bScaled);
        expect(r.normInf()).toBeLessThan(1e-14 * ldexpMatrix(A, e).normInf() * x.normInf());
        // The mixed matrix has subnormal entries at 1e-308, which no longer represent the base.
        if (name !== 'mixed') expect(x.sub(xTrue).normInf()).toBeLessThan(1e-14);
      });

      it(`rank, both methods: full rank, ${label}`, () => {
        const A = scaledBase(name, s);
        expect(A.rank()).toBe(3);
        expect(A.rank({ method: 'elimination' })).toBe(3);
      });

      it(`cond1 does not depend on the scale, ${label}`, () => {
        if (name === 'mixed' && s < 1e-290) return; // the entries are subnormal: another matrix
        const ref = Matrix.from(BASES[name]).cond1();
        expect(Math.abs(scaledBase(name, s).cond1() / ref - 1)).toBeLessThan(1e-12);
      });

      it(`inverse: residual, or overflow to Infinity without NaN, ${label}`, () => {
        const A = scaledBase(name, s);
        const inv = A.inverse();
        expect(inv.data.some(Number.isNaN)).toBe(false);
        if (inv.data.every(Number.isFinite)) {
          expect(A.matmul(inv).sub(Matrix.identity(3)).normInf()).toBeLessThan(1e-8);
        } else {
          // Only if the inverse really is out of range: its largest entry is 1/s times the one
          // of the base's inverse.
          const biggest = Matrix.from(BASES[name]).inverse().normInf() / 3;
          expect(biggest / s).toBeGreaterThan(1e307);
        }
      });
    }
  }
});

describe('the scale does not change the results, bit for bit', () => {
  // A power of two scale is exact, and elimination commutes with it: no rounding differs.
  const M = Matrix.from([
    [0.3, -0.7, 0.2, 0.9, -0.1],
    [0.5, 0.1, -0.8, 0.4, 0.6],
    [-0.2, 0.9, 0.3, -0.5, 0.7],
    [0.8, -0.4, 0.6, 0.1, -0.9],
    [0.1, 0.2, -0.3, 0.7, 0.5]
  ]);
  const b = Vector.from([1, -2, 3, -4, 5]);

  for (const e of [-1000, -950, 950, 1000]) {
    it(`solve and inverse with entries scaled by 2^${e}`, () => {
      const A = M.mult(2 ** (e / 2)).mult(2 ** (e / 2));
      const sb = b.mult(2 ** (e / 2)).mult(2 ** (e / 2));
      expect(Array.from(A.solve(sb).data)).toEqual(Array.from(M.solve(b).data));
      // The inverse scales the other way; at these scales its entries are still normal numbers.
      const inv = A.inverse();
      expect(Array.from(ldexpMatrix(inv, e).data)).toEqual(Array.from(M.inverse().data));
    });
  }
});

describe('determinant at the limits of the double range', () => {
  it('is exact when it is representable, however large or small the pivots are', () => {
    // The partial products of the pivots (1e400, then 1e-400 below) used to overflow.
    expect(Matrix.diag([1e200, 1e200, 1e-200, 1e-200]).determinant()).toBeCloseTo(1, 14);
    expect(Matrix.diag([1e-200, 1e-200, 1e200, 1e200]).determinant()).toBeCloseTo(1, 14);
    expect(Matrix.diag([1e308, 1e-300]).determinant()).toBeCloseTo(1e8, 6);
    // Entries near the limits, and a determinant that is representable.
    expect(Matrix.from([[1.5e308]]).determinant()).toBe(1.5e308);
    expect(Matrix.from([[1e-310]]).determinant()).toBe(1e-310);
    expect(Matrix.from([[-1e308]]).determinant()).toBe(-1e308);
  });

  it('keeps the sign of the permutation', () => {
    expect(
      Matrix.from([
        [0, 1e308],
        [1e-300, 0]
      ]).determinant()
    ).toBeCloseTo(-1e8, 6);
    expect(
      Matrix.from([
        [0, 1.5e308],
        [1, 0]
      ]).determinant()
    ).toBe(-1.5e308);
  });

  it('overflows to ±Infinity and underflows to ±0 only if the true value does', () => {
    // The scaled determinant is representable (it is a number of order 1), the true one is not.
    expect(Matrix.identity(3).mult(1e300).determinant()).toBe(Infinity);
    expect(Matrix.identity(3).mult(1e308).determinant()).toBe(Infinity);
    expect(
      Matrix.from([
        [1e308, 1e308],
        [1e308, -1e308]
      ]).determinant()
    ).toBe(-Infinity);
    expect(Matrix.identity(3).mult(1e-300).determinant()).toBe(0);
    expect(Matrix.identity(3).mult(1e-308).determinant()).toBe(0);
    expect(
      Matrix.from([
        [1e-300, 1e-300],
        [1e-300, -1e-300]
      ]).determinant()
    ).toBe(-0);
  });

  it('is 0 for a singular matrix with entries near the limits', () => {
    expect(
      Matrix.from([
        [1e308, 1e308],
        [1e308, 1e308]
      ]).determinant()
    ).toBe(0);
    expect(
      Matrix.from([
        [1e-308, 1e-308],
        [1e-308, 1e-308]
      ]).determinant()
    ).toBe(0);
  });
});

describe('solve, inverse and rank in the cases that used to fail', () => {
  const H = () =>
    Matrix.from([
      [1e308, 1e308],
      [1e308, -1e308]
    ]);

  it('solve: the elimination doubles an entry that is near the limit', () => {
    // Used to return [1, 0]: the entry 1e308 - (-1e308) of U overflowed.
    const x = H().solve(Vector.from([1e308, 0]));
    expect(x.data[0]).toBeCloseTo(0.5, 15);
    expect(x.data[1]).toBeCloseTo(0.5, 15);
  });

  it('lu returns the factors of the matrix, with Infinity where U overflows', () => {
    const { L, U } = H().lu();
    expect(Array.from(L.data)).toEqual([1, 0, 1, 1]);
    expect(U.get(0, 0)).toBe(1e308);
    expect(U.get(0, 1)).toBe(1e308);
    expect(U.get(1, 0)).toBe(0);
    expect(U.get(1, 1)).toBe(-Infinity); // -2e308, out of range; solve() is not affected
  });

  it('lu with out works in place on a matrix near the limit', () => {
    const out = Matrix.identity(2).lu();
    const A = H();
    const dec = A.lu(out);
    expect(dec).toBe(out);
    expect(dec.U.get(0, 0)).toBe(1e308);
    const B = Matrix.from([
      [4e307, 1e307],
      [1e307, 5e307]
    ]);
    const inPlace = B.copy();
    const f = inPlace.lu({ L: Matrix.identity(2), U: inPlace, perm: [0, 1], sign: 1 });
    const g = B.lu();
    expect(Array.from(f.U.data)).toEqual(Array.from(g.U.data));
  });

  it('inverse of the matrix', () => {
    const inv = H().inverse();
    expect(inv.get(0, 0)).toBeCloseTo(0.5e-308, 320);
    expect(inv.get(1, 1)).toBeCloseTo(-0.5e-308, 320);
    expect(H().matmul(inv).sub(Matrix.identity(2)).normInf()).toBeLessThan(1e-15);
  });

  it('a singular matrix near the limit is still reported singular', () => {
    const S = Matrix.from([
      [1e308, 1e308],
      [1e308, 1e308]
    ]);
    expect(() => S.lu()).toThrow('Matrix.lu: matrix is singular.');
    expect(() => S.solve(Vector.from([1, 1]))).toThrow('Matrix.solve: matrix is singular.');
    expect(() => S.inverse()).toThrow('Matrix.inverse: matrix is singular.');
    expect(S.cond1()).toBe(Infinity);
    expect(S.rank()).toBe(1);
    expect(S.rank({ method: 'elimination' })).toBe(1);
  });

  it('rank, both methods, of a rank-deficient matrix near the limits', () => {
    for (const s of [1e300, 1.5e308, 1e-300, 1e-308]) {
      const A = Matrix.from([
        [1, 2, 3],
        [2, 4, 6],
        [1, 0, 1]
      ]).mult(s / 6);
      expect(A.rank()).toBe(2);
      expect(A.rank({ method: 'elimination' })).toBe(2);
    }
  });

  it('rank: atol is absolute, whatever the scale', () => {
    const A = Matrix.identity(3).mult(1e-300);
    expect(A.rank({ atol: 1e-310 })).toBe(3);
    expect(A.rank({ atol: 1e-299 })).toBe(0);
    expect(A.rank({ method: 'elimination', atol: 1e-310 })).toBe(3);
    expect(A.rank({ method: 'elimination', atol: 1e-299 })).toBe(0);
    const B = Matrix.identity(3).mult(1e308);
    expect(B.rank({ atol: 1e307 })).toBe(3);
    expect(B.rank({ method: 'elimination', atol: 1e307 })).toBe(3);
  });

  it('cond1 of a matrix whose 1-norm overflows', () => {
    expect(H().cond1()).toBeCloseTo(2, 14);
  });
});

describe('solve with a right-hand side near the limits', () => {
  it('a huge or tiny right-hand side with an ordinary matrix', () => {
    const A = Matrix.from([
      [2, 1],
      [1, 3]
    ]);
    const big = A.solve(Vector.from([1e308, 1e308]));
    expect(big.data[0]).toBeCloseTo(0.4e308, -293);
    expect(big.data[1]).toBeCloseTo(0.2e308, -293);
    const tiny = A.solve(Vector.from([1e-308, 1e-308]));
    expect(tiny.data[0] / 1e-308).toBeCloseTo(0.4, 14);
    expect(tiny.data[1] / 1e-308).toBeCloseTo(0.2, 14);
  });

  it('a solution that is out of range is ±Infinity, not NaN', () => {
    const A = Matrix.from([
      [3e-310, 1e-310],
      [1e-310, 2e-310]
    ]);
    expect(Array.from(A.solve(Vector.from([1, 1])).data)).toEqual([Infinity, Infinity]);
    expect(Array.from(A.solve(Vector.from([1, -1])).data)).toEqual([Infinity, -Infinity]);
    expect(A.inverse().data.some(Number.isNaN)).toBe(false);
  });
});

describe('a matrix whose entries span the whole range of doubles', () => {
  // Scaling it into a safe range would flush the small entries to zero: it is not scaled.
  const D = Matrix.diag([1e308, 1e-300]);

  it('lu, solve, inverse and determinant still work', () => {
    const { U } = D.lu();
    expect(Array.from(U.data)).toEqual([1e308, 0, 0, 1e-300]);
    const x = D.solve(Vector.from([1e308, 1e-300]));
    expect(Array.from(x.data)).toEqual([1, 1]);
    const inv = D.inverse();
    expect(inv.get(0, 0)).toBeCloseTo(1e-308, 321);
    expect(inv.get(1, 1) / 1e300).toBeCloseTo(1, 14);
    expect(D.determinant()).toBeCloseTo(1e8, 6);
    // The default cutoff is relative to the largest entry: 1e-300 is below it, as documented.
    expect(D.rank({ method: 'elimination' })).toBe(1);
    expect(D.rank({ method: 'elimination', rtol: 0 })).toBe(2);
  });
});
