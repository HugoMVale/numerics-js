import { describe, expect, it } from 'vitest';
import { BandedMatrix } from '../../src/linalg/BandedMatrix.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import { Vector } from '../../src/linalg/Vector.js';
import { tridiagonalLU, solveTridiagonalLU } from '../../src/linalg/tridiagonal.js';
import fixtures from './BandedMatrix.scipy.json' with { type: 'json' };

const EPS = Number.EPSILON;

/** `max|x - y| / max|y|`. */
function relInf(x: ArrayLike<number>, y: ArrayLike<number>): number {
  let diff = 0;
  let ref = 0;
  for (let i = 0; i < y.length; i++) {
    diff = Math.max(diff, Math.abs(x[i] - y[i]));
    ref = Math.max(ref, Math.abs(y[i]));
  }
  return diff / ref;
}

// Factors are compared entry by entry when the matrix is well conditioned enough that rounding
// cannot change a pivot choice; otherwise only the solution and the residual are compared.
const wellConditioned = (cond1: number) => cond1 < 1e4;

describe('BandedMatrix vs scipy.linalg.solve_banded and LAPACK dgbtrf', () => {
  for (const f of fixtures) {
    describe(`${f.id} (n = ${f.n}, kl = ${f.kl}, ku = ${f.ku})`, () => {
      const B = BandedMatrix.fromBands(Matrix.from(f.ab), f.kl, f.ku);

      it('solves like solve_banded', () => {
        const x = B.solve(f.b);
        // forward error is bounded by cond * eps for both implementations
        expect(relInf(x.data, f.x)).toBeLessThanOrEqual(Math.max(1e-13, 20 * f.cond1 * EPS));
        const r = B.mulVec(x).sub(Vector.from(f.b));
        const residual = r.normInf() / Vector.from(f.b).normInf();
        expect(residual).toBeLessThanOrEqual(Math.max(1e-13, f.scipyResidual * 100));
      });

      it('reproduces the factors and row interchanges of dgbtrf', () => {
        const dec = B.lu();
        expect(dec.scale).toBe(1);
        if (wellConditioned(f.cond1)) {
          expect(Array.from(dec.ipiv)).toEqual(f.ipiv);
          const ref = Matrix.from(f.lu); // (2 kl + ku + 1) x n, LAPACK layout
          const ld = 2 * f.kl + f.ku + 1;
          let max = 0;
          for (let j = 0; j < f.n; j++) {
            for (let r = 0; r < ld; r++) {
              max = Math.max(max, Math.abs(dec.ab[r + j * ld] - ref.get(r, j)));
            }
          }
          expect(max).toBeLessThanOrEqual(1e-12 * Math.max(1, f.cond1));
        }
      });

      it('reuses the factorization for several right-hand sides', () => {
        const dec = B.lu();
        const b2 = f.b.map((v, i) => (i % 2 === 0 ? -v : 2 * v + 1));
        const A = B.toMatrix();
        const x2 = BandedMatrix.solveLU(dec, b2);
        const r = A.mulVec(x2).sub(Vector.from(b2));
        // normwise backward error: ||r|| / (||A|| ||x|| + ||b||)
        const backward = r.normInf() / (A.normInf() * x2.normInf() + Vector.from(b2).normInf());
        expect(backward).toBeLessThanOrEqual(50 * EPS);
      });
    });
  }
});

describe('tridiagonalLU vs LAPACK dgttrf', () => {
  for (const f of fixtures) {
    const t = (f as { tridiagonal?: Record<string, number[]> }).tridiagonal;
    if (t === undefined) continue;
    it(`matches dgttrf for ${f.id}`, () => {
      const lu = tridiagonalLU(t.dl, t.d, t.du);
      expect(Array.from(lu.ipiv)).toEqual(t.ipiv);
      for (const [mine, ref] of [
        [lu.dl, t.luDl],
        [lu.d, t.luD],
        [lu.du, t.luDu],
        [lu.du2, t.luDu2]
      ] as const) {
        expect(relInf(mine, ref) || 0).toBeLessThanOrEqual(1e-13);
      }
      const x = solveTridiagonalLU(lu, f.b);
      expect(relInf(x.data, f.x)).toBeLessThanOrEqual(Math.max(1e-13, 20 * f.cond1 * EPS));
    });
  }
});
