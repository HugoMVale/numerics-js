import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import fixtures from './Matrix.svd.scipy.json' with { type: 'json' };

const EPS = Number.EPSILON;

describe('Matrix SVD family vs scipy.linalg / numpy.linalg reference values', () => {
  for (const fixture of fixtures) {
    it(`matches SciPy for ${fixture.id}: ${fixture.description}`, () => {
      const A = Matrix.from(fixture.matrix);
      const m = A.rows;
      const n = A.cols;
      const k = Math.min(m, n);
      const sRef = fixture.scipyS;
      const rank = fixture.scipyRank;
      // Condition number of the part of A that is kept by the rank cutoff: it sets how much the
      // solution and the pseudoinverse can differ between two backward-stable implementations.
      const kappaEff = sRef[0] / sRef[rank - 1];
      const solutionTol = 1e-13 + 20 * EPS * kappaEff;

      // Singular values: absolute agreement relative to the largest one (the backward-stable
      // guarantee both LAPACK and this implementation give).
      const { U, s, V } = A.svd();
      expect(s.size).toBe(k);
      for (let i = 0; i < k; i++) {
        expect(Math.abs(s.get(i) - sRef[i])).toBeLessThanOrEqual(50 * EPS * sRef[0]);
      }
      expect(A.singularValues().data).toEqual(s.data);

      // Factorization identities. SciPy's own residual sizes the reconstruction tolerance.
      const reconstruction =
        U.matmul(Matrix.diag(s)).matmul(V.transpose()).sub(A).norm() / A.norm();
      expect(reconstruction).toBeLessThanOrEqual(
        Math.max(1e-13, fixture.scipyReconstructionResidual * 10)
      );
      expect(U.transpose().matmul(U).sub(Matrix.identity(k)).norm()).toBeLessThanOrEqual(1e-13);
      expect(V.transpose().matmul(V).sub(Matrix.identity(k)).norm()).toBeLessThanOrEqual(1e-13);

      // Rank (numpy.linalg.matrix_rank convention) and 2-norm condition number.
      expect(A.rankSvd()).toBe(rank);
      if (fixture.scipyCond2 !== null) {
        const tol = Math.max(1e-12, 100 * EPS * fixture.scipyCond2);
        expect(Math.abs(A.cond2() - fixture.scipyCond2) / fixture.scipyCond2).toBeLessThanOrEqual(
          tol
        );
      } else {
        // Rank deficient (cond_2 reported as null): the smallest singular value is rounding noise.
        expect(A.cond2()).toBeGreaterThan(1e12);
      }

      // Minimum-norm least squares (LAPACK gelsd, same cutoff).
      const res = A.lstsq(fixture.rhs);
      const xRef = fixture.scipyLstsqX;
      expect(res.rank).toBe(fixture.scipyLstsqRank);
      let dx = 0;
      let nx = 0;
      for (let j = 0; j < n; j++) {
        dx += (res.x.get(j) - xRef[j]) ** 2;
        nx += xRef[j] ** 2;
      }
      expect(Math.sqrt(dx) / Math.sqrt(nx)).toBeLessThanOrEqual(solutionTol);

      // Pseudoinverse (scipy.linalg.pinv, stored for the small cases).
      if (fixture.scipyPinv !== null) {
        const P = A.pinv();
        const Pref = Matrix.from(fixture.scipyPinv);
        expect(P.rows).toBe(n);
        expect(P.cols).toBe(m);
        expect(P.sub(Pref).norm() / Pref.norm()).toBeLessThanOrEqual(solutionTol);
      }
    });
  }
});
