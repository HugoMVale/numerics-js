import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import fixtures from './Matrix.qrPivot.scipy.json' with { type: 'json' };

describe('Matrix.qrPivot() vs scipy.linalg.qr(pivoting=True) reference values', () => {
  for (const fixture of fixtures) {
    it(`matches SciPy for ${fixture.id}: ${fixture.description}`, () => {
      const matrix = Matrix.from(fixture.matrix);
      const dec = matrix.qrPivot();
      const Q = Matrix.qrPivotQ(dec);
      const R = Matrix.qrPivotR(dec);
      const k = Math.min(matrix.rows, matrix.cols);

      const permuted = new Matrix(matrix.rows, matrix.cols);
      dec.perm.forEach((src, dst) => permuted.setCol(dst, matrix.col(src)));
      const reconstructionResidual = Q.matmul(R).sub(permuted).norm() / matrix.norm();
      const orthogonalityResidual = Q.transpose().matmul(Q).sub(Matrix.identity(k)).norm();

      expect(Q.rows).toBe(matrix.rows);
      expect(Q.cols).toBe(k);
      expect(R.rows).toBe(k);
      expect(R.cols).toBe(matrix.cols);
      expect(reconstructionResidual).toBeLessThanOrEqual(
        Math.max(1e-12, fixture.scipyResidual * 10)
      );
      expect(orthogonalityResidual).toBeLessThanOrEqual(1e-12);

      // With a numerically rank-deficient matrix, the trailing columns are
      // determined by rounding noise: only the factorization identities apply.
      if (fixture.id !== 'rank_deficient') {
        // Same pivot order, and (since both follow LAPACK's Householder
        // conventions) the same compact factors, signs included.
        expect(dec.perm).toEqual(fixture.scipyPerm);
        const scaleQR = Math.max(...fixture.scipyQR.flat().map(Math.abs));
        for (let i = 0; i < matrix.rows; i++) {
          for (let j = 0; j < matrix.cols; j++) {
            expect(Math.abs(dec.QR.get(i, j) - fixture.scipyQR[i][j])).toBeLessThanOrEqual(
              1e-9 * scaleQR
            );
          }
        }
        for (let j = 0; j < k; j++) {
          expect(dec.tau.get(j)).toBeCloseTo(fixture.scipyTau[j], 9);
        }
        for (let i = 0; i < k; i++) {
          for (let j = 0; j < matrix.cols; j++) {
            expect(Math.abs(R.get(i, j) - fixture.scipyR[i][j])).toBeLessThanOrEqual(
              1e-9 * scaleQR
            );
          }
        }
      }
    });
  }
});
