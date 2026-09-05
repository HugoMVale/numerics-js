import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/linalg/Matrix.js';
import fixtures from './fixtures/Matrix.qr.scipy.json' with { type: 'json' };

describe('Matrix.qr() vs scipy.linalg.qr reference values', () => {
    for (const fixture of fixtures) {
        it(`matches SciPy for ${fixture.id}: ${fixture.description}`, () => {
            const matrix = Matrix.from(fixture.matrix);
            const { Q, R } = matrix.qr();
            const reconstructionResidual = Q.matmul(R).sub(matrix).norm() / matrix.norm();
            const orthogonalityResidual = Q.transpose().matmul(Q).sub(Matrix.identity(Q.rows)).norm();

            expect(Q.rows).toBe(matrix.rows);
            expect(Q.cols).toBe(matrix.rows);
            expect(R.rows).toBe(matrix.rows);
            expect(R.cols).toBe(matrix.cols);
            expect(reconstructionResidual).toBeLessThanOrEqual(Math.max(1e-10, fixture.scipyResidual * 10));
            expect(orthogonalityResidual).toBeLessThanOrEqual(
                Math.max(1e-10, fixture.scipyOrthogonalityResidual * 10)
            );

            if (fixture.id !== 'rank_deficient') {
                const determinedColumns = Math.min(matrix.rows, matrix.cols);
                for (let i = 0; i < matrix.rows; i++) {
                    for (let j = 0; j < determinedColumns; j++) {
                        expect(Math.abs(Q.get(i, j))).toBeCloseTo(Math.abs(fixture.scipyQ[i][j]), 9);
                    }
                }
                for (let i = 0; i < matrix.rows; i++) {
                    for (let j = 0; j < matrix.cols; j++) {
                        expect(Math.abs(R.get(i, j))).toBeCloseTo(Math.abs(fixture.scipyR[i][j]), 9);
                    }
                }
            }

            for (let i = 1; i < R.rows; i++) {
                for (let j = 0; j < Math.min(i, R.cols); j++) expect(R.get(i, j)).toBe(0);
            }
        });
    }
});