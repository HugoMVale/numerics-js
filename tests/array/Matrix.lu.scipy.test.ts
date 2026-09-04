import { describe, expect, it } from 'vitest';
import { Matrix } from '../../src/array/Matrix.js';
import fixtures from './fixtures/Matrix.lu.scipy.json' with { type: 'json' };

describe('Matrix.lu() vs scipy.linalg.lu reference values', () => {
    for (const fixture of fixtures) {
        it(`matches SciPy for ${fixture.id}: ${fixture.description}`, () => {
            const matrix = Matrix.from(fixture.matrix);
            const result = matrix.lu();

            expect(result.perm).toEqual(fixture.scipyPerm);
            expect(result.sign).toBe(fixture.scipySign);

            if (fixture.id === 'hilbert_6x6' || fixture.id === 'wide_dynamic_range') {
                const permuted = Matrix.from(fixture.scipyPerm.map((row) => fixture.matrix[row]));
                const residual = permuted.sub(result.L.matmul(result.U)).norm() / matrix.norm();
                expect(residual).toBeLessThanOrEqual(Math.max(1e-10, fixture.scipyResidual * 10));
            } else {
                expect(result.L.allClose(Matrix.from(fixture.scipyL), 1e-12, 1e-12)).toBe(true);
                expect(result.U.allClose(Matrix.from(fixture.scipyU), 1e-12, 1e-12)).toBe(true);
            }
        });
    }
});