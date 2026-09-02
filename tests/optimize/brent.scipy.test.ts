import { describe, expect, it } from 'vitest';
import { brent } from '../../src/optimize/brent.js';
import type { ScalarFunction } from '../../src/types.js';
import fixtures from './fixtures/brent.scipy.json' with { type: 'json' };

// Must mirror CASES in scripts/benchmarks/generate_brent_fixtures.py
// (same id, objective, and bracket endpoints).
const OBJECTIVES: Record<string, ScalarFunction> = {
    quadratic: (x) => (x - 2) ** 2,
    quartic: (x) => x ** 4 - x + 1,
    absolute_value: (x) => Math.abs(x - 0.5),
    oscillatory_local_minimum: (x) => Math.sin(5 * x) + (x - 1) ** 2,
    sharp_minimum: (x) => Math.exp(50 * (x - 0.3) ** 2),
};

const EVALUATION_FACTOR = 1.1;
const RESULT_TOLERANCE_FACTOR = 10;

describe('brent vs scipy.optimize.minimize_scalar Brent reference values', () => {
    for (const fixture of fixtures) {
        it(`matches SciPy Brent for ${fixture.id}: ${fixture.description}`, () => {
            const fn = OBJECTIVES[fixture.id];
            expect(fn, `no matching objective registered for id "${fixture.id}"`).toBeDefined();

            const result = brent(fn, fixture.xa, fixture.xb, { tolX: fixture.tolX });

            expect(result.success).toBe(true);
            expect(Math.abs(result.x - fixture.scipyX)).toBeLessThanOrEqual(
                RESULT_TOLERANCE_FACTOR * fixture.tolX
            );
            expect(Math.abs(result.fx - fixture.scipyFx)).toBeLessThanOrEqual(
                RESULT_TOLERANCE_FACTOR * fixture.tolX
            );
            expect(result.evaluations).toBeLessThanOrEqual(
                fixture.scipyEvaluations * EVALUATION_FACTOR
            );
        });
    }
});