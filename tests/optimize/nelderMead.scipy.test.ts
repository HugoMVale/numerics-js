import { describe, expect, it } from 'vitest';
import { nelderMead } from '../../src/optimize/nelderMead.js';
import { TEST_FUNCTIONS_MULTIVAR } from './testFunctions.js';
import fixtures from './fixtures/nelderMead.scipy.json' with { type: 'json' };

// Must mirror CASES in scripts/benchmarks/generate_nelder_mead_fixtures.py
// (same id, objective, and starting point), which in turn mirror
// TEST_FUNCTIONS_MULTIVAR in tests/optimize/testFunctions.ts for N = 2.
const N = 2;

const X_TOLERANCE = 2e-7;
const F_TOLERANCE_FACTOR = 10;
const EVALUATION_FACTOR = 1.1;

describe('nelderMead vs scipy.optimize.minimize Nelder-Mead reference values', () => {
    for (const fixture of fixtures) {
        it(`matches SciPy Nelder-Mead for ${fixture.id}: ${fixture.description}`, () => {
            const data = TEST_FUNCTIONS_MULTIVAR[fixture.id];
            expect(data, `no matching test function registered for id "${fixture.id}"`).toBeDefined();

            const x0 = data.initialPoint(N);
            expect(Array.from(x0.data)).toEqual(fixture.x0);

            const result = nelderMead(data.fn, x0, { tolX: fixture.tolX, tolF: fixture.tolF });

            expect(result.success, result.message).toBe(true);
            for (let i = 0; i < N; i++) {
                expect(Math.abs(result.x.get(i) - fixture.scipyX[i])).toBeLessThanOrEqual(
                    X_TOLERANCE
                );
            }
            expect(Math.abs(result.fx - fixture.scipyFx)).toBeLessThanOrEqual(
                F_TOLERANCE_FACTOR * fixture.tolF
            );
            expect(result.evaluations).toBeLessThanOrEqual(
                fixture.scipyEvaluations * EVALUATION_FACTOR
            );
        });
    }
});
