import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { quasiNewton } from '../../src/roots/newton.js';
import fixtures from './fixtures/quasiNewton.scipy.json' with { type: 'json' };
import {
    fExample65,
    fPowellSingular,
    fRosenbrock,
    fTrigonometric,
} from './testFunctions.js';

// Must mirror CASES in scripts/benchmarks/generate_quasi_newton_fixtures.py.
const TEST_FUNCTIONS: Record<string, (x: Vector) => Vector> = {
    rosenbrock: fRosenbrock.f,
    powell_singular: fPowellSingular.f,
    trigonometric: fTrigonometric.f,
    trigonometric_5: fTrigonometric.f,
    example65: fExample65.f,
};

const X_TOLERANCE = 2e-6;
const F_TOLERANCE = 1e-7;
const MIRROR_TOLERANCE = 1e-12;

// Allowed ratio of numerics-js to SciPy function evaluations. Without Broyden updates the solver
// evaluates a full finite-difference Jacobian at every iteration (n evaluations), whereas SciPy's
// hybr updates its Jacobian by rank-1 corrections, so ratios above 1 are possible on problems that
// converge in few iterations.
const EVALUATION_FACTORS = { withoutBroyden: 3, withBroyden: 1.5 };

describe('test functions mirror the SciPy reference functions', () => {
    for (const fixture of fixtures) {
        it(`${fixture.id}: f(x0) and f(scipyX) agree with the Python implementation`, () => {
            const f = TEST_FUNCTIONS[fixture.id];
            expect(f, `no matching test function registered for id "${fixture.id}"`).toBeDefined();

            expect(f(Vector.from(fixture.x0)).allClose(
                Vector.from(fixture.fx0), MIRROR_TOLERANCE, MIRROR_TOLERANCE,
            )).toBe(true);
            expect(f(Vector.from(fixture.scipyX)).allClose(
                Vector.from(fixture.scipyFx), MIRROR_TOLERANCE, MIRROR_TOLERANCE,
            )).toBe(true);
        });
    }
});

describe('quasiNewton vs scipy.optimize.root hybr reference values', () => {
    for (const broydenUpdate of [false, true]) {
        const evaluationFactor = broydenUpdate
            ? EVALUATION_FACTORS.withBroyden
            : EVALUATION_FACTORS.withoutBroyden;

        for (const fixture of fixtures) {
            it(`matches SciPy hybr for ${fixture.id} (broydenUpdate = ${broydenUpdate}): ${fixture.description}`, () => {
                const data = TEST_FUNCTIONS[fixture.id];
                expect(data, `no matching test function registered for id "${fixture.id}"`).toBeDefined();

                const result = quasiNewton(data, Vector.from(fixture.x0), {
                    tolx: fixture.tolx,
                    tolf: fixture.tolf,
                    globalMethod: 'dogleg',
                    broydenUpdate,
                });

                expect(result.success, result.message).toBe(true);
                if (result.message === '||sclf*f(x)||∞ ≤ tolf') {
                    // With tolf = 0 this can only be an exactly zero residual.
                    expect(result.fx.normInf()).toBe(0);
                } else {
                    expect(result.message).toContain('||Δx/max(x, 1/sclx)||∞ ≤ tolx');
                }
                expect(result.x.size).toBe(fixture.scipyX.length);
                for (let i = 0; i < fixture.scipyX.length; i++) {
                    expect(Math.abs(result.x.get(i) - fixture.scipyX[i])).toBeLessThanOrEqual(
                        X_TOLERANCE
                    );
                }
                expect(result.fx.normInf()).toBeLessThanOrEqual(F_TOLERANCE);
                expect(result.evaluationsFunction).toBeLessThanOrEqual(
                    fixture.scipyEvaluations * evaluationFactor
                );
            });
        }
    }
});