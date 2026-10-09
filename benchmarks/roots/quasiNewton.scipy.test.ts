import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { quasiNewton } from '../../src/roots/newton.js';
import fixtures from './quasiNewton.scipy.json' with { type: 'json' };
import {
    fExample65,
    fHelicalValley,
    fPowellBadlyScaled,
    fPowellSingular,
    fRosenbrock,
    fTrigonometric,
} from '../../tests/roots/testFunctions.js';

// Must mirror CASES in generate_quasi_newton_fixtures.py.
const TEST_FUNCTIONS: Record<string, (x: Vector) => Vector> = {
    rosenbrock: fRosenbrock.f,
    rosenbrock_100: fRosenbrock.f,
    powell_singular: fPowellSingular.f,
    powell_badly_scaled: fPowellBadlyScaled.f,
    helical_valley: fHelicalValley.f,
    trigonometric: fTrigonometric.f,
    trigonometric_5: fTrigonometric.f,
    example65: fExample65.f,
};

const X_TOLERANCE = 2e-6;
const F_TOLERANCE = 1e-7;
const MIRROR_TOLERANCE = 1e-12;

// Allowed ratio of numerics-js to SciPy function evaluations. SciPy's hybr updates its Jacobian by
// rank-1 corrections instead of recomputing it at every iteration, so quasiNewton is run with
// Broyden updates as well; the comparison of evaluation counts is not meaningful otherwise. The
// largest observed ratio is 1.08 (helical valley).
const EVALUATION_FACTOR = 1.5;

describe('test functions mirror the SciPy reference functions', () => {
    for (const fixture of fixtures) {
        it(`${fixture.id}: f agrees with the Python implementation at x0, scipyX and generic points`, () => {
            const f = TEST_FUNCTIONS[fixture.id];
            expect(f, `no matching test function registered for id "${fixture.id}"`).toBeDefined();

            expect(f(Vector.from(fixture.x0)).allClose(
                Vector.from(fixture.fx0), MIRROR_TOLERANCE, MIRROR_TOLERANCE,
            )).toBe(true);
            expect(f(Vector.from(fixture.scipyX)).allClose(
                Vector.from(fixture.scipyFx), MIRROR_TOLERANCE, MIRROR_TOLERANCE,
            )).toBe(true);

            // x0 and the solution can be special points (e.g. a term vanishes there), so also
            // compare at generic points.
            for (const point of fixture.mirrorPoints) {
                expect(f(Vector.from(point.x)).allClose(
                    Vector.from(point.fx), MIRROR_TOLERANCE, MIRROR_TOLERANCE,
                )).toBe(true);
            }
        });
    }
});

describe('quasiNewton vs scipy.optimize.root hybr reference values', () => {
    for (const fixture of fixtures) {
        it(`matches SciPy hybr for ${fixture.id} (sclx ${fixture.sclx === null ? 'default' : 'given'}): ${fixture.description}`, () => {
            const data = TEST_FUNCTIONS[fixture.id];
            expect(data, `no matching test function registered for id "${fixture.id}"`).toBeDefined();

            const result = quasiNewton(data, Vector.from(fixture.x0), {
                tolx: fixture.tolx,
                tolf: fixture.tolf,
                globalMethod: 'dogleg',
                broydenUpdate: true,
                // Scaling is set explicitly only for some problems (see the generator script).
                ...(fixture.sclx === null ? {} : { sclx: Vector.from(fixture.sclx) }),
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
                fixture.scipyEvaluations * EVALUATION_FACTOR
            );
        });
    }
});