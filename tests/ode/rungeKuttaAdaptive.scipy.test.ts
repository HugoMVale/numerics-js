import { describe, expect, it } from 'vitest';
import { Array1D } from '../../src/array/array1d.js';
import { rungeKuttaAdaptive } from '../../src/ode/rungeKuttaAdaptive.js';
import type { DerivativeFunction, RungeKuttaAdaptiveMethod } from '../../src/ode/types.js';
import fixtures from './fixtures/rungeKuttaAdaptive.scipy.json' with { type: 'json' };

// Must mirror CASES in scripts/benchmarks/generate_runge_kutta_adaptive_fixtures.py
// (same id, derivative, time span, and initial state).
const DERIVATIVES: Record<string, DerivativeFunction> = {
    exponential_decay: (_t, y, out) => {
        out.data[0] = -y.data[0];
        return out;
    },
    logistic_growth: (_t, y, out) => {
        out.data[0] = 10 * y.data[0] * (1 - y.data[0]);
        return out;
    },
    harmonic_oscillator: (_t, y, out) => {
        out.data[0] = y.data[1];
        out.data[1] = -y.data[0];
        return out;
    },
    backward_exponential: (_t, y, out) => {
        out.data[0] = -2 * y.data[0];
        return out;
    },
};

const EVALUATION_FACTOR = 1.01;
const FINAL_STATE_TOLERANCE_FACTOR = 10;

describe('rungeKuttaAdaptive vs scipy.integrate.solve_ivp reference values', () => {
    for (const fixture of fixtures) {
        for (const method of ['rk23', 'rk45'] as const) {
            it(`matches SciPy ${method.toUpperCase()} for ${fixture.id}: ${fixture.description}`, () => {
                const f = DERIVATIVES[fixture.id];
                expect(f, `no matching derivative registered for id "${fixture.id}"`).toBeDefined();

                const result = rungeKuttaAdaptive(
                    method as RungeKuttaAdaptiveMethod,
                    f,
                    fixture.t0,
                    fixture.tEnd,
                    new Array1D(fixture.y0),
                    { atol: fixture.atol, rtol: fixture.rtol }
                );
                const reference = fixture.scipy[method];
                const finalY = result.y.row(result.y.rows - 1).data;

                for (let i = 0; i < finalY.length; i++) {
                    const tolerance = FINAL_STATE_TOLERANCE_FACTOR * (
                        fixture.atol + fixture.rtol * Math.max(Math.abs(finalY[i]), Math.abs(reference.finalY[i]))
                    );
                    expect(Math.abs(finalY[i] - reference.finalY[i])).toBeLessThanOrEqual(tolerance);
                }

                expect(result.evaluations).toBeLessThanOrEqual(reference.evaluations * EVALUATION_FACTOR);
            });
        }
    }
});