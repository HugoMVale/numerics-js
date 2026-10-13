import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { rungeKuttaAdaptive } from '../../src/ode/rungeKuttaAdaptive.js';
import type { DerivativeFunction, OdeEvent } from '../../src/ode/types.js';
import fixtures from './rungeKuttaAdaptive.events.scipy.json' with { type: 'json' };

// Must mirror CASES in generate_runge_kutta_adaptive_events_fixtures.py
// (same id, derivative, time span, initial state and step, and events).
const oscillator: DerivativeFunction = (_t, y, out) => {
    out.data[0] = y.data[1];
    out.data[1] = -y.data[0];
    return out;
};

const CASES: Record<string, { f: DerivativeFunction; events: OdeEvent[] }> = {
    oscillator_zero_crossings: {
        f: oscillator,
        events: [{ fn: (_t, y) => y.data[0] }],
    },
    decay_threshold_terminal: {
        f: (_t, y, out) => {
            out.data[0] = -y.data[0];
            return out;
        },
        events: [{ fn: (_t, y) => y.data[0] - 0.25, direction: 'falling', terminal: true }],
    },
    logistic_rising: {
        f: (_t, y, out) => {
            out.data[0] = 10 * y.data[0] * (1 - y.data[0]);
            return out;
        },
        events: [{ fn: (_t, y) => y.data[0] - 0.5, direction: 'rising' }],
    },
    oscillator_terminal_count: {
        f: oscillator,
        events: [{ fn: (_t, y) => y.data[0], terminal: 3 }],
    },
    backward_growth: {
        f: (_t, y, out) => {
            out.data[0] = -2 * y.data[0];
            return out;
        },
        events: [{ fn: (_t, y) => y.data[0] - 1 }],
    },
    oscillator_two_events: {
        f: oscillator,
        events: [{ fn: (_t, y) => y.data[0] }, { fn: (_t, y) => y.data[1] - 0.5, terminal: 2 }],
    },
};

// With the same initial step, numerics-js and SciPy take the same steps up to ~1e-11 (event times agree to ~1e-10 or better), and the dense
// outputs are the same polynomials, so event times agree far inside the integration tolerance.
const TIME_TOLERANCE = 1e-8;
const STATE_TOLERANCE = 1e-8;

describe('rungeKuttaAdaptive events vs scipy.integrate.solve_ivp reference values', () => {
    for (const fixture of fixtures) {
        for (const method of ['rk23', 'rk45'] as const) {
            it(`matches SciPy ${method.toUpperCase()} events for ${fixture.id}: ${fixture.description}`, () => {
                const definition = CASES[fixture.id];
                expect(definition, `no matching case registered for id "${fixture.id}"`).toBeDefined();

                const result = rungeKuttaAdaptive(method, definition.f, fixture.t0, fixture.tEnd, new Vector(fixture.y0), {
                    atol: fixture.atol,
                    rtol: fixture.rtol,
                    h0: fixture.h0,
                    events: definition.events,
                });
                const reference = fixture.scipy[method];

                expect(result.success).toBe(true);
                expect(result.message === 'Integration stopped by a terminal event.').toBe(reference.terminated);
                expect(Math.abs(result.t.get(result.t.size - 1) - reference.finalT)).toBeLessThan(TIME_TOLERANCE);

                definition.events.forEach((_event, index) => {
                    const found = result.events!.filter((e) => e.index === index);
                    const expectedTimes = reference.tEvents[index];
                    const expectedStates = reference.yEvents[index];
                    expect(found.map((e) => e.t).length, `event ${index} count`).toBe(expectedTimes.length);
                    found.forEach((e, k) => {
                        expect(Math.abs(e.t - expectedTimes[k])).toBeLessThan(TIME_TOLERANCE);
                        e.y.data.forEach((value, i) => {
                            expect(Math.abs(value - expectedStates[k][i])).toBeLessThan(STATE_TOLERANCE);
                        });
                    });
                });
            });
        }
    }
});
