import { describe, expect, it } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { trBdf2Adaptive } from '../../src/ode/trBdf2Adaptive.js';
import type { DerivativeFunction, JacobianFunction } from '../../src/ode/types.js';
import fixtures from './trBdf2Adaptive.scipy.json' with { type: 'json' };

// Must mirror CASES in generate_tr_bdf2_adaptive_fixtures.py
// (same id, derivative, time span, state, and tolerances).
const LAMBDA = -1e5;
const MU = 1000;

const DERIVATIVES: Record<string, DerivativeFunction> = {
    prothero_robinson: (t, y, out) => {
        out.data[0] = LAMBDA * (y.data[0] - Math.sin(t)) + Math.cos(t);
        return out;
    },
    robertson: (_t, y, out) => {
        const [a, b, c] = y.data;
        out.data[0] = -0.04 * a + 1e4 * b * c;
        out.data[1] = 0.04 * a - 1e4 * b * c - 3e7 * b * b;
        out.data[2] = 3e7 * b * b;
        return out;
    },
    van_der_pol_mu1000: (_t, y, out) => {
        out.data[0] = y.data[1];
        out.data[1] = MU * (1 - y.data[0] * y.data[0]) * y.data[1] - y.data[0];
        return out;
    },
    stiff_linear_chain: (_t, y, out) => {
        out.data[0] = -1000 * y.data[0] + y.data[1];
        out.data[1] = -y.data[1] + y.data[2];
        out.data[2] = -0.1 * y.data[2];
        return out;
    },
};

// Analytic Jacobians; the solver must give the same answer with or without them.
const JACOBIANS: Record<string, JacobianFunction> = {
    prothero_robinson: (_t, _y, J) => J.set(0, 0, LAMBDA),
    robertson: (_t, y, J) => {
        const [, b, c] = y.data;
        J.set(0, 0, -0.04).set(0, 1, 1e4 * c).set(0, 2, 1e4 * b);
        J.set(1, 0, 0.04).set(1, 1, -1e4 * c - 6e7 * b).set(1, 2, -1e4 * b);
        J.set(2, 1, 6e7 * b);
        return J;
    },
    van_der_pol_mu1000: (_t, y, J) => {
        const [x, v] = y.data;
        J.set(0, 1, 1);
        J.set(1, 0, -2 * MU * x * v - 1).set(1, 1, MU * (1 - x * x));
        return J;
    },
    stiff_linear_chain: (_t, _y, J) => {
        J.set(0, 0, -1000).set(0, 1, 1);
        J.set(1, 1, -1).set(1, 2, 1);
        J.set(2, 2, -0.1);
        return J;
    },
};

// The tolerances only control the local error of each step, so the global error at tEnd
// exceeds `atol + rtol * |y|` by an amount that grows with the number of steps and with the
// sensitivity of the problem. For this second-order method the observed worst ratio is about
// 110 (Van der Pol, mu = 1000, at the tight setting, where phase errors accumulate over many
// steps). The factor guards against gross errors; the order and the error estimator are
// verified separately in tests/ode/trBdf2Adaptive.test.ts.
const FINAL_STATE_TOLERANCE_FACTOR = 200;

describe('trBdf2Adaptive vs scipy.integrate.solve_ivp reference values', () => {
    for (const fixture of fixtures) {
        for (const setting of fixture.settings) {
            for (const withJacobian of [false, true]) {
                const label = withJacobian ? 'analytic Jacobian' : 'finite-difference Jacobian';
                it(`matches the SciPy Radau reference for ${fixture.id}, ${setting.name} tolerances (${label}): ${fixture.description}`, () => {
                    const f = DERIVATIVES[fixture.id];
                    expect(f, `no matching derivative registered for id "${fixture.id}"`).toBeDefined();

                    const result = trBdf2Adaptive(f, fixture.t0, fixture.tEnd, new Vector(fixture.y0), {
                        atol: setting.atol,
                        rtol: setting.rtol,
                        jac: withJacobian ? JACOBIANS[fixture.id] : undefined,
                    });
                    expect(result.success, result.message).toBe(true);

                    const finalY = result.y.row(result.y.rows - 1).data;
                    const referenceY = fixture.reference.finalY;
                    expect(finalY.length).toBe(referenceY.length);

                    for (let i = 0; i < finalY.length; i++) {
                        const tolerance = FINAL_STATE_TOLERANCE_FACTOR * (
                            setting.atol + setting.rtol * Math.max(Math.abs(finalY[i]), Math.abs(referenceY[i]))
                        );
                        expect(Math.abs(finalY[i] - referenceY[i])).toBeLessThanOrEqual(tolerance);
                    }

                    // Sanity bounds on the work done. The Jacobian must be reused across steps. A
                    // factorization is needed whenever the step size changes, which can be on nearly
                    // every attempt during a fast transient, so only a loose bound applies to the LUs.
                    const steps = result.t.size - 1;
                    expect(result.evaluationsJacobian).toBeLessThan(steps);
                    expect(result.decompositions).toBeLessThanOrEqual(2 * steps);
                });
            }
        }
    }
});
