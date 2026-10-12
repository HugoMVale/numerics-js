import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';
import { trBdf2Adaptive } from '../../src/ode/trBdf2Adaptive.js';
import type { DerivativeFunction, JacobianFunction } from '../../src/ode/types.js';

const lastRow = (result: { y: Matrix }): Float64Array => result.y.row(result.y.rows - 1).data;

// Linear, non-stiff: y' = -y.
const decay: DerivativeFunction = (_t, y, out) => {
    out.data[0] = -y.data[0];
    return out;
};

// Prothero-Robinson problem with exact solution y(t) = sin(t) when y(0) = 0.
const LAMBDA = -1e5;
const protheroRobinson: DerivativeFunction = (t, y, out) => {
    out.data[0] = LAMBDA * (y.data[0] - Math.sin(t)) + Math.cos(t);
    return out;
};
const protheroRobinsonJac: JacobianFunction = (_t, _y, J) => J.set(0, 0, LAMBDA);

describe('trBdf2Adaptive', () => {
    it('returns the initial state unchanged when tEnd === t0', () => {
        const y0 = new Vector([3, 4]);
        const result = trBdf2Adaptive(decay, 2, 2, y0);

        expect(result.method).toBe('trbdf2');
        expect(result.success).toBe(true);
        expect(result.message).toBe('Integration successful.');
        expect(result.t.size).toBe(1);
        expect(result.t.data[0]).toBe(2);
        expect(Array.from(result.y.row(0).data)).toEqual([3, 4]);
        expect(result.evaluations).toBe(0);
        expect(result.evaluationsJacobian).toBe(0);
        expect(result.decompositions).toBe(0);
    });

    it('integrates a non-stiff problem accurately and reports a well-formed result', () => {
        const y0 = new Vector([1]);
        const result = trBdf2Adaptive(decay, 0, 1, y0, { atol: 1e-9, rtol: 1e-9 });

        expect(result.method).toBe('trbdf2');
        expect(result.success).toBe(true);
        expect(result.message).toBe('Integration successful.');
        expect(result.t.size).toBe(result.y.rows);
        expect(result.y.cols).toBe(1);
        expect(result.t.data[0]).toBe(0);
        expect(result.t.data[result.t.size - 1]).toBe(1);
        for (let i = 1; i < result.t.size; i++) {
            expect(result.t.data[i]).toBeGreaterThan(result.t.data[i - 1]);
        }
        expect(result.y.row(0).data[0]).toBe(1);
        expect(lastRow(result)[0]).toBeCloseTo(Math.exp(-1), 5);
        expect(Array.from(y0.data)).toEqual([1]); // input is never mutated
    });

    it('matches the exact solution of a stiff linear system', () => {
        // y1' = -100 y1 + y2, y2' = -y2
        // y2 = y2(0) e^-t,  y1 = y1(0) e^-100t + y2(0)/99 (e^-t - e^-100t)
        const A = Matrix.from([[-100, 1], [0, -1]]);
        const f: DerivativeFunction = (_t, y, out) => out.set(A.mulVec(y).data);
        let jacCalls = 0;
        const jac: JacobianFunction = (_t, _y, J) => {
            jacCalls++;
            J.data.set(A.data);
            return J;
        };
        const y0 = new Vector([1, 2]);
        const tEnd = 3;
        const exact = [
            1 * Math.exp(-100 * tEnd) + (2 / 99) * (Math.exp(-tEnd) - Math.exp(-100 * tEnd)),
            2 * Math.exp(-tEnd),
        ];

        for (const options of [{}, { jac }]) {
            const result = trBdf2Adaptive(f, 0, tEnd, y0, { atol: 1e-9, rtol: 1e-7, ...options });
            expect(result.success).toBe(true);
            const y = lastRow(result);
            expect(y[0]).toBeCloseTo(exact[0], 5);
            expect(y[1]).toBeCloseTo(exact[1], 5);
            expect(result.evaluationsJacobian).toBeGreaterThanOrEqual(1);
        }
        expect(jacCalls).toBeGreaterThanOrEqual(1);
    });

    it('takes few steps on a very stiff problem where an explicit method cannot', () => {
        const result = trBdf2Adaptive(protheroRobinson, 0, 2, new Vector([0]), { atol: 1e-8, rtol: 1e-5 });

        expect(result.success).toBe(true);
        expect(lastRow(result)[0]).toBeCloseTo(Math.sin(2), 6);
        // Explicit Runge-Kutta needs on the order of |lambda| * T = 2e5 steps for stability.
        expect(result.t.size - 1).toBeLessThan(50);
    });

    it('recovers from an initial stiff transient', () => {
        const result = trBdf2Adaptive(protheroRobinson, 0, 2, new Vector([1]), { atol: 1e-8, rtol: 1e-5 });

        expect(result.success).toBe(true);
        expect(lastRow(result)[0]).toBeCloseTo(Math.sin(2), 6);
        expect(result.t.size - 1).toBeLessThan(1000);
    });

    it('damps very stiff modes in a single large step (L-stability)', () => {
        // The trapezoidal rule would give y1 ~ -1 here; an L-stable method gives y1 ~ 0.
        const f: DerivativeFunction = (_t, y, out) => {
            out.data[0] = -1e8 * y.data[0];
            return out;
        };
        const result = trBdf2Adaptive(f, 0, 2, new Vector([1]), { atol: 10, rtol: 1e-3, h0: 1, hMax: 1 });

        expect(result.success).toBe(true);
        expect(result.t.data[1]).toBe(1);
        expect(Math.abs(result.y.get(1, 0))).toBeLessThan(1e-6);
    });

    it('converges with an error that shrinks like tol^(2/3), consistent with second order', () => {
        const rtols = [1e-3, 1e-5, 1e-7];
        const errors = rtols.map((rtol) => {
            const result = trBdf2Adaptive(decay, 0, 5, new Vector([1]), { atol: 1e-14, rtol });
            return Math.abs(lastRow(result)[0] - Math.exp(-5));
        });
        const slope = Math.log(errors[0] / errors[2]) / Math.log(rtols[0] / rtols[2]);

        expect(errors[1]).toBeLessThan(errors[0]);
        expect(errors[2]).toBeLessThan(errors[1]);
        expect(slope).toBeGreaterThan(0.5);
        expect(slope).toBeLessThan(0.85);
    });

    it('integrates backward in time', () => {
        // y' = -2 y from t = 2 to t = -1, with y(2) = e^-4, so y(-1) = e^2.
        const f: DerivativeFunction = (_t, y, out) => {
            out.data[0] = -2 * y.data[0];
            return out;
        };
        const result = trBdf2Adaptive(f, 2, -1, new Vector([Math.exp(-4)]), { atol: 1e-10, rtol: 1e-8 });

        expect(result.success).toBe(true);
        expect(result.t.data[result.t.size - 1]).toBe(-1);
        for (let i = 1; i < result.t.size; i++) {
            expect(result.t.data[i]).toBeLessThan(result.t.data[i - 1]);
        }
        expect(lastRow(result)[0] / Math.exp(2)).toBeCloseTo(1, 4);
    });

    it('preserves linear invariants on the stiff Robertson problem', () => {
        const f: DerivativeFunction = (_t, y, out) => {
            const [a, b, c] = y.data;
            out.data[0] = -0.04 * a + 1e4 * b * c;
            out.data[1] = 0.04 * a - 1e4 * b * c - 3e7 * b * b;
            out.data[2] = 3e7 * b * b;
            return out;
        };
        const result = trBdf2Adaptive(f, 0, 1e3, new Vector([1, 0, 0]), { atol: 1e-10, rtol: 1e-6 });

        expect(result.success).toBe(true);
        for (let i = 0; i < result.y.rows; i++) {
            const row = result.y.row(i).data;
            expect(Math.abs(row[0] + row[1] + row[2] - 1)).toBeLessThan(1e-9);
        }
        // Species 1 is monotonically consumed and stays within (0, 1].
        const first = result.y.col(0).data;
        for (let i = 1; i < first.length; i++) {
            expect(first[i]).toBeLessThanOrEqual(first[i - 1] + 1e-9);
        }
    });

    it('spends no derivative evaluation on the first stage of a step (smoothed first stage)', () => {
        // For f = 0 each Newton solve converges at its first correction, so a step costs exactly
        // two evaluations (one per implicit stage); the first stage reuses the previous last stage.
        let calls = 0;
        const f: DerivativeFunction = (_t, _y, out) => {
            calls++;
            return out.fill(0);
        };
        const result = trBdf2Adaptive(f, 0, 1, new Vector([1]), { h0: 0.25, hMax: 0.25, jac: (_t, _y, J) => J });
        const steps = result.t.size - 1;

        expect(result.success).toBe(true);
        expect(steps).toBe(4);
        expect(result.evaluations).toBe(calls);
        expect(result.evaluations).toBe(1 + 2 * steps); // initial f(t0, y0), then two per step
        expect(result.evaluationsJacobian).toBe(1);
        expect(result.decompositions).toBe(1); // constant step size: a single factorization
    });

    describe('Jacobian handling', () => {
        it('accepts an analytic Jacobian and avoids finite-difference evaluations', () => {
            const options = { atol: 1e-8, rtol: 1e-5 };
            const numeric = trBdf2Adaptive(protheroRobinson, 0, 2, new Vector([0]), options);
            const analytic = trBdf2Adaptive(protheroRobinson, 0, 2, new Vector([0]), { ...options, jac: protheroRobinsonJac });

            expect(analytic.success).toBe(true);
            expect(analytic.evaluationsJacobian).toBeGreaterThanOrEqual(1);
            expect(numeric.evaluations).toBeGreaterThan(analytic.evaluations);
            expect(lastRow(analytic)[0]).toBeCloseTo(lastRow(numeric)[0], 6);
        });

        it('calls the Jacobian with a zeroed n x n buffer at the current point', () => {
            const calls: { t: number; y: number[]; rows: number; cols: number; zero: boolean }[] = [];
            const f: DerivativeFunction = (_t, y, out) => {
                out.data[0] = y.data[1];
                out.data[1] = -y.data[0] - 50 * y.data[1];
                return out;
            };
            const jac: JacobianFunction = (t, y, J) => {
                calls.push({ t, y: Array.from(y.data), rows: J.rows, cols: J.cols, zero: J.data.every((v) => v === 0) });
                J.set(0, 1, 1);
                J.set(1, 0, -1);
                J.set(1, 1, -50);
                return J;
            };
            const result = trBdf2Adaptive(f, 0, 1, new Vector([1, 0]), { jac });

            expect(result.success).toBe(true);
            expect(calls.length).toBe(result.evaluationsJacobian);
            expect(calls[0]).toEqual({ t: 0, y: [1, 0], rows: 2, cols: 2, zero: true });
            expect(calls.every((c) => c.rows === 2 && c.cols === 2 && c.zero)).toBe(true);
        });

        it('counts every derivative evaluation, including finite-difference Jacobians', () => {
            let calls = 0;
            const f: DerivativeFunction = (t, y, out) => {
                calls++;
                return protheroRobinson(t, y, out);
            };
            const result = trBdf2Adaptive(f, 0, 2, new Vector([0]), { atol: 1e-8, rtol: 1e-5, h0: 1e-6 });

            expect(result.evaluations).toBe(calls);
            expect(result.evaluationsJacobian).toBeGreaterThanOrEqual(1);
        });

        it('reuses the Jacobian and LU factors across steps', () => {
            const result = trBdf2Adaptive(decay, 0, 5, new Vector([1]), { atol: 1e-9, rtol: 1e-9 });
            const steps = result.t.size - 1;

            expect(result.success).toBe(true);
            expect(steps).toBeGreaterThan(50);
            expect(result.evaluationsJacobian).toBeLessThanOrEqual(2);
            expect(result.decompositions).toBeLessThan(steps / 4);
            expect(result.decompositions).toBeGreaterThanOrEqual(result.evaluationsJacobian);
        });

        it('refreshes a poor Jacobian when the Newton iteration struggles', () => {
            let calls = 0;
            const badFirstJac: JacobianFunction = (_t, _y, J) => {
                calls++;
                if (calls > 1) J.set(0, 0, LAMBDA); // the first call leaves J = 0
                return J;
            };
            const result = trBdf2Adaptive(protheroRobinson, 0, 2, new Vector([0]), { atol: 1e-8, rtol: 1e-5, jac: badFirstJac });

            expect(result.success).toBe(true);
            expect(result.evaluationsJacobian).toBeGreaterThanOrEqual(2);
            expect(lastRow(result)[0]).toBeCloseTo(Math.sin(2), 5);
        });
    });

    describe('limits and failures', () => {
        it('returns an unsuccessful result when the Newton iteration cannot converge', () => {
            const f: DerivativeFunction = (t, y, out) => {
                out.data[0] = t > 0.5 ? Number.NaN : -y.data[0];
                return out;
            };
            const result = trBdf2Adaptive(f, 0, 1, new Vector([1]), { hMin: 1e-6 });

            expect(result.success).toBe(false);
            expect(result.message).toMatch(/underflowed below hMin/);
            expect(result.message).toMatch(/Newton/);
            expect(result.t.size).toBe(result.y.rows);
            expect(result.t.data[result.t.size - 1]).toBeLessThanOrEqual(0.5);
        });

        it('treats a singular iteration matrix as a failed step', () => {
            const nanJac: JacobianFunction = (_t, _y, J) => J.fill(Number.NaN);
            const result = trBdf2Adaptive(decay, 0, 1, new Vector([1]), { jac: nanJac, hMin: 1e-9 });

            expect(result.success).toBe(false);
            expect(result.message).toMatch(/underflowed below hMin/);
            expect(result.decompositions).toBe(0);
        });

        it('returns an unsuccessful result when step size underflows because of error control', () => {
            const f: DerivativeFunction = (_t, y, out) => {
                out.data[0] = Math.sin(1 / (y.data[0] + 1e-3)) * 1e8;
                return out;
            };
            const result = trBdf2Adaptive(f, 0, 1, new Vector([1]), { atol: 1e-14, rtol: 1e-14, hMin: 1e-9 });

            expect(result.success).toBe(false);
            expect(result.message).toMatch(/underflowed below hMin/);
            expect(result.t.size).toBe(result.y.rows);
        });

        it('returns an unsuccessful result when maxSteps is exceeded', () => {
            const result = trBdf2Adaptive(decay, 0, 1, new Vector([1]), { h0: 0.1, hMax: 0.1, maxSteps: 1 });

            expect(result.success).toBe(false);
            expect(result.message).toBe('trBdf2Adaptive: exceeded maxSteps (1) without reaching tEnd.');
            expect(result.t.size).toBe(result.y.rows);
            expect(result.t.data[result.t.size - 1]).toBeCloseTo(0.1, 12);
        });

        it('succeeds when the final allowed attempt reaches tEnd', () => {
            const f: DerivativeFunction = (_t, _y, out) => out.fill(0);
            const result = trBdf2Adaptive(f, 0, 1, new Vector([1]), { h0: 1, maxSteps: 1 });

            expect(result.success).toBe(true);
            expect(result.t.data[result.t.size - 1]).toBe(1);
        });

        it('accepts an explicit initial step size h0 of either sign', () => {
            for (const h0 of [0.01, -0.01]) {
                const result = trBdf2Adaptive(decay, 0, 1, new Vector([1]), { atol: 1e-9, rtol: 1e-9, h0 });
                expect(result.success).toBe(true);
                expect(lastRow(result)[0]).toBeCloseTo(Math.exp(-1), 5);
            }
        });

        it('never exceeds hMax', () => {
            const result = trBdf2Adaptive(decay, 0, 1, new Vector([1]), { hMax: 0.05 });

            expect(result.success).toBe(true);
            for (let i = 1; i < result.t.size; i++) {
                expect(result.t.data[i] - result.t.data[i - 1]).toBeLessThanOrEqual(0.05 + 1e-12);
            }
        });
    });

    describe('input validation', () => {
        const y0 = new Vector([1]);

        it('rejects a state without components', () => {
            expect(() => trBdf2Adaptive(decay, 0, 1, new Vector(0))).toThrow(RangeError);
        });

        it('rejects h0 = 0', () => {
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { h0: 0 })).toThrow(RangeError);
        });

        it.each([
            { atol: Number.NaN, rtol: 1e-6 },
            { atol: 1e-6, rtol: Number.NaN },
            { atol: Number.POSITIVE_INFINITY, rtol: 1e-6 },
            { atol: 1e-6, rtol: Number.POSITIVE_INFINITY },
            { atol: -1e-6, rtol: 1e-6 },
            { atol: 1e-6, rtol: -1e-6 },
        ])('rejects invalid tolerances ($atol, $rtol)', (options) => {
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, options)).toThrow(RangeError);
        });

        it('rejects atol = rtol = 0', () => {
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { atol: 0, rtol: 0 })).toThrow(RangeError);
        });

        it('rejects non-positive hMin and hMax, and hMin > hMax', () => {
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { hMin: 0 })).toThrow(RangeError);
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { hMax: -1 })).toThrow(RangeError);
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { hMax: 1, hMin: 2 })).toThrow(RangeError);
        });

        it('rejects non-positive or non-integer maxSteps', () => {
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { maxSteps: 0 })).toThrow(RangeError);
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { maxSteps: 1.5 })).toThrow(RangeError);
        });

        it('rejects safety outside (0, 1]', () => {
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { safety: 1.5 })).toThrow(RangeError);
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { safety: 0 })).toThrow(RangeError);
        });

        it('rejects minScale > maxScale and non-positive minScale', () => {
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { minScale: 5, maxScale: 2 })).toThrow(RangeError);
            expect(() => trBdf2Adaptive(decay, 0, 1, y0, { minScale: 0 })).toThrow(RangeError);
        });
    });
});
