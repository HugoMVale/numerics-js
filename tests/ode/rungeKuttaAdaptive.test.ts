import { describe, it, expect } from 'vitest';
import { Array1D } from '../../src/array/array1d.js';
import { rungeKuttaAdaptive, rungeKuttaAdaptiveStep } from '../../src/ode/rungeKuttaAdaptive.js';
import type { DerivativeFunction } from '../../src/ode/types.js';

const methods = ['rk23', 'rk45'] as const;

describe('rungeKuttaAdaptive additional coverage', () => {
    it('returns the initial state unchanged when tEnd === t0', () => {
        const f: DerivativeFunction = (t, y, out) => { out.data[0] = -y.data[0]; return out; };
        const y0 = new Array1D([3, 4]);
        const result = rungeKuttaAdaptive('rk45', f, 2, 2, y0);

        expect(result.t.size).toBe(1);
        expect(result.t.data[0]).toBe(2);
        expect(result.y.rows).toBe(1);
        expect(result.y.cols).toBe(2);
        expect(Array.from(result.y.row(0).data)).toEqual([3, 4]);
    });

    it.each(methods)('(%s) accepts an explicit initial step size h0', (method) => {
        const f: DerivativeFunction = (t, y, out) => { out.data[0] = -y.data[0]; return out; };
        const y0 = new Array1D([1]);
        const result = rungeKuttaAdaptive(method, f, 0, 1, y0, 1e-8, 1e-8, 0.01);

        const finalY = result.y.row(result.y.rows - 1).data[0];
        expect(finalY).toBeCloseTo(Math.exp(-1), 6);
    });

    it.each(methods)('(%s) rejects an overly large first step and shrinks h to succeed', (method) => {
        // Nonlinear logistic growth: dy/dt = 10*y*(1-y). A first step of 0.5
        // (half the whole integration domain) is far too coarse for this
        // tolerance and forces the controller through several reject+shrink
        // cycles before converging.
        //
        // Note: a linear f = -y is a poor choice for this kind of test --
        // for the exponential test equation the BS23 embedded error
        // estimator is an exact polynomial in h*lambda that can (and does,
        // e.g. at h*lambda = -1) pass through zero for a finite step, so a
        // wildly inaccurate single step can be "accepted" with err === 0.
        // That's a property of the estimator on the linear test problem,
        // not a bug, but it makes a flaky/misleading regression test.
        const f: DerivativeFunction = (t, y, out) => {
            out.data[0] = 10 * y.data[0] * (1 - y.data[0]);
            return out;
        };
        const y0 = new Array1D([0.01]);
        const result = rungeKuttaAdaptive(method, f, 0, 1, y0, 1e-8, 1e-8, 0.5);

        // Reference solution via the closed-form logistic curve.
        const exact = 1 / (1 + (1 / 0.01 - 1) * Math.exp(-10));
        const finalY = result.y.row(result.y.rows - 1).data[0];

        expect(finalY).toBeCloseTo(exact, 6);
        // A handful of accepted points implies rejection+shrink cycles
        // happened along the way -- a single oversized step could not have
        // produced this many accepted intermediate points.
        expect(result.t.size).toBeGreaterThan(5);
    });

    it('throws a RangeError when step size underflows below hMin', () => {
        const f: DerivativeFunction = (t, y, out) => {
            out.data[0] = Math.sin(1 / (y.data[0] + 1e-3)) * 1e8;
            return out;
        };
        const y0 = new Array1D([1]);

        expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, 1e-14, 1e-14, undefined, Infinity, 1e-9))
            .toThrowError(/underflowed below hMin/);
    });

    describe('input validation', () => {
        const f: DerivativeFunction = (t, y, out) => { out.data[0] = -y.data[0]; return out; };
        const y0 = new Array1D([1]);

        it('rejects h0 = 0', () => {
            expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, 1e-6, 1e-6, 0)).toThrow(RangeError);
        });

        it('rejects non-positive hMin', () => {
            expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, 1e-6, 1e-6, undefined, Infinity, 0)).toThrow(RangeError);
        });

        it('rejects non-positive hMax', () => {
            expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, 1e-6, 1e-6, undefined, -1)).toThrow(RangeError);
        });

        it('rejects hMin > hMax', () => {
            expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, 1e-6, 1e-6, undefined, 1, 2)).toThrow(RangeError);
        });

        it('rejects non-positive maxSteps', () => {
            expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, 1e-6, 1e-6, undefined, Infinity, 1e-12, 0)).toThrow(RangeError);
        });

        it('rejects safety outside (0, 1]', () => {
            expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, 1e-6, 1e-6, undefined, Infinity, 1e-12, 1e5, 1.5)).toThrow(RangeError);
            expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, 1e-6, 1e-6, undefined, Infinity, 1e-12, 1e5, 0)).toThrow(RangeError);
        });

        it('rejects minScale > maxScale', () => {
            expect(() => rungeKuttaAdaptive('rk45', f, 0, 1, y0, 1e-6, 1e-6, undefined, Infinity, 1e-12, 1e5, 0.9, 5, 2)).toThrow(RangeError);
        });
    });

    it('rungeKuttaAdaptiveStep throws for an unrecognized method', () => {
        const f: DerivativeFunction = (t, y, out) => { out.data[0] = -y.data[0]; return out; };
        const y0 = new Array1D([1]);
        const out = new Array1D(1);
        const scratch = {
            k1: new Array1D(1), k2: new Array1D(1), k3: new Array1D(1), k4: new Array1D(1),
            k5: new Array1D(1), k6: new Array1D(1), k7: new Array1D(1), yTemp: new Array1D(1),
        };

        expect(() => rungeKuttaAdaptiveStep('bogus' as any, f, 0, y0, 0.1, out, scratch)).toThrow(RangeError);
    });

    it('throws for an unrecognized method at the top level', () => {
        const f: DerivativeFunction = (t, y, out) => { out.data[0] = -y.data[0]; return out; };
        const y0 = new Array1D([1]);

        expect(() => rungeKuttaAdaptive('bogus' as any, f, 0, 1, y0)).toThrow(RangeError);
    });

    it.each(methods)('(%s) produces a monotonically increasing t for forward integration', (method) => {
        const f: DerivativeFunction = (t, y, out) => { out.data[0] = -y.data[0]; return out; };
        const y0 = new Array1D([1]);
        const result = rungeKuttaAdaptive(method, f, 0, 5, y0, 1e-6, 1e-6);

        for (let i = 1; i < result.t.size; i++) {
            expect(result.t.data[i]).toBeGreaterThan(result.t.data[i - 1]);
        }
        expect(result.y.rows).toBe(result.t.size);
        expect(result.y.cols).toBe(1);
    });
});