import { describe, expect, it } from 'vitest';
import { Array1D } from '../../src/array/array1d.js';
import { rungeKuttaFixed, rungeKuttaStep, type RungeKuttaMethod } from '../../src/ode/rungeKuttaFixed.js';
import type { DerivativeFunction } from '../../src/ode/types.js';

/** dy/dt = y -> y(t) = y0 * e^t */
const exponential: DerivativeFunction = (_t, y, out) => out.set(y.data);

/** dy/dt = -y -> y(t) = y0 * e^-t */
const decay: DerivativeFunction = (_t, y, out) => out.set(y.data).multSelf(-1);

/** Simple harmonic oscillator: y0' = y1, y1' = -y0 -> y0(t) = cos t, y1(t) = -sin t (with y0(0)=1, y1(0)=0) */
const shm: DerivativeFunction = (_t, y, out) => {
    out.set(0, y.get(1));
    out.set(1, -y.get(0));
    return out;
};

const methods: RungeKuttaMethod[] = ['euler', 'midpoint', 'trapezoid', 'rk4'];
const orderOf: Record<RungeKuttaMethod, number> = {
    euler: 1,
    midpoint: 2,
    trapezoid: 2,
    rk4: 4,
};

describe('rungeKuttaStep', () => {
    it('euler: matches y + h*f(t,y) exactly', () => {
        const y = new Array1D([2]);
        const out = new Array1D(1);
        rungeKuttaStep('euler', exponential, 0, y, 0.1, out);
        expect(out.get(0)).toBeCloseTo(2 + 0.1 * 2, 12);
    });

    it('midpoint: matches the hand-derived closed form for dy/dt = y', () => {
        // k1 = y0; k2 = y0*(1 + h/2); y1 = y0 + h*k2
        const y0 = 3;
        const h = 0.2;
        const expected = y0 + h * (y0 * (1 + h / 2));
        const out = new Array1D(1);
        rungeKuttaStep('midpoint', exponential, 0, new Array1D([y0]), h, out);
        expect(out.get(0)).toBeCloseTo(expected, 12);
    });

    it('trapezoid: matches the hand-derived closed form for dy/dt = y', () => {
        // k1 = y0; k2 = y0*(1+h); y1 = y0 + h/2*(k1+k2)
        const y0 = 3;
        const h = 0.2;
        const k1 = y0;
        const k2 = y0 * (1 + h);
        const expected = y0 + (h / 2) * (k1 + k2);
        const out = new Array1D(1);
        rungeKuttaStep('trapezoid', exponential, 0, new Array1D([y0]), h, out);
        expect(out.get(0)).toBeCloseTo(expected, 12);
    });

    it('rk4: matches the truncated Taylor series of e^h for dy/dt = y', () => {
        const y0 = 1;
        const h = 0.1;
        const expected = y0 * (1 + h + h ** 2 / 2 + h ** 3 / 6 + h ** 4 / 24);
        const out = new Array1D(1);
        rungeKuttaStep('rk4', exponential, 0, new Array1D([y0]), h, out);
        expect(out.get(0)).toBeCloseTo(expected, 12);
    });

    it('returns the same `out` reference it was given', () => {
        const out = new Array1D(1);
        const returned = rungeKuttaStep('euler', exponential, 0, new Array1D([1]), 0.1, out);
        expect(returned).toBe(out);
    });

    it('works with the default scratch (none supplied)', () => {
        const out = new Array1D(1);
        rungeKuttaStep('rk4', exponential, 0, new Array1D([1]), 0.1, out);
        expect(out.get(0)).toBeCloseTo(Math.E ** 0.1, 6);
    });

    it('does not mutate the input state vector `y`', () => {
        const y = new Array1D([5]);
        const out = new Array1D(1);
        rungeKuttaStep('rk4', exponential, 0, y, 0.1, out);
        expect(y.get(0)).toBe(5);
    });

    it('is safe to call with `out === y` (aliased buffers)', () => {
        const y = new Array1D([1]);
        rungeKuttaStep('rk4', exponential, 0, y, 0.1, y);
        expect(y.get(0)).toBeCloseTo(Math.E ** 0.1, 6);
    });

    it('throws a RangeError for an unrecognized method', () => {
        const out = new Array1D(1);
        expect(() =>
            rungeKuttaStep('bogus' as unknown as RungeKuttaMethod, exponential, 0, new Array1D([1]), 0.1, out)
        ).toThrow(RangeError);
    });
});

describe('rungeKuttaFixed', () => {
    it.each(methods)('%s: records the initial state as the first row, unchanged', (method) => {
        const y0 = new Array1D([1, 2, 3]);
        const f: DerivativeFunction = (_t, y, out) => out.reset();
        const result = rungeKuttaFixed(method, f, 0, 1, y0, 0.5);
        expect(Array.from(result.y.row(0).data)).toEqual([1, 2, 3]);
    });

    it.each(methods)('%s: produces the expected time grid for an exact multiple of h', (method) => {
        const f: DerivativeFunction = (_t, y, out) => out.reset();
        const result = rungeKuttaFixed(method, f, 0, 1, new Array1D([0]), 0.25);
        expect(result.t.size).toBe(5);
        expect(Array.from(result.t.data)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    });

    it.each([
        ['euler', 1],
        ['midpoint', 2],
        ['trapezoid', 2],
        ['rk4', 4],
    ] as const)('%s: reports every derivative evaluation', (method, stagesPerStep) => {
        let calls = 0;
        const f: DerivativeFunction = (_t, _y, out) => {
            calls++;
            return out.reset();
        };
        const result = rungeKuttaFixed(method, f, 0, 1, new Array1D([0]), 0.5);

        expect(result.evaluations).toBe(calls);
        expect(result.evaluations).toBe(2 * stagesPerStep);
    });

    it.each([
        ['euler', 1],
        ['midpoint', 2],
        ['trapezoid', 2],
        ['rk4', 4],
    ] as const)('%s: counts a shortened final step', (method, stagesPerStep) => {
        let calls = 0;
        const f: DerivativeFunction = (_t, _y, out) => {
            calls++;
            return out.reset();
        };
        const result = rungeKuttaFixed(method, f, 0, 1, new Array1D([0]), 0.6);

        expect(result.evaluations).toBe(calls);
        expect(result.evaluations).toBe(2 * stagesPerStep);
    });

    it.each(methods)('%s: adds one partial final step when h does not evenly divide the span', (method) => {
        const f: DerivativeFunction = (_t, y, out) => out.reset();
        const result = rungeKuttaFixed(method, f, 0, 1, new Array1D([0]), 0.3);
        // 3 full steps of 0.3 (0, 0.3, 0.6, 0.9) + 1 partial step to reach 1.0
        expect(result.t.size).toBe(5);
        const t = Array.from(result.t.data);
        expect(t[0]).toBe(0);
        expect(t[3]).toBeCloseTo(0.9, 9);
        expect(t[4]).toBe(1); // last point is always forced to exactly tEnd
    });

    it.each(methods)('%s: the last recorded time is exactly tEnd, never an accumulated approximation', (method) => {
        const f: DerivativeFunction = (_t, y, out) => out.reset();
        const result = rungeKuttaFixed(method, f, 0, 1, new Array1D([0]), 1 / 3);
        expect(result.t.data[result.t.size - 1]).toBe(1);
    });

    it.each(methods)('%s: converges at the theoretically expected order for dy/dt = y', (method) => {
        const errorAt = (h: number) => {
            const result = rungeKuttaFixed(method, exponential, 0, 1, new Array1D([1]), h);
            const last = result.y.get(result.y.rows - 1, 0);
            return Math.abs(last - Math.E);
        };
        const errBig = errorAt(0.1);
        const errSmall = errorAt(0.05);
        const expectedRatio = 2 ** orderOf[method];
        // Empirical ratio should be reasonably close to the theoretical one (loose bound
        // since we're only halving h once and floating point / higher-order terms add noise).
        expect(errBig / errSmall).toBeGreaterThan(expectedRatio * 0.7);
        expect(errBig / errSmall).toBeLessThan(expectedRatio * 1.3);
    });

    it('rk4 integrates a 2D simple harmonic oscillator accurately over a full period', () => {
        const result = rungeKuttaFixed('rk4', shm, 0, 2 * Math.PI, new Array1D([1, 0]), 0.01);
        const last = result.y.row(result.y.rows - 1);
        expect(last.get(0)).toBeCloseTo(1, 6); // back to y0 = cos(2*pi) = 1
        expect(last.get(1)).toBeCloseTo(0, 6); // -sin(2*pi) = 0
    });

    it('integrates backwards when h < 0 and t0 > tEnd', () => {
        const result = rungeKuttaFixed('rk4', decay, 1, 0, new Array1D([Math.E ** -1]), -0.1);
        const last = result.y.get(result.y.rows - 1, 0);
        expect(last).toBeCloseTo(1, 5); // e^-t at t=0 is 1
        expect(result.t.data[result.t.size - 1]).toBe(0);
        // times should be strictly decreasing
        const t = Array.from(result.t.data);
        for (let i = 1; i < t.length; i++) expect(t[i]).toBeLessThan(t[i - 1]);
    });

    it('does not mutate y0', () => {
        const y0 = new Array1D([1, 2]);
        rungeKuttaFixed('rk4', shm, 0, 1, y0, 0.1);
        expect(Array.from(y0.data)).toEqual([1, 2]);
    });

    it('two independent calls do not interfere with each other (no shared/leaked scratch state)', () => {
        const r1 = rungeKuttaFixed('rk4', exponential, 0, 1, new Array1D([1]), 0.1);
        const r2 = rungeKuttaFixed('rk4', decay, 0, 1, new Array1D([1]), 0.1);
        expect(r1.y.get(r1.y.rows - 1, 0)).toBeCloseTo(Math.E, 5);
        expect(r2.y.get(r2.y.rows - 1, 0)).toBeCloseTo(1 / Math.E, 5);
    });

    it('throws a RangeError when h is 0', () => {
        expect(() => rungeKuttaFixed('rk4', exponential, 0, 1, new Array1D([1]), 0)).toThrow(RangeError);
    });

    it('throws a RangeError when the sign of h does not match the integration direction', () => {
        expect(() => rungeKuttaFixed('rk4', exponential, 0, 1, new Array1D([1]), -0.1)).toThrow(RangeError);
        expect(() => rungeKuttaFixed('rk4', exponential, 1, 0, new Array1D([1]), 0.1)).toThrow(RangeError);
    });

    it('throws a clear RangeError when y0 has dimension 0', () => {
        // Asserting the message (not just `RangeError`) matters here: without an explicit
        // guard in rungeKuttaFixed, `new Array2D(nTimes, 0)` throws its own RangeError
        // ("cols must be a positive integer"), which would make this test pass for the
        // wrong reason and mask a missing/removed check.
        expect(() => rungeKuttaFixed('rk4', exponential, 0, 1, new Array1D(0), 0.1)).toThrow(
            /y0 must have at least one component/
        );
    });

    it('returns a single row when t0 === tEnd', () => {
        const result = rungeKuttaFixed('rk4', exponential, 5, 5, new Array1D([1]), 0.1);
        expect(result.t.size).toBe(1);
        expect(result.y.rows).toBe(1);
        expect(result.t.data[0]).toBe(5);
        expect(result.y.get(0, 0)).toBe(1);
    });

    it('throws a RangeError for an unrecognized method', () => {
        expect(() =>
            rungeKuttaFixed('bogus' as unknown as RungeKuttaMethod, exponential, 0, 1, new Array1D([1]), 0.1)
        ).toThrow(RangeError);
    });
});