import { describe, expect, it } from 'vitest';
import { quad } from '../../src/integrate/quad.js';
import {
    fSin, fIdentity, fOscillatory,
    fExpMinusX, fExpX, fGaussian, fExpMinusAbsX, fArctanDeriv
} from './testFunctions.js';

describe('quad - finite domains', () => {
    it('integrates standard functions accurately without breakpoints', () => {
        const res = quad(fSin, 0, Math.PI, { tol: 1e-10 });
        expect(res.value).toBeCloseTo(2, 10);
        expect(res.converged).toBe(true);
    });

    it('integrates a piecewise function exactly using breakpoints', () => {
        const fPiecewise = (x: number) => (x < 1 ? 1 : 2);
        const res = quad(fPiecewise, 0, 2, { breakpoints: [1], tol: 1e-10 });
        expect(res.value).toBeCloseTo(3, 10);
        expect(res.converged).toBe(true);
    });

    it('filters out-of-bounds breakpoints and sorts the rest', () => {
        const res = quad(fIdentity, 0, 2, { breakpoints: [1.5, -1, 0.5, 3] });
        expect(res.value).toBeCloseTo(2, 10);
        expect(res.subintervals).toBeGreaterThanOrEqual(3);
    });

    it('supports reversed bounds (a > b) while processing breakpoints', () => {
        const forward = quad(fIdentity, 0, 2, { breakpoints: [1] });
        const reverse = quad(fIdentity, 2, 0, { breakpoints: [1] });
        expect(reverse.value).toBeCloseTo(-forward.value, 10);
    });

    it('throws RangeError for NaN integration limits', () => {
        expect(() => quad(fSin, 0, NaN)).toThrow(
            'quad: a and b must not be NaN'
        );
    });
});

describe('quad - infinite domains', () => {
    it('integrates over a positive semi-infinite domain [a, Infinity]', () => {
        // Integral of exp(-x) from 0 to Infinity is 1
        const res = quad(fExpMinusX, 0, Infinity, { tol: 1e-10 });
        expect(res.value).toBeCloseTo(1, 10);
        expect(res.converged).toBe(true);
    });

    it('integrates over a negative semi-infinite domain [-Infinity, b]', () => {
        // Integral of exp(x) from -Infinity to 0 is 1
        const res = quad(fExpX, -Infinity, 0, { tol: 1e-10 });
        expect(res.value).toBeCloseTo(1, 10);
        expect(res.converged).toBe(true);
    });

    it('integrates over a fully infinite domain [-Infinity, Infinity]', () => {
        // Integral of exp(-x^2) from -Infinity to Infinity is sqrt(pi)
        const res = quad(fGaussian, -Infinity, Infinity, { tol: 1e-10 });
        expect(res.value).toBeCloseTo(Math.sqrt(Math.PI), 8);
        expect(res.converged).toBe(true);

        // Integral of 4/(1+x^2) from -Infinity to Infinity is 4 * pi
        const res2 = quad(fArctanDeriv, -Infinity, Infinity, { tol: 1e-10 });
        expect(res2.value).toBeCloseTo(4 * Math.PI, 10);
        expect(res2.converged).toBe(true);
    });

    it('maps breakpoints correctly into infinite domains', () => {
        // Integral of exp(-|x|) from -Infinity to Infinity is 2.
        // The derivative is discontinuous at x = 0, so adding 0 as a breakpoint 
        // ensures the underlying adaptive logic cleanly resolves the sharp peak.
        const res = quad(fExpMinusAbsX, -Infinity, Infinity, {
            breakpoints: [0],
            tol: 1e-10
        });

        expect(res.value).toBeCloseTo(2, 10);
        expect(res.converged).toBe(true);
        // Ensure that splitting the infinite domain generated multiple subintervals
        expect(res.subintervals).toBeGreaterThan(1);
    });

    it('supports reversed infinite bounds', () => {
        const forward = quad(fExpMinusX, 0, Infinity);
        const reverse = quad(fExpMinusX, Infinity, 0);
        expect(reverse.value).toBeCloseTo(-forward.value, 10);
    });
});