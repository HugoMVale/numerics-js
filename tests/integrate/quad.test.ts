import { describe, expect, it } from 'vitest';
import { quad } from '../../src/integrate/quad.js';
import {
    fSin, fIdentity,
    fExpMinusX, fExpX, fGaussian, fExpMinusAbsX, fArctanDeriv,
    fEndpointSingularity
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

describe('quad - user-defined transform', () => {
    // x = t^2 removes the 1/sqrt(x) endpoint singularity: integrand becomes 2.
    const sqrtTransform = {
        map: (t: number) => ({ x: t * t, dxDt: 2 * t }),
        inverse: Math.sqrt
    };

    it('removes an endpoint singularity and needs far fewer evaluations', () => {
        const plain = quad(fEndpointSingularity, 0, 1);
        const res = quad(fEndpointSingularity, 0, 1, { transform: sqrtTransform });
        expect(res.value).toBeCloseTo(2, 12);
        expect(res.converged).toBe(true);
        expect(res.evaluations).toBe(15);
        expect(res.evaluations).toBeLessThan(plain.evaluations);
    });

    it('supports a decreasing map and ignores the sign of dxDt', () => {
        // Integral of x^-2 from 1 to Infinity is 1, with x = 1/t (decreasing).
        const f = (x: number) => 1 / (x * x);
        const negative = { map: (t: number) => ({ x: 1 / t, dxDt: -1 / (t * t) }), inverse: (x: number) => 1 / x };
        const positive = { map: (t: number) => ({ x: 1 / t, dxDt: 1 / (t * t) }), inverse: (x: number) => 1 / x };

        const r1 = quad(f, 1, Infinity, { transform: negative, tol: 1e-10 });
        const r2 = quad(f, 1, Infinity, { transform: positive, tol: 1e-10 });
        expect(r1.value).toBeCloseTo(1, 10);
        expect(r1.converged).toBe(true);
        expect(r2.value).toBe(r1.value);
    });

    it('takes precedence over the automatic infinite-limit handling', () => {
        // Same integral as the built-in path, but through an explicit x = -log(t) map.
        let mapCalls = 0;
        const expTransform = {
            map: (t: number) => { mapCalls++; return { x: -Math.log(t), dxDt: -1 / t }; },
            inverse: (x: number) => Math.exp(-x)
        };
        const res = quad(fExpMinusX, 0, Infinity, { transform: expTransform, tol: 1e-10 });
        expect(mapCalls).toBeGreaterThan(0);
        expect(res.value).toBeCloseTo(1, 10);
        expect(res.converged).toBe(true);
        // exp(-x) * (1/t) with x = -log(t) is the constant 1 on (0, 1).
        expect(res.evaluations).toBe(15);
    });

    it('calls inverse with +/-Infinity for infinite limits', () => {
        const seen: number[] = [];
        const tanTransform = {
            map: (t: number) => ({ x: Math.tan(t), dxDt: 1 / (Math.cos(t) ** 2) }),
            inverse: (x: number) => { seen.push(x); return Math.atan(x); }
        };
        const res = quad(fArctanDeriv, -Infinity, Infinity, { transform: tanTransform, tol: 1e-10 });
        expect(seen).toContain(Infinity);
        expect(seen).toContain(-Infinity);
        expect(res.value).toBeCloseTo(4 * Math.PI, 10);
    });

    it('maps breakpoints through the user inverse', () => {
        const fPiecewise = (x: number) => (x < 1 ? 1 : 2);
        const res = quad(fPiecewise, 0, 2, {
            transform: sqrtTransform,
            breakpoints: [1],
            tol: 1e-10
        });
        expect(res.value).toBeCloseTo(3, 10);
        expect(res.converged).toBe(true);
        expect(res.subintervals).toBeGreaterThanOrEqual(2);
    });

    it('handles breakpoints with a decreasing map', () => {
        // Integral of exp(-|x-1|) over [1/2, Infinity) = (1 - e^-0.5) + 1, kink at x = 1.
        const f = (x: number) => Math.exp(-Math.abs(x - 1));
        const decreasing = { map: (t: number) => ({ x: 1 / t, dxDt: -1 / (t * t) }), inverse: (x: number) => 1 / x };
        const res = quad(f, 0.5, Infinity, { transform: decreasing, breakpoints: [1], tol: 1e-10 });
        expect(res.value).toBeCloseTo(2 - Math.exp(-0.5), 9);
        expect(res.converged).toBe(true);
    });

    it('ignores breakpoints that do not map strictly inside the t-interval', () => {
        const res = quad(fEndpointSingularity, 0, 1, {
            transform: sqrtTransform,
            breakpoints: [0, 1, -3, 7, NaN, Infinity]
        });
        expect(res.value).toBeCloseTo(2, 12);
        expect(res.subintervals).toBe(1);
    });

    it('drops breakpoints whose image is non-finite or outside the t-interval', () => {
        const f = (x: number) => Math.sin(x);
        const baseline = quad(f, 0, 1, { transform: sqrtTransform });
        // x = 0.5 is inside [0, 1], but a sloppy inverse maps it outside [0, 1] (or to NaN).
        for (const bad of [10, NaN, Infinity, -2]) {
            const sloppy = { map: sqrtTransform.map, inverse: (x: number) => (x === 0.5 ? bad : Math.sqrt(x)) };
            const res = quad(f, 0, 1, { transform: sloppy, breakpoints: [0.5] });
            expect(res.value).toBe(baseline.value);
            expect(res.subintervals).toBe(baseline.subintervals);
        }
    });

    it('supports reversed bounds', () => {
        const forward = quad(fEndpointSingularity, 0, 1, { transform: sqrtTransform });
        const reverse = quad(fEndpointSingularity, 1, 0, { transform: sqrtTransform });
        expect(reverse.value).toBe(-forward.value);
    });

    it('validates the transform', () => {
        expect(() => quad(fSin, 0, 1, { transform: { map: 1 as any, inverse: Math.sqrt } }))
            .toThrow(TypeError);
        expect(() => quad(fSin, 0, 1, { transform: { map: sqrtTransform.map, inverse: undefined as any } }))
            .toThrow('transform.map and transform.inverse must be functions');
        expect(() => quad(fSin, 0, 1, { transform: null as any })).toThrow(TypeError);
    });

    it('throws RangeError when inverse yields non-finite or equal t-limits', () => {
        expect(() => quad(fSin, 0, 1, { transform: { map: sqrtTransform.map, inverse: () => NaN } }))
            .toThrow(RangeError);
        expect(() => quad(fSin, 0, 1, { transform: { map: sqrtTransform.map, inverse: () => Infinity } }))
            .toThrow(RangeError);
        expect(() => quad(fSin, 0, 1, { transform: { map: sqrtTransform.map, inverse: () => 0.5 } }))
            .toThrow('finite, distinct');
    });

    it('reports the real x, not the internal t, when f is non-finite', () => {
        // x = 2t maps t in [0, 0.5] onto x in [0, 1]; f blows up everywhere,
        // so the very first evaluation (t = 0.25, x = 0.5) triggers it.
        const badTransform = { map: (t: number) => ({ x: 2 * t, dxDt: 2 }), inverse: (x: number) => x / 2 };
        expect(() => quad(() => Infinity, 0, 1, { transform: badTransform }))
            .toThrow('f(0.5)');
        expect(() => quad(() => Infinity, 0, 1, { transform: badTransform }))
            .not.toThrow('f(0.25)');
    });

    it('returns 0 for a === b without touching the transform', () => {
        const res = quad(fSin, 1, 1, { transform: sqrtTransform });
        expect(res).toEqual({ value: 0, error: 0, evaluations: 0, converged: true, subintervals: 0 });
    });
});