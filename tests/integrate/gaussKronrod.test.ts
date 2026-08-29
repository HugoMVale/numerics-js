import { describe, expect, it } from 'vitest';
import { gaussKronrod, gaussKronrod15 } from '../../src/integrate/gaussKronrod.js';

describe('gaussKronrod15', () => {
    it('computes a single-panel estimate for polynomials accurately', () => {
        let evaluations = 0;
        const evalF = (x: number) => {
            evaluations++;
            return x * x;
        };

        const result = gaussKronrod15(evalF, 0, 1);
        expect(result.kronrod).toBeCloseTo(1 / 3, 10);
        expect(evaluations).toBe(15);
    });
});

describe('gaussKronrod', () => {
    it('integrates standard functions accurately', () => {
        const r1 = gaussKronrod(Math.sin, 0, Math.PI, 1e-10);
        expect(r1.value).toBeCloseTo(2, 10);
        expect(r1.converged).toBe(true);

        const r2 = gaussKronrod((x) => 4 / (1 + x * x), 0, 1, 1e-12);
        expect(r2.value).toBeCloseTo(Math.PI, 12);
        expect(r2.converged).toBe(true);
    });

    it('handles identical integration bounds', () => {
        const res = gaussKronrod((x) => x, 2, 2);
        expect(res).toEqual({
            value: 0,
            error: 0,
            evaluations: 0,
            converged: true,
            subintervals: 0,
        });
    });

    it('supports reversed bounds (a > b)', () => {
        const forward = gaussKronrod(Math.sin, 0, Math.PI);
        const reverse = gaussKronrod(Math.sin, Math.PI, 0);

        expect(reverse.value).toBeCloseTo(-forward.value, 10);
        expect(reverse.evaluations).toBe(forward.evaluations);
    });

    it('respects maxSubintervals and sets converged to false when exceeded', () => {
        const oscillatory = (x: number) => Math.sin(100 * x);
        const res = gaussKronrod(oscillatory, 0, Math.PI, 1e-15, 3);

        expect(res.converged).toBe(false);
        expect(res.subintervals).toBe(3);
    });

    it('throws TypeError for invalid integrand input', () => {
        expect(() => gaussKronrod('not a function' as any, 0, 1)).toThrow(TypeError);
    });

    it('throws RangeError for infinite or NaN integration limits', () => {
        expect(() => gaussKronrod(Math.sin, 0, Infinity)).toThrow(
            'gaussKronrod: a and b must be finite numbers'
        );
        expect(() => gaussKronrod(Math.sin, NaN, 1)).toThrow(
            'gaussKronrod: a and b must be finite numbers'
        );
    });

    it('throws an Error when the function produces non-finite values', () => {
        const singular = (x: number) => (x === 0.5 ? Infinity : 1 / x);
        expect(() => gaussKronrod(singular, 0, 1)).toThrow(
            /gaussKronrod: f\(0\.5\) returned a non-finite value \(Infinity\); cannot integrate\./
        );
    });
});