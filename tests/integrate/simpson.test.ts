import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/array/Vector.js';
import { simpson } from '../../src/integrate/simpson.js';

describe('simpson', () => {
    describe('degenerate sizes', () => {
        it('returns 0 for an empty vector', () => {
            expect(simpson(new Vector(0))).toBe(0);
        });

        it('returns 0 for a single-point vector', () => {
            expect(simpson(Vector.from([5]))).toBe(0);
        });

        it('falls back to the trapezoidal rule for exactly 2 points', () => {
            const y = Vector.from([1, 3]);
            // trapz([1, 3], dx=1) = (1+3)/2 = 2
            expect(simpson(y)).toBeCloseTo(2);
        });

        it('falls back to trapz with x for exactly 2 points', () => {
            const y = Vector.from([1, 3]);
            const x = Vector.from([0, 2]);
            // trapz: (2-0) * (1+3)/2 = 4
            expect(simpson(y, x)).toBeCloseTo(4);
        });
    });

    describe('uniform spacing, even number of intervals (exact case)', () => {
        it('matches the classic 3-point Simpson formula', () => {
            // simpson([1, 4, 1], dx=1) = (1/3)(1 + 4*4 + 1) = 6
            expect(simpson(Vector.from([1, 4, 1]))).toBeCloseTo(6);
        });

        it('integrates a quadratic exactly (3 points)', () => {
            // y = x^2 over [0, 2], dx = 1. True integral = 8/3.
            const y = Vector.from([0, 1, 4]);
            expect(simpson(y)).toBeCloseTo(8 / 3);
        });

        it('integrates a quadratic exactly (5 points, 2 pairs)', () => {
            // y = x^2 over [0, 4], dx = 1. True integral = 64/3.
            const y = Vector.from([0, 1, 4, 9, 16]);
            expect(simpson(y)).toBeCloseTo(64 / 3);
        });

        it('integrates a constant function exactly', () => {
            const y = Vector.from([2, 2, 2, 2, 2]);
            expect(simpson(y, undefined, 1)).toBeCloseTo(8);
        });

        it('integrates a linear function exactly', () => {
            // y = x over [0, 4], dx = 1. True integral = 8.
            const y = Vector.from([0, 1, 2, 3, 4]);
            expect(simpson(y)).toBeCloseTo(8);
        });

        it('integrates a cubic exactly when the interval count is even', () => {
            // Simpson's rule is exact for cubics too (degree <= 3), but only
            // in the "pure" case with no trailing-trapz correction, i.e. an
            // even number of intervals. y = x^3 over [0, 4], dx = 1
            // (4 intervals, even) -> true integral = 4^4 / 4 = 64.
            const y = Vector.from([0, 1, 8, 27, 64]);
            expect(simpson(y)).toBeCloseTo(64);
        });

        it('scales with dx', () => {
            // Same shape as [1, 4, 1] but with dx = 2 should double the result.
            const y = Vector.from([1, 4, 1]);
            expect(simpson(y, undefined, 2)).toBeCloseTo(12);
        });
    });

    describe('uniform spacing, odd number of intervals (trailing correction)', () => {
        it('integrates a quadratic exactly even with an odd interval count', () => {
            // y = x^2 over [0, 3], dx = 1 (3 intervals, odd). The trailing
            // correction is exact for quadratics, so the whole result is
            // exact too: true integral = 9.
            const y = Vector.from([0, 1, 4, 9]);
            expect(simpson(y)).toBeCloseTo(9);
        });

        it('is only approximate for a cubic, unlike the even-interval case', () => {
            // y = x^3 over [0, 3], dx = 1 (3 intervals, odd). The trailing
            // correction fits a quadratic (not a cubic) through the last 3
            // points, integrated over just the asymmetric final interval,
            // so it's not exact here the way the even-interval case above is.
            // True integral = 81/4 = 20.25; the correction gives 20.5.
            const y = Vector.from([0, 1, 8, 27]);
            const result = simpson(y);
            expect(result).toBeCloseTo(20.5);
            expect(result).not.toBeCloseTo(20.25, 1);
        });

        it('combines a 1/3-rule pair with a trailing correction (5 intervals)', () => {
            // y = x^3 over [0, 5], dx = 1 (5 intervals, odd): one exact
            // 1/3-rule pair over [0,2], then the trailing correction over
            // [2,5]. True integral = 5^4/4 = 156.25; approximate here.
            const y = Vector.from([0, 1, 8, 27, 64, 125]);
            expect(simpson(y)).toBeCloseTo(156.5);
        });
    });

    describe('non-uniform spacing (x)', () => {
        it('integrates a quadratic exactly even with irregular spacing', () => {
            // y = x^2 at x = [0, 1, 3]. True integral over [0,3] = 9.
            // Simpson's parabolic rule is exact for quadratics regardless
            // of interval spacing.
            const y = Vector.from([0, 1, 9]);
            const x = Vector.from([0, 1, 3]);
            expect(simpson(y, x)).toBeCloseTo(9);
        });

        it('matches the uniform-dx result when x has constant spacing', () => {
            const y = Vector.from([0, 1, 4, 9, 16]);
            const x = Vector.from([0, 1, 2, 3, 4]);
            expect(simpson(y, x)).toBeCloseTo(simpson(y, undefined, 1));
        });

        it('integrates a quadratic exactly with an odd interval count under irregular spacing', () => {
            // y = x^2 at x = [0, 1, 3, 6] (3 intervals, odd, irregular
            // widths 1, 2, 3). The trailing correction is exact for
            // quadratics regardless of spacing, so this whole result is
            // exact: true integral over [0,6] = 6^3/3 = 72.
            const y = Vector.from([0, 1, 9, 36]);
            const x = Vector.from([0, 1, 3, 6]);
            expect(simpson(y, x)).toBeCloseTo(72);
        });

        it('handles an odd number of intervals with irregular spacing', () => {
            // 4 points -> 3 intervals (odd), last interval wider than the
            // rest. Not a quadratic, so just checked against a hand-worked
            // value rather than a known closed-form integral.
            const y = Vector.from([0, 1, 4, 9]);
            const x = Vector.from([0, 1, 2, 4]); // last interval has width 2
            expect(simpson(y, x)).toBeCloseTo(143 / 9);
        });

        it('throws a RangeError when x and y sizes differ', () => {
            const y = Vector.from([1, 2, 3]);
            const x = Vector.from([0, 1]);
            expect(() => simpson(y, x)).toThrow(RangeError);
            expect(() => simpson(y, x)).toThrow(/same size/);
        });
    });
});