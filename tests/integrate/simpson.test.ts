import { describe, it, expect } from 'vitest';
import { Array1D } from '../../src/array/array1d.js';
import { simpson } from '../../src/integrate/simpson.js';

describe('simpson', () => {
    describe('degenerate sizes', () => {
        it('returns 0 for an empty vector', () => {
            expect(simpson(new Array1D(0))).toBe(0);
        });

        it('returns 0 for a single-point vector', () => {
            expect(simpson(Array1D.from([5]))).toBe(0);
        });

        it('falls back to the trapezoidal rule for exactly 2 points', () => {
            const y = Array1D.from([1, 3]);
            // trapz([1, 3], dx=1) = (1+3)/2 = 2
            expect(simpson(y)).toBeCloseTo(2);
        });

        it('falls back to trapz with x for exactly 2 points', () => {
            const y = Array1D.from([1, 3]);
            const x = Array1D.from([0, 2]);
            // trapz: (2-0) * (1+3)/2 = 4
            expect(simpson(y, x)).toBeCloseTo(4);
        });
    });

    describe('uniform spacing, even number of intervals (exact case)', () => {
        it('matches the classic 3-point Simpson formula', () => {
            // simpson([1, 4, 1], dx=1) = (1/3)(1 + 4*4 + 1) = 6
            expect(simpson(Array1D.from([1, 4, 1]))).toBeCloseTo(6);
        });

        it('integrates a quadratic exactly (3 points)', () => {
            // y = x^2 over [0, 2], dx = 1. True integral = 8/3.
            const y = Array1D.from([0, 1, 4]);
            expect(simpson(y)).toBeCloseTo(8 / 3);
        });

        it('integrates a quadratic exactly (5 points, 2 pairs)', () => {
            // y = x^2 over [0, 4], dx = 1. True integral = 64/3.
            const y = Array1D.from([0, 1, 4, 9, 16]);
            expect(simpson(y)).toBeCloseTo(64 / 3);
        });

        it('integrates a constant function exactly', () => {
            const y = Array1D.from([2, 2, 2, 2, 2]);
            expect(simpson(y, undefined, 1)).toBeCloseTo(8);
        });

        it('integrates a linear function exactly', () => {
            // y = x over [0, 4], dx = 1. True integral = 8.
            const y = Array1D.from([0, 1, 2, 3, 4]);
            expect(simpson(y)).toBeCloseTo(8);
        });

        it('integrates a cubic exactly when the interval count is even', () => {
            // Simpson's rule is exact for cubics too (degree <= 3), but only
            // in the "pure" case with no trailing-trapz correction, i.e. an
            // even number of intervals. y = x^3 over [0, 4], dx = 1
            // (4 intervals, even) -> true integral = 4^4 / 4 = 64.
            const y = Array1D.from([0, 1, 8, 27, 64]);
            expect(simpson(y)).toBeCloseTo(64);
        });

        it('scales with dx', () => {
            // Same shape as [1, 4, 1] but with dx = 2 should double the result.
            const y = Array1D.from([1, 4, 1]);
            expect(simpson(y, undefined, 2)).toBeCloseTo(12);
        });
    });

    describe('uniform spacing, odd number of intervals (trailing trapz correction)', () => {
        it('applies a trapezoidal correction on the last interval', () => {
            // y = x^3 over [0, 3], dx = 1, points at x = 0,1,2,3.
            // Pair covers [0,2] via Simpson; [2,3] via trapz.
            // Simpson pair: (2/6)*(0 + 4*1 + 8) = 4
            // Trapz tail:   (8 + 27)/2 = 17.5
            // Total: 21.5
            const y = Array1D.from([0, 1, 8, 27]);
            expect(simpson(y)).toBeCloseTo(21.5);
        });

        it('is close to, but not required to exactly match, the true integral for a cubic', () => {
            // Simpson's rule is exact for cubics too, but only when applied
            // "purely" (even interval count, no trailing correction). Here
            // the interval count is odd, so the true integral of x^3 over
            // [0, 3] (81/4 = 20.25) is NOT reproduced exactly, since the
            // last interval only gets trapz accuracy. Compare against the
            // even-interval-count cubic test above, which IS exact.
            const y = Array1D.from([0, 1, 8, 27]);
            const result = simpson(y);
            expect(result).not.toBeCloseTo(20.25, 1);
            expect(Math.abs(result - 20.25)).toBeLessThan(2);
        });
    });

    describe('non-uniform spacing (x)', () => {
        it('integrates a quadratic exactly even with irregular spacing', () => {
            // y = x^2 at x = [0, 1, 3]. True integral over [0,3] = 9.
            // Simpson's parabolic rule is exact for quadratics regardless
            // of interval spacing.
            const y = Array1D.from([0, 1, 9]);
            const x = Array1D.from([0, 1, 3]);
            expect(simpson(y, x)).toBeCloseTo(9);
        });

        it('matches the uniform-dx result when x has constant spacing', () => {
            const y = Array1D.from([0, 1, 4, 9, 16]);
            const x = Array1D.from([0, 1, 2, 3, 4]);
            expect(simpson(y, x)).toBeCloseTo(simpson(y, undefined, 1));
        });

        it('handles an odd number of intervals with a trailing trapz step', () => {
            // 4 points -> 3 intervals (odd): one Simpson pair + one trapz tail.
            const y = Array1D.from([0, 1, 4, 9]);
            const x = Array1D.from([0, 1, 2, 4]); // last interval has width 2
            // Pair [0,1,2] uniform h=1: (2/6)*(0 + 4*1 + 4) = 8/3
            // Tail [2,4] width 2: 2*(4+9)/2 = 13
            expect(simpson(y, x)).toBeCloseTo(8 / 3 + 13);
        });

        it('throws a RangeError when x and y sizes differ', () => {
            const y = Array1D.from([1, 2, 3]);
            const x = Array1D.from([0, 1]);
            expect(() => simpson(y, x)).toThrow(RangeError);
            expect(() => simpson(y, x)).toThrow(/same size/);
        });
    });
});
