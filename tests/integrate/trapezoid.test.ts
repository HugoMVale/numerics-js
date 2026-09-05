import { describe, it, expect } from 'vitest';
import { Vector } from '../../src/linalg/Vector.js';
import { trapezoid } from '../../src/integrate/trapezoid.js';

describe('trapz', () => {
    describe('degenerate sizes', () => {
        it('returns 0 for an empty vector', () => {
            expect(trapezoid(new Vector(0))).toBe(0);
        });

        it('returns 0 for a single-point vector', () => {
            expect(trapezoid(Vector.from([5]))).toBe(0);
        });

        it('returns 0 for a single point even with x provided', () => {
            expect(trapezoid(Vector.from([5]), Vector.from([2]))).toBe(0);
        });
    });

    describe('uniform spacing (dx)', () => {
        it('defaults to dx = 1 when no x or dx is given', () => {
            // np.trapz([1, 2, 3]) == 4
            expect(trapezoid(Vector.from([1, 2, 3]))).toBeCloseTo(4);
        });

        it('scales the result by dx', () => {
            // np.trapz([1, 2, 3], dx=2) == 8
            expect(trapezoid(Vector.from([1, 2, 3]), undefined, 2)).toBeCloseTo(8);
        });

        it('integrates a constant function exactly', () => {
            // Area of a flat line y=3 over [0, 4] (dx=1, 5 samples) is 12.
            const y = Vector.from([3, 3, 3, 3, 3]);
            expect(trapezoid(y, undefined, 1)).toBeCloseTo(12);
        });

        it('integrates a linear function exactly (trapezoidal rule is exact for lines)', () => {
            // y = x over [0, 4], dx = 1 -> exact integral is 8.
            const y = Vector.from([0, 1, 2, 3, 4]);
            expect(trapezoid(y, undefined, 1)).toBeCloseTo(8);
        });

        it('approximates a convex function with a known error direction', () => {
            // y = x^2 over [0, 1] with dx = 0.5: trapz should overestimate
            // the true integral (1/3) since the rule is exact only for lines.
            const y = Vector.from([0, 0.25, 1]);
            const result = trapezoid(y, undefined, 0.5);
            expect(result).toBeCloseTo(0.375);
            expect(result).toBeGreaterThan(1 / 3);
        });

        it('handles negative dx by negating the result', () => {
            const y = Vector.from([1, 2, 3]);
            expect(trapezoid(y, undefined, -1)).toBeCloseTo(-4);
        });
    });

    describe('non-uniform spacing (x)', () => {
        it('matches the uniform-dx result when x has constant spacing', () => {
            const y = Vector.from([0, 0.25, 1]);
            const x = Vector.from([0, 0.5, 1]);
            expect(trapezoid(y, x)).toBeCloseTo(trapezoid(y, undefined, 0.5));
        });

        it('handles irregular sample spacing', () => {
            // Two trapezoids: [0,1] width 1, height avg (0+2)/2=1 -> area 1
            //                 [1,4] width 3, height avg (2+2)/2=2 -> area 6
            const y = Vector.from([0, 2, 2]);
            const x = Vector.from([0, 1, 4]);
            expect(trapezoid(y, x)).toBeCloseTo(7);
        });

        it('supports decreasing x (negative contributions)', () => {
            const y = Vector.from([1, 2, 3]);
            const xAsc = Vector.from([0, 1, 2]);
            const xDesc = Vector.from([2, 1, 0]);
            expect(trapezoid(y, xDesc)).toBeCloseTo(-trapezoid(y, xAsc));
        });

        it('ignores dx when x is provided', () => {
            const y = Vector.from([0, 1, 2]);
            const x = Vector.from([0, 1, 2]);
            expect(trapezoid(y, x, 100)).toBeCloseTo(trapezoid(y, x));
        });

        it('throws a RangeError when x and y sizes differ', () => {
            const y = Vector.from([1, 2, 3]);
            const x = Vector.from([0, 1]);
            expect(() => trapezoid(y, x)).toThrow(RangeError);
            expect(() => trapezoid(y, x)).toThrow(/same size/);
        });
    });
});
