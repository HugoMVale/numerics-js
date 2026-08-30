import { describe, expect, it } from 'vitest';
import { interp, LinearInterpolator } from '../../src/interpolate/interp1.js';
import { Array1D } from '../../src/array/array1d.js';

describe('interp', () => {
    const xp = [1, 2, 3];
    const fp = [3, 2, 0];

    describe('scalar x', () => {
        it('returns the exact fp value at a knot', () => {
            expect(interp(1, xp, fp)).toBe(3);
            expect(interp(2, xp, fp)).toBe(2);
            expect(interp(3, xp, fp)).toBe(0);
        });

        it('linearly interpolates between two knots', () => {
            expect(interp(1.5, xp, fp)).toBeCloseTo(2.5);
            expect(interp(2.5, xp, fp)).toBeCloseTo(1);
        });

        it('interpolates correctly across unevenly-spaced intervals', () => {
            // the [1,5] interval is 4x wider than [0,1], so a naive
            // "average slope" assumption would give the wrong answer here
            const uxp = [0, 1, 5];
            const ufp = [0, 10, 10];
            expect(interp(3, uxp, ufp)).toBeCloseTo(10); // flat second segment
            expect(interp(0.5, uxp, ufp)).toBeCloseTo(5); // steep first segment
        });

        it('clamps to fp[0] below the range by default', () => {
            expect(interp(0, xp, fp)).toBe(3);
            expect(interp(-100, xp, fp)).toBe(3);
        });

        it('clamps to fp[last] above the range by default', () => {
            expect(interp(3.5, xp, fp)).toBe(0);
            expect(interp(1000, xp, fp)).toBe(0);
        });

        it('uses a custom left value below the range', () => {
            expect(interp(0, xp, fp, { left: -1 })).toBe(-1);
        });

        it('uses a custom right value above the range', () => {
            expect(interp(4, xp, fp, { right: 99 })).toBe(99);
        });

        it('propagates NaN', () => {
            expect(interp(NaN, xp, fp)).toBeNaN();
        });

        it('handles a single-point xp/fp as a constant function', () => {
            expect(interp(5, [2], [7])).toBe(7);
            expect(interp(2, [2], [7])).toBe(7);
            expect(interp(-5, [2], [7])).toBe(7);
        });

        it('handles duplicate x-coordinates without dividing by zero', () => {
            // ties resolve to the later (rightmost) knot with that x-value
            expect(interp(2, [1, 2, 2, 3], [0, 10, 20, 30])).toBe(20);
        });
    });

    describe('array x', () => {
        it('accepts a plain array and returns an Array1D', () => {
            const result = interp([0, 1, 1.5, 2.72, 3.14], xp, fp);
            expect(result).toBeInstanceOf(Array1D);
            expect(result.toArray()[0]).toBeCloseTo(3);
            expect(result.toArray()[1]).toBeCloseTo(3);
            expect(result.toArray()[2]).toBeCloseTo(2.5);
            expect(result.toArray()[3]).toBeCloseTo(0.56);
            expect(result.toArray()[4]).toBeCloseTo(0);
        });

        it('accepts an Array1D for x', () => {
            const result = interp(Array1D.from([1.5, 2.5]), xp, fp);
            expect(result.toArray()).toEqual([2.5, 1]);
        });

        it('accepts Array1D for xp and fp as well', () => {
            const result = interp([1.5, 2.5], Array1D.from(xp), Array1D.from(fp));
            expect(result.toArray()).toEqual([2.5, 1]);
        });

        it('returns an empty Array1D for empty x', () => {
            const result = interp([], xp, fp);
            expect(result.size).toBe(0);
        });
    });

    describe('validation', () => {
        it('throws if xp is empty', () => {
            expect(() => interp(1, [], [])).toThrow(RangeError);
        });

        it('throws if xp and fp have different lengths', () => {
            expect(() => interp(1, [1, 2], [1])).toThrow(RangeError);
        });

        it('throws by default if xp is not monotonically increasing', () => {
            expect(() => interp(1, [3, 2, 1], [1, 2, 3])).toThrow(RangeError);
        });

        it('allows non-decreasing xp (duplicates are not an error)', () => {
            expect(() => interp(1, [1, 1, 2], [1, 2, 3])).not.toThrow();
        });

        it('skips the sortedness check when checkSorted is false', () => {
            // xp is not actually sorted; result is unspecified, but it must
            // not throw and must still return a number.
            expect(() => interp(1, [3, 2, 1], [1, 2, 3], { checkSorted: false })).not.toThrow();
            expect(typeof interp(1, [3, 2, 1], [1, 2, 3], { checkSorted: false })).toBe('number');
        });
    });
});

describe('Interpolator1D', () => {
    const xp = [1, 2, 3];
    const fp = [3, 2, 0];

    it('evaluates a scalar the same way as interp', () => {
        const f = new LinearInterpolator(xp, fp);
        expect(f.eval(2.5)).toBeCloseTo(1);
        expect(f.eval(1.5)).toBeCloseTo(interp(1.5, xp, fp));
    });

    it('evaluates an array the same way as interp', () => {
        const f = new LinearInterpolator(xp, fp);
        const result = f.eval([0, 1, 1.5, 2.72, 3.14]);
        const expected = interp([0, 1, 1.5, 2.72, 3.14], xp, fp);
        expect(result.toArray()).toEqual(expected.toArray());
    });

    it('accepts an Array1D for x in eval', () => {
        const f = new LinearInterpolator(xp, fp);
        const result = f.eval(Array1D.from([1.5, 2.5]));
        expect(result.toArray()).toEqual([2.5, 1]);
    });

    it('exposes the number of underlying data points via size', () => {
        expect(new LinearInterpolator(xp, fp).size).toBe(3);
        expect(new LinearInterpolator([2], [7]).size).toBe(1);
    });

    it('respects custom left/right clamp values', () => {
        const f = new LinearInterpolator(xp, fp, -1, 99);
        expect(f.eval(0)).toBe(-1);
        expect(f.eval(4)).toBe(99);
    });

    it('can be reused across many eval calls without re-validating', () => {
        const f = new LinearInterpolator(xp, fp);
        expect(f.eval(1)).toBe(3);
        expect(f.eval(2)).toBe(2);
        expect(f.eval(3)).toBe(0);
        expect(f.eval(1.5)).toBeCloseTo(2.5);
    });

    it('throws in the constructor for invalid input, not on eval', () => {
        expect(() => new LinearInterpolator([], [])).toThrow(RangeError);
        expect(() => new LinearInterpolator([1, 2], [1])).toThrow(RangeError);
        expect(() => new LinearInterpolator([3, 2, 1], [1, 2, 3])).toThrow(RangeError);
    });

    it('skips the sortedness check when checkSorted is false', () => {
        expect(() => new LinearInterpolator([3, 2, 1], [1, 2, 3], undefined, undefined, false)).not.toThrow();
    });

    it('accepts Array1D directly for xp and fp', () => {
        const f = new LinearInterpolator(Array1D.from(xp), Array1D.from(fp));
        expect(f.eval(2.5)).toBeCloseTo(1);
    });

    describe('derivative', () => {
        it('returns the segment slope and zero outside clamped bounds', () => {
            const f = new LinearInterpolator(xp, fp);
            expect(f.derivative(1.5)).toBeCloseTo(-1);
            expect(f.derivative(2.5)).toBeCloseTo(-2);
            expect(f.derivative(0)).toBe(0);
            expect(f.derivative(4)).toBe(0);
        });

        it('preserves the input shape and propagates NaN', () => {
            const f = new LinearInterpolator(xp, fp);
            expect(f.derivative([1.5, 2.5]).toArray()).toEqual([-1, -2]);
            expect(f.derivative(NaN)).toBeNaN();
        });
    });

    describe('integrate', () => {
        it('returns 0 for equal bounds', () => {
            const f = new LinearInterpolator(xp, fp);
            expect(f.integrate(2, 2)).toBe(0);
        });

        it('integrates exactly over a single segment via the trapezoid rule', () => {
            const f = new LinearInterpolator(xp, fp);
            expect(f.integrate(1, 2)).toBeCloseTo((3 + 2) / 2);
        });

        it('integrates exactly across multiple segments, including interior knots', () => {
            const f = new LinearInterpolator(xp, fp);
            // [1,2] trapezoid + [2,3] trapezoid
            expect(f.integrate(1, 3)).toBeCloseTo((3 + 2) / 2 + (2 + 0) / 2);
        });

        it('integrates correctly over a sub-interval that starts/ends mid-segment', () => {
            const f = new LinearInterpolator(xp, fp);
            // f(1.5) = 2.5, f(2.5) = 1; area = [1.5,2] trapezoid + [2,2.5] trapezoid
            const expected = (2.5 + 2) / 2 * 0.5 + (2 + 1) / 2 * 0.5;
            expect(f.integrate(1.5, 2.5)).toBeCloseTo(expected);
        });

        it('negates the result and swaps bounds when b < a', () => {
            const f = new LinearInterpolator(xp, fp);
            expect(f.integrate(3, 1)).toBeCloseTo(-f.integrate(1, 3));
        });

        it('uses the clamp values when integrating outside the xp range', () => {
            const f = new LinearInterpolator(xp, fp);
            // below range: flat at fp[0] = 3
            expect(f.integrate(-1, 1)).toBeCloseTo(2 * 3);
            // above range: flat at fp[last] = 0
            expect(f.integrate(3, 5)).toBeCloseTo(0);
        });

        it('handles unevenly-spaced intervals correctly', () => {
            const f = new LinearInterpolator([0, 1, 5], [0, 10, 10]);
            // [0,1]: triangle area 5; [1,5]: flat rectangle area 40
            expect(f.integrate(0, 5)).toBeCloseTo(45);
        });

        it('is exact for a constant (single-point) interpolant', () => {
            const f = new LinearInterpolator([2], [7]);
            expect(f.integrate(0, 4)).toBeCloseTo(28);
        });

        it('throws if either bound is NaN', () => {
            const f = new LinearInterpolator(xp, fp);
            expect(() => f.integrate(NaN, 2)).toThrow(RangeError);
            expect(() => f.integrate(1, NaN)).toThrow(RangeError);
        });
    });
});