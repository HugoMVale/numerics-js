import { describe, expect, it } from 'vitest';
import { interp1D, LinearInterpolator1D, interpND, LinearInterpolatorND } from '../../src/interpolate/interp1.js';
import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';

describe('interp1D', () => {
    const xp = [1, 2, 3];
    const fp = [3, 2, 0];

    describe('scalar x', () => {
        it('returns the exact fp value at a knot', () => {
            expect(interp1D(1, xp, fp)).toBe(3);
            expect(interp1D(2, xp, fp)).toBe(2);
            expect(interp1D(3, xp, fp)).toBe(0);
        });

        it('linearly interpolates between two knots', () => {
            expect(interp1D(1.5, xp, fp)).toBeCloseTo(2.5);
            expect(interp1D(2.5, xp, fp)).toBeCloseTo(1);
        });

        it('interpolates correctly across unevenly-spaced intervals', () => {
            // the [1,5] interval is 4x wider than [0,1], so a naive
            // "average slope" assumption would give the wrong answer here
            const uxp = [0, 1, 5];
            const ufp = [0, 10, 10];
            expect(interp1D(3, uxp, ufp)).toBeCloseTo(10); // flat second segment
            expect(interp1D(0.5, uxp, ufp)).toBeCloseTo(5); // steep first segment
        });

        it('clamps to fp[0] below the range by default', () => {
            expect(interp1D(0, xp, fp)).toBe(3);
            expect(interp1D(-100, xp, fp)).toBe(3);
        });

        it('clamps to fp[last] above the range by default', () => {
            expect(interp1D(3.5, xp, fp)).toBe(0);
            expect(interp1D(1000, xp, fp)).toBe(0);
        });

        it('uses a custom left value below the range', () => {
            expect(interp1D(0, xp, fp, { left: -1 })).toBe(-1);
        });

        it('uses a custom right value above the range', () => {
            expect(interp1D(4, xp, fp, { right: 99 })).toBe(99);
        });

        it('propagates NaN', () => {
            expect(interp1D(NaN, xp, fp)).toBeNaN();
        });

        it('handles a single-point xp/fp as a constant function', () => {
            expect(interp1D(5, [2], [7])).toBe(7);
            expect(interp1D(2, [2], [7])).toBe(7);
            expect(interp1D(-5, [2], [7])).toBe(7);
        });

        it('handles duplicate x-coordinates without dividing by zero', () => {
            // ties resolve to the later (rightmost) knot with that x-value
            expect(interp1D(2, [1, 2, 2, 3], [0, 10, 20, 30])).toBe(20);
        });
    });

    describe('array x', () => {
        it('accepts a plain array and returns an Vector', () => {
            const result = interp1D([0, 1, 1.5, 2.72, 3.14], xp, fp);
            expect(result).toBeInstanceOf(Vector);
            expect(result.toArray()[0]).toBeCloseTo(3);
            expect(result.toArray()[1]).toBeCloseTo(3);
            expect(result.toArray()[2]).toBeCloseTo(2.5);
            expect(result.toArray()[3]).toBeCloseTo(0.56);
            expect(result.toArray()[4]).toBeCloseTo(0);
        });

        it('accepts an Vector for x', () => {
            const result = interp1D(Vector.from([1.5, 2.5]), xp, fp);
            expect(result.toArray()).toEqual([2.5, 1]);
        });

        it('accepts Vector for xp and fp as well', () => {
            const result = interp1D([1.5, 2.5], Vector.from(xp), Vector.from(fp));
            expect(result.toArray()).toEqual([2.5, 1]);
        });

        it('returns an empty Vector for empty x', () => {
            const result = interp1D([], xp, fp);
            expect(result.size).toBe(0);
        });
    });

    describe('validation', () => {
        it('throws if xp is empty', () => {
            expect(() => interp1D(1, [], [])).toThrow(RangeError);
        });

        it('throws if xp and fp have different lengths', () => {
            expect(() => interp1D(1, [1, 2], [1])).toThrow(RangeError);
        });

        it('throws by default if xp is not monotonically increasing', () => {
            expect(() => interp1D(1, [3, 2, 1], [1, 2, 3])).toThrow(RangeError);
        });

        it('allows non-decreasing xp (duplicates are not an error)', () => {
            expect(() => interp1D(1, [1, 1, 2], [1, 2, 3])).not.toThrow();
        });

        it('skips the sortedness check when checkSorted is false', () => {
            // xp is not actually sorted; result is unspecified, but it must
            // not throw and must still return a number.
            expect(() => interp1D(1, [3, 2, 1], [1, 2, 3], { checkSorted: false })).not.toThrow();
            expect(typeof interp1D(1, [3, 2, 1], [1, 2, 3], { checkSorted: false })).toBe('number');
        });
    });
});

describe('LinearInterpolator1D', () => {
    const xp = [1, 2, 3];
    const fp = [3, 2, 0];

    it('evaluates a scalar the same way as interp', () => {
        const f = new LinearInterpolator1D(xp, fp);
        expect(f.eval(2.5)).toBeCloseTo(1);
        expect(f.eval(1.5)).toBeCloseTo(interp1D(1.5, xp, fp));
    });

    it('evaluates an array the same way as interp', () => {
        const f = new LinearInterpolator1D(xp, fp);
        const result = f.eval([0, 1, 1.5, 2.72, 3.14]);
        const expected = interp1D([0, 1, 1.5, 2.72, 3.14], xp, fp);
        expect(result.toArray()).toEqual(expected.toArray());
    });

    it('accepts an Vector for x in eval', () => {
        const f = new LinearInterpolator1D(xp, fp);
        const result = f.eval(Vector.from([1.5, 2.5]));
        expect(result.toArray()).toEqual([2.5, 1]);
    });

    it('exposes the number of underlying data points via size', () => {
        expect(new LinearInterpolator1D(xp, fp).size).toBe(3);
        expect(new LinearInterpolator1D([2], [7]).size).toBe(1);
    });

    it('respects custom left/right clamp values', () => {
        const f = new LinearInterpolator1D(xp, fp, { left: -1, right: 99 });
        expect(f.eval(0)).toBe(-1);
        expect(f.eval(4)).toBe(99);
    });

    it('can be reused across many eval calls without re-validating', () => {
        const f = new LinearInterpolator1D(xp, fp);
        expect(f.eval(1)).toBe(3);
        expect(f.eval(2)).toBe(2);
        expect(f.eval(3)).toBe(0);
        expect(f.eval(1.5)).toBeCloseTo(2.5);
    });

    it('throws in the constructor for invalid input, not on eval', () => {
        expect(() => new LinearInterpolator1D([], [])).toThrow(RangeError);
        expect(() => new LinearInterpolator1D([1, 2], [1])).toThrow(RangeError);
        expect(() => new LinearInterpolator1D([3, 2, 1], [1, 2, 3])).toThrow(RangeError);
    });

    it('skips the sortedness check when checkSorted is false', () => {
        expect(() => new LinearInterpolator1D([3, 2, 1], [1, 2, 3], { checkSorted: false })).not.toThrow();
    });

    it('accepts Vector directly for xp and fp', () => {
        const f = new LinearInterpolator1D(Vector.from(xp), Vector.from(fp));
        expect(f.eval(2.5)).toBeCloseTo(1);
    });

    describe('derivative', () => {
        it('returns the segment slope and zero outside clamped bounds', () => {
            const f = new LinearInterpolator1D(xp, fp);
            expect(f.derivative(1.5)).toBeCloseTo(-1);
            expect(f.derivative(2.5)).toBeCloseTo(-2);
            expect(f.derivative(0)).toBe(0);
            expect(f.derivative(4)).toBe(0);
        });

        it('preserves the input shape and propagates NaN', () => {
            const f = new LinearInterpolator1D(xp, fp);
            expect(f.derivative([1.5, 2.5]).toArray()).toEqual([-1, -2]);
            expect(f.derivative(NaN)).toBeNaN();
        });
    });

    describe('integrate', () => {
        it('returns 0 for equal bounds', () => {
            const f = new LinearInterpolator1D(xp, fp);
            expect(f.integrate(2, 2)).toBe(0);
        });

        it('integrates exactly over a single segment via the trapezoid rule', () => {
            const f = new LinearInterpolator1D(xp, fp);
            expect(f.integrate(1, 2)).toBeCloseTo((3 + 2) / 2);
        });

        it('integrates exactly across multiple segments, including interior knots', () => {
            const f = new LinearInterpolator1D(xp, fp);
            // [1,2] trapezoid + [2,3] trapezoid
            expect(f.integrate(1, 3)).toBeCloseTo((3 + 2) / 2 + (2 + 0) / 2);
        });

        it('integrates correctly over a sub-interval that starts/ends mid-segment', () => {
            const f = new LinearInterpolator1D(xp, fp);
            // f(1.5) = 2.5, f(2.5) = 1; area = [1.5,2] trapezoid + [2,2.5] trapezoid
            const expected = (2.5 + 2) / 2 * 0.5 + (2 + 1) / 2 * 0.5;
            expect(f.integrate(1.5, 2.5)).toBeCloseTo(expected);
        });

        it('negates the result and swaps bounds when b < a', () => {
            const f = new LinearInterpolator1D(xp, fp);
            expect(f.integrate(3, 1)).toBeCloseTo(-f.integrate(1, 3));
        });

        it('uses the clamp values when integrating outside the xp range', () => {
            const f = new LinearInterpolator1D(xp, fp);
            // below range: flat at fp[0] = 3
            expect(f.integrate(-1, 1)).toBeCloseTo(2 * 3);
            // above range: flat at fp[last] = 0
            expect(f.integrate(3, 5)).toBeCloseTo(0);
        });

        it('handles unevenly-spaced intervals correctly', () => {
            const f = new LinearInterpolator1D([0, 1, 5], [0, 10, 10]);
            // [0,1]: triangle area 5; [1,5]: flat rectangle area 40
            expect(f.integrate(0, 5)).toBeCloseTo(45);
        });

        it('is exact for a constant (single-point) interpolant', () => {
            const f = new LinearInterpolator1D([2], [7]);
            expect(f.integrate(0, 4)).toBeCloseTo(28);
        });

        it('throws if either bound is NaN', () => {
            const f = new LinearInterpolator1D(xp, fp);
            expect(() => f.integrate(NaN, 2)).toThrow(RangeError);
            expect(() => f.integrate(1, NaN)).toThrow(RangeError);
        });
    });
});

describe('interpND', () => {
    // Two independent scalar series packed as columns, so every ND result
    // can be cross-checked against the already-verified interp1D/scalar path.
    const xp = [1, 2, 3];
    const fpA = [3, 2, 0];
    const fpB = [10, 20, 40];
    const fp = Matrix.from([
        [3, 10],
        [2, 20],
        [0, 40],
    ]);

    describe('scalar x', () => {
        it('matches componentwise interp1D at knots and midpoints', () => {
            for (const x of [1, 2, 3, 1.5, 2.5]) {
                const v = interpND(x, xp, fp);
                expect(v).toBeInstanceOf(Vector);
                expect(v.toArray()[0]).toBeCloseTo(interp1D(x, xp, fpA));
                expect(v.toArray()[1]).toBeCloseTo(interp1D(x, xp, fpB));
            }
        });

        it('clamps to fp.row(0) / fp.row(last) by default', () => {
            expect(interpND(0, xp, fp).toArray()).toEqual([3, 10]);
            expect(interpND(4, xp, fp).toArray()).toEqual([0, 40]);
        });

        it('propagates NaN to every component', () => {
            const v = interpND(NaN, xp, fp);
            expect(v.toArray().every(Number.isNaN)).toBe(true);
        });

        it('accepts a broadcast scalar for left/right', () => {
            const v = interpND(0, xp, fp, { left: -1 });
            expect(v.toArray()).toEqual([-1, -1]);
        });

        it('accepts a per-component vector for left/right', () => {
            const v = interpND(0, xp, fp, { left: [-1, -2] });
            expect(v.toArray()).toEqual([-1, -2]);
        });

        it('rejects a left/right vector of the wrong length', () => {
            expect(() => interpND(0, xp, fp, { left: [-1, -2, -3] })).toThrow(RangeError);
        });
    });

    describe('array x', () => {
        it('returns a Matrix with one row per query point', () => {
            const m = interpND([1, 1.5, 3], xp, fp);
            expect(m).toBeInstanceOf(Matrix);
            expect(m.rows).toBe(3);
            expect(m.cols).toBe(2);
            expect(m.row(1).toArray()[0]).toBeCloseTo(interp1D(1.5, xp, fpA));
            expect(m.row(1).toArray()[1]).toBeCloseTo(interp1D(1.5, xp, fpB));
        });

        it('throws for an empty query (Matrix cannot have 0 rows)', () => {
            expect(() => interpND([], xp, fp)).toThrow(RangeError);
        });
    });

    describe('validation', () => {
        it('throws if xp.length !== fp.rows', () => {
            expect(() => interpND(1, [1, 2], fp)).toThrow(RangeError);
        });

        it('throws by default if xp is not monotonically increasing', () => {
            expect(() => interpND(1, [3, 2, 1], fp)).toThrow(RangeError);
        });
    });
});

describe('LinearInterpolatorND', () => {
    const xp = [1, 2, 3];
    const fpA = [3, 2, 0];
    const fpB = [10, 20, 40];
    const fp = Matrix.from([
        [3, 10],
        [2, 20],
        [0, 40],
    ]);

    it('evaluates a scalar the same way as interpND', () => {
        const f = new LinearInterpolatorND(xp, fp);
        expect(f.eval(2.5).toArray()).toEqual(interpND(2.5, xp, fp).toArray());
    });

    it('evaluates an array the same way as interpND', () => {
        const f = new LinearInterpolatorND(xp, fp);
        const result = f.eval([0, 1.5, 3.14]);
        const expected = interpND([0, 1.5, 3.14], xp, fp);
        expect(result.toArray()).toEqual(expected.toArray());
    });

    it('exposes size and dim', () => {
        const f = new LinearInterpolatorND(xp, fp);
        expect(f.size).toBe(3);
        expect(f.dim).toBe(2);
    });

    it('respects custom left/right clamp values', () => {
        const f = new LinearInterpolatorND(xp, fp, { left: [-1, -2], right: [99, 98] });
        expect(f.eval(0).toArray()).toEqual([-1, -2]);
        expect(f.eval(4).toArray()).toEqual([99, 98]);
    });

    it('throws in the constructor for invalid input, not on eval', () => {
        expect(() => new LinearInterpolatorND([1, 2], fp)).toThrow(RangeError);
        expect(() => new LinearInterpolatorND([3, 2, 1], fp)).toThrow(RangeError);
    });

    describe('derivative', () => {
        it('matches componentwise LinearInterpolator1D derivatives', () => {
            const f = new LinearInterpolatorND(xp, fp);
            const fA = new LinearInterpolator1D(xp, fpA);
            const fB = new LinearInterpolator1D(xp, fpB);
            for (const x of [1.5, 2.5, 0, 4]) {
                const d = f.derivative(x);
                expect(d.toArray()[0]).toBeCloseTo(fA.derivative(x));
                expect(d.toArray()[1]).toBeCloseTo(fB.derivative(x));
            }
        });

        it('throws for an empty query array', () => {
            const f = new LinearInterpolatorND(xp, fp);
            expect(() => f.derivative([])).toThrow(RangeError);
        });
    });

    describe('integrate', () => {
        it('matches componentwise LinearInterpolator1D integrals', () => {
            const f = new LinearInterpolatorND(xp, fp);
            const fA = new LinearInterpolator1D(xp, fpA);
            const fB = new LinearInterpolator1D(xp, fpB);

            const result = f.integrate(1, 3);
            expect(result.toArray()[0]).toBeCloseTo(fA.integrate(1, 3));
            expect(result.toArray()[1]).toBeCloseTo(fB.integrate(1, 3));
        });

        it('negates and swaps bounds when b < a', () => {
            const f = new LinearInterpolatorND(xp, fp);
            const fwd = f.integrate(1, 3);
            const rev = f.integrate(3, 1);
            expect(rev.toArray()[0]).toBeCloseTo(-fwd.toArray()[0]);
            expect(rev.toArray()[1]).toBeCloseTo(-fwd.toArray()[1]);
        });

        it('returns the zero vector for equal bounds', () => {
            const f = new LinearInterpolatorND(xp, fp);
            expect(f.integrate(2, 2).toArray()).toEqual([0, 0]);
        });

        it('throws if either bound is NaN', () => {
            const f = new LinearInterpolatorND(xp, fp);
            expect(() => f.integrate(NaN, 2)).toThrow(RangeError);
            expect(() => f.integrate(1, NaN)).toThrow(RangeError);
        });
    });
});