import { describe, it, expect } from 'vitest';
import { PchipInterpolator1D, PchipInterpolatorND } from '../../src/interpolate/pchip.js';
import { Vector } from '../../src/linalg/Vector.js';
import { Matrix } from '../../src/linalg/Matrix.js';

describe('PchipInterpolator1D', () => {
    it('produces the values shown in its documentation example', () => {
        const pchip = new PchipInterpolator1D([1, 2, 3], [3, 2, 0]);

        expect(pchip.eval(2.5)).toBeCloseTo(1.1458333333333333);
        expect(pchip.eval([0, 1.5, 3.14]).data).toEqual(
            new Float64Array([3, 2.6041666666666665, 0])
        );
    });

    it('evaluates exactly at the knots', () => {
        const xp = [0, 1, 2, 3, 4];
        const fp = [0, 2, 0, -1, 3];
        const pchip = new PchipInterpolator1D(xp, fp);

        expect(pchip.eval(0)).toBe(0);
        expect(pchip.eval(1)).toBe(2);
        expect(pchip.eval(2)).toBe(0);
        expect(pchip.eval(4)).toBe(3);
    });

    it('preserves monotonicity and prevents overshoots (step function)', () => {
        const xp = [0, 1, 2, 3, 4];
        const fp = [0, 0, 1, 1, 1]; // Step function
        const pchip = new PchipInterpolator1D(xp, fp);

        // Before the step, should be exactly 0
        expect(pchip.eval(0.5)).toBeCloseTo(0);
        // During the step, should be monotonic (between 0 and 1)
        const midVal = pchip.eval(1.5);
        expect(midVal).toBeGreaterThan(0);
        expect(midVal).toBeLessThan(1);
        // After the step, should be exactly 1 (no overshoot)
        expect(pchip.eval(2.5)).toBeCloseTo(1);
        expect(pchip.eval(3.5)).toBeCloseTo(1);
    });

    it('returns zero derivative in flat regions and clamped bounds', () => {
        const xp = [0, 1, 2, 3];
        const fp = [5, 5, 2, 0];
        const pchip = new PchipInterpolator1D(xp, fp);

        // Derivative between two equal values must be 0
        expect(pchip.derivative(0)).toBeCloseTo(0);
        expect(pchip.derivative(0.5)).toBeCloseTo(0);
        expect(pchip.derivative(1)).toBeCloseTo(0);

        // Derivative falling off the flat region should be negative
        expect(pchip.derivative(1.5)).toBeLessThan(0);

        // Clamped out-of-bounds derivatives should be 0
        expect(pchip.derivative(-1)).toBe(0);
        expect(pchip.derivative(4)).toBe(0);
    });

    it('applies left and right clamping for evaluation', () => {
        const xp = [0, 1, 2];
        const fp = [10, 20, 30];
        const pchip = new PchipInterpolator1D(xp, fp, { left: 5, right: 35 }); // Custom bounds

        expect(pchip.eval(-1)).toBe(5);
        expect(pchip.eval(3)).toBe(35);

        const defaultPchip = new PchipInterpolator1D(xp, fp);
        expect(defaultPchip.eval(-1)).toBe(10); // Defaults to fp[0]
        expect(defaultPchip.eval(3)).toBe(30);  // Defaults to fp[last]
    });

    it('handles vectorization (array inputs) for eval and derivative', () => {
        const xp = [0, 1, 2];
        const fp = [0, 1, 8];
        const pchip = new PchipInterpolator1D(xp, fp);

        const xQuery = [0, 1, 2, 3];

        const evalRes = pchip.eval(xQuery);
        expect(evalRes).toBeInstanceOf(Vector);
        expect(evalRes.data[0]).toBe(0);
        expect(evalRes.data[1]).toBe(1);
        expect(evalRes.data[2]).toBe(8);
        expect(evalRes.data[3]).toBe(8); // Clamped

        const derivRes = pchip.derivative(xQuery);
        expect(derivRes).toBeInstanceOf(Vector);
        expect(derivRes.data[3]).toBe(0); // Out of bounds derivative is 0
    });

    it('falls back to linear interpolation for exactly 2 points', () => {
        const xp = [0, 2];
        const fp = [0, 10];
        const pchip = new PchipInterpolator1D(xp, fp);

        expect(pchip.eval(1)).toBeCloseTo(5);
        expect(pchip.derivative(0)).toBeCloseTo(5);
        expect(pchip.derivative(1)).toBeCloseTo(5);
        expect(pchip.derivative(2)).toBeCloseTo(5);
    });

    it('supports one point as a constant interpolant', () => {
        const pchip = new PchipInterpolator1D([2], [7]);
        expect(pchip.size).toBe(1);
        expect(pchip.eval(2)).toBe(7);
        expect(pchip.eval([0, 2, 4]).toArray()).toEqual([7, 7, 7]);
        expect(pchip.derivative(2)).toBe(0);
        expect(pchip.integrate(0, 4)).toBe(28);
    });

    it('rejects duplicate x-coordinates because PCHIP requires distinct knots', () => {
        expect(() => new PchipInterpolator1D([0, 1, 1], [0, 1, 2])).toThrow(RangeError);
    });

    it('integrates exact cubic segments over bounds', () => {
        const xp = [0, 1, 2];
        const fp = [0, 1, 0]; // Triangular-shaped data
        const pchip = new PchipInterpolator1D(xp, fp);

        // PCHIP fits a smooth curve, so the area differs from standard 
        // trapezoidal linear interpolation (which would be 1.0)
        const area = pchip.integrate(0, 2);

        // Split integral should equal the whole
        const leftHalf = pchip.integrate(0, 1);
        const rightHalf = pchip.integrate(1, 2);
        expect(leftHalf + rightHalf).toBeCloseTo(area);

        // Reversing bounds inverts the sign
        expect(pchip.integrate(2, 0)).toBeCloseTo(-area);
    });

    it('computes exact rectangular areas for constant out-of-bounds integration', () => {
        const xp = [0, 2];
        const fp = [10, 10]; // Flat line
        const pchip = new PchipInterpolator1D(xp, fp, { left: 10, right: 10 });

        // Interior
        expect(pchip.integrate(0, 2)).toBeCloseTo(20);

        // Out of bounds on the left (width 2 * height 10 = 20)
        expect(pchip.integrate(-2, 0)).toBeCloseTo(20);

        // Out of bounds on the right (width 3 * height 10 = 30)
        expect(pchip.integrate(2, 5)).toBeCloseTo(30);

        // Spanning across everything: [-1, 3] = total width 4 * height 10 = 40
        expect(pchip.integrate(-1, 3)).toBeCloseTo(40);
    });

    it('returns 0 for zero-width integration intervals', () => {
        const pchip = new PchipInterpolator1D([0, 1, 2], [1, 5, 2]);
        expect(pchip.integrate(1, 1)).toBe(0);
        expect(pchip.integrate(-5, -5)).toBe(0);
    });

    it('throws RangeError for NaN integration bounds', () => {
        const pchip = new PchipInterpolator1D([0, 1], [0, 1]);
        expect(() => pchip.integrate(NaN, 1)).toThrow(RangeError);
        expect(() => pchip.integrate(0, NaN)).toThrow(RangeError);
    });

});

describe('PchipInterpolatorND', () => {
    // Two independent scalar series packed as columns of fp, so every ND
    // result can be cross-checked against the already-verified 1D path.
    // fpA is the class's own doc-example series; fpB is a different
    // shape-preserving step function, chosen so a column mix-up (reading
    // the wrong column, or reusing fpA's derivatives for fpB) would be
    // caught by the componentwise comparison.
    const xp = [1, 2, 3];
    const fpA = [3, 2, 0];
    const fpB = [0, 0, 5];
    const fp = Matrix.from([
        [3, 0],
        [2, 0],
        [0, 5],
    ]);

    it('matches componentwise PchipInterpolator1D at knots and midpoints', () => {
        const f = new PchipInterpolatorND(xp, fp);
        const fA = new PchipInterpolator1D(xp, fpA);
        const fB = new PchipInterpolator1D(xp, fpB);

        for (const x of [1, 2, 3, 1.5, 2.5]) {
            const v = f.eval(x);
            expect(v).toBeInstanceOf(Vector);
            expect(v.toArray()[0]).toBeCloseTo(fA.eval(x));
            expect(v.toArray()[1]).toBeCloseTo(fB.eval(x));
        }
    });

    it('evaluates exactly at the knots', () => {
        const f = new PchipInterpolatorND(xp, fp);
        expect(f.eval(1).toArray()).toEqual([3, 0]);
        expect(f.eval(2).toArray()).toEqual([2, 0]);
        expect(f.eval(3).toArray()).toEqual([0, 5]);
    });

    it('exposes size and dim', () => {
        const f = new PchipInterpolatorND(xp, fp);
        expect(f.size).toBe(3);
        expect(f.dim).toBe(2);
    });

    it('applies left/right clamping, scalar and per-component', () => {
        const f = new PchipInterpolatorND(xp, fp, { left: -1, right: [9, 8] });
        expect(f.eval(0).toArray()).toEqual([-1, -1]);
        expect(f.eval(4).toArray()).toEqual([9, 8]);
    });

    it('handles vectorization for eval and derivative, matching the 1D path', () => {
        const f = new PchipInterpolatorND(xp, fp);
        const fA = new PchipInterpolator1D(xp, fpA);
        const fB = new PchipInterpolator1D(xp, fpB);

        const evalRes = f.eval([0, 1, 2.5, 4]);
        expect(evalRes).toBeInstanceOf(Matrix);
        expect(evalRes.rows).toBe(4);
        expect(evalRes.cols).toBe(2);
        for (let i = 0; i < 4; i++) {
            const x = [0, 1, 2.5, 4][i];
            expect(evalRes.row(i).toArray()[0]).toBeCloseTo(fA.eval(x));
            expect(evalRes.row(i).toArray()[1]).toBeCloseTo(fB.eval(x));
        }

        const derivRes = f.derivative([0, 1, 2.5, 4]);
        for (let i = 0; i < 4; i++) {
            const x = [0, 1, 2.5, 4][i];
            expect(derivRes.row(i).toArray()[0]).toBeCloseTo(fA.derivative(x));
            expect(derivRes.row(i).toArray()[1]).toBeCloseTo(fB.derivative(x));
        }
    });

    it('throws for an empty query array (Matrix cannot have 0 rows)', () => {
        const f = new PchipInterpolatorND(xp, fp);
        expect(() => f.eval([])).toThrow(RangeError);
        expect(() => f.derivative([])).toThrow(RangeError);
    });

    it('falls back to linear interpolation for exactly 2 points', () => {
        const f = new PchipInterpolatorND([0, 2], Matrix.from([[0, 10], [10, 0]]));
        expect(f.eval(1).toArray()).toEqual([5, 5]);
        expect(f.derivative(1).toArray()).toEqual([5, -5]);
    });

    it('supports one point as a constant interpolant', () => {
        const f = new PchipInterpolatorND([2], Matrix.from([[7, -3]]));
        expect(f.size).toBe(1);
        expect(f.eval(2).toArray()).toEqual([7, -3]);
        expect(f.derivative(2).toArray()).toEqual([0, 0]);
        expect(f.integrate(0, 4).toArray()).toEqual([28, -12]);
    });

    it('rejects duplicate x-coordinates because PCHIP requires distinct knots', () => {
        expect(() => new PchipInterpolatorND([0, 1, 1], Matrix.from([[0, 0], [1, 1], [2, 2]]))).toThrow(RangeError);
    });

    describe('integrate', () => {
        it('matches componentwise PchipInterpolator1D integrals', () => {
            const f = new PchipInterpolatorND(xp, fp);
            const fA = new PchipInterpolator1D(xp, fpA);
            const fB = new PchipInterpolator1D(xp, fpB);

            const result = f.integrate(1, 3);
            expect(result.toArray()[0]).toBeCloseTo(fA.integrate(1, 3));
            expect(result.toArray()[1]).toBeCloseTo(fB.integrate(1, 3));

            // Split integral should equal the whole, per component
            const left = f.integrate(1, 2);
            const right = f.integrate(2, 3);
            expect(left.toArray()[0] + right.toArray()[0]).toBeCloseTo(result.toArray()[0]);
            expect(left.toArray()[1] + right.toArray()[1]).toBeCloseTo(result.toArray()[1]);
        });

        it('negates and swaps bounds when b < a', () => {
            const f = new PchipInterpolatorND(xp, fp);
            const fwd = f.integrate(1, 3);
            const rev = f.integrate(3, 1);
            expect(rev.toArray()[0]).toBeCloseTo(-fwd.toArray()[0]);
            expect(rev.toArray()[1]).toBeCloseTo(-fwd.toArray()[1]);
        });

        it('computes exact rectangular areas for constant out-of-bounds integration', () => {
            const f = new PchipInterpolatorND([0, 2], Matrix.from([[10, 1], [10, 1]]), { left: [10, 1], right: [10, 1] });
            expect(f.integrate(-2, 0).toArray()).toEqual([20, 2]);
            expect(f.integrate(2, 5).toArray()).toEqual([30, 3]);
        });

        it('returns the zero vector for zero-width integration intervals', () => {
            const f = new PchipInterpolatorND(xp, fp);
            expect(f.integrate(1, 1).toArray()).toEqual([0, 0]);
        });

        it('throws RangeError for NaN integration bounds', () => {
            const f = new PchipInterpolatorND(xp, fp);
            expect(() => f.integrate(NaN, 1)).toThrow(RangeError);
            expect(() => f.integrate(0, NaN)).toThrow(RangeError);
        });
    });
});