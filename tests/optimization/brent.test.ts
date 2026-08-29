import { describe, expect, it, vi } from 'vitest';
import { brent } from '../../src/optimize/brent.js';
import { isClose } from '../../src/misc.js'
import type { ScalarFunction } from '../../src/types.js';

describe('fminBrent', () => {
    it('finds the minimum of a simple parabola', () => {
        const fn = (x: number) => (x - 2) ** 2 + 1;
        const result = brent(fn, -5, 5);
        expect(result.x).toBeCloseTo(2, 6);
    });

    it('finds the minimum of x^4 - x + 1', () => {
        // Matches the docstring example from the Python reference implementation.
        const fn = (x: number) => x ** 4 - x + 1;
        const result = brent(fn, -3, 3);
        expect(result).toMatchObject({
            method: 'Brent',
            success: true,
            message: '|dx| <= tolX',
            nFev: 19,
            nIter: 18,
        });
        expect(result.x).toBeCloseTo(0.6299605249, 6);
        expect(result.f).toBeCloseTo(0.5275296058, 6);
    });

    it('works when the minimum sits near the left edge of the bracket', () => {
        const fn = (x: number) => (x + 4.9) ** 2;
        const result = brent(fn, -5, 5);
        expect(result.x).toBeCloseTo(-4.9, 5);
    });

    it('works when the minimum sits near the right edge of the bracket', () => {
        const fn = (x: number) => (x - 4.9) ** 2;
        const result = brent(fn, -5, 5);
        expect(result.x).toBeCloseTo(4.9, 5);
    });

    it('is insensitive to the order of the bracket endpoints', () => {
        const fn = (x: number) => (x - 1.234) ** 2;
        const forward = brent(fn, -10, 10);
        const reversed = brent(fn, 10, -10);
        expect(forward.x).toBeCloseTo(1.234, 6);
        expect(reversed.x).toBeCloseTo(1.234, 6);
    });

    it('converges at least as tightly when tolX is smaller', () => {
        // A pure quadratic is fit exactly by the parabolic step, so use a quartic
        // (as in the x^4 - x + 1 case) where the tolerance genuinely gates how many
        // refinement steps are taken.
        const fn = (x: number) => (x - Math.PI) ** 4;
        const loose = brent(fn, 0, 10, 1e-1);
        const tight = brent(fn, 0, 10, 1e-12);
        expect(Math.abs(tight.x - Math.PI)).toBeLessThanOrEqual(Math.abs(loose.x - Math.PI));
        expect(tight.x).toBeCloseTo(Math.PI, 8);
    });

    it('handles a minimum at x = 0', () => {
        const fn = (x: number) => x ** 2;
        const result = brent(fn, -1, 1);
        expect(result.x).toBeCloseTo(0, 6);
    });

    it('handles asymmetric brackets', () => {
        const fn = (x: number) => (x - 0.1) ** 2;
        const result = brent(fn, -100, 1);
        expect(result.x).toBeCloseTo(0.1, 5);
    });

    it('finds a local minimum of a non-convex function within the bracket', () => {
        // Has a local min near x ≈ -1.42 and another near x ≈ 1.42 within [-3, 3];
        // starting near the left half of the bracket should land on the left one.
        const fn = (x: number) => x ** 4 - 4 * x ** 2;
        const result = brent(fn, -3, 0);
        expect(result.x).toBeCloseTo(-Math.sqrt(2), 4);
    });

    it('warns and still returns a value when maxIter is exhausted', () => {
        const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
        const fn = (x: number) => (x - 2) ** 2;

        const result = brent(fn, -5, 5, 1e-14, 1);

        expect(warnSpy).toHaveBeenCalledOnce();
        expect(warnSpy.mock.calls[0][0]).toContain('maxIter');
        expect(result.success).toBe(false);
        expect(result.x).toBeDefined();

        warnSpy.mockRestore();
    });

    it('respects a custom maxIter without throwing', () => {
        const fn = (x: number) => (x - 2) ** 2;
        expect(() => brent(fn, -5, 5, 1e-8, 5)).not.toThrow();
    });

    it('handles a flat (constant) function without infinite looping', () => {
        const fn = () => 42;
        expect(() => brent(fn, -1, 1)).not.toThrow();
    });

    it('only evaluates fn within [xa, xb] (no bracket escape)', () => {
        const lo = -2;
        const hi = 3;
        const fn = vi.fn((x: number) => {
            expect(x).toBeGreaterThanOrEqual(lo);
            expect(x).toBeLessThanOrEqual(hi);
            return (x - 0.5) ** 2;
        });

        brent(fn, lo, hi);
        expect(fn).toHaveBeenCalled();
    });
});

/**
 * Test problems ported from polykin's test_brent.py (Python reference
 * implementation this module was translated from). Each case brackets a
 * minimum and checks that fminBrent converges to it within tolerance.
 */
interface BrentTestCase {
    f: ScalarFunction;
    xa: number;
    xb: number;
    xmin: number;
}

const TEST_FUNCTIONS: Record<string, BrentTestCase> = {
    quadratic: {
        f: (x) => (x - 2.0) ** 2,
        xa: -5.0,
        xb: 5.0,
        xmin: 2.0,
    },
    scaled_quadratic: {
        f: (x) => 1e6 * (x - 1e-3) ** 2,
        xa: -1.0,
        xb: 1.0,
        xmin: 1e-3,
    },
    quartic_flat: {
        f: (x) => (x - 1.0) ** 4,
        xa: -2.0,
        xb: 3.0,
        xmin: 1.0,
    },
    absolute_value: {
        f: (x) => Math.abs(x - 0.5),
        xa: -1.0,
        xb: 2.0,
        xmin: 0.5,
    },
    multi_minima: {
        f: (x) => Math.sin(5 * x) + (x - 1) ** 2,
        xa: -2.0,
        xb: 3.0,
        xmin: 0.9467389984,
    },
    degenerate_parabola: {
        // Exact minimizer of (x-1)^2 + eps*x is 1 - eps/2.
        f: (x) => (x - 1) ** 2 + 1e-12 * x,
        xa: 0.0,
        xb: 2.0,
        xmin: 1.0 - 5e-13,
    },
    sharp_minimum: {
        f: (x) => Math.exp(50 * (x - 0.3) ** 2),
        xa: 0.0,
        xb: 1.0,
        xmin: 0.3,
    },
    boundary_minimum: {
        f: (x) => (x + 2) ** 2,
        xa: -2.0,
        xb: 5.0,
        xmin: -2.0,
    },
};

describe('fminBrent (test problems ported from polykin test_brent.py)', () => {
    const tolX = 1e-6;

    for (const [name, { f, xa, xb, xmin }] of Object.entries(TEST_FUNCTIONS)) {
        it(`finds the minimum for "${name}"`, () => {
            const result = brent(f, xa, xb, tolX);
            expect(
                isClose(result.x, xmin, 2 * tolX),
                `Incorrect minimum for ${name}: x=${result.x}, expected ${xmin}`
            ).toBe(true);
        });
    }
});
