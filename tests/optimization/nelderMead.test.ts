import { describe, expect, it } from 'vitest';
import { Array1D } from '../../src/array/array1d.js';
import { Array2D } from '../../src/array/array2d.js';
import { minimizeNelderMead } from '../../src/optimization/nelderMead.js';
import { TEST_FUNCTIONS_MULTIVAR } from './testFunctions';

describe.each(Object.entries(TEST_FUNCTIONS_MULTIVAR))('fminNelderMead — %s', (_name, data) => {
    const N = 2; // test in 2D for simplicity, but should work in any dimension
    const tolx = 1e-6;

    it('converges to the known global minimum', () => {
        const x0 = data.initialPoint(N);
        const res = minimizeNelderMead(data.fn, x0, tolx);

        expect(res.success, res.message).toBe(true);
        expect(Math.abs(res.f - data.globalMinimum)).toBeLessThanOrEqual(2 * tolx);
    });

    it('stops early and reports success when the callback requests it', () => {
        const x0 = data.initialPoint(N);
        const res = minimizeNelderMead(
            data.fn,
            x0,
            tolx,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            (nIter) => ({ stop: nIter >= 3, success: true })
        );

        expect(res.success).toBe(true);
        expect(res.message).toMatch(/callback/i);
        expect(res.nIter).toBe(3);
    });
});

describe('fminNelderMead — options and edge cases', () => {
    it('throws for a zero-dimensional initial guess', () => {
        expect(() => minimizeNelderMead(() => 0, [])).toThrow(RangeError);
    });

    it('accepts an Array1D as the initial guess', () => {
        const res = minimizeNelderMead(
            (x) => x.get(0) ** 2 + x.get(1) ** 2,
            Array1D.from([3, -4])
        );
        expect(res.success).toBe(true);
        expect(res.f).toBeCloseTo(0, 5);
    });

    it('respects a custom maxiter and reports failure without throwing', () => {
        const res = minimizeNelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, [10, 10], 1e-8, 1e-8, undefined, 2);
        expect(res.success).toBe(false);
        expect(res.nIter).toBe(2);
        expect(res.message).toMatch(/maximum number of iterations/i);
    });

    it('respects a custom maxfeval and reports failure without throwing', () => {
        const res = minimizeNelderMead(
            (x) => x.get(0) ** 2 + x.get(1) ** 2,
            [10, 10],
            1e-8,
            1e-8,
            undefined,
            undefined,
            5
        );
        expect(res.success).toBe(false);
        expect(res.nFev).toBeGreaterThanOrEqual(5);
        expect(res.message).toMatch(/maximum number of function evaluations/i);
    });

    it('honors a user-supplied sclx', () => {
        const res = minimizeNelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, [1, 1], 1e-8, 1e-8, [1, 1]);
        expect(res.success).toBe(true);
        expect(res.f).toBeCloseTo(0, 5);
    });

    it('supports the non-adaptive parameter scheme', () => {
        const res = minimizeNelderMead((x) => x.get(0) ** 2 + x.get(1) ** 2, [5, -3], 1e-8, 1e-8, undefined, undefined, undefined, false);
        expect(res.success).toBe(true);
        expect(res.f).toBeCloseTo(0, 5);
    });

    it('passes the callback an (N+1) x N Array2D and an (N+1) Array1D', () => {
        let seenRows = -1;
        let seenCols = -1;
        let seenFxDim = -1;

        minimizeNelderMead(
            (x) => x.get(0) ** 2 + x.get(1) ** 2,
            [5, -3],
            1e-8,
            1e-8,
            undefined,
            undefined,
            undefined,
            undefined,
            (nIter, x: Array2D, fx: Array1D) => {
                if (nIter === 1) {
                    seenRows = x.rows;
                    seenCols = x.cols;
                    seenFxDim = fx.size;
                }
                return { stop: nIter >= 1, success: true };
            }
        );

        expect(seenRows).toBe(3); // N+1 vertices for N=2
        expect(seenCols).toBe(2);
        expect(seenFxDim).toBe(3);
    });
});
