import { beforeEach, describe, expect, it } from 'vitest';
import { besselJ, besselJZero, clearBesselCache } from '../../src/special/bessel.js';

// Reference values computed with mpmath (40 digits), printed with 17 significant digits.
const J_REFERENCE: [n: number, x: number, value: number][] = [
    [0, 1, 0.76519768655796655],
    [1, 1, 0.44005058574493352],
    [2, 1, 0.11490348493190048],
    [3, 0.5, 0.0025637299945872441],
    [5, 2.5, 0.01950162513450322],
    [20, 5, 2.7703300521289417e-11],
    [5, 0.001, 2.6041665581597244e-19],
    [10, 9.5, 0.16502640472619116],
    [10, 10, 0.20748610663335886],
    [10, 10.5, 0.24774553753592743],
    [30, 30, 0.14393585001030721],
    [0, 50, 0.055812327669251815],
    [100, 50, 1.1159273690838093e-21],
    [2, 100, -0.021528757344505366],
    [50, 100, -0.038698339728525383],
    [1, 1000, 0.0047283119070895239],
];

const ZERO_REFERENCE: [n: number, m: number, value: number][] = [
    [0, 1, 2.4048255576957728],
    [0, 2, 5.5200781102863106],
    [0, 3, 8.6537279129110122],
    [1, 1, 3.8317059702075123],
    [1, 2, 7.0155866698156188],
    [2, 1, 5.1356223018406826],
    [2, 20, 65.159273190757798],
    [5, 3, 15.700174079711671],
    [30, 2, 41.092778663153428],
    [100, 1, 108.83616589840977],
];

describe('besselJ', () => {
    it('matches high-precision reference values', () => {
        for (const [n, x, expected] of J_REFERENCE) {
            const actual = besselJ(n, x);
            // Absolute accuracy ~1e-15 where J is O(0.1); relative accuracy where it decays.
            expect(Math.abs(actual - expected), `J_${n}(${x})`).toBeLessThanOrEqual(
                1e-15 + 1e-13 * Math.abs(expected)
            );
        }
    });

    it('is exact at x = 0', () => {
        expect(besselJ(0, 0)).toBe(1);
        expect(besselJ(1, 0)).toBe(0);
        expect(besselJ(7, 0)).toBe(0);
        expect(besselJ(0, -0)).toBe(1);
    });

    it('applies the parity identity J_n(-x) = (-1)^n J_n(x)', () => {
        for (const n of [0, 1, 2, 3, 10, 11]) {
            for (const x of [0.3, 2.5, 25, 250]) {
                expect(besselJ(n, -x)).toBe((n % 2 === 0 ? 1 : -1) * besselJ(n, x));
            }
        }
    });

    it('handles non-finite arguments', () => {
        expect(besselJ(0, NaN)).toBeNaN();
        expect(besselJ(3, NaN)).toBeNaN();
        expect(besselJ(0, Infinity)).toBe(0);
        expect(besselJ(2, -Infinity)).toBe(0);
        expect(besselJ(3, -Infinity)).toBe(-0); // odd order: limit approached from the negative side
    });

    it('is finite and accurate for tiny arguments (no overflow in the recurrence)', () => {
        expect(besselJ(0, 1e-300)).toBe(1);
        expect(besselJ(1, 1e-300) / 5e-301).toBeCloseTo(1, 14);
        expect(besselJ(2, 1e-100) / 1.25e-201).toBeCloseTo(1, 14);
        expect(besselJ(2, 1e-300)).toBe(0); // true value 1.25e-601 underflows
        expect(besselJ(5, 1e-310)).toBe(0);
    });

    it('is continuous across the internal method boundaries', () => {
        // x = 1 (series -> recurrence), n = x (forward -> Miller).
        const below = besselJ(3, 1 - 1e-12);
        const above = besselJ(3, 1 + 1e-12);
        expect(Math.abs(below - above)).toBeLessThan(1e-11);
        for (const n of [2, 10, 100, 500]) {
            const lo = besselJ(n, n - 1e-9);
            const at = besselJ(n, n);
            const hi = besselJ(n, n + 1e-9);
            expect(Math.abs(lo - at)).toBeLessThan(1e-8);
            expect(Math.abs(hi - at)).toBeLessThan(1e-8);
        }
    });

    it('satisfies the three-term recurrence J_{n-1} + J_{n+1} = (2n/x) J_n', () => {
        for (const x of [0.7, 3.3, 15, 80, 400, 3000]) {
            for (const n of [1, 2, 5, 12, 40, 150]) {
                const lhs = besselJ(n - 1, x) + besselJ(n + 1, x);
                const rhs = (2 * n / x) * besselJ(n, x);
                expect(Math.abs(lhs - rhs), `n=${n}, x=${x}`).toBeLessThan(1e-13 * Math.max(1, 2 * n / x));
            }
        }
    });

    it('satisfies the Neumann sum rule J_0^2 + 2 sum_k J_k^2 = 1, including large x', () => {
        for (const x of [0.5, 7, 60, 1000]) {
            let sum = besselJ(0, x) ** 2;
            for (let k = 1; k < x + 20 * Math.cbrt(x) + 60; k++) sum += 2 * besselJ(k, x) ** 2;
            expect(Math.abs(sum - 1), `x=${x}`).toBeLessThan(1e-12);
        }
    });

    it('satisfies the Jacobi-Anger expansion cos(x) = J_0 + 2 sum (-1)^k J_{2k}', () => {
        for (const x of [2, 30, 200]) {
            let sum = besselJ(0, x);
            for (let k = 1; 2 * k < x + 20 * Math.cbrt(x) + 60; k++) {
                sum += 2 * (k % 2 === 0 ? 1 : -1) * besselJ(2 * k, x);
            }
            expect(Math.abs(sum - Math.cos(x)), `x=${x}`).toBeLessThan(1e-12);
        }
    });

    it('stays accurate for n >= 2 at large x (regression: Miller start order)', () => {
        // The old implementation was off by O(1) here.
        expect(besselJ(2, 100)).toBeCloseTo(-0.021528757344505366, 13);
        expect(besselJ(5, 1000)).toBeCloseTo(0.005025406945233186, 13);
    });

    it('stays accurate for J_0 and J_1 beyond the old sampling cap (regression)', () => {
        // x = 1e4 exceeded the old fixed cap of 4000 trapezoid steps.
        expect(besselJ(0, 1e4)).toBeCloseTo(-0.007096160353388801, 12);
        expect(besselJ(1, 1e4)).toBeCloseTo(0.00364745075552958, 12);
    });

    it('works for large orders', () => {
        expect(besselJ(1000, 1)).toBe(0); // underflows
        expect(besselJ(1000, 1000)).toBeCloseTo(0.04473067294796404, 13);
        expect(Number.isFinite(besselJ(1e6, 1e6))).toBe(true);
    });

    it('throws RangeError for invalid orders', () => {
        for (const n of [-1, 1.5, NaN, Infinity, -Infinity, 1e6 + 1, 1e9]) {
            expect(() => besselJ(n, 5), `n=${n}`).toThrow(RangeError);
        }
        expect(() => besselJ(-1, NaN)).toThrow(RangeError);
    });
});

describe('besselJZero', () => {
    beforeEach(() => {
        clearBesselCache();
    });

    it('matches high-precision reference values', () => {
        for (const [n, m, expected] of ZERO_REFERENCE) {
            const actual = besselJZero(n, m);
            expect(Math.abs(actual - expected) / expected, `j_{${n},${m}}`).toBeLessThan(1e-14);
        }
    });

    it('returns zeros that are roots of J_n', () => {
        for (const n of [0, 1, 2, 7, 25]) {
            for (const m of [1, 2, 5, 17]) {
                const z = besselJZero(n, m);
                const slope = Math.abs(besselJ(n - 1 >= 0 ? n - 1 : 1, z) - (n / z) * besselJ(n, z));
                expect(Math.abs(besselJ(n, z)) / Math.max(slope, 1e-3), `n=${n}, m=${m}`).toBeLessThan(5e-14);
            }
        }
    });

    it('returns strictly increasing zeros with no missed index (large m, n >= 2)', () => {
        // The old implementation skipped zeros beyond x ~ 60 for n >= 2.
        for (const n of [0, 2, 5, 30]) {
            let previous = besselJZero(n, 1);
            for (let m = 2; m <= 120; m++) {
                const z = besselJZero(n, m);
                expect(z).toBeGreaterThan(previous + 3.11); // gaps between zeros exceed 3.116
                previous = z;
            }
        }
        expect(besselJZero(2, 20)).toBeCloseTo(65.159273190757798, 12);
        expect(besselJZero(2, 50)).toBeCloseTo(159.42406617141825, 10);
    });

    it('gives the same value regardless of request order and cache state', () => {
        const direct = besselJZero(3, 9);
        clearBesselCache();
        const sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9].map((m) => besselJZero(3, m));
        expect(sequence[8]).toBe(direct);
        clearBesselCache();
        const outOfOrder = [besselJZero(3, 9), besselJZero(3, 2), besselJZero(3, 20), besselJZero(3, 2)];
        expect(outOfOrder[0]).toBe(direct);
        expect(outOfOrder[1]).toBe(sequence[1]);
        expect(outOfOrder[3]).toBe(outOfOrder[1]);
    });

    it('keeps orders independent', () => {
        const j0 = besselJZero(0, 3);
        const j1 = besselJZero(1, 3);
        expect(besselJZero(0, 3)).toBe(j0);
        expect(besselJZero(1, 3)).toBe(j1);
        expect(j0).not.toBe(j1);
    });

    it('throws RangeError for invalid order or index, leaving the cache usable', () => {
        for (const n of [-1, 1.5, NaN, Infinity, 1e6 + 1]) {
            expect(() => besselJZero(n, 1), `n=${n}`).toThrow(RangeError);
        }
        for (const m of [0, -3, 1.5, NaN, Infinity]) {
            expect(() => besselJZero(0, m), `m=${m}`).toThrow(RangeError);
        }
        expect(besselJZero(0, 1)).toBeCloseTo(2.404825557695773, 14);
    });
});
