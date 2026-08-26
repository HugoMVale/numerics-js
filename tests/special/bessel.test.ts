import { describe, it, expect, vi, beforeEach } from 'vitest';
import { bessel } from '../../src/special/bessel';
import * as roots from '../../src/roots/brent.js';

describe('Bessel Function Module', () => {

    beforeEach(() => {
        // Clear the internal cache and reset spies before each test
        bessel.zerosCache = [];
        vi.restoreAllMocks();
    });

    describe('J(n, x) - Evaluation', () => {
        it('should throw RangeError for invalid orders', () => {
            expect(() => bessel.J(-1, 5)).toThrow(RangeError);
            expect(() => bessel.J(1.5, 5)).toThrow(RangeError);
        });

        it('should handle x = 0 correctly', () => {
            expect(bessel.J(0, 0)).toBe(1);
            expect(bessel.J(1, 0)).toBe(0);
        });

        it('should handle negative x using parity identity', () => {
            const x: number = 2.5;
            expect(bessel.J(0, -x)).toBeCloseTo(bessel.J(0, x), 10);
            expect(bessel.J(1, -x)).toBeCloseTo(-bessel.J(1, x), 10);
        });

        it('should accurately compute J_n(x) for n=0, n=1', () => {
            expect(bessel.J(0, 1)).toBeCloseTo(0.7651976865, 5);
            expect(bessel.J(1, 1)).toBeCloseTo(0.4400505857, 5);
        });

        it('should accurately compute J_n(x) for n >= 2', () => {
            expect(bessel.J(2, 1)).toBeCloseTo(0.1149034849, 5);
            expect(bessel.J(5, 2.5)).toBeCloseTo(0.0195025791, 5);
        });
    });

    describe('getZero(n, m) - Mathematical Roots and Caching', () => {
        it('should throw RangeError for invalid index m', () => {
            expect(() => bessel.getZero(0, 0)).toThrow(RangeError);
        });

        it('should accurately find the first few zeros of J_0', () => {
            expect(bessel.getZero(0, 1)).toBeCloseTo(2.4048, 4);
            expect(bessel.getZero(0, 2)).toBeCloseTo(5.5201, 4);
            expect(bessel.getZero(0, 3)).toBeCloseTo(8.6537, 4);
        });

        it('should accurately find the first few zeros of J_1', () => {
            expect(bessel.getZero(1, 1)).toBeCloseTo(3.8317, 4);
            expect(bessel.getZero(1, 2)).toBeCloseTo(7.0156, 4);
        });

        it('should utilize the cache for previously computed zeros', () => {
            const spy = vi.spyOn(roots, 'brent');

            bessel.getZero(0, 1); // Computes and caches
            expect(spy).toHaveBeenCalled();

            spy.mockClear(); // Reset the call tracker on the spy

            const cachedRoot: number = bessel.getZero(0, 1); // Retrieves from cache
            expect(spy).not.toHaveBeenCalled(); // Proves bisection was skipped
            expect(cachedRoot).toBeCloseTo(2.4048, 4);
        });

        it('should resume scanning from the highest cached zero', () => {
            bessel.getZero(0, 1); // First root cached ~ 2.4048

            const spy = vi.spyOn(roots, 'brent');
            bessel.getZero(0, 2);

            expect(spy).toHaveBeenCalledTimes(1);
            const callArgs = spy.mock.calls[0];
            const lowerBracket: number = callArgs[1];

            // The step size is 0.5. It should resume from cache[1] + 0.5 (~2.9)
            expect(lowerBracket).toBeGreaterThan(2.8);
        });
    });
});