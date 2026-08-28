import { describe, it, expect, vi, afterEach } from 'vitest';
import { bisection } from '../src/roots/bisection';
import { secant } from '../src/roots/secant';
import { brent } from '../src/roots/brent';

// A simple test function: f(x) = x^2 - 4, which has roots at x = 2 and x = -2
const f = (x: number): number => x * x - 4;

describe('bisection', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('should find a root within the given tolerance', () => {
        const root = bisection(f, 0, 5);
        expect(root).toBeCloseTo(2, 5);
    });

    it('should swap bounds if a > b', () => {
        const root = bisection(f, 5, 0);
        expect(root).toBeCloseTo(2, 5);
    });

    it('should return exactly a if f(a) is 0', () => {
        expect(bisection(f, 2, 5)).toBe(2);
    });

    it('should return exactly b if f(b) is 0', () => {
        expect(bisection(f, 0, 2)).toBe(2);
    });

    it('should throw an error if a and b are the same', () => {
        expect(() => bisection(f, 1, 1)).toThrow('bisection: a and b must be different');
    });

    it('should throw an error if fn(a) and fn(b) have the same sign', () => {
        expect(() => bisection(f, 3, 5)).toThrow('bisection: fn(a) and fn(b) must have opposite signs');
    });

    it('should warn and return mid if maxIter is reached without convergence', () => {
        const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
        // Only allow 2 iterations, which won't be enough to reach the default 1e-8 tolerance
        const root = bisection(f, 0, 5, 1e-8, 2);

        expect(consoleSpy).toHaveBeenCalledWith(
            expect.stringContaining('bisection: reached maxIter (2) without converging')
        );
        expect(root).toBeGreaterThan(0);
        expect(root).toBeLessThan(5);
    });
});

describe('secant', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('should find a root within the given tolerance', () => {
        const root = secant(f, 0, 5);
        expect(root).toBeCloseTo(2, 5);
    });

    it('should return exactly x0 if f(x0) is 0', () => {
        expect(secant(f, 2, 5)).toBe(2);
    });

    it('should return exactly x1 if f(x1) is 0', () => {
        expect(secant(f, 0, 2)).toBe(2);
    });

    it('should throw an error if x0 and x1 are the same', () => {
        expect(() => secant(f, 1, 1)).toThrow('secant: x0 and x1 must be different');
    });

    it('should throw an error if the denominator becomes zero (flat slope)', () => {
        // f(1) = -3, f(-1) = -3, so f1 - f0 = 0
        expect(() => secant(f, -1, 1)).toThrow('secant: zero denominator encountered');
    });

    it('should warn and return the last estimate if maxIter is reached', () => {
        const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
        // Only allow 1 iteration
        const root = secant(f, 0, 5, 1e-8, 1);

        expect(consoleSpy).toHaveBeenCalledWith(
            expect.stringContaining('secant: reached maxIter (1) without converging')
        );
        expect(root).toBeDefined();
    });
});

describe('brent', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('should find a root within the given tolerance', () => {
        const root = brent(f, 0, 5);
        expect(root).toBeCloseTo(2, 5);
    });

    it('should find the negative root within the given tolerance', () => {
        const root = brent(f, -5, 0);
        expect(root).toBeCloseTo(-2, 5);
    });

    it('should return exactly xa if f(xa) is 0', () => {
        expect(brent(f, 2, 5)).toBe(2);
    });

    it('should return exactly xb if f(xb) is 0', () => {
        expect(brent(f, 0, 2)).toBe(2);
    });

    it('should throw an error if fn(xa) and fn(xb) have the same sign', () => {
        expect(() => brent(f, 3, 5)).toThrow(
            'brent: root is not bracketed (fn(xa) and fn(xb) must have opposite signs)'
        );
    });

    it('should warn and return the last estimate if maxIter is reached', () => {
        const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
        // Only allow 1 iteration, which won't be enough to reach the default tolerances
        const root = brent(f, 0, 5, 1e-8, 1e-8, 1);

        expect(consoleSpy).toHaveBeenCalledWith(
            expect.stringContaining('brent: reached maxIter (1) without converging')
        );
        expect(root).toBeGreaterThan(0);
        expect(root).toBeLessThan(5);
    });

    it('should converge early when |f(x)| is within tolF', () => {
        // A wide bracket but a loose function-value tolerance should let the
        // function-value stop criterion trigger before the x-tolerance would.
        const root = brent(f, 0, 5, 1e-15, 1e-1);
        expect(f(root)).toBeLessThanOrEqual(1e-1);
    });
});