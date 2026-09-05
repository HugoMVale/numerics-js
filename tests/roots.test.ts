import { describe, it, expect, vi, afterEach } from 'vitest';
import { bisection } from '../src/roots/bisection.js';
import { secant } from '../src/roots/secant.js';
import { brent } from '../src/roots/brent.js';

// A simple test function: f(x) = x^2 - 4, which has roots at x = 2 and x = -2
const f = (x: number): number => x * x - 4;

describe('bisection', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('should find a root within the given tolerance', () => {
        const result = bisection(f, 0, 5);
        expect(result.x).toBeCloseTo(2, 5);
        expect(result.method).toBe('bisection');
        expect(result.fx).toBeCloseTo(f(result.x), 10);
        expect(result.evaluations).toBeGreaterThan(0);
    });

    it('should swap bounds if a > b', () => {
        const result = bisection(f, 5, 0);
        expect(result.x).toBeCloseTo(2, 5);
    });

    it('should return exactly a if f(a) is 0', () => {
        const result = bisection(f, 2, 5);
        expect(result.x).toBe(2);
        expect(result.fx).toBe(0);
        expect(result.evaluations).toBe(2);
    });

    it('should return exactly b if f(b) is 0', () => {
        const result = bisection(f, 0, 2);
        expect(result.x).toBe(2);
        expect(result.fx).toBe(0);
        expect(result.evaluations).toBe(2);
    });

    it('should throw an error if a and b are the same', () => {
        expect(() => bisection(f, 1, 1)).toThrow('bisection: a and b must be different');
    });

    it('should throw an error if f(a) and f(b) have the same sign', () => {
        expect(() => bisection(f, 3, 5)).toThrow('bisection: f(a) and f(b) must have opposite signs');
    });

    it('should return mid with success false if maxIter is reached without convergence', () => {
        // Only allow 2 iterations, which won't be enough to reach the default 1e-8 tolerance
        const result = bisection(f, 0, 5, { maxIter: 2 });

        expect(result.success).toBe(false);
        expect(result.message).toContain('did not converge: reached maxIter (2)');
        expect(result.x).toBeGreaterThan(0);
        expect(result.x).toBeLessThan(5);
    });
});

describe('secant', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('should find a root within the given tolerance', () => {
        const result = secant(f, 0, 5);
        expect(result.x).toBeCloseTo(2, 5);
        expect(result.method).toBe('secant');
        expect(result.fx).toBeCloseTo(f(result.x), 10);
        expect(result.evaluations).toBeGreaterThan(0);
    });

    it('should return exactly x0 if f(x0) is 0', () => {
        const result = secant(f, 2, 5);
        expect(result.x).toBe(2);
        expect(result.fx).toBe(0);
        expect(result.evaluations).toBe(2);
    });

    it('should return exactly x1 if f(x1) is 0', () => {
        const result = secant(f, 0, 2);
        expect(result.x).toBe(2);
        expect(result.fx).toBe(0);
        expect(result.evaluations).toBe(2);
    });

    it('should throw an error if x0 and x1 are the same', () => {
        expect(() => secant(f, 1, 1)).toThrow('secant: x0 and x1 must be different');
    });

    it('should return success false if the denominator becomes zero (flat slope)', () => {
        // f(1) = -3, f(-1) = -3, so f1 - f0 = 0
        const result = secant(f, -1, 1);
        expect(result.success).toBe(false);
        expect(result.message).toContain('did not converge: zero denominator encountered');
    });

    it('should return the last estimate with success false if maxIter is reached', () => {
        // Only allow 1 iteration
        const result = secant(f, 0, 5, { maxIter: 1 });

        expect(result.success).toBe(false);
        expect(result.message).toContain('did not converge: reached maxIter (1)');
        expect(result.x).toBeDefined();
    });
});

describe('brent', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('should find a root within the given tolerance', () => {
        const result = brent(f, 0, 5);
        expect(result.x).toBeCloseTo(2, 5);
        expect(result.method).toBe('brent');
        expect(result.fx).toBeCloseTo(f(result.x), 10);
        expect(result.evaluations).toBeGreaterThan(0);
    });

    it('should find the negative root within the given tolerance', () => {
        const result = brent(f, -5, 0);
        expect(result.x).toBeCloseTo(-2, 5);
    });

    it('should return exactly xa if f(xa) is 0', () => {
        const result = brent(f, 2, 5);
        expect(result.x).toBe(2);
        expect(result.fx).toBe(0);
        expect(result.evaluations).toBe(1);
    });

    it('should return exactly xb if f(xb) is 0', () => {
        const result = brent(f, 0, 2);
        expect(result.x).toBe(2);
        expect(result.fx).toBe(0);
        expect(result.evaluations).toBe(2);
    });

    it('should throw an error if f(xa) and f(xb) have the same sign', () => {
        expect(() => brent(f, 3, 5)).toThrow(
            'brent: root is not bracketed (f(xa) and f(xb) must have opposite signs)'
        );
    });

    it('should return the last estimate with success false if maxIter is reached', () => {
        // Only allow 1 iteration, which won't be enough to reach the default tolerances
        const result = brent(f, 0, 5, { maxIter: 1 });

        expect(result.success).toBe(false);
        expect(result.message).toContain('did not converge: reached maxIter (1)');
        expect(result.x).toBeGreaterThan(0);
        expect(result.x).toBeLessThan(5);
    });

    it('should converge early when |f(x)| is within tolF', () => {
        // A wide bracket but a loose function-value tolerance should let the
        // function-value stop criterion trigger before the x-tolerance would.
        const result = brent(f, 0, 5, { tolX: 1e-15, tolF: 1e-1 });
        expect(f(result.x)).toBeLessThanOrEqual(1e-1);
    });
});