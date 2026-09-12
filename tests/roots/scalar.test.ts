import { describe, it, expect, vi, afterEach } from 'vitest';
import { bisection } from '../../src/roots/bisection.js';
import { secant } from '../../src/roots/secant.js';
import { brent } from '../../src/roots/brent.js';

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

    it('should swap bounds if xa > xb', () => {
        const result = bisection(f, 5, 0);
        expect(result.x).toBeCloseTo(2, 5);
    });

    it('should return exactly xa if f(xa) is 0', () => {
        const result = bisection(f, 2, 5);
        expect(result.x).toBe(2);
        expect(result.fx).toBe(0);
        expect(result.evaluations).toBe(2);
    });

    it('should return exactly xb if f(xb) is 0', () => {
        const result = bisection(f, 0, 2);
        expect(result.x).toBe(2);
        expect(result.fx).toBe(0);
        expect(result.evaluations).toBe(2);
    });

    it('should return the midpoint when it is an exact root', () => {
        const result = bisection(f, 0, 4);
        expect(result.x).toBe(2);
        expect(result.fx).toBe(0);
        expect(result.message).toBe('converged: exact root found');
        expect(result.evaluations).toBe(3);
    });

    it('should throw an error if xa and xb are the same', () => {
        expect(() => bisection(f, 1, 1)).toThrow('bisection: xa and xb must be different');
    });

    it('should throw an error if f(xa) and f(xb) have the same sign', () => {
        expect(() => bisection(f, 3, 5)).toThrow('bisection: f(xa) and f(xb) must have opposite signs');
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

    it('should return the updated estimate when it is an exact root', () => {
        const result = secant((x) => x - 2, 0, 5);
        expect(result.x).toBe(2);
        expect(result.fx).toBe(0);
        expect(result.message).toBe('converged: exact root found');
        expect(result.evaluations).toBe(3);
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

    it('should use inverse quadratic interpolation for a nonlinear root', () => {
        const result = brent((x) => 2 * x ** 3 + 4 * x ** 2 + x - 2, 0, 1);
        expect(result.success).toBe(true);
        expect(result.x).toBeCloseTo(0.5369737681, 8);
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

    it('should converge early when |f(x)| is within tolf', () => {
        // A wide bracket but a loose function-value tolerance should let the
        // function-value stop criterion trigger before the x-tolerance would.
        const result = brent(f, 0, 5, { tolx: 1e-15, tolf: 1e-1 });
        expect(f(result.x)).toBeLessThanOrEqual(1e-1);
    });

    it('should converge by bracket width when function-value tolerance is disabled', () => {
        const result = brent((x) => x * x - 2, 0, 2, { tolf: 0 });
        expect(result.success).toBe(true);
        expect(result.message).toBe('converged: bracket half-width below tolx');
        expect(result.x).toBeCloseTo(Math.SQRT2, 8);
    });
});