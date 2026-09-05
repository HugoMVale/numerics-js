import type { ScalarFunction } from '../types.js';
import type { RootResult } from './types.js';

/**
 * Finds a root of fn using the secant method, starting from two initial guesses.
 *
 * Unlike bisection, the secant method does not require a bracketing interval
 * (fn(x0) and fn(x1) need not have opposite signs), and typically converges
 * faster (superlinear, order ~1.618) when it converges. It is not guaranteed
 * to converge for all inputs.
 *
 * @param fn Function to find a root of.
 * @param x0 First initial guess.
 * @param x1 Second initial guess (should differ from x0).
 * @param options Tuning options.
 * @param options.tolX Stop when |x1 - x0| (the step size) is below this. Defaults to `1e-8`.
 * @param options.maxIter Maximum number of iterations. Defaults to `50`.
 * @returns Result containing success status, a message, the approximate root, function value, and evaluation count.
 * @throws {Error} If x0 and x1 are equal.
 *
 * @example
 * ```ts
 * // Test for scalar root-finding methods.
 * function f1(x: number): number {
 *     return 2 * x ** 3 + 4 * x ** 2 + x - 2;
 * }
 * const result = secant(f1, 0, 1);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   method: 'secant',
 *   success: true,
 *   message: 'converged: step size below tolX',
 *   evaluations: 10,
 *   x: 0.5369737680962301,
 *   fx: -2.220446049250313e-16
 * }
 * ```
 *
 * @example
 * ```ts
 * // Overriding a single option; unspecified options keep their defaults.
 * const result = secant(f1, 0, 1, { maxIter: 100 });
 * ```
 */
export function secant(
    fn: ScalarFunction,
    x0: number,
    x1: number,
    options: {
        tolX?: number;
        maxIter?: number;
    } = {}
): RootResult {
    const { tolX = 1e-8, maxIter = 50 } = options;

    if (x0 === x1) {
        throw new Error('secant: x0 and x1 must be different');
    }

    let evaluations = 0;
    let f0 = fn(x0);
    evaluations++;
    let f1 = fn(x1);
    evaluations++;

    if (f0 === 0) {
        return {
            method: 'secant',
            success: true,
            message: 'converged: exact root at x0',
            evaluations,
            x: x0,
            fx: f0
        };
    }
    if (f1 === 0) {
        return {
            method: 'secant',
            success: true,
            message: 'converged: exact root at x1',
            evaluations,
            x: x1,
            fx: f1
        };
    }

    for (let k = 0; k < maxIter; k++) {
        const denom = f1 - f0;

        if (denom === 0) {
            return {
                method: 'secant',
                success: false,
                message: `did not converge: zero denominator encountered (fn(x0)=${f0}, fn(x1)=${f1})`,
                evaluations,
                x: x1,
                fx: f1
            };
        }

        const x2 = x1 - (f1 * (x1 - x0)) / denom;

        if (Math.abs(x2 - x1) < tolX) {
            const fx2 = fn(x2);
            evaluations++;
            return {
                method: 'secant',
                success: true,
                message: 'converged: step size below tolX',
                evaluations,
                x: x2,
                fx: fx2
            };
        }

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = fn(x1);
        evaluations++;

        if (f1 === 0) {
            return {
                method: 'secant',
                success: true,
                message: 'converged: exact root found',
                evaluations,
                x: x1,
                fx: f1
            };
        }
    }

    return {
        method: 'secant',
        success: false,
        message: `did not converge: reached maxIter (${maxIter}) without meeting tolX`,
        evaluations,
        x: x1,
        fx: f1
    };
}