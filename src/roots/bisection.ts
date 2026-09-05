import type { ScalarFunction } from '../types.js';
import type { BisectionOptions, RootResult } from './types.js';

/**
 * Finds a root of fn in the interval [a, b] using the bisection method.
 *
 * @param fn Continuous function to find a root of.
 * @param a Left endpoint of the bracketing interval.
 * @param b Right endpoint of the bracketing interval.
 * @param options Tuning options.
 * @param options.tolX Stop when the interval half-width is below this. Defaults to `1e-8`.
 * @param options.maxIter Maximum number of iterations. Defaults to `50`.
 * @returns Result containing success status, a message, the approximate root, function value, and evaluation count.
 * @throws {Error} If a and b are equal, or if fn(a) and fn(b) don't bracket a root.
 *
 * @example
 * ```ts
 * // Test for scalar root-finding methods.
 * function f1(x: number): number {
 *     return 2 * x ** 3 + 4 * x ** 2 + x - 2;
 * }
 * const result = bisection(f1, 0, 1);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   method: 'bisection',
 *   success: true,
 *   message: 'converged: interval half-width below tolX',
 *   evaluations: 29,
 *   x: 0.5369737669825554,
 *   fx: -7.824495273922594e-9
 * }
 * ```
 *
 * @example
 * ```ts
 * // Overriding a single option; unspecified options keep their defaults.
 * const result = bisection(f1, 0, 1, { maxIter: 100 });
 * ```
 */
export function bisection(
    fn: ScalarFunction,
    a: number,
    b: number,
    options: BisectionOptions = {}
): RootResult {
    const { tolX = 1e-8, maxIter = 50 } = options;

    if (a === b) {
        throw new Error('bisection: a and b must be different');
    }
    if (a > b) {
        [a, b] = [b, a];
    }

    let evaluations = 0;
    let fa = fn(a);
    evaluations++;
    const fb = fn(b);
    evaluations++;

    if (fa === 0) {
        return {
            method: 'bisection',
            success: true,
            message: 'converged: exact root at a',
            evaluations,
            x: a,
            fx: fa
        };
    }
    if (fb === 0) {
        return {
            method: 'bisection',
            success: true,
            message: 'converged: exact root at b',
            evaluations,
            x: b,
            fx: fb
        };
    }

    if (Math.sign(fa) === Math.sign(fb)) {
        throw new Error(
            `bisection: fn(a) and fn(b) must have opposite signs (got fn(a)=${fa}, fn(b)=${fb})`
        );
    }

    let mid = (a + b) / 2;
    let fmid = fa;

    for (let k = 0; k < maxIter; k++) {
        mid = (a + b) / 2;
        fmid = fn(mid);
        evaluations++;

        if (fmid === 0) {
            return {
                method: 'bisection',
                success: true,
                message: 'converged: exact root found',
                evaluations,
                x: mid,
                fx: fmid
            };
        }
        if ((b - a) / 2 < tolX) {
            return {
                method: 'bisection',
                success: true,
                message: 'converged: interval half-width below tolX',
                evaluations,
                x: mid,
                fx: fmid
            };
        }

        if (Math.sign(fmid) === Math.sign(fa)) {
            a = mid;
            fa = fmid;
        } else {
            b = mid;
        }
    }

    return {
        method: 'bisection',
        success: false,
        message: `did not converge: reached maxIter (${maxIter}) without meeting tolX`,
        evaluations,
        x: mid,
        fx: fmid
    };
}