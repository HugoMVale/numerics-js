import type { ScalarRootResult } from './types.js';

const METHOD = 'bisection';

/**
 * Finds a root of a scalar function `f(x)` using the bisection method,
 * given a bracketing interval.
 *
 * @param f Continuous function to find a root of.
 * @param xa Left endpoint of the bracketing interval.
 * @param xb Right endpoint of the bracketing interval.
 * @param options Optional settings.
 * @param options.tolX Stop when the interval half-width is below this. Defaults to `1e-8`.
 * @param options.maxIter Maximum number of iterations. Defaults to `50`.
 * @returns Result containing success status, a message, the approximate root, function value, and evaluation count.
 * @throws {Error} If `xa` and `xb` are equal, or if `f(xa)` and `f(xb)` don't bracket a root.
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
    f: (x: number) => number,
    xa: number,
    xb: number,
    options: {
        tolX?: number;
        maxIter?: number;
    } = {}
): ScalarRootResult {
    const { tolX = 1e-8, maxIter = 50 } = options;

    if (xa === xb) {
        throw new Error(`${METHOD}: xa and xb must be different`);
    }
    if (xa > xb) {
        [xa, xb] = [xb, xa];
    }

    let evaluations = 0;
    let fa = f(xa);
    evaluations++;
    const fb = f(xb);
    evaluations++;

    if (fa === 0) {
        return {
            method: METHOD,
            success: true,
            message: 'converged: exact root at xa',
            evaluations,
            x: xa,
            fx: fa
        };
    }
    if (fb === 0) {
        return {
            method: METHOD,
            success: true,
            message: 'converged: exact root at xb',
            evaluations,
            x: xb,
            fx: fb
        };
    }

    if (Math.sign(fa) === Math.sign(fb)) {
        throw new Error(
            `${METHOD}: f(xa) and f(xb) must have opposite signs (got f(xa)=${fa}, f(xb)=${fb})`
        );
    }

    let mid = (xa + xb) / 2;
    let fmid = fa;

    for (let k = 0; k < maxIter; k++) {
        mid = (xa + xb) / 2;
        fmid = f(mid);
        evaluations++;

        if (fmid === 0) {
            return {
                method: METHOD,
                success: true,
                message: 'converged: exact root found',
                evaluations,
                x: mid,
                fx: fmid
            };
        }
        if ((xb - xa) / 2 < tolX) {
            return {
                method: METHOD,
                success: true,
                message: 'converged: interval half-width below tolX',
                evaluations,
                x: mid,
                fx: fmid
            };
        }

        if (Math.sign(fmid) === Math.sign(fa)) {
            xa = mid;
            fa = fmid;
        } else {
            xb = mid;
        }
    }

    return {
        method: METHOD,
        success: false,
        message: `did not converge: reached maxIter (${maxIter}) without meeting tolX`,
        evaluations,
        x: mid,
        fx: fmid
    };
}