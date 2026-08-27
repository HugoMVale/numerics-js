import type { MathFunction } from '../types.js';

/**
 * Finds a root of fn using the secant method, starting from two initial guesses.
 *
 * Unlike bisection, the secant method does not require a bracketing interval
 * (fn(x0) and fn(x1) need not have opposite signs), and typically converges
 * faster (superlinear, order ~1.618) when it converges. It is not guaranteed
 * to converge for all inputs.
 *
 * @param fn - Function to find a root of.
 * @param x0 - First initial guess.
 * @param x1 - Second initial guess (should differ from x0).
 * @param tolX - Stop when |x1 - x0| (the step size) is below this.
 * @param maxIterations - Maximum number of iterations.
 * @returns Approximate root.
 * @throws {Error} If x0 and x1 are equal, or if a zero derivative estimate is encountered.
 *
 * @example
 * ```ts
 * // Test for scalar root-finding methods.
 * function f1(x: number): number {
 *     return 2 * x ** 3 + 4 * x ** 2 + x - 2;
 * }
 * const root = secant(f1, 0, 1);
 * console.log(`x = ${root.toFixed(3)}`);
 * ```
 *
 * Output:
 * ```text
 * x = 0.557
 * ```
 */
export function secant(
    fn: MathFunction,
    x0: number,
    x1: number,
    tolX: number = 1e-8,
    maxIterations: number = 100
): number {
    if (x0 === x1) {
        throw new Error('secant: x0 and x1 must be different');
    }

    let f0 = fn(x0);
    let f1 = fn(x1);

    if (f0 === 0) return x0;
    if (f1 === 0) return x1;

    for (let k = 0; k < maxIterations; k++) {
        const denom = f1 - f0;

        if (denom === 0) {
            throw new Error(
                `secant: zero denominator encountered (fn(x0)=${f0}, fn(x1)=${f1}); cannot continue`
            );
        }

        const x2 = x1 - (f1 * (x1 - x0)) / denom;

        if (Math.abs(x2 - x1) < tolX) {
            return x2;
        }

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = fn(x1);

        if (f1 === 0) {
            return x1;
        }
    }

    console.warn(
        `secant: reached maxIterations (${maxIterations}) without converging to tolerance ${tolX}`
    );
    return x1;
}