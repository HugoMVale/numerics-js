import type { ScalarFunction } from '../types';
import { copysign } from '../misc';

/**
 * Finds a root of fn using Brent's method, given a bracketing interval.
 *
 * Brent's method combines bisection, the secant method, and inverse
 * quadratic interpolation. It is as robust as bisection (guaranteed to
 * converge, given a valid bracket) and typically converges as fast as
 * the secant method.
 *
 * Unlike the secant method, Brent's method requires fn(xa) and fn(xb) to
 * have opposite signs (i.e. the root must be bracketed by [xa, xb]).
 *
 * @param fn - Function to find a root of.
 * @param xa - One end of the bracketing interval.
 * @param xb - Other end of the bracketing interval.
 * @param tolX - Stop when the bracket half-width is below this (absolute x tolerance).
 * @param tolF - Stop when |fn(x)| is below this (absolute function-value tolerance).
 * @param maxIterations - Maximum number of iterations.
 * @returns Approximate root.
 * @throws {Error} If fn(xa) and fn(xb) do not have opposite signs.
 *
 * @example
 * ```ts
 * // Test for scalar root-finding methods.
 * function f1(x: number): number {
 *     return 2 * x ** 3 + 4 * x ** 2 + x - 2;
 * }
 * const root = brent(f1, 0, 1);
 * console.log(`x = ${root.toFixed(3)}`);
 * ```
 *
 * Output:
 * ```text
 * x = 0.557
 * ```
 */
export function brent(
    fn: ScalarFunction,
    xa: number,
    xb: number,
    tolX: number = 1e-8,
    tolF: number = 1e-8,
    maxIterations: number = 100
): number {
    const eps = Number.EPSILON;

    let fa = fn(xa);
    if (Math.abs(fa) <= tolF) return xa;

    let fb = fn(xb);
    if (Math.abs(fb) <= tolF) return xb;

    if (fa * fb > 0) {
        throw new Error(
            'brent: root is not bracketed (fn(xa) and fn(xb) must have opposite signs)'
        );
    }

    let xc = xb;
    let fc = fb;
    let d = xb - xa;
    let e = d;

    for (let k = 0; k < maxIterations; k++) {
        if (fb * fc > 0) {
            xc = xa;
            fc = fa;
            d = xb - xa;
            e = d;
        }

        if (Math.abs(fc) < Math.abs(fb)) {
            xa = xb; fa = fb;
            xb = xc; fb = fc;
            xc = xa; fc = fa;
        }

        const tol1 = 2 * eps * Math.abs(xb) + 0.5 * tolX;
        const m = 0.5 * (xc - xb);

        if (Math.abs(fb) <= tolF) {
            return xb;
        }

        if (Math.abs(m) <= tol1) {
            return xb;
        }

        if (Math.abs(e) >= tol1 && Math.abs(fa) > Math.abs(fb)) {
            const s = fb / fa;
            let p: number;
            let q: number;

            if (xa === xc) {
                p = 2 * m * s;
                q = 1 - s;
            } else {
                const q1 = fa / fc;
                const r = fb / fc;
                p = s * (2 * m * q1 * (q1 - r) - (xb - xa) * (r - 1));
                q = (q1 - 1) * (r - 1) * (s - 1);
            }

            if (p > 0) {
                q = -q;
            }
            p = Math.abs(p);

            const min1 = 3 * m * q - Math.abs(tol1 * q);
            const min2 = Math.abs(e * q);

            if (2 * p < Math.min(min1, min2)) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = d;
            }
        } else {
            d = m;
            e = d;
        }

        xa = xb;
        fa = fb;

        if (Math.abs(d) > tol1) {
            xb += d;
        } else {
            xb += copysign(tol1, m);
        }

        fb = fn(xb);
    }

    console.warn(
        `brent: reached maxIterations (${maxIterations}) without converging to tolX=${tolX}, tolF=${tolF}`
    );
    return xb;
}