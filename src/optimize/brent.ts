import type { ScalarFunction } from '../types.js';
import { copysign } from '../misc.js';

/**
 * Result of a `minimizeBrent` optimize.
 */
export interface BrentResult {
    /** Name of the optimize method used. */
    method: string;
    /** Whether the optimize terminated successfully. */
    success: boolean;
    /** Human-readable description of the termination reason. */
    message: string;
    /** Number of function evaluations performed. */
    nFev: number;
    /** Number of iterations performed. */
    nIter: number;
    /** Location of the minimum. */
    x: number;
    /** Function value at `x`. */
    f: number;
}

/**
 * Finds a local minimum of fn using Brent's method, given a bracketing interval.
 *
 * Brent's method is a derivative-free optimize algorithm that combines
 * golden-section search with inverse parabolic interpolation. It maintains a
 * bracketing interval known to contain a local minimum and iteratively
 * narrows it: when the function behaves smoothly it attempts a fast
 * parabolic step, and otherwise falls back to the more robust golden-section
 * step.
 *
 * @param fn - Function to minimize.
 * @param xa - One end of the bracketing interval.
 * @param xb - Other end of the bracketing interval.
 * @param tolX - Stop when the search interval shrinks to approximately this width.
 * @param maxIter - Maximum number of iterations.
 * @returns Optimization result containing the minimum, function value, and convergence information.
 *
 * @example
 * ```ts
 * const f = (x: number): number => x ** 4 - x + 1;
 * const result = minimizeBrent(f, -3.0, 3.0);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   method: 'Brent',
 *   success: true,
 *   message: '|dx| <= tolX',
 *   nFev: 19,
 *   nIter: 18,
 *   x: 0.62996052,
 *   f: 0.52752961
 * }
 * ```
 */
export function minimizeBrent(
    fn: ScalarFunction,
    xa: number,
    xb: number,
    tolX: number = 1e-8,
    maxIter: number = 100
): BrentResult {
    // Golden ratio constant: (3 - sqrt(5)) / 2
    const GOLDEN = 0.38196601125010515179;
    const EPS = Number.EPSILON;

    let a = Math.min(xa, xb);
    let b = Math.max(xa, xb);

    let x = a + GOLDEN * (b - a);
    let v = x;
    let w = x;

    let fx = fn(x);
    let fv = fx;
    let fw = fx;
    let nFev = 1;

    let d = 0;
    let e = 0;

    for (let k = 0; k < maxIter; k++) {
        const xm = 0.5 * (a + b);
        const tol1 = EPS * Math.abs(x) + tolX / 3.0;
        const tol2 = 2.0 * tol1;

        if (Math.abs(x - xm) <= tol2 - 0.5 * (b - a)) {
            return {
                method: 'Brent',
                success: true,
                message: '|dx| <= tolX',
                nFev,
                nIter: k,
                x,
                f: fx,
            };
        }

        let p = 0;
        let q = 0;
        let r = 0;
        let parabolic = false;

        if (Math.abs(e) > tol1) {
            // Fit parabola through (v, fv), (w, fw), (x, fx)
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) {
                p = -p;
            }
            q = Math.abs(q);
            r = e;
            e = d;

            // Is the parabolic step acceptable?
            if (Math.abs(p) < Math.abs(0.5 * q * r) && p > q * (a - x) && p < q * (b - x)) {
                d = p / q;
                const u = x + d;
                // Convergence check for u
                if (u - a < tol2 || b - u < tol2) {
                    d = copysign(tol1, xm - x);
                }
                parabolic = true;
            }
        }

        if (!parabolic) {
            // Golden-section step
            e = x < xm ? b - x : a - x;
            d = GOLDEN * e;
        }

        // Numerical safety: ensure step is at least tol1
        const u = Math.abs(d) >= tol1 ? x + d : x + copysign(tol1, d);
        const fu = fn(u);
        nFev++;

        // Update points
        if (fu <= fx) {
            if (u >= x) {
                a = x;
            } else {
                b = x;
            }
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) {
                a = u;
            } else {
                b = u;
            }
            if (fu <= fw || w === x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v === x || v === w) {
                v = u; fv = fu;
            }
        }
    }

    console.warn(
        `minimizeBrent: reached maxIter (${maxIter}) without converging to tolerance ${tolX}`
    );
    return {
        method: 'Brent',
        success: false,
        message: `Maximum number of iterations (${maxIter}) reached.`,
        nFev,
        nIter: maxIter,
        x,
        f: fx,
    };
}