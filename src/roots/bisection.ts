import type { ScalarRootResult } from './types.js';
import { validateDistinctFinite, validateMaxIter, validateTolerance } from './validation.js';

const METHOD = 'bisection';

/** Options for {@link bisection}. */
export interface BisectionOptions {
  /**
   * Absolute tolerance on `x`. The iteration stops when the half-width of the bracketing
   * interval is below this threshold. Must be finite and non-negative.
   * @default 1e-8
   */
  tolx?: number;
  /**
   * Maximum number of iterations (bisection steps). Must be a finite, strictly positive integer.
   * @default 50
   */
  maxIter?: number;
}

/** Midpoint of `[xa, xb]` that cannot overflow for large finite endpoints. */
function stableMidpoint(xa: number, xb: number): number {
  return xa < 0 !== xb < 0 ? (xa + xb) / 2 : xa + (xb - xa) / 2;
}

/**
 * Finds a root of a scalar function `f(x)` using the bisection method,
 * given a bracketing interval.
 *
 * The bracket is halved at every iteration, so the interval half-width after `k` iterations is
 * `|xb - xa| / 2^k`, regardless of the shape of `f`. If `f` returns `NaN` at an endpoint or at a
 * midpoint, the search stops and the result has `success: false`.
 *
 * @param f Continuous function to find a root of.
 * @param xa Left endpoint of the bracketing interval (the endpoints are swapped if `xa > xb`).
 * @param xb Right endpoint of the bracketing interval.
 * @param options Optional settings; see {@link BisectionOptions}.
 * @returns Result containing success status, a message, the approximate root, function value, and evaluation count.
 * @throws {RangeError} If `xa` or `xb` is not finite, if `xa` and `xb` are equal, if `f(xa)` and
 * `f(xb)` don't bracket a root, or if an option is out of range.
 *
 * @example
 * ```ts
 * import { bisection } from 'numerics-js/roots';
 *
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
 *   message: 'converged: interval half-width below tolx',
 *   evaluations: 29,
 *   x: 0.5369737669825554,
 *   fx: -7.824495273922594e-9
 * }
 * ```
 *
 * @example
 * ```ts
 * import { bisection } from 'numerics-js/roots';
 *
 * // Overriding a single option; unspecified options keep their defaults.
 * function f1(x: number): number {
 *     return 2 * x ** 3 + 4 * x ** 2 + x - 2;
 * }
 * const result = bisection(f1, 0, 1, { tolx: 1e-6 });
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   method: 'bisection',
 *   success: true,
 *   message: 'converged: interval half-width below tolx',
 *   evaluations: 22,
 *   x: 0.5369729995727539,
 *   fx: -0.0000053995149571495205
 * }
 * ```
 */
export function bisection(
  f: (x: number) => number,
  xa: number,
  xb: number,
  options: BisectionOptions = {}
): ScalarRootResult {
  const { tolx = 1e-8, maxIter = 50 } = options;

  validateDistinctFinite(METHOD, 'xa', xa, 'xb', xb);
  validateTolerance(METHOD, 'tolx', tolx);
  validateMaxIter(METHOD, maxIter);

  if (xa > xb) {
    [xa, xb] = [xb, xa];
  }

  let evaluations = 0;
  const result = (success: boolean, message: string, x: number, fx: number): ScalarRootResult => ({
    method: METHOD,
    success,
    message,
    evaluations,
    x,
    fx
  });

  let fa = f(xa);
  evaluations++;
  const fb = f(xb);
  evaluations++;

  if (fa === 0) {
    return result(true, 'converged: exact root at xa', xa, fa);
  }
  if (fb === 0) {
    return result(true, 'converged: exact root at xb', xb, fb);
  }
  if (Number.isNaN(fa)) {
    return result(false, 'did not converge: f(xa) is NaN', xa, fa);
  }
  if (Number.isNaN(fb)) {
    return result(false, 'did not converge: f(xb) is NaN', xb, fb);
  }

  if (Math.sign(fa) === Math.sign(fb)) {
    throw new RangeError(
      `${METHOD}: f(xa) and f(xb) must have opposite signs (got f(xa)=${fa}, f(xb)=${fb})`
    );
  }

  let mid = xa;
  let fmid = fa;

  for (let k = 0; k < maxIter; k++) {
    mid = stableMidpoint(xa, xb);
    fmid = f(mid);
    evaluations++;

    if (fmid === 0) {
      return result(true, 'converged: exact root found', mid, fmid);
    }
    if (Number.isNaN(fmid)) {
      return result(false, `did not converge: f is NaN at x=${mid}`, mid, fmid);
    }
    if ((xb - xa) / 2 < tolx) {
      return result(true, 'converged: interval half-width below tolx', mid, fmid);
    }

    if (Math.sign(fmid) === Math.sign(fa)) {
      xa = mid;
      fa = fmid;
    } else {
      xb = mid;
    }
  }

  return result(
    false,
    `did not converge: reached maxIter (${maxIter}) without meeting tolx`,
    mid,
    fmid
  );
}
