import { copysign } from '../math.js';
import type { ScalarRootResult } from './types.js';
import { validateDistinctFinite, validateMaxIter, validateTolerance } from './validation.js';

const METHOD = 'brent';

/** Options for {@link brent}. */
export interface BrentOptions {
  /**
   * Absolute tolerance on `x`. The iteration stops when the width of the bracketing interval
   * is below this threshold (plus a small relative term, `4*eps*|x|`, that accounts for the
   * floating-point resolution). Must be finite and non-negative.
   * @default 1e-8
   */
  tolx?: number;
  /**
   * Absolute tolerance on the function value. The iteration stops as soon as `|f(x)|` is
   * at or below this threshold. Use `0` to disable this criterion and rely on `tolx` alone.
   * Must be finite and non-negative.
   * @default 1e-8
   */
  tolf?: number;
  /**
   * Maximum number of iterations. Must be a finite, strictly positive integer.
   * @default 50
   */
  maxIter?: number;
  /**
   * Known value of `f(xa)`. If given, `f(xa)` is not evaluated, which saves a function
   * evaluation when the caller already has it. It must be consistent with `f`.
   */
  fa?: number;
  /** Known value of `f(xb)`; same meaning as `fa`. */
  fb?: number;
}

/**
 * Finds a root of a scalar function `f(x)` using Brent's method,
 * given a bracketing interval.
 *
 * Brent's method combines bisection, the secant method, and inverse
 * quadratic interpolation. It is as robust as bisection (guaranteed to
 * converge, given a valid bracket) and typically converges as fast as
 * the secant method.
 *
 * Unlike the secant method, Brent's method requires `f(xa)` and `f(xb)` to
 * have opposite signs (i.e. the root must be bracketed by `[xa, xb]`), unless one of them is
 * already within `tolf` of zero. If `f` returns `NaN` at an endpoint or at a trial point, the
 * search stops and the result has `success: false`.
 *
 * @param f Function to find a root of.
 * @param xa One end of the bracketing interval.
 * @param xb Other end of the bracketing interval.
 * @param options Optional settings; see {@link BrentOptions}.
 * @returns Result containing success status, a message, the approximate root, function value, and
 * evaluation count. The count includes only calls to `f` actually made, i.e. it excludes the
 * endpoint values supplied through `options.fa` and `options.fb`.
 * @throws {RangeError} If `xa` or `xb` is not finite, if `xa` and `xb` are equal, if `f(xa)` and
 * `f(xb)` do not have opposite signs (and neither is within `tolf` of zero), or if an option is
 * out of range.
 *
 * @example
 * ```ts
 * import { brent } from 'numerics-js/roots';
 *
 * // Test for scalar root-finding methods.
 * function f1(x: number): number {
 *     return 2 * x ** 3 + 4 * x ** 2 + x - 2;
 * }
 * const result = brent(f1, 0, 1);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   method: 'brent',
 *   success: true,
 *   message: 'converged: |f(x)| below tolf',
 *   evaluations: 9,
 *   x: 0.5369737681040232,
 *   fx: 5.475264686083392e-11
 * }
 * ```
 *
 * @example
 * ```ts
 * import { brent } from 'numerics-js/roots';
 *
 * // Overriding a single option; unspecified options keep their defaults.
 * function f1(x: number): number {
 *     return 2 * x ** 3 + 4 * x ** 2 + x - 2;
 * }
 * const result = brent(f1, 0, 1, { tolx: 1e-15, tolf: 1e-12 });
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   method: 'brent',
 *   success: true,
 *   message: 'converged: |f(x)| below tolf',
 *   evaluations: 10,
 *   x: 0.5369737680962301,
 *   fx: -2.220446049250313e-16
 * }
 * ```
 */
export function brent(
  f: (x: number) => number,
  xa: number,
  xb: number,
  options: BrentOptions = {}
): ScalarRootResult {
  const { tolx = 1e-8, tolf = 1e-8, maxIter = 50, fa: faKnown, fb: fbKnown } = options;

  validateDistinctFinite(METHOD, 'xa', xa, 'xb', xb);
  validateTolerance(METHOD, 'tolx', tolx);
  validateTolerance(METHOD, 'tolf', tolf);
  validateMaxIter(METHOD, maxIter);

  const eps = Number.EPSILON;

  let evaluations = 0;
  const result = (success: boolean, message: string, x: number, fx: number): ScalarRootResult => ({
    method: METHOD,
    success,
    message,
    evaluations,
    x,
    fx
  });

  let fa: number;
  if (faKnown === undefined) {
    fa = f(xa);
    evaluations++;
  } else {
    fa = faKnown;
  }
  if (Math.abs(fa) <= tolf) {
    return result(true, 'converged: |f(xa)| below tolf', xa, fa);
  }

  let fb: number;
  if (fbKnown === undefined) {
    fb = f(xb);
    evaluations++;
  } else {
    fb = fbKnown;
  }
  if (Math.abs(fb) <= tolf) {
    return result(true, 'converged: |f(xb)| below tolf', xb, fb);
  }

  if (Number.isNaN(fa)) {
    return result(false, 'did not converge: f(xa) is NaN', xa, fa);
  }
  if (Number.isNaN(fb)) {
    return result(false, 'did not converge: f(xb) is NaN', xb, fb);
  }

  if (Math.sign(fa) === Math.sign(fb)) {
    throw new RangeError(
      `${METHOD}: root is not bracketed (f(xa) and f(xb) must have opposite signs)`
    );
  }

  let xc = xb;
  let fc = fb;
  let d = xb - xa;
  let e = d;

  for (let k = 0; k < maxIter; k++) {
    if (Math.sign(fb) === Math.sign(fc)) {
      xc = xa;
      fc = fa;
      d = xb - xa;
      e = d;
    }

    if (Math.abs(fc) < Math.abs(fb)) {
      xa = xb;
      fa = fb;
      xb = xc;
      fb = fc;
      xc = xa;
      fc = fa;
    }

    const tol1 = 2 * eps * Math.abs(xb) + 0.5 * tolx;
    const m = 0.5 * (xc - xb);

    if (Math.abs(fb) <= tolf) {
      return result(true, 'converged: |f(x)| below tolf', xb, fb);
    }

    if (Math.abs(m) <= tol1) {
      return result(true, 'converged: bracket width below tolx', xb, fb);
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

    fb = f(xb);
    evaluations++;

    if (Number.isNaN(fb)) {
      return result(false, `did not converge: f is NaN at x=${xb}`, xb, fb);
    }
  }

  return result(
    false,
    `did not converge: reached maxIter (${maxIter}) without meeting tolx/tolf`,
    xb,
    fb
  );
}
