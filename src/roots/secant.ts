import type { ScalarRootResult } from './types.js';
import { validateDistinctFinite, validateMaxIter, validateTolerance } from './validation.js';

const METHOD = 'secant';

/** Options for {@link secant}. */
export interface SecantOptions {
  /**
   * Absolute tolerance on `x`. The iteration stops when the distance between two successive
   * iterates (the step size) is below this threshold. Must be finite and non-negative.
   * @default 1e-8
   */
  tolx?: number;
  /**
   * Maximum number of iterations. Must be a finite, strictly positive integer.
   * @default 50
   */
  maxIter?: number;
}

/**
 * Finds a root of a scalar function `f(x)` using the secant method,
 * starting from two initial guesses.
 *
 * Unlike bisection, the secant method does not require a bracketing interval
 * (`f(x0)` and `f(x1)` need not have opposite signs), and typically converges
 * faster (superlinear, order `~1.618`) when it converges. It is not guaranteed
 * to converge for all inputs. The search stops with `success: false` if two
 * successive function values are equal (the secant is horizontal) or if `f`
 * returns a non-finite value.
 *
 * @param f Function to find a root of.
 * @param x0 First initial guess.
 * @param x1 Second initial guess (must differ from `x0`).
 * @param options Optional settings; see {@link SecantOptions}.
 * @returns Result containing success status, a message, the approximate root, function value, and evaluation count.
 * @throws {RangeError} If `x0` or `x1` is not finite, if `x0` and `x1` are equal, or if an
 * option is out of range.
 *
 * @example
 * ```ts
 * import { secant } from 'numerics-js/roots';
 *
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
 *   message: 'converged: step size below tolx',
 *   evaluations: 10,
 *   x: 0.5369737680962301,
 *   fx: -2.220446049250313e-16
 * }
 * ```
 *
 * @example
 * ```ts
 * import { secant } from 'numerics-js/roots';
 *
 * // Overriding a single option; unspecified options keep their defaults.
 * function f1(x: number): number {
 *     return 2 * x ** 3 + 4 * x ** 2 + x - 2;
 * }
 * const result = secant(f1, 0, 1, { tolx: 1e-6 });
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   method: 'secant',
 *   success: true,
 *   message: 'converged: step size below tolx',
 *   evaluations: 9,
 *   x: 0.5369737680249199,
 *   fx: -5.010138970362732e-10
 * }
 * ```
 */
export function secant(
  f: (x: number) => number,
  x0: number,
  x1: number,
  options: SecantOptions = {}
): ScalarRootResult {
  const { tolx = 1e-8, maxIter = 50 } = options;

  validateDistinctFinite(METHOD, 'x0', x0, 'x1', x1);
  validateTolerance(METHOD, 'tolx', tolx);
  validateMaxIter(METHOD, maxIter);

  let evaluations = 0;
  const result = (success: boolean, message: string, x: number, fx: number): ScalarRootResult => ({
    method: METHOD,
    success,
    message,
    evaluations,
    x,
    fx
  });

  let f0 = f(x0);
  evaluations++;
  let f1 = f(x1);
  evaluations++;

  if (f0 === 0) {
    return result(true, 'converged: exact root at x0', x0, f0);
  }
  if (f1 === 0) {
    return result(true, 'converged: exact root at x1', x1, f1);
  }
  if (!Number.isFinite(f0)) {
    return result(false, `did not converge: f(x0) is not finite (${f0})`, x0, f0);
  }
  if (!Number.isFinite(f1)) {
    return result(false, `did not converge: f(x1) is not finite (${f1})`, x1, f1);
  }

  for (let k = 0; k < maxIter; k++) {
    const denom = f1 - f0;

    if (denom === 0) {
      return result(
        false,
        `did not converge: zero denominator encountered (f(x0)=${f0}, f(x1)=${f1})`,
        x1,
        f1
      );
    }

    const x2 = x1 - (f1 * (x1 - x0)) / denom;

    if (Math.abs(x2 - x1) < tolx) {
      const fx2 = f(x2);
      evaluations++;
      return result(true, 'converged: step size below tolx', x2, fx2);
    }

    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = f(x1);
    evaluations++;

    if (f1 === 0) {
      return result(true, 'converged: exact root found', x1, f1);
    }
    if (!Number.isFinite(f1)) {
      return result(false, `did not converge: f is not finite (${f1}) at x=${x1}`, x1, f1);
    }
  }

  return result(
    false,
    `did not converge: reached maxIter (${maxIter}) without meeting tolx`,
    x1,
    f1
  );
}
