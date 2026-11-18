import {
  CONVERGED_MESSAGE,
  countedIntegrand,
  emptyIntervalResult,
  ENDPOINT_SINGULARITY_MESSAGE,
  midpoint,
  NonFiniteIntegrandError,
  resolveGaussKronrodOptions,
  subintervalLimitMessage,
  validateIntegrand
} from './common.js';
import { gaussKronrod15, type LocalGaussKronrodResult } from './kronrod.js';
import type { GaussKronrodOptions, GaussKronrodResult } from './types.js';

/** A subinterval `[a, b]` of the adaptive partition, with its local G7-K15 results. */
interface Subinterval {
  a: number;
  b: number;
  kronrod: number;
  error: number;
}

/**
 * Plain adaptive G7-K15 engine of {@link gaussKronrod}, without extrapolation ({@link quad} has
 * its own engine, see `extrapolatedIntegral`).
 *
 * At each step the subinterval with the largest error estimate is bisected, until
 * `error <= atol + rtol * |value|` for the whole integral or the subinterval limit is reached.
 *
 * Bisecting toward an integrable endpoint singularity (e.g. `x^-0.9` at `x = 0`) keeps splitting
 * the subinterval that touches it. After about 45 levels the outermost nodes round onto the
 * endpoint, where `f` is not finite. If that is where the integrand fails, i.e. the non-finite
 * abscissa coincides with an endpoint of the subinterval being split, the engine stops with
 * `success = false` and the estimates found so far instead of throwing. A non-finite value at any
 * other abscissa is a genuinely bad integrand and still throws. Integrands that are finite at
 * the rounded-onto endpoints are not affected: they simply continue.
 *
 * The totals are re-summed from the subintervals at every step instead of being updated
 * incrementally (`total += left + right - parent`). The incremental update accumulates
 * rounding errors that are relatively large once the error estimate has shrunk by many
 * orders of magnitude: the estimate can even become negative and stop the loop too early.
 * The extra `O(n)` sum is free, since selecting the worst subinterval is already `O(n)`.
 *
 * @param f Integrand.
 * @param a Lower limit of integration, finite and smaller than `b`.
 * @param b Upper limit of integration, finite.
 * @param options Validated options, see `resolveGaussKronrodOptions`.
 * @returns Integral over `[a, b]` (ascending orientation).
 */
function integrateAdaptive(
  f: (x: number) => number,
  a: number,
  b: number,
  options: Required<GaussKronrodOptions>
): GaussKronrodResult {
  const { atol, rtol, maxSubintervals } = options;

  const integrand = countedIntegrand('gaussKronrod', f);
  const evalF = integrand.evaluate;

  const first = gaussKronrod15(evalF, a, b);
  const subintervals: Subinterval[] = [{ a, b, kronrod: first.kronrod, error: first.error }];

  for (;;) {
    let value = 0;
    let error = 0;
    let worstIdx = 0;
    for (let i = 0; i < subintervals.length; i++) {
      value += subintervals[i].kronrod;
      error += subintervals[i].error;
      if (subintervals[i].error > subintervals[worstIdx].error) worstIdx = i;
    }

    const success = error <= atol + rtol * Math.abs(value);
    if (success || subintervals.length >= maxSubintervals) {
      return {
        success,
        message: success ? CONVERGED_MESSAGE : subintervalLimitMessage(maxSubintervals),
        evaluations: integrand.evaluations,
        subintervals: subintervals.length,
        value,
        error
      };
    }

    const worst = subintervals[worstIdx];
    const mid = midpoint(worst.a, worst.b);
    let left: LocalGaussKronrodResult;
    let right: LocalGaussKronrodResult;
    try {
      left = gaussKronrod15(evalF, worst.a, mid);
      right = gaussKronrod15(evalF, mid, worst.b);
    } catch (e) {
      const atEndpoint =
        e instanceof NonFiniteIntegrandError &&
        (e.point === worst.a || e.point === mid || e.point === worst.b);
      if (!atEndpoint) throw e;
      return {
        success: false,
        message: ENDPOINT_SINGULARITY_MESSAGE,
        evaluations: integrand.evaluations,
        subintervals: subintervals.length,
        value,
        error
      };
    }

    subintervals[worstIdx] = { a: worst.a, b: mid, kronrod: left.kronrod, error: left.error };
    subintervals.push({ a: mid, b: worst.b, kronrod: right.kronrod, error: right.error });
  }
}

/**
 * Numerically integrate `f` over `[a, b]` using adaptive Gauss-Kronrod
 * (G7-K15) quadrature.
 *
 * This function is suitable for numerically integrating smooth functions over finite
 * intervals. For functions with discontinuities or sharp features, or infinite domains,
 * use {@link quad} instead.
 *
 * The interval is bisected adaptively until the estimated error of the whole integral
 * satisfies `error <= atol + rtol * |value|`. The value is the sum of the Kronrod estimates
 * over the subintervals, and the error the sum of their local estimates `|kronrod - gauss|`.
 * There is no extrapolation: for integrands with singularities, kinks or jumps, the error
 * estimate can be too small (it can claim success while missing the tolerance) and many
 * subintervals are needed; use {@link quad}, which extrapolates, for those.
 *
 * If `a > b`, the integral over `[b, a]` is computed and its sign reversed.
 *
 * @param f Scalar function `f` to integrate.
 * @param a Lower bound of integration `a`.
 * @param b Upper bound of integration `b`.
 * @param options Optional settings, see {@link GaussKronrodOptions}.
 * @returns Quadrature output including final value, estimated error, and diagnostics.
 * @throws {TypeError} If `f` is not a function.
 * @throws {RangeError} If `a` or `b` are non-finite numbers, `atol` or `rtol` is not a
 * finite non-negative number, or `maxSubintervals` is not a finite positive integer.
 * These checks are made also when `a === b`.
 * @throws {Error} If `f` evaluates to a non-finite value, except when bisection reaches
 * floating-point resolution at an endpoint where `f` is not finite: then `success` is `false`.
 *
 * @example
 * ```ts
 * import { gaussKronrod } from 'numerics-js/integrate';
 *
 * // Integrate an oscillatory function that requires adaptive subdivision.
 * const result = gaussKronrod(x => Math.sin(100 * x), 0, 1);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   message: 'Estimated error is at most `atol + rtol*|value|`.',
 *   evaluations: 465,
 *   subintervals: 16,
 *   value: 0.001376811277122893,
 *   error: 2.005802135987561e-9
 * }
 * ```
 *
 * @example
 * ```ts
 * // A purely relative tolerance is invariant under rescaling of the integrand:
 * // the same subintervals are used whether the integral is of order 1e-12 or 1e12.
 * const small = gaussKronrod(x => 1e-12 * Math.exp(Math.sin(30 * x)), 0, 1, { atol: 0, rtol: 1e-6 });
 * const large = gaussKronrod(x => 1e12 * Math.exp(Math.sin(30 * x)), 0, 1, { atol: 0, rtol: 1e-6 });
 * console.log(small.subintervals, large.subintervals);
 * ```
 *
 * Output:
 * ```text
 * 10 10
 * ```
 */
export function gaussKronrod(
  f: (x: number) => number,
  a: number,
  b: number,
  options: GaussKronrodOptions = {}
): GaussKronrodResult {
  // All arguments are validated up front, so that invalid input is reported even when
  // the interval is empty.
  validateIntegrand('gaussKronrod', f);
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    throw new RangeError(`gaussKronrod: a and b must be finite numbers, got ${a} and ${b}`);
  }
  const resolved = resolveGaussKronrodOptions('gaussKronrod', options);

  if (a === b) {
    return emptyIntervalResult();
  }

  const sign = a > b ? -1 : 1;
  const result = integrateAdaptive(f, Math.min(a, b), Math.max(a, b), resolved);
  result.value *= sign;
  return result;
}
