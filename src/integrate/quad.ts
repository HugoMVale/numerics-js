import { Vector, type VectorLike } from '../linalg/Vector.js';
import { describeStatus, extrapolatedIntegral } from './adaptiveExtrapolation.js';
import {
  emptyIntervalResult,
  NonFiniteIntegrandError,
  resolveGaussKronrodOptions,
  validateIntegrand
} from './common.js';
import type { GaussKronrodOptions } from './types.js';

/** Result of a {@link quad} numerical integration. */
export interface QuadResult {
  /**
   * Whether the requested error tolerance was reached for the whole integral, i.e.
   * `error <= atol + rtol * |value|`; see `message`. It is `false` if the subinterval limit
   * was reached, if roundoff error makes the tolerance unattainable, if the integrand
   * misbehaves at a point of the range (a singularity or discontinuity that cannot be
   * resolved; try `breakpoints`) or if the integral is probably divergent. Then `value`
   * and `error` are only the best estimates found, and for a divergent integral the value
   * can be meaningless (e.g. `-1` with an error of `1e-15` for `1/x^2` on `(0, 1]`).
   */
  success: boolean;
  /** Message describing the reason for termination. */
  message: string;
  /** Total number of integrand evaluations performed across all subintervals. */
  evaluations: number;
  /** Total number of final subintervals used by the adaptive algorithm. */
  subintervals: number;
  /**
   * Estimated value of the definite integral. If the extrapolation was used, it is the
   * extrapolated value, not the sum of the integrals over the final subintervals.
   */
  value: number;
  /**
   * Estimated absolute integration error of `value`. It is not a guaranteed bound.
   *
   * Without extrapolation it is the sum of the local estimates, each of them the difference
   * `|kronrod - gauss|` of the two rules rescaled as in QUADPACK (it is larger for rougher
   * integrands, and never below the roundoff level `50 eps ∫|f|`). With extrapolation it is
   * the estimate of the epsilon algorithm: the spread of its last three extrapolated results,
   * not smaller than `5 eps |value|`. For smooth integrands it usually exceeds the actual
   * error by orders of magnitude; for integrands with singularities, kinks or jumps it is
   * usually a realistic estimate, but it can be too small, by an order of magnitude when it is
   * close to the roundoff level. A narrow feature that falls between all the nodes of a
   * subinterval can go unnoticed and make the estimate far too small; use `breakpoints` to
   * resolve it.
   */
  error: number;
}

/**
 * A user-defined change of variables `x = map(t)` for {@link quad}.
 *
 * `quad(f, a, b, { transform })` still integrates `f` over `[a, b]` in the
 * original variable `x`; the transform only changes the variable in which the
 * adaptive quadrature is carried out:
 *
 * `∫ f(x) dx = ∫ f(map(t).x) · |map(t).dxDt| dt`, over `t` between
 * `inverse(a)` and `inverse(b)`.
 *
 * Requirements:
 * - `map` is a strictly monotone bijection from the `t`-interval onto `[a, b]`
 *   (it may be increasing or decreasing). Not verified at run time.
 * - `inverse` is consistent with `map`, i.e. `inverse(map(t).x) === t`. Not verified at
 *   run time.
 * - `inverse(a)` and `inverse(b)` are finite and distinct. If `a` or `b` is
 *   infinite, `inverse` must accept `±Infinity` and return the matching finite
 *   `t`-limit. Verified: {@link quad} throws a `RangeError` otherwise.
 * - `map(t).x` and `map(t).dxDt` are finite for `t` strictly inside the
 *   `t`-interval. The endpoints themselves are never evaluated. Verified: {@link quad}
 *   throws an `Error` otherwise.
 */
export interface QuadTransform {
  /**
   * Maps the integration variable `t` to the original variable `x` and
   * returns `dx/dt`. Only the magnitude of `dxDt` is used; the orientation
   * of the mapping is handled internally, so its sign does not matter.
   */
  map: (t: number) => { x: number; dxDt: number };
  /**
   * Inverse mapping `x -> t`. Used to derive the `t`-interval from the
   * integration limits and to translate breakpoints.
   */
  inverse: (x: number) => number;
}

/** Options for {@link quad}. */
export interface QuadOptions extends GaussKronrodOptions {
  /**
   * Specific points within the integration interval where the function
   * might have discontinuities or sharp features. The integration interval
   * is split at these points. Breakpoints outside the integration limits
   * are ignored. A `Vector`, a plain `number[]` or a `Float64Array`.
   * @default []
   */
  breakpoints?: VectorLike;
  /**
   * Optional user-defined change of variables (see {@link QuadTransform}),
   * e.g. to remove an endpoint singularity or to handle an infinite domain
   * differently. When provided, it takes precedence over the automatic
   * handling of infinite limits.
   */
  transform?: QuadTransform;
}

/**
 * Change of variables for `[a, ∞)`: `x = a + (1 - t) / t`, `|dx/dt| = 1 / t²`, with
 * `t ∈ (0, 1]` (`t = 1` is `x = a`; `t → 0` is `x → ∞`).
 *
 * @internal
 */
function upperUnboundedTransform(a: number): QuadTransform {
  return {
    map: (t: number) => ({
      x: a + (1 - t) / t,
      dxDt: 1 / (t * t)
    }),
    inverse: (x: number) => 1 / (1 + x - a)
  };
}

/**
 * Change of variables for `(-∞, b]`: `x = b - (1 - t) / t`, `|dx/dt| = 1 / t²`, with
 * `t ∈ (0, 1]` (`t = 1` is `x = b`; `t → 0` is `x → -∞`).
 *
 * @internal
 */
function lowerUnboundedTransform(b: number): QuadTransform {
  return {
    map: (t: number) => ({
      x: b - (1 - t) / t,
      dxDt: 1 / (t * t)
    }),
    inverse: (x: number) => 1 / (1 + b - x)
  };
}

/**
 * Change of variables for `(-∞, ∞)`: `x = t / (1 - t²)`, `|dx/dt| = (1 + t²) / (1 - t²)²`, with
 * `t ∈ (-1, 1)` (`t = 0` is `x = 0`; `t → ±1` is `x → ±∞`).
 *
 * @internal
 */
function unboundedTransform(): QuadTransform {
  return {
    map: (t: number) => {
      const t2 = t * t;
      const denom = 1 - t2;
      return {
        x: t / denom,
        dxDt: (1 + t2) / (denom * denom)
      };
    },
    inverse: (x: number) => {
      if (x === 0) return 0;
      // The closed form below is ∞/∞ for infinite x.
      if (!Number.isFinite(x)) return Math.sign(x);
      return (Math.sqrt(1 + 4 * x * x) - 1) / (2 * x);
    }
  };
}

/**
 * Numerically integrate `f` over `[a, b]` using adaptive quadrature.
 *
 * This function serves as a general-purpose integrator. It natively supports
 * infinite integration limits (`Infinity` and `-Infinity`). It also
 * supports splitting the integration range at known `breakpoints` to handle
 * piecewise functions or sharp features. Finally, a user-defined change of
 * variables (`transform`) can be supplied, e.g. to remove an endpoint
 * singularity; see {@link QuadTransform}.
 *
 * The integration is performed with the adaptive Gauss-Kronrod (G7-K15) rule and
 * extrapolation by Wynn's epsilon algorithm, as in QUADPACK's `dqagpe` (the routine
 * behind `scipy.integrate.quad` with `points`, and, without breakpoints, `dqagse`). The
 * integral is estimated on subintervals, which are bisected where the local error
 * is largest; the sequence of the successive approximations is then extrapolated, which
 * greatly accelerates the convergence for integrands with integrable endpoint or interior
 * singularities (`x^-0.9`, `log(x)`), kinks and jumps, for which plain bisection needs many
 * subintervals. For smooth integrands the extrapolation costs nothing.
 *
 * All pieces defined by the integration limits and the breakpoints share a single error budget
 * and a single subinterval limit. The algorithm stops when the estimated error of the whole
 * integral satisfies `error <= atol + rtol * |value|`. QUADPACK's tests also detect when
 * this is not possible, and `success` is `false` with a `message` that says why: the limit
 * `maxSubintervals` was reached; roundoff error prevents the tolerance (for example,
 * `rtol` close to the machine precision); the integrand misbehaves at a point of the
 * range (a singularity or discontinuity inside it: add it to `breakpoints`); or the integral
 * is probably divergent (for example `1/x^2` on `(0, 1]`). In these cases `value` and `error`
 * are the best estimates found, and for a divergent integral the value is meaningless. The
 * detection of divergence is heuristic: a divergent integral can also end as one of the
 * other failures (`1/x` ends on the subinterval limit).
 *
 * Because of the extrapolation, the `value` of the result is in general not the sum of the
 * integrals over the final subintervals, and `error` is not the sum of their errors.
 *
 * If `a > b`, the integral over `[b, a]` is computed and its sign reversed; breakpoints are then
 * still given in the original variable, in any order.
 *
 * @param f Scalar function `f` to integrate.
 * @param a Lower bound of integration `a`. Can be `-Infinity`.
 * @param b Upper bound of integration `b`. Can be `Infinity`.
 * @param options Optional settings, see {@link QuadOptions}: error tolerances (`atol`, `rtol`),
 * `maxSubintervals`, `breakpoints` and a user-defined `transform`. A user-defined transform
 * takes precedence over the automatic handling of infinite limits.
 * @returns Quadrature output including aggregated value, error, and diagnostics.
 * @throws {TypeError} If `f` is not a function, if `transform` is given but `map` or
 * `inverse` is not a function, or if `options.breakpoints` is not a `VectorLike`.
 * @throws {RangeError} If `a` or `b` are `NaN`, if `atol` or `rtol` is not finite and
 * non-negative, if `maxSubintervals` is not a finite positive integer or is smaller than the
 * number of initial subintervals defined by the limits and the breakpoints, or if
 * `transform.inverse` returns non-finite or equal values for the integration limits.
 * Except for the last two, these checks are made also when `a === b`.
 * @throws {Error} If `f` evaluates to a non-finite value, or, when a transform is active, if
 * `transform.map` returns a non-finite `x` or `dxDt` or the transformed integrand overflows.
 * The message reports the untransformed `x`, not the internal `t`. The one exception is a
 * non-finite value exactly at an endpoint of the subinterval being bisected, which happens when
 * bisection toward an endpoint singularity reaches floating-point resolution: `success` is then
 * `false`, see {@link QuadResult.success}.
 *
 * @example
 * ```ts
 * import { quad } from 'numerics-js/integrate';
 *
 * // Integrate a function over a finite interval.
 * const result = quad(x => x * x, 0, 1);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   message: 'Estimated error is at most `atol + rtol*|value|`.',
 *   evaluations: 15,
 *   subintervals: 1,
 *   value: 0.3333333333333333,
 *   error: 3.700743415417188e-15
 * }
 * ```
 *
 * @example
 * ```ts
 * // Split the interval at a known discontinuity.
 * const result = quad(x => (x < 1 ? 1 : 2), 0, 2, {
 *     breakpoints: [1]
 * });
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   message: 'Estimated error is at most `atol + rtol*|value|`.',
 *   evaluations: 30,
 *   subintervals: 2,
 *   value: 3,
 *   error: 3.3306690738754696e-14
 * }
 * ```
 *
 * @example
 * ```ts
 * // Integrate over a semi-infinite domain.
 * const result = quad(x => Math.exp(-x), 0, Infinity);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   message: 'Estimated error is at most `atol + rtol*|value|`.',
 *   evaluations: 135,
 *   subintervals: 5,
 *   value: 1,
 *   error: 5.8426077140981e-11
 * }
 * ```
 *
 * @example
 * ```ts
 * // Remove the endpoint singularity of 1/sqrt(x) with the substitution x = t^2.
 * // The transformed integrand f(t^2) * 2t is the constant 2. Without the transform the same
 * // call also succeeds, thanks to the extrapolation, but needs 165 evaluations and 6
 * // subintervals (and the value is 2.0000000000000013).
 * const result = quad(x => 1 / Math.sqrt(x), 0, 1, {
 *     transform: {
 *         map: t => ({ x: t * t, dxDt: 2 * t }),
 *         inverse: Math.sqrt
 *     }
 * });
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   message: 'Estimated error is at most `atol + rtol*|value|`.',
 *   evaluations: 15,
 *   subintervals: 1,
 *   value: 2,
 *   error: 2.220446049250313e-14
 * }
 * ```
 *
 * @example
 * ```ts
 * // A decreasing map (x = 1/t) is fine; only the magnitude of dxDt is used.
 * // Infinite limits are passed to `inverse` as `Infinity` (1 / Infinity = 0).
 * const result = quad(x => 1 / (x * x), 1, Infinity, {
 *     transform: {
 *         map: t => ({ x: 1 / t, dxDt: -1 / (t * t) }),
 *         inverse: x => 1 / x
 *     }
 * });
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   message: 'Estimated error is at most `atol + rtol*|value|`.',
 *   evaluations: 15,
 *   subintervals: 1,
 *   value: 1,
 *   error: 1.1102230246251565e-14
 * }
 * ```
 *
 * @example
 * ```ts
 * // An integrable singularity at the endpoint of a semi-infinite interval:
 * // Gamma(0.1) = 9.513507698668732.
 * const result = quad(x => x ** -0.9 * Math.exp(-x), 0, Infinity, { rtol: 1e-6 });
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   message: 'Estimated error is at most `atol + rtol*|value|`.',
 *   evaluations: 285,
 *   subintervals: 10,
 *   value: 9.513507687893949,
 *   error: 0.000003934779215342132
 * }
 * ```
 *
 * @example
 * ```ts
 * // A divergent integral is reported as a failure: the value is meaningless.
 * const result = quad(x => 1 / (x * x), 0, 1);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: false,
 *   message: 'The integral is probably divergent, or converges too slowly.',
 *   evaluations: 135,
 *   subintervals: 5,
 *   value: -1,
 *   error: 1.1102230246251565e-15
 * }
 * ```
 */
export function quad(
  f: (x: number) => number,
  a: number,
  b: number,
  options: QuadOptions = {}
): QuadResult {
  // All arguments are validated up front, so that invalid input is reported even when
  // the interval is empty.
  validateIntegrand('quad', f);
  if (Number.isNaN(a) || Number.isNaN(b)) {
    throw new RangeError(`quad: a and b must not be NaN, got ${a} and ${b}.`);
  }
  const { breakpoints = [], transform: userTransform, ...toleranceOptions } = options;
  const resolved = resolveGaussKronrodOptions('quad', toleranceOptions);
  if (
    userTransform !== undefined &&
    (typeof userTransform?.map !== 'function' || typeof userTransform?.inverse !== 'function')
  ) {
    throw new TypeError('quad: transform.map and transform.inverse must be functions.');
  }
  const breakpointsv = Vector.asVector(breakpoints);

  if (a === b) {
    return emptyIntervalResult();
  }

  const sign = a > b ? -1 : 1;
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);

  // A user-defined transform takes precedence over the automatic one.
  let transform: QuadTransform | null = userTransform ?? null;
  if (!transform) {
    if (lo === -Infinity && hi === Infinity) {
      transform = unboundedTransform();
    } else if (lo === -Infinity) {
      transform = lowerUnboundedTransform(hi);
    } else if (hi === Infinity) {
      transform = upperUnboundedTransform(lo);
    }
  }

  let tA = lo;
  let tB = hi;
  let targetF = f;

  let validBreakpoints = Array.from(breakpointsv.data).filter(
    (bp) => Number.isFinite(bp) && bp > lo && bp < hi
  );

  if (transform) {
    const { map, inverse } = transform;

    // The map may be increasing or decreasing, so order the t-limits.
    const t1 = inverse(lo);
    const t2 = inverse(hi);
    if (!Number.isFinite(t1) || !Number.isFinite(t2) || t1 === t2) {
      throw new RangeError(
        `quad: transform.inverse must return finite, distinct values for the integration limits (got ${t1} and ${t2}).`
      );
    }
    tA = Math.min(t1, t2);
    tB = Math.max(t1, t2);

    // Only the magnitude of dx/dt matters: the t-range is always ascending.
    // The checks are made here, in x-space, so that the errors name the real x and the
    // actual culprit (the map or f) rather than the transformed integrand that the
    // engine's own check would see.
    targetF = (t: number) => {
      const { x, dxDt } = map(t);
      if (!Number.isFinite(x) || !Number.isFinite(dxDt)) {
        throw new NonFiniteIntegrandError(
          `quad: transform.map(${t}) returned a non-finite x (${x}) or dxDt (${dxDt}); cannot integrate.`,
          t
        );
      }
      const fx = f(x);
      if (!Number.isFinite(fx)) {
        throw new NonFiniteIntegrandError(
          `quad: f(${x}) returned a non-finite value (${fx}) after applying the transform (t=${t}); cannot integrate.`,
          t
        );
      }
      const y = fx * Math.abs(dxDt);
      if (!Number.isFinite(y)) {
        throw new NonFiniteIntegrandError(
          `quad: f(${x}) * |dxDt| overflows (f = ${fx}, dxDt = ${dxDt}) after applying the transform (t=${t}); cannot integrate.`,
          t
        );
      }
      return y;
    };

    // Keep only breakpoints that map strictly inside the t-interval. The
    // comparison also rejects NaN/Infinity images and rounding overshoot.
    validBreakpoints = validBreakpoints.map(inverse).filter((t) => t > tA && t < tB);
  }

  validBreakpoints.sort((x, y) => x - y);
  // Drop repeated points (e.g. breakpoints that collide after the mapping): they would
  // define zero-width subintervals.
  const points = [tA];
  for (const p of [...validBreakpoints, tB]) {
    if (p > points[points.length - 1]) points.push(p);
  }
  if (points.length - 1 > resolved.maxSubintervals) {
    throw new RangeError(
      `quad: maxSubintervals (${resolved.maxSubintervals}) is smaller than the number of initial subintervals (${points.length - 1}) defined by the integration limits and the breakpoints.`
    );
  }

  const r = extrapolatedIntegral('quad', targetF, points, resolved);
  return {
    success: r.status === 0,
    message: describeStatus(r.status, resolved.maxSubintervals, r.endpointSingularity),
    evaluations: r.evaluations,
    subintervals: r.subintervals,
    value: sign * r.value,
    error: r.error
  };
}
