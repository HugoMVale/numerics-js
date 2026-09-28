import { gaussKronrod } from './gaussKronrod.js';

/** Result of a {@link quad} numerical integration. */
export interface QuadResult {
    /** Estimated value of the definite integral. */
    value: number;
    /** Estimated absolute integration error. */
    error: number;
    /** Total number of integrand evaluations performed across all intervals. */
    evaluations: number;
    /** Whether the requested error tolerance was reached in all subintervals. */
    converged: boolean;
    /** Total number of final subintervals used by the adaptive algorithms. */
    subintervals: number;
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
 * Requirements (documented, not verified at run time):
 * - `map` is a strictly monotone bijection from the `t`-interval onto `[a, b]`
 *   (it may be increasing or decreasing).
 * - `inverse` is consistent with `map`, i.e. `inverse(map(t).x) === t`.
 * - `inverse(a)` and `inverse(b)` are finite and distinct. If `a` or `b` is
 *   infinite, `inverse` must accept `±Infinity` and return the matching finite
 *   `t`-limit.
 * - `map(t).x` and `map(t).dxDt` are finite for `t` strictly inside the
 *   `t`-interval. The endpoints themselves are never evaluated.
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
export interface QuadOptions {
    /** Absolute error tolerance for the whole interval. Defaults to `1e-8`. */
    tol?: number;
    /** Safety limit on panel count. Defaults to `200`. */
    maxSubintervals?: number;
    /** 
     * Specific points within the integration interval where the function 
     * might have discontinuities or sharp features. The integration interval 
     * is split at these points. Breakpoints outside the integration limits 
     * are ignored.
     */
    breakpoints?: number[];
    /**
     * Optional user-defined change of variables (see {@link QuadTransform}),
     * e.g. to remove an endpoint singularity or to handle an infinite domain
     * differently. When provided, it takes precedence over the automatic
     * handling of infinite limits.
     */
    transform?: QuadTransform;
}

/** @internal */
function getPositiveInfiniteTransform(a: number): QuadTransform {
    return {
        map: (t: number) => ({
            x: a + (1 - t) / t,
            dxDt: 1 / (t * t)
        }),
        inverse: (x: number) => 1 / (1 + x - a)
    };
}

/** @internal */
function getNegativeInfiniteTransform(b: number): QuadTransform {
    return {
        map: (t: number) => ({
            x: b - (1 - t) / t,
            dxDt: 1 / (t * t)
        }),
        inverse: (x: number) => 1 / (1 + b - x)
    };
}

/** @internal */
function getFullyInfiniteTransform(): QuadTransform {
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
 * Within each interval, the integration is performed using the Gauss-Kronrod (G7-K15)
 * adaptive rule.
 *
 * @param f Scalar function `f` to integrate.
 * @param a Lower bound of integration `a`. Can be `-Infinity`.
 * @param b Upper bound of integration `b`. Can be `Infinity`.
 * @param options Optional settings including error tolerance, breakpoints and
 * a user-defined `transform`. A user-defined transform takes precedence over
 * the automatic handling of infinite limits.
 * @returns Quadrature output including aggregated value, error, and diagnostics.
 * @throws {RangeError} If `a` or `b` are `NaN`, or if `transform.inverse`
 * returns non-finite or equal values for the integration limits.
 * @throws {TypeError} If `transform` is given but `map` or `inverse` is not a function.
 * 
 * @example
 * ```ts
 * // Integrate a function over a finite interval.
 * const result = quad(x => x * x, 0, 1);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   value: 0.3333333333333333,
 *   error: 0,
 *   evaluations: 15,
 *   converged: true,
 *   subintervals: 1
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
 *   value: 3,
 *   error: 0,
 *   evaluations: 30,
 *   converged: true,
 *   subintervals: 2
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
 *   value: 1.0000000000000002,
 *   error: 4.507393744129824e-11,
 *   evaluations: 135,
 *   converged: true,
 *   subintervals: 5
 * }
 * ```
 *
 * @example
 * ```ts
 * // Remove the endpoint singularity of 1/sqrt(x) with the substitution x = t^2.
 * // The transformed integrand f(t^2) * 2t is the constant 2.
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
 *   value: 2,
 *   error: 0,
 *   evaluations: 15,
 *   converged: true,
 *   subintervals: 1
 * }
 * ```
 * Without the transform the same call needs 1395 evaluations and 47
 * subintervals, and is only accurate to about 1e-9.
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
 *   value: 1,
 *   error: 0,
 *   evaluations: 15,
 *   converged: true,
 *   subintervals: 1
 * }
 * ```
 */
export function quad(
    f: (x: number) => number,
    a: number,
    b: number,
    options: QuadOptions = {}
): QuadResult {
    const { breakpoints = [], transform: userTransform, ...gkOptions } = options;

    if (Number.isNaN(a) || Number.isNaN(b)) {
        throw new RangeError('quad: a and b must not be NaN');
    }
    if (userTransform !== undefined
        && (typeof userTransform?.map !== 'function' || typeof userTransform?.inverse !== 'function')) {
        throw new TypeError('quad: transform.map and transform.inverse must be functions');
    }
    if (a === b) {
        return { value: 0, error: 0, evaluations: 0, converged: true, subintervals: 0 };
    }

    const sign = a > b ? -1 : 1;
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);

    // A user-defined transform takes precedence over the automatic one.
    let transform: QuadTransform | null = userTransform ?? null;
    if (!transform) {
        if (lo === -Infinity && hi === Infinity) {
            transform = getFullyInfiniteTransform();
        } else if (lo === -Infinity) {
            transform = getNegativeInfiniteTransform(hi);
        } else if (hi === Infinity) {
            transform = getPositiveInfiniteTransform(lo);
        }
    }

    let tA = lo;
    let tB = hi;
    let targetF = f;

    let validBreakpoints = breakpoints.filter(bp => Number.isFinite(bp) && bp > lo && bp < hi);

    if (transform) {
        const { map, inverse } = transform;

        // The map may be increasing or decreasing, so order the t-limits.
        const t1 = inverse(lo);
        const t2 = inverse(hi);
        if (!Number.isFinite(t1) || !Number.isFinite(t2) || t1 === t2) {
            throw new RangeError(
                `quad: transform.inverse must return finite, distinct values for the integration limits (got ${t1} and ${t2})`
            );
        }
        tA = Math.min(t1, t2);
        tB = Math.max(t1, t2);

        // Only the magnitude of dx/dt matters: the t-range is always ascending.
        targetF = (t: number) => {
            const { x, dxDt } = map(t);
            return f(x) * Math.abs(dxDt);
        };

        // Keep only breakpoints that map strictly inside the t-interval. The
        // comparison also rejects NaN/Infinity images and rounding overshoot.
        validBreakpoints = validBreakpoints
            .map(inverse)
            .filter(t => t > tA && t < tB);
    }

    validBreakpoints.sort((x, y) => x - y);
    const points = [tA, ...validBreakpoints, tB];

    let totalValue = 0;
    let totalError = 0;
    let totalEvaluations = 0;
    let totalSubintervals = 0;
    let allConverged = true;

    for (let i = 0; i < points.length - 1; i++) {
        const p1 = points[i];
        const p2 = points[i + 1];

        const res = gaussKronrod(targetF, p1, p2, gkOptions);

        totalValue += res.value;
        totalError += res.error;
        totalEvaluations += res.evaluations;
        totalSubintervals += res.subintervals;
        if (!res.converged) allConverged = false;
    }

    return {
        value: sign * totalValue,
        error: totalError,
        evaluations: totalEvaluations,
        converged: allConverged,
        subintervals: totalSubintervals,
    };
}