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
}

/** 
 * Defines a variable substitution mapping t -> x to handle infinite domains. 
 * @internal 
 */
interface Transformation {
    /** Maps integration variable t to the original variable x, returning x and dx/dt. */
    mapTtoX: (t: number) => { x: number; dxDt: number };
    /** Inverse mapping from x to t, used to translate user-provided breakpoints. */
    mapXtoT: (x: number) => number;
    tMin: number;
    tMax: number;
}

/** @internal */
function getPositiveInfiniteTransform(a: number): Transformation {
    return {
        mapTtoX: (t: number) => ({
            x: a + (1 - t) / t,
            dxDt: 1 / (t * t)
        }),
        mapXtoT: (x: number) => 1 / (1 + x - a),
        tMin: 0,
        tMax: 1
    };
}

/** @internal */
function getNegativeInfiniteTransform(b: number): Transformation {
    return {
        mapTtoX: (t: number) => ({
            x: b - (1 - t) / t,
            dxDt: 1 / (t * t)
        }),
        mapXtoT: (x: number) => 1 / (1 + b - x),
        tMin: 0,
        tMax: 1
    };
}

/** @internal */
function getFullyInfiniteTransform(): Transformation {
    return {
        mapTtoX: (t: number) => {
            const t2 = t * t;
            const denom = 1 - t2;
            return {
                x: t / denom,
                dxDt: (1 + t2) / (denom * denom)
            };
        },
        mapXtoT: (x: number) => {
            if (x === 0) return 0;
            return (Math.sqrt(1 + 4 * x * x) - 1) / (2 * x);
        },
        tMin: -1,
        tMax: 1
    };
}

/**
 * Numerically integrate `fn` over [`a`, `b`] using adaptive quadrature.
 * 
 * This function serves as a general-purpose integrator. It natively supports
 * infinite integration limits (`Infinity` and `-Infinity`). It also 
 * supports splitting the integration range at known `breakpoints` to handle 
 * piecewise functions or sharp features.
 * 
 * Within each interval, the integration is performed using the Gauss-Kronrod (G7-K15)
 * adaptive rule.
 *
 * @param fn Scalar function to integrate.
 * @param a Lower bound of integration. Can be `-Infinity`.
 * @param b Upper bound of integration. Can be `Infinity`.
 * @param options Optional settings including error tolerance and breakpoints.
 * @returns Quadrature output including aggregated value, error, and diagnostics.
 * @throws {RangeError} If `a` or `b` are `NaN`.
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
 */
export function quad(
    fn: (x: number) => number,
    a: number,
    b: number,
    options: QuadOptions = {}
): QuadResult {
    const { breakpoints = [], ...gkOptions } = options;

    if (Number.isNaN(a) || Number.isNaN(b)) {
        throw new RangeError('quad: a and b must not be NaN');
    }
    if (a === b) {
        return { value: 0, error: 0, evaluations: 0, converged: true, subintervals: 0 };
    }

    const sign = a > b ? -1 : 1;
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);

    let transform: Transformation | null = null;
    if (lo === -Infinity && hi === Infinity) {
        transform = getFullyInfiniteTransform();
    } else if (lo === -Infinity) {
        transform = getNegativeInfiniteTransform(hi);
    } else if (hi === Infinity) {
        transform = getPositiveInfiniteTransform(lo);
    }

    let tA = lo;
    let tB = hi;
    let targetFn = fn;

    let validBreakpoints = breakpoints.filter(bp => Number.isFinite(bp) && bp > lo && bp < hi);

    if (transform) {
        tA = transform.tMin;
        tB = transform.tMax;

        targetFn = (t: number) => {
            const { x, dxDt } = transform!.mapTtoX(t);
            return fn(x) * dxDt;
        };

        validBreakpoints = validBreakpoints.map(transform.mapXtoT);
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

        const res = gaussKronrod(targetFn, p1, p2, gkOptions);

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