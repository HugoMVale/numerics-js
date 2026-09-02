import { ScalarFunction } from '../types.js';

/** Result of a {@link gaussKronrod} quadrature. */
export interface GaussKronrodResult {
    /** Estimated value of the definite integral. */
    value: number;
    /** Estimated absolute integration error. */
    error: number;
    /** Number of integrand evaluations performed. */
    evaluations: number;
    /** Whether the requested error tolerance was reached. */
    converged: boolean;
    /** Number of final subintervals used by the adaptive algorithm. */
    subintervals: number;
}

interface LocalGaussKronrodResult {
    kronrod: number;
    gauss: number;
    error: number;
}

/** Options for {@link gaussKronrod}. */
export interface GaussKronrodOptions {
    /** Absolute error tolerance for the whole interval. Defaults to `1e-8`. */
    tol?: number;
    /** Safety limit on panel count. Defaults to `200`. */
    maxSubintervals?: number;
}

const KRONROD_NODES: readonly number[] = [
    0.000000000000000000000000000000000,
    0.207784955007898467600689403773245,
    0.405845151377397166906606412076961,
    0.586087235467691130294144838258730,
    0.741531185599394439863864773280788,
    0.864864423359769072789712788640926,
    0.949107912342758524526189684047851,
    0.991455371120812639206854697526329,
];

const KRONROD_WEIGHTS: readonly number[] = [
    0.209482141084727828012999174891714,
    0.204432940075298892414161999234649,
    0.190350578064785409913256402421014,
    0.169004726639267902826583426598550,
    0.140653259715525918745189590510238,
    0.104790010322250183839876322541518,
    0.063092092629978553290700663189204,
    0.022935322010529224963732008058970,
];

const GAUSS_WEIGHTS: readonly number[] = [
    0.417959183673469387755102040816327,
    0,
    0.381830050505118944950369775488975,
    0,
    0.279705391489276667901467771423780,
    0,
    0.129484966168869693270611432679082,
    0,
];

interface Panel {
    a: number;
    b: number;
    kronrod: number;
    error: number;
}

/**
 * Applies the 7-15 Gauss-Kronrod rule to one panel and returns the
 * embedded Gauss and Kronrod estimates.
 * @internal
 *
 * @param evalF Instrumented function evaluator.
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 * @returns Local quadrature estimates and estimated local error.
 */
export function gaussKronrod15(
    evalF: ScalarFunction,
    a: number,
    b: number
): LocalGaussKronrodResult {
    const halfLength = (b - a) / 2;
    const center = (a + b) / 2;

    let kronrodSum = 0;
    let gaussSum = 0;

    for (let i = 0; i < KRONROD_NODES.length; i++) {
        const dx = halfLength * KRONROD_NODES[i];
        const kw = KRONROD_WEIGHTS[i];
        const gw = GAUSS_WEIGHTS[i];

        if (i === 0) {
            const f0 = evalF(center);
            kronrodSum += kw * f0;
            gaussSum += gw * f0;
        } else {
            const fPlus = evalF(center + dx);
            const fMinus = evalF(center - dx);
            kronrodSum += kw * (fPlus + fMinus);
            gaussSum += gw * (fPlus + fMinus);
        }
    }

    const kronrod = halfLength * kronrodSum;
    const gauss = halfLength * gaussSum;

    return { kronrod, gauss, error: Math.abs(kronrod - gauss) };
}

/**
 * Numerically integrate `f` over [`a`, `b`] using adaptive Gauss-Kronrod
 * (G7-K15) quadrature.
 *
 * This function is suitable for numerically integrating smooth functions over finite 
 * intervals. For functions with discontinuities or sharp features, or infinite domains, 
 * use {@link quad} instead.
 *
 *
 * @param fn Scalar function to integrate.
 * @param a Lower bound of integration.
 * @param b Upper bound of integration.
 * @param options Optional settings.
 * @param options.tol Absolute error tolerance for the whole interval. Defaults to `1e-8`.
 * @param options.maxSubintervals Safety limit on panel count. Defaults to `200`.
 * @returns Quadrature output including final value, estimated error, and diagnostics.
 * @throws {RangeError} If `a` or `b` are non-finite numbers.
 * @throws {Error} If `f` evaluates to a non-finite value.
 *
 * @example
 * ```ts
 * // Integrate an oscillatory function that requires adaptive subdivision.
 * const result = gaussKronrod(x => Math.sin(100 * x), 0, 1);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   value: 0.0013768112771228931,
 *   error: 2.0058021368820277e-9,
 *   evaluations: 465,
 *   converged: true,
 *   subintervals: 16
 * }
 * ```
 */
export function gaussKronrod(
    fn: ScalarFunction,
    a: number,
    b: number,
    options: GaussKronrodOptions = {}
): GaussKronrodResult {
    const { tol = 1e-8, maxSubintervals = 200 } = options;

    if (!Number.isFinite(a) || !Number.isFinite(b)) {
        throw new RangeError('gaussKronrod: a and b must be finite numbers');
    }
    if (a === b) {
        return { value: 0, error: 0, evaluations: 0, converged: true, subintervals: 0 };
    }

    const sign = a > b ? -1 : 1;
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);

    let evaluations = 0;
    const evalF = (x: number): number => {
        evaluations++;
        const y = fn(x);
        if (!Number.isFinite(y)) {
            throw new Error(`gaussKronrod: f(${x}) returned a non-finite value (${y}); cannot integrate.`);
        }
        return y;
    };

    const first = gaussKronrod15(evalF, lo, hi);
    const panels: Panel[] = [{ a: lo, b: hi, kronrod: first.kronrod, error: first.error }];

    let totalValue = first.kronrod;
    let totalError = first.error;
    let converged = true;

    while (totalError > tol) {
        if (panels.length >= maxSubintervals) {
            converged = false;
            break;
        }

        let worstIdx = 0;
        for (let i = 1; i < panels.length; i++) {
            if (panels[i].error > panels[worstIdx].error) worstIdx = i;
        }
        const worst = panels[worstIdx];

        const mid = (worst.a + worst.b) / 2;
        const left = gaussKronrod15(evalF, worst.a, mid);
        const right = gaussKronrod15(evalF, mid, worst.b);

        totalValue += left.kronrod + right.kronrod - worst.kronrod;
        totalError += left.error + right.error - worst.error;

        panels[worstIdx] = { a: worst.a, b: mid, kronrod: left.kronrod, error: left.error };
        panels.push({ a: mid, b: worst.b, kronrod: right.kronrod, error: right.error });
    }

    return {
        value: sign * totalValue,
        error: totalError,
        evaluations,
        converged,
        subintervals: panels.length,
    };
}