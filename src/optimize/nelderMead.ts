import { Vector } from '../array/Vector.js';
import { Matrix } from '../array/Matrix.js';

/**
 * Result of a {@link nelderMead} optimization.
 */
export interface NelderMeadResult {
    /** Whether the optimize terminated successfully. */
    success: boolean;
    /** Human-readable description of the termination reason. */
    message: string;
    /** Number of function evaluations performed. */
    evaluations: number;
    /** Best point found. */
    x: Vector;
    /** Function value at `x`. */
    fx: number;
}

/**
 * Options for {@link nelderMead}.
 */
export interface NelderMeadOptions {
    /**
     * Absolute tolerance for `x`. The algorithm terminates when the maximum
     * scaled distance between the simplex vertices is less than `tolX`.
     * Defaults to `1e-8`.
     */
    tolX?: number;
    /**
     * Absolute tolerance for `f`. The algorithm terminates when the maximum
     * difference between the function values at the simplex vertices is less
     * than `tolF`. Defaults to `1e-8`.
     */
    tolF?: number;
    /**
     * Positive scaling factors for the components of `x`, as a plain array
     * or an `Vector`. Ideally, these should be chosen so that `scale*x` is
     * of order 1 near the solution for all components. If omitted, scaling
     * is inferred from `x0` as `1 / max(|x0_i|, 1)`.
     */
    scale?: number[] | Vector;
    /** Maximum number of iterations. Defaults to `200*N`. */
    maxIter?: number;
    /** Maximum number of function evaluations. Defaults to `200*N`. */
    maxFunEvals?: number;
    /**
     * Whether to use the adaptive parameter scheme proposed by Gao (2012).
     * If `false`, the standard Nelder-Mead parameters are used. Defaults to
     * `true`.
     */
    adaptive?: boolean;
    /**
     * Optional callback invoked at each iteration as
     * `callback(nIter, x, fx) -> {stop, success}`, where `x` is the
     * `(N+1) x N` matrix of simplex vertices (one per row) and `fx` the
     * corresponding `N+1` function values. Neither should be mutated by the
     * callback. If `stop` is `true`, the iteration is terminated; `success`
     * then decides whether the result is reported as successful.
     */
    callback?: (nIter: number, x: Matrix, fx: Vector) => { stop: boolean; success: boolean };
}

/**
 * Infers default scaling factors from the initial guess: `1 / max(|x0_i|, 1)`.
 * @param x0 Initial guess.
 * @returns Default scaling factors.
 */
function defaultScale(x0: Vector): Vector {
    const s = new Vector(x0.size);
    for (let i = 0; i < x0.size; i++) {
        s.set(i, 1 / Math.max(Math.abs(x0.get(i)), 1));
    }
    return s;
}

/**
 * Finds the index of the smallest value, and the indices of the two largest
 * values, in a single pass over `fx`. Requires `fx.size >= 2`.
 * @param fx Function values at the simplex vertices.
 * @returns `{imin, imax, imax2}`, where `imax` is the largest and `imax2`
 * the second largest.
 */
function simplexExtremes(fx: Vector): { imin: number; imax: number; imax2: number } {
    const d = fx.data;

    let imin = 0;
    for (let k = 1; k < d.length; k++) {
        if (d[k] < d[imin]) imin = k;
    }

    let imax: number, imax2: number;
    if (d[0] > d[1]) {
        imax = 0;
        imax2 = 1;
    } else {
        imax = 1;
        imax2 = 0;
    }
    for (let k = 2; k < d.length; k++) {
        if (d[k] > d[imax]) {
            imax2 = imax;
            imax = k;
        } else if (d[k] > d[imax2]) {
            imax2 = k;
        }
    }

    return { imin, imax, imax2 };
}

/**
 * Finds the minimum of a multivariate function using the Nelder-Mead simplex
 * algorithm.
 *
 * The Nelder-Mead simplex algorithm is a derivative-free optimize method
 * for unconstrained minimization of multivariate functions. It maintains a
 * simplex of `N+1` vertices in `N`-dimensional space and iteratively updates
 * this simplex based on the function values at the vertices.
 *
 * The initial simplex is aligned with the coordinate axes. For each
 * coordinate, the corresponding simplex vertex offset is:
 *
 * `dx_i = 0.05 * max(|x0_i|, 1/scale_i)`
 *
 * where `x0` is the initial guess and `scale_i` is the scaling factor
 * associated with variable `x_i`. Therefore, it is important that `x0`
 * and/or `scale` reflect the expected scale of the variables. If `scale` is
 * not provided, the variable scaling is inferred from `x0`.
 *
 * Results are validated against `scipy.optimize.minimize` with
 * `method="Nelder-Mead"` (see `tests/optimize/nelderMead.scipy.test.ts`);
 * see `tests/optimize/nelderMead.scipy.benchmark.md` for the comparison
 * table.
 *
 * References:
 * - Nelder, J. A.; Mead, R. A Simplex Method for Function Minimization.
 *   Comput. J. 1965, 7, 308-313.
 * - Gao, F.; Han, L. Implementing the Nelder-Mead Simplex Algorithm with
 *   Adaptive Parameters. Comput. Optim. Appl. 2012, 51, 259-277.
 *
 *
 * @param fn Objective function to minimize.
 * @param x0 Initial guess for the optimum. If no user-defined `scale` is
 * provided, the scaling factors will be determined from this value.
 * @param options Optional settings; see {@link NelderMeadOptions}.
 * @returns The optimize result.
 *
 * @example
 * ```ts
 * // Find the minimum of f(x) = (x0-100)^2 + (x1-1e10)^2
 * const fn = (x: Vector): number => (x.get(0) - 1e2) ** 2 + (x.get(1) - 1e10) ** 2;
 * const result = nelderMead(fn, [1, 1e8]);
 * // or, with explicit options:
 * // const result = nelderMead(fn, [1, 1e8], { tolX: 1e-10, adaptive: false });
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   message: 'Function value spread is less than `tolF`.',
 *   evaluations: 225,
 *   x: Vector [ 99.99998642148125, 9999999999.999956 ],
 *   fx: 2.1088669603067145e-9
 * }
 * ```
 */
export function nelderMead(
    fn: (x: Vector) => number,
    x0: number[] | Vector,
    options: NelderMeadOptions = {}
): NelderMeadResult {
    const {
        tolX = 1e-8,
        tolF = 1e-8,
        scale,
        maxIter,
        maxFunEvals,
        adaptive = true,
        callback,
    } = options;

    const x0v = x0 instanceof Vector ? x0.copy() : Vector.from(x0);
    const n = x0v.size;
    if (n === 0) {
        throw new RangeError('nelderMead: x0 must have at least one component');
    }

    const xScale =
        scale !== undefined
            ? Vector.from(Array.from(scale instanceof Vector ? scale.data : scale, Math.abs))
            : defaultScale(x0v);

    const iterLimit = maxIter ?? 200 * n;
    const fevalLimit = maxFunEvals ?? 200 * n;

    // Algorithm parameters
    let a: number, b: number, c: number, d: number;
    if (adaptive) {
        // Dimension-dependent parameters from Gao (2012)
        a = 1.0;
        b = 1.0 + 2.0 / n;
        c = 0.75 - 1.0 / (2 * n);
        d = 1.0 - 1.0 / n;
    } else {
        a = 1.0;
        b = 2.0;
        c = 0.5;
        d = 0.5;
    }

    // Initialize the simplex vertices: (n+1) x n matrix, one vertex per row
    const x = new Matrix(n + 1, n);
    x.setRow(0, x0v);
    for (let k = 0; k < n; k++) {
        const v = x0v.copy();
        const step = 0.05 * Math.max(Math.abs(x0v.get(k)), 1 / xScale.get(k));
        v.set(k, v.get(k) + step);
        x.setRow(k + 1, v);
    }

    // Evaluate the function at the simplex vertices
    const fx = new Vector(n + 1);
    for (let k = 0; k <= n; k++) {
        fx.data[k] = fn(x.row(k));
    }
    let nFev = n + 1;

    // Main optimize loop
    let nIter = 0;
    let success = false;
    let message = '';
    let xmin = x.row(0);
    let fmin = fx.data[0];

    while (true) {
        nIter++;

        const { imin, imax, imax2 } = simplexExtremes(fx);
        xmin = x.row(imin);
        fmin = fx.data[imin];
        const fmax2 = fx.data[imax2];
        const fmax = fx.data[imax];

        if (callback !== undefined) {
            const { stop, success: cbSuccess } = callback(nIter, x, fx);
            if (stop) {
                success = cbSuccess;
                message = 'Terminated by user callback.';
                break;
            }
        }

        let maxDist = 0;
        for (let k = 0; k <= n; k++) {
            for (let j = 0; j < n; j++) {
                const dj = Math.abs(xScale.get(j) * (x.get(k, j) - xmin.get(j)));
                if (dj > maxDist) maxDist = dj;
            }
        }
        if (maxDist < tolX) {
            success = true;
            message = 'Maximum distance between simplex vertices is less than `tolX`.';
            break;
        }

        if (fmax - fmin < tolF) {
            success = true;
            message = 'Function value spread is less than `tolF`.';
            break;
        }

        if (nIter === iterLimit) {
            message = `Maximum number of iterations (${iterLimit}) reached.`;
            break;
        }

        if (nFev >= fevalLimit) {
            message = `Maximum number of function evaluations (${fevalLimit}) reached.`;
            break;
        }

        // Centroid excluding the worst point
        const xc = new Vector(n);
        for (let k = 0; k <= n; k++) {
            if (k === imax) continue;
            xc.addSelf(x.row(k));
        }
        xc.multSelf(1 / n);

        const xWorst = x.row(imax);

        // Reflection
        const xr = xc.copy().multSelf(1 + a).addScaled(xWorst, -a);
        const fr = fn(xr);
        nFev++;

        let doShrink = false;
        if (fmin <= fr && fr < fmax2) {
            x.setRow(imax, xr);
            fx.data[imax] = fr;
        } else if (fr < fmin) {
            const xe = xc.copy().multSelf(1 + a * b).addScaled(xWorst, -a * b);
            const fe = fn(xe);
            nFev++;
            if (fe < fr) {
                x.setRow(imax, xe);
                fx.data[imax] = fe;
            } else {
                x.setRow(imax, xr);
                fx.data[imax] = fr;
            }
        } else if (fmax2 <= fr && fr < fmax) {
            const xoc = xc.copy().multSelf(1 + a * c).addScaled(xWorst, -a * c);
            const foc = fn(xoc);
            nFev++;
            if (foc <= fr) {
                x.setRow(imax, xoc);
                fx.data[imax] = foc;
            } else {
                doShrink = true;
            }
        } else {
            const xic = xc.copy().multSelf(1 - a * c).addScaled(xWorst, a * c);
            const fic = fn(xic);
            nFev++;
            if (fic < fmax) {
                x.setRow(imax, xic);
                fx.data[imax] = fic;
            } else {
                doShrink = true;
            }
        }

        if (doShrink) {
            for (let k = 0; k <= n; k++) {
                if (k === imin) continue;
                const xs = x.row(k).multSelf(d).addScaled(xmin, 1 - d);
                x.setRow(k, xs);
                fx.data[k] = fn(xs);
                nFev++;
            }
        }
    }

    return {
        success,
        message,
        evaluations: nFev,
        x: xmin,
        fx: fmin,
    };
}