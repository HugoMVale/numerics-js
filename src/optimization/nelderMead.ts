import { Array1D } from '../array/array1d.js';
import { Array2D } from '../array/array2d.js';

/**
 * Result of a `fminNelderMead` optimization.
 */
export interface NelderMeadResult {
    /** Name of the optimization method used. */
    method: string;
    /** Whether the optimization terminated successfully. */
    success: boolean;
    /** Human-readable description of the termination reason. */
    message: string;
    /** Number of function evaluations performed. */
    nfeval: number;
    /** Number of iterations performed. */
    niter: number;
    /** Best point found. */
    x: Array1D;
    /** Function value at `x`. */
    f: number;
}

/**
 * Optional settings for `fminNelderMead`.
 */
export interface NelderMeadOptions {
    /**
     * Absolute tolerance for `x`. The algorithm terminates when the maximum
     * scaled distance between the simplex vertices is less than `tolx`.
     * @default 1e-8
     */
    tolx?: number;
    /**
     * Absolute tolerance for `f`. The algorithm terminates when the maximum
     * difference between the function values at the simplex vertices is less
     * than `tolf`.
     * @default 1e-8
     */
    tolf?: number;
    /**
     * Positive scaling factors for the components of `x`. Ideally, these
     * should be chosen so that `sclx*x` is of order 1 near the solution for
     * all components. By default, scaling is inferred from `x0` as
     * `1 / max(|x0_i|, 1)`.
     */
    sclx?: number[] | Float64Array;
    /**
     * Maximum number of iterations.
     * @default 200*N
     */
    maxiter?: number;
    /**
     * Maximum number of function evaluations.
     * @default 200*N
     */
    maxfeval?: number;
    /**
     * Whether to use the adaptive parameter scheme proposed by Gao (2012).
     * If `false`, the standard Nelder-Mead parameters are used.
     * @default true
     */
    adaptive?: boolean;
    /**
     * Optional callback invoked at each iteration as
     * `callback(niter, x, fx) -> {stop, success}`, where `x` is the
     * `(N+1) x N` matrix of simplex vertices (one per row) and `fx` the
     * corresponding `N+1` function values. Neither should be mutated by the
     * callback. If `stop` is `true`, the iteration is terminated; `success`
     * then decides whether the result is reported as successful.
     */
    callback?: (niter: number, x: Array2D, fx: Array1D) => { stop: boolean; success: boolean };
}

/**
 * Infers default scaling factors from the initial guess: `1 / max(|x0_i|, 1)`.
 * @param x0 - Initial guess.
 * @returns Default scaling factors.
 */
function defaultScale(x0: Array1D): Array1D {
    const s = new Array1D(x0.dim);
    for (let i = 0; i < x0.dim; i++) {
        s.set(i, 1 / Math.max(Math.abs(x0.get(i)), 1));
    }
    return s;
}

/**
 * Finds the index of the smallest value, and the indices of the two largest
 * values, in a single pass over `fx`. Requires `fx.dim >= 2`.
 * @param fx - Function values at the simplex vertices.
 * @returns `{imin, imax, imax2}`, where `imax` is the largest and `imax2`
 * the second largest.
 */
function simplexExtremes(fx: Array1D): { imin: number; imax: number; imax2: number } {
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
 * The Nelder-Mead simplex algorithm is a derivative-free optimization method
 * for unconstrained minimization of multivariate functions. It maintains a
 * simplex of `N+1` vertices in `N`-dimensional space and iteratively updates
 * this simplex based on the function values at the vertices.
 *
 * The initial simplex is aligned with the coordinate axes. For each
 * coordinate, the corresponding simplex vertex offset is:
 *
 * `dx_i = 0.05 * max(|x0_i|, 1/sclx_i)`
 *
 * where `x0` is the initial guess and `sclx_i` is the scaling factor
 * associated with variable `x_i`. Therefore, it is important that `x0`
 * and/or `sclx` reflect the expected scale of the variables. If `sclx` is
 * not provided, the variable scaling is inferred from `x0`.
 *
 * References:
 * - Nelder, J. A.; Mead, R. A Simplex Method for Function Minimization.
 *   Comput. J. 1965, 7, 308-313.
 * - Gao, F.; Han, L. Implementing the Nelder-Mead Simplex Algorithm with
 *   Adaptive Parameters. Comput. Optim. Appl. 2012, 51, 259-277.
 *
 * @param f - Objective function to minimize.
 * @param x0 - Initial guess for the optimum. If no user-defined scale
 * `sclx` is provided, the scaling factors will be determined from this
 * value.
 * @param options - Optional algorithm settings, see `NelderMeadOptions`.
 * @returns The optimization result.
 *
 * @example
 * ```ts
 * // Find the minimum of f(x) = (x0-100)^2 + (x1-1e10)^2
 * const result = fminNelderMead(
 *     (x) => (x.get(0) - 1e2) ** 2 + (x.get(1) - 1e10) ** 2,
 *     [1, 1e8]
 * );
 * ```
 */
export function fminNelderMead(
    f: (x: Array1D) => number,
    x0: number[] | Float64Array | Array1D,
    options: NelderMeadOptions = {}
): NelderMeadResult {
    const { tolx = 1e-8, tolf = 1e-8, sclx, maxiter, maxfeval, adaptive = true, callback } = options;

    const x0v = x0 instanceof Array1D ? x0.copy() : Array1D.from(x0);
    const n = x0v.dim;
    if (n === 0) {
        throw new RangeError('fminNelderMead: x0 must have at least one component');
    }

    const scale = sclx !== undefined ? Array1D.from(Array.from(sclx, Math.abs)) : defaultScale(x0v);

    const maxIter = maxiter ?? 200 * n;
    const maxFeval = maxfeval ?? 200 * n;

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
    const x = new Array2D(n + 1, n);
    x.setRow(0, x0v);
    for (let k = 0; k < n; k++) {
        const v = x0v.copy();
        const step = 0.05 * Math.max(Math.abs(x0v.get(k)), 1 / scale.get(k));
        v.set(k, v.get(k) + step);
        x.setRow(k + 1, v);
    }

    // Evaluate the function at the simplex vertices
    const fx = new Array1D(n + 1);
    for (let k = 0; k <= n; k++) {
        fx.data[k] = f(x.row(k));
    }
    let nfeval = n + 1;

    // Main optimization loop
    let niter = 0;
    let success = false;
    let message = '';
    let xmin = x.row(0);
    let fmin = fx.data[0];

    while (true) {
        niter++;

        const { imin, imax, imax2 } = simplexExtremes(fx);
        xmin = x.row(imin);
        fmin = fx.data[imin];
        const fmax2 = fx.data[imax2];
        const fmax = fx.data[imax];

        if (callback !== undefined) {
            const { stop, success: cbSuccess } = callback(niter, x, fx);
            if (stop) {
                success = cbSuccess;
                message = 'Terminated by user callback.';
                break;
            }
        }

        let maxDist = 0;
        for (let k = 0; k <= n; k++) {
            for (let j = 0; j < n; j++) {
                const dj = Math.abs(scale.get(j) * (x.get(k, j) - xmin.get(j)));
                if (dj > maxDist) maxDist = dj;
            }
        }
        if (maxDist < tolx) {
            success = true;
            message = 'Maximum distance between simplex vertices is less than `tolx`.';
            break;
        }

        if (fmax - fmin < tolf) {
            success = true;
            message = 'Function value spread is less than `tolf`.';
            break;
        }

        if (niter === maxIter) {
            message = `Maximum number of iterations (${maxIter}) reached.`;
            break;
        }

        if (nfeval >= maxFeval) {
            message = `Maximum number of function evaluations (${maxFeval}) reached.`;
            break;
        }

        // Centroid excluding the worst point
        const xc = new Array1D(n);
        for (let k = 0; k <= n; k++) {
            if (k === imax) continue;
            xc.addSelf(x.row(k));
        }
        xc.multSelf(1 / n);

        const xWorst = x.row(imax);

        // Reflection
        const xr = xc.copy().multSelf(1 + a).addScaled(xWorst, -a);
        const fr = f(xr);
        nfeval++;

        let shrink = false;
        if (fmin <= fr && fr < fmax2) {
            x.setRow(imax, xr);
            fx.data[imax] = fr;
        } else if (fr < fmin) {
            const xe = xc.copy().multSelf(1 + a * b).addScaled(xWorst, -a * b);
            const fe = f(xe);
            nfeval++;
            if (fe < fr) {
                x.setRow(imax, xe);
                fx.data[imax] = fe;
            } else {
                x.setRow(imax, xr);
                fx.data[imax] = fr;
            }
        } else if (fmax2 <= fr && fr < fmax) {
            const xoc = xc.copy().multSelf(1 + a * c).addScaled(xWorst, -a * c);
            const foc = f(xoc);
            nfeval++;
            if (foc <= fr) {
                x.setRow(imax, xoc);
                fx.data[imax] = foc;
            } else {
                shrink = true;
            }
        } else {
            const xic = xc.copy().multSelf(1 - a * c).addScaled(xWorst, a * c);
            const fic = f(xic);
            nfeval++;
            if (fic < fmax) {
                x.setRow(imax, xic);
                fx.data[imax] = fic;
            } else {
                shrink = true;
            }
        }

        if (shrink) {
            for (let k = 0; k <= n; k++) {
                if (k === imin) continue;
                const xs = x.row(k).multSelf(d).addScaled(xmin, 1 - d);
                x.setRow(k, xs);
                fx.data[k] = f(xs);
                nfeval++;
            }
        }
    }

    return {
        method: 'Nelder-Mead',
        success,
        message,
        nfeval,
        niter,
        x: xmin,
        f: fmin,
    };
}
