import { Vector } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import { jacobianForward, scaleVector } from '../numdiff.js';
import { levenbergMarquardt, type LevenbergMarquardtOptions } from './levenbergMarquardt.js';

const EPS = Number.EPSILON;

/**
 * Result of {@link curveFit}.
 */
export interface CurveFitResult {
    /** Descriptive method name. */
    method: string;
    /** Whether the fit converged; see {@link LevenbergMarquardtResult.success}. */
    success: boolean;
    /** Message describing the result or error. */
    message: string;
    /**
     * Number of evaluations of the vector of (weighted) residuals, i.e. of the
     * model at all data points, including the finite-difference evaluations
     * and the ones needed for the final Jacobian.
     */
    evaluationsFunction: number;
    /** Number of user-supplied Jacobian evaluations, including the one for the final covariance. */
    evaluationsJacobian: number;
    /** Number of accepted Levenberg-Marquardt steps. */
    iterations: number;
    /** Optimal parameters (or best estimate, if not converged). */
    p: Vector;
    /**
     * Estimated covariance matrix of the parameters (`n x n`, in the original
     * parameter order). Entries are `Infinity` if the covariance cannot be
     * determined: the Jacobian is rank-deficient (some parameters are
     * redundant), or there are no degrees of freedom (`dof = 0`) and
     * `absoluteSigma` is `false`. They are `NaN` if the residuals or the
     * Jacobian at `p` are not finite.
     */
    cov: Matrix;
    /** Standard errors of the parameters, `sqrt(diag(cov))`. */
    stderr: Vector;
    /** Residuals `ydata - f(xdata, p)` at the returned parameters (unweighted). */
    residuals: Vector;
    /**
     * Weighted residual sum of squares, `sum(((ydata - f(xdata, p)) / sigma)²)`.
     * Without `sigma`, it is the plain residual sum of squares.
     */
    rss: number;
    /** Degrees of freedom, `m - n`: the number of data points minus the number of parameters. */
    dof: number;
}

/** Options for {@link curveFit}. */
export interface CurveFitOptions extends Omit<LevenbergMarquardtOptions, 'jac' | 'sclx'> {
    /**
     * Standard deviations of the errors of `ydata`, used to weight the
     * residuals as `(f(x, p) - y) / sigma`. Must be strictly positive. By
     * default, all points have the same weight.
     */
    sigma?: Vector | number[];
    /**
     * Whether `sigma` holds absolute standard deviations, so that the
     * covariance is `inv(J^T J)`. If `false`, `sigma` only gives relative
     * weights and the covariance is scaled by the residual variance,
     * `rss/dof`, as in `scipy.optimize.curve_fit`; then, multiplying `sigma`
     * by a constant does not change `cov`.
     * @default false
     */
    absoluteSigma?: boolean;
    /**
     * Positive scaling factors for the parameters, with the same meaning as
     * `sclx` in {@link levenbergMarquardt}: ideally `sclx*p` is of order 1 at
     * the solution. If omitted, MINPACK's adaptive scaling is used.
     */
    sclx?: Vector | number[];
    /**
     * Function returning the gradient of the model with respect to the
     * parameters, `∂f/∂p` (`n` components), at a single point `x`. By
     * default, the Jacobian is approximated using forward finite differences.
     * Like in {@link levenbergMarquardt}, it is checked against finite
     * differences at `p0` if `jacCheck` is `true`.
     */
    jac?: (x: number, p: Vector) => Vector | number[];
}

/**
 * Fits a model function to data by nonlinear least squares, using the
 * Levenberg-Marquardt algorithm ({@link levenbergMarquardt}).
 *
 * It finds the parameters `p` that minimize `sum(((f(x_i, p) - y_i) /
 * sigma_i)²)`, and estimates their covariance matrix from the Jacobian at
 * the solution: `cov = s² * inv(J^T J)`, where `J` is the (weighted)
 * Jacobian and `s² = rss/dof` — or `s² = 1` if `absoluteSigma` is `true` —
 * as in `scipy.optimize.curve_fit` (with `method='lm'`). `J^T J` is
 * never formed: the inverse is computed from the pivoted QR decomposition of
 * `J`, which also detects rank deficiency.
 *
 * The model must be defined at all parameter values visited by the
 * algorithm: a trial point at which the model is not finite is rejected and
 * the trust region is reduced. The parameter vector passed to `f` is reused
 * internally and must not be retained or modified. There must be at least as
 * many data points as parameters.
 *
 * Parameters are unbounded, and the reliability of the fit depends on the
 * quality of the initial guess `p0`.
 *
 * @param f Model function `f(x, p)`, returning the predicted `y` at `x` for
 * the parameter vector `p`.
 * @param xdata Independent variable values, `m` points.
 * @param ydata Observed values, `m` points.
 * @param p0 Initial guess for the parameters, `n <= m` components.
 * @param options Configuration options; see {@link CurveFitOptions}.
 * @returns A {@link CurveFitResult} with the optimal parameters, their
 * covariance and diagnostics.
 * @throws {RangeError} If `xdata` and `ydata` (or `sigma`) have different
 * lengths, if `p0` has no components, if there are fewer data points than
 * parameters, if `jac` does not return `n` components, or for the invalid
 * options listed in {@link levenbergMarquardt}.
 * @throws {Error} If `sigma` or `sclx` contains a value that is not strictly positive.
 *
 * @example
 * ```ts
 * import { curveFit } from 'numerics-js/optimize';
 *
 * // Fit `y = a*exp(-b*t)` to noisy data.
 * const t = [0, 1, 2, 3, 4, 5];
 * const y = [3.02, 1.81, 1.12, 0.66, 0.41, 0.24];
 * const result = curveFit((t, p) => p.get(0) * Math.exp(-p.get(1) * t), t, y, [1, 1]);
 * console.log(result.p.toString());
 * console.log(result.stderr.toString());
 * ```
 *
 * Output:
 * ```text
 * Vector(3.016152409051229, 0.5027330972776392)
 * Vector(0.01154965335247648, 0.0035871308494804487)
 * ```
 */
export function curveFit(
    f: (x: number, p: Vector) => number,
    xdata: Vector | number[],
    ydata: Vector | number[],
    p0: Vector | number[],
    options: CurveFitOptions = {}
): CurveFitResult {
    const x = xdata instanceof Vector ? xdata : Vector.from(xdata);
    const y = ydata instanceof Vector ? ydata : Vector.from(ydata);
    const p0v = p0 instanceof Vector ? p0.copy() : Vector.from(p0);
    const m = x.size;
    const n = p0v.size;

    if (n === 0) {
        throw new RangeError('curveFit: p0 must have at least one component');
    }
    if (y.size !== m) {
        throw new RangeError(`curveFit: xdata and ydata must have the same length, got ${m} and ${y.size}`);
    }
    if (m < n) {
        throw new RangeError(`curveFit: at least as many data points as parameters are required, got ${m} points and ${n} parameters`);
    }

    // Weights, 1/sigma.
    const { sigma: sigmaOption, absoluteSigma = false, jac, ...lmOptions } = options;
    let sigma: Vector;
    if (sigmaOption === undefined) {
        sigma = Vector.ones(m);
    } else {
        sigma = sigmaOption instanceof Vector ? sigmaOption : Vector.from(sigmaOption);
        if (sigma.size !== m) {
            throw new RangeError(`curveFit: sigma must have the same length as ydata (${m}), got ${sigma.size}`);
        }
        for (let i = 0; i < m; i++) {
            if (!(sigma.get(i) > 0)) throw new Error('curveFit: sigma must contain strictly positive values');
        }
    }
    const unweighted = sigmaOption === undefined;

    // Weighted residuals, r_i = (f(x_i, p) - y_i)/sigma_i.
    const residualFunction = (p: Vector): Vector => {
        const r = new Vector(m);
        for (let i = 0; i < m; i++) {
            const ri = f(x.data[i], p) - y.data[i];
            r.data[i] = unweighted ? ri : ri / sigma.data[i];
        }
        return r;
    };

    // Weighted Jacobian, J_ij = (∂f/∂p_j)(x_i, p)/sigma_i.
    const jacobianFunction = jac
        ? (p: Vector): Matrix => {
              const J = new Matrix(m, n);
              for (let i = 0; i < m; i++) {
                  const g = jac(x.data[i], p);
                  const size = g instanceof Vector ? g.size : g.length;
                  if (size !== n) {
                      throw new RangeError(`curveFit: jac must return ${n} components, got ${size}`);
                  }
                  const w = unweighted ? 1 : 1 / sigma.data[i];
                  for (let j = 0; j < n; j++) J.data[i * n + j] = (g instanceof Vector ? g.data[j] : g[j]) * w;
              }
              return J;
          }
        : undefined;

    const lm = levenbergMarquardt(residualFunction, p0v, { ...lmOptions, jac: jacobianFunction });
    let nFev = lm.evaluationsFunction;
    let nJev = lm.evaluationsJacobian;

    const p = lm.x;
    const dof = m - n;
    const rss = 2 * lm.cost;
    const residuals = new Vector(m);
    for (let i = 0; i < m; i++) residuals.data[i] = -lm.fx.data[i] * (unweighted ? 1 : sigma.data[i]);

    // Jacobian at the solution (the one stored in the result may be one iteration behind).
    let J: Matrix;
    if (jacobianFunction) {
        J = jacobianFunction(p);
        nJev += 1;
    } else {
        const sclx = lmOptions.sclx === undefined ? scaleVector(p0v) : Vector.from(Array.from(lmOptions.sclx instanceof Vector ? lmOptions.sclx.data : lmOptions.sclx));
        J = jacobianForward(
            (pv) => {
                nFev += 1;
                return residualFunction(pv);
            },
            p,
            { fx: lm.fx, sclx, epsf: lmOptions.epsf }
        );
    }

    const cov = parameterCovariance(J, rss, dof, absoluteSigma, residuals.allFinite() && J.allFinite());
    const stderr = cov.diag().sqrt();

    return {
        method: lm.method.replace('Levenberg-Marquardt', 'Curve fit by Levenberg-Marquardt'),
        success: lm.success,
        message: lm.message,
        evaluationsFunction: nFev,
        evaluationsJacobian: nJev,
        iterations: lm.iterations,
        p,
        cov,
        stderr,
        residuals,
        rss,
        dof,
    };
}

/**
 * Covariance matrix of the parameters, `s² * inv(J^T J)`, from the pivoted QR
 * decomposition `J P = Q R`, as `P inv(R) inv(R)^T P^T`.
 *
 * @param J Weighted Jacobian (`m x n`, `m >= n`).
 * @param rss Weighted residual sum of squares.
 * @param dof Degrees of freedom, `m - n`.
 * @param absoluteSigma If `true`, `s² = 1`; otherwise `s² = rss/dof`.
 * @param finite Whether the residuals and the Jacobian are finite; if not, the result is `NaN`.
 * @returns The `n x n` covariance matrix; `Infinity` entries if `J` is rank-deficient or if
 * `s²` is undefined (`dof = 0` and not `absoluteSigma`); `NaN` entries if not `finite`.
 */
function parameterCovariance(J: Matrix, rss: number, dof: number, absoluteSigma: boolean, finite: boolean): Matrix {
    const n = J.cols;
    const filled = (value: number) => new Matrix(n, n).fill(value);

    if (!finite) return filled(NaN);
    if (!absoluteSigma && dof === 0) return filled(Infinity);

    const dec = J.qrPivot();
    const R = Matrix.qrPivotR(dec);

    // Numerical rank from the (nonincreasing) diagonal of R.
    const tol = Math.max(J.rows, n) * EPS * Math.abs(R.get(0, 0));
    for (let j = 0; j < n; j++) {
        if (!(Math.abs(R.get(j, j)) > tol)) return filled(Infinity);
    }

    // inv(R), column by column, then C = inv(R) inv(R)^T.
    const Rinv = new Matrix(n, n);
    const e = new Vector(n);
    for (let j = 0; j < n; j++) {
        e.fill(0);
        e.set(j, 1);
        Rinv.setCol(j, R.solveUpper(e));
    }
    const C = Rinv.matmul(Rinv.transpose());

    const s2 = absoluteSigma ? 1 : rss / dof;
    const cov = new Matrix(n, n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            cov.set(dec.perm[i], dec.perm[j], s2 * C.get(i, j));
        }
    }
    return cov;
}
