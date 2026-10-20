import { Vector, type VectorLike } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
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
   * Number of evaluations of the vector of (weighted) residuals, i.e. number
   * of calls to the model `f`, including the finite-difference evaluations
   * and, if `jacFinal` is `true`, the ones needed for the final Jacobian.
   */
  evaluationsFunction: number;
  /**
   * Number of user-supplied Jacobian evaluations, including, if `jacFinal`
   * is `true`, the one at the solution (for the covariance).
   */
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
   * residuals as `(y - f(x, p)) / sigma`. A `Vector`, a plain `number[]` or a
   * `Float64Array`, with strictly positive components and the same length as `ydata`.
   * By default, all points have the same weight.
   */
  sigma?: VectorLike;
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
  sclx?: Vector;
  /**
   * Function returning the Jacobian of the model with respect to the
   * parameters, `∂f/∂p`, at all data points: an `m x n` `Matrix` whose entry
   * `(i, j)` is `∂f_i/∂p_j`, where `f_i = f(x, p)_i`. Note that this is the
   * Jacobian of the model, not of the residuals `y - f`, as in
   * `scipy.optimize.curve_fit`. The arguments `x` and `p` must not be modified.
   * By default, the Jacobian is approximated using forward finite differences.
   * Like in {@link levenbergMarquardt}, it is checked against finite
   * differences at `p0` if `jacCheck` is `true`.
   */
  jac?: (x: Vector, p: Vector) => Matrix;
  /**
   * Whether to evaluate the Jacobian at the returned parameters, to compute
   * the covariance from it. If `false`, the covariance is computed from the
   * last Jacobian evaluated during the iterations, which may be one step
   * behind the solution (as in `scipy.optimize.curve_fit`), at no extra
   * cost. If `true`, the covariance is exact, at the cost of `n`
   * evaluations of the residuals (or one call to `jac`). See
   * {@link LevenbergMarquardtOptions.jacFinal}.
   * @default false
   */
  jacFinal?: boolean;
}

/**
 * Fits a model function to data by nonlinear least squares, using the Levenberg-Marquardt
 * algorithm ({@link levenbergMarquardt}).
 *
 * It finds the parameters `p` that minimize `sum(((y_i - f(x_i, p)) / sigma_i)²)`, and
 * estimates their covariance matrix from the Jacobian near the solution:
 * `cov = s² * inv(J^T J)`, where `J` is the (weighted) Jacobian and `s² = rss/dof` or
 * `s² = 1` if `absoluteSigma` is `true`. By default, `J` is the last Jacobian of the
 * iterations, which is one step behind the solution if the last step was accepted; with
 * `jacFinal: true`, it is evaluated at the solution. `J^T J` is never formed: the inverse
 * is computed from the pivoted QR decomposition of `J`, which also detects rank deficiency.
 *
 * The model is vectorized: it is called with all the data points at once, `f(xdata, p)`, and
 * must return the predictions at all of them. The model must be defined at all parameter
 * values visited by the algorithm: a trial point at which the model is not finite is rejected
 * and the trust region is reduced. The vectors passed to `f` (and to `jac`) are reused
 * internally and must not be retained or modified. There must be at least as many data points
 * as parameters.
 *
 * Parameters are unbounded, and the reliability of the fit depends on the quality of the
 * initial guess `p0`.
 *
 * @param f Model function `f(x, p)`, returning a `Vector` with the predicted `y` at every
 * point of `x` (the same length as `x`) for the parameter vector `p`.
 * @param xdata Independent variable values, `m` points: a `Vector`, a plain `number[]` or a
 * `Float64Array`. The model `f` receives them as a `Vector`.
 * @param ydata Observed values, `m` points, in any of the forms accepted for `xdata`.
 * @param p0 Initial guess for the parameters, a `Vector` with `n <= m` components.
 * @param options Configuration options; see {@link CurveFitOptions}.
 * @returns A {@link CurveFitResult} with the optimal parameters, their covariance and
 * diagnostics.
 * @throws {TypeError} If `xdata`, `ydata` or `options.sigma` is not a `VectorLike`.
 * @throws {RangeError} If `xdata` and `ydata` (or `sigma`) have different lengths, if `p0`
 * has no components, if there are fewer data points than parameters, if `f` does not return
 * `m` components, if `jac` does not return an `m x n` matrix, or for the invalid options
 * listed in {@link levenbergMarquardt}.
 * @throws {Error} If `sigma` or `sclx` contains a value that is not strictly positive.
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { curveFit } from 'numerics-js/optimize';
 *
 * // Fit `y = a*exp(-b*t)` to noisy data.
 * const t = Vector.from([0, 1, 2, 3, 4, 5]);
 * const y = Vector.from([3.02, 1.81, 1.12, 0.66, 0.41, 0.24]);
 * const model = (t: Vector, p: Vector) => t.map((ti) => p.get(0) * Math.exp(-p.get(1) * ti));
 * const result = curveFit(model, t, y, Vector.from([1, 1]));
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   method: 'Curve fit by Levenberg-Marquardt (Trust region, Adaptive scaling)',
 *   success: true,
 *   message: 'Both the actual and predicted relative reductions in ||f||² are at most tolf.',
 *   evaluationsFunction: 23,
 *   evaluationsJacobian: 0,
 *   iterations: 7,
 *   p: Vector {
 *     data: Float64Array(2) [ 3.0161524090313265, 0.5027330972728501 ]
 *   },
 *   cov: Matrix {
 *     _rows: 2,
 *     _cols: 2,
 *     data: Float64Array(4) [
 *       0.00013339448303097227,
 *       0.000021832629224564343,
 *       0.000021832629224564343,
 *       0.000012867500506506627
 *     ]
 *   },
 *   stderr: Vector {
 *     data: Float64Array(2) [ 0.011549652939849416, 0.0035871298424376315 ]
 *   },
 *   residuals: Vector {
 *     data: Float64Array(6) [
 *       0.0038475909686734866,
 *       -0.014395838984609277,
 *       0.016468173379435402,
 *       -0.007499051654432876,
 *       0.006246367153699139,
 *       -0.004220565755919725
 *     ]
 *   },
 *   rss: 0.0006063109244228651,
 *   dof: 4
 * }
 * ```
 */
export function curveFit(
  f: (x: Vector, p: Vector) => Vector,
  xdata: VectorLike,
  ydata: VectorLike,
  p0: Vector,
  options: CurveFitOptions = {}
): CurveFitResult {
  const x = Vector.asVector(xdata);
  const y = Vector.asVector(ydata);
  const p0v = p0.copy();
  const m = x.size;
  const n = p0v.size;

  if (n === 0) {
    throw new RangeError('curveFit: p0 must have at least one component');
  }
  if (y.size !== m) {
    throw new RangeError(
      `curveFit: xdata and ydata must have the same length, got ${m} and ${y.size}`
    );
  }
  if (m < n) {
    throw new RangeError(
      `curveFit: at least as many data points as parameters are required, got ${m} points and ${n} parameters`
    );
  }

  // Weights, 1/sigma.
  const {
    sigma: sigmaOption,
    absoluteSigma = false,
    jac,
    jacFinal = false,
    ...lmOptions
  } = options;
  let sigma: Vector;
  if (sigmaOption === undefined) {
    sigma = Vector.ones(m);
  } else {
    sigma = Vector.asVector(sigmaOption);
    if (sigma.size !== m) {
      throw new RangeError(
        `curveFit: sigma must have the same length as ydata (${m}), got ${sigma.size}`
      );
    }
    for (let i = 0; i < m; i++) {
      if (!(sigma.get(i) > 0))
        throw new Error('curveFit: sigma must contain strictly positive values');
    }
  }
  const unweighted = sigmaOption === undefined;

  // Weighted residuals, r_i = (y_i - f(x_i, p))/sigma_i.
  const residualFunction = (p: Vector): Vector => {
    const fx = f(x, p);
    if (fx.size !== m) {
      throw new RangeError(
        `curveFit: f must return one value per data point (${m}), got ${fx.size}`
      );
    }
    const r = new Vector(m);
    for (let i = 0; i < m; i++) {
      const ri = y.data[i] - fx.data[i];
      r.data[i] = unweighted ? ri : ri / sigma.data[i];
    }
    return r;
  };

  // Jacobian of the weighted residuals, J_ij = -(∂f_i/∂p_j)(x, p)/sigma_i. The user-supplied `jac`
  // returns ∂f/∂p, whereas the residuals are y - f, hence the minus sign.
  const jacobianFunction = jac
    ? (p: Vector): Matrix => {
        const Jf = jac(x, p);
        if (Jf.rows !== m || Jf.cols !== n) {
          throw new RangeError(
            `curveFit: jac must return a ${m}x${n} matrix, got ${Jf.rows}x${Jf.cols}`
          );
        }
        const J = new Matrix(m, n);
        for (let i = 0; i < m; i++) {
          const w = unweighted ? 1 : 1 / sigma.data[i];
          for (let j = 0; j < n; j++) J.data[i * n + j] = -Jf.data[i * n + j] * w;
        }
        return J;
      }
    : undefined;

  const lm = levenbergMarquardt(residualFunction, p0v, {
    ...lmOptions,
    jac: jacobianFunction,
    jacFinal
  });

  const p = lm.x;
  const dof = m - n;
  const rss = 2 * lm.cost;
  const residuals = new Vector(m);
  for (let i = 0; i < m; i++) residuals.data[i] = lm.fx.data[i] * (unweighted ? 1 : sigma.data[i]);

  // Weighted Jacobian at the solution (or at the last iteration, if `jacFinal` is `false`).
  // It is `null` only if the solver failed at the initial guess.
  const J = lm.Jx;

  const finite = J !== null && J.allFinite() && residuals.allFinite();
  const cov = finite ? parameterCovariance(J, rss, dof, absoluteSigma) : new Matrix(n, n).fill(NaN);
  const stderr = cov.diag().sqrt();

  return {
    method: lm.method.replace('Levenberg-Marquardt', 'Curve fit by Levenberg-Marquardt'),
    success: lm.success,
    message: lm.message,
    evaluationsFunction: lm.evaluationsFunction,
    evaluationsJacobian: lm.evaluationsJacobian,
    iterations: lm.iterations,
    p,
    cov,
    stderr,
    residuals,
    rss,
    dof
  };
}

/**
 * Covariance matrix of the parameters, `s² * inv(J^T J)`, from the pivoted QR
 * decomposition `J P = Q R`, as `P inv(R) inv(R)^T P^T`.
 *
 * @param J Weighted Jacobian (`m x n`, `m >= n`), with finite entries.
 * @param rss Weighted residual sum of squares, finite.
 * @param dof Degrees of freedom, `m - n`.
 * @param absoluteSigma If `true`, `s² = 1`; otherwise `s² = rss/dof`.
 * @returns The `n x n` covariance matrix; `Infinity` entries if `J` is rank-deficient or if
 * `s²` is undefined (`dof = 0` and not `absoluteSigma`).
 */
function parameterCovariance(J: Matrix, rss: number, dof: number, absoluteSigma: boolean): Matrix {
  const n = J.cols;
  const filled = (value: number) => new Matrix(n, n).fill(value);

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
