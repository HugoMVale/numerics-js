import { Vector, type VectorLike } from '../linalg/Vector.js';
import { Matrix, type QRPivotDecomposition } from '../linalg/Matrix.js';
import { jacobianForward, scaleVector } from '../numdiff.js';
import { enorm, lmpar } from './lmpar.js';

const EPS = Number.EPSILON;
const SQRT_EPS = Math.sqrt(EPS);

// Constants of the MINPACK trust-region logic.
const P1 = 0.1;
const P5 = 0.5;
const P25 = 0.25;
const P75 = 0.75;
const P0001 = 1e-4;

/**
 * Result of {@link levenbergMarquardt}.
 */
export interface LevenbergMarquardtResult {
  /** Descriptive method name, including the scaling strategy used. */
  method: string;
  /**
   * Whether the solver met one of the convergence criteria (`tolf`, `tolx`
   * or `tolg`), or stopped at machine precision with a negligible residual
   * (`||f(x)|| <= sqrt(eps)*||f(x0)||`). A `tolx` termination may also be
   * reported when progress has merely become slow; inspect `message`, `fx`
   * and `cost`. It is `false` if the evaluation budget is exhausted, or if
   * the solver stops at machine precision while the residual is still
   * non-negligible (so that no better point can be reached, whether or not
   * `x` is a minimizer of a nonzero-residual problem).
   */
  success: boolean;
  /** Message describing the result or error. */
  message: string;
  /**
   * Number of evaluations of `f`, including those made by the
   * finite-difference Jacobian (also the final one, if `jacFinal` is
   * `true`) and by `jacCheck`.
   */
  evaluationsFunction: number;
  /** Number of user-supplied Jacobian evaluations. */
  evaluationsJacobian: number;
  /** Number of accepted steps. */
  iterations: number;
  /** The solution vector (or best estimate, if not converged). */
  x: Vector;
  /** Residual vector `f(x)` at the returned solution. */
  fx: Vector;
  /** Cost function `0.5*||f(x)||²` at the returned solution. */
  cost: number;
  /**
   * Last evaluated Jacobian matrix (`m x n`). By default, it may be one
   * iteration behind `x` and `fx`, because the Jacobian is evaluated at the
   * start of each iteration. If the option `jacFinal` is `true`, it is
   * evaluated at the returned `x` (and may then contain non-finite values,
   * without this affecting `success`). It is `null` if no Jacobian was
   * evaluated.
   */
  Jx: Matrix | null;
}

/** Options for {@link levenbergMarquardt}. */
export interface LevenbergMarquardtOptions {
  /**
   * Tolerance for the relative reduction of the sum of squares. The
   * algorithm terminates when both the actual and the predicted relative
   * reductions of `||f||²` are at most `tolf`. Note that this is a
   * different criterion from the residual-norm test called `tolf` in
   * `quasiNewton`.
   * @default sqrt(eps) ≈ 1.49e-8
   */
  tolf?: number;
  /**
   * Tolerance for the relative step size. The algorithm terminates when the
   * trust-region radius is at most `tolx*||D*x||`, where `D` is the diagonal
   * scaling matrix. If too large, the algorithm may terminate before
   * reaching a minimum.
   * @default sqrt(eps) ≈ 1.49e-8
   */
  tolx?: number;
  /**
   * Tolerance for orthogonality between the residual and the columns of
   * the Jacobian. The algorithm terminates when the cosine of the angle
   * between `f` and every column of the Jacobian is at most `tolg`, in
   * absolute value.
   * @default 0
   */
  tolg?: number;
  /**
   * Maximum number of evaluations of `f`, including those used by the
   * finite-difference Jacobian. It is checked between trial steps, so
   * the actual count may exceed it by the evaluations of one Jacobian.
   * Must be a finite, strictly positive integer.
   * @default 100*(n+1)
   */
  maxFunEvals?: number;
  /**
   * Maximum number of accepted steps. Must be a strictly positive integer
   * or `Infinity`.
   * @default Infinity
   */
  maxIter?: number;
  /**
   * Positive scaling factors for the components of `x`, with the same
   * meaning as in `quasiNewton`: ideally `sclx*x` is of order 1 near the
   * solution. If provided, they define a fixed trust-region metric
   * (`D = diag(sclx)`) and set the finite-difference steps. If omitted,
   * the metric is MINPACK's adaptive scaling, based on the norms of the
   * Jacobian columns, and the finite-difference steps are scaled from `x0`
   * via {@link scaleVector}. A `Vector`, a plain `number[]` or a `Float64Array`.
   */
  sclx?: VectorLike;
  /**
   * Factor determining the initial trust-region radius, `trustLenFactor *
   * ||D*x0||` (or `trustLenFactor` itself if `x0` is zero). Values
   * between 0.1 and 100 are generally appropriate: a smaller value makes
   * the first steps more cautious.
   * @default 100
   */
  trustLenFactor?: number;
  /**
   * Machine precision of the function values. If `undefined`, machine
   * precision of the 64-bit floating-point type is assumed. If the number
   * of reliable base-10 digits in `f`'s results is `n`, then `epsf` is
   * approximately `10^(-n)`.
   */
  epsf?: number;
  /**
   * Function to compute the `m x n` Jacobian of `f`. By default, the
   * Jacobian is approximated using forward finite differences
   * ({@link jacobianForward}), which costs `n` evaluations of `f`. In that
   * case, setting `epsf` appropriately is essential.
   */
  jac?: (x: Vector) => Matrix;
  /**
   * If `true` and `jac` is provided, `jac`'s result at `x0` is checked
   * against a forward finite-difference approximation, to help catch
   * errors in the user-provided Jacobian (at the cost of `n` extra
   * function evaluations).
   * @default true
   */
  jacCheck?: boolean;
  /**
   * Whether to make sure that the returned Jacobian `Jx` is evaluated at the returned
   * solution `x`. If `false`, `Jx` is the last Jacobian evaluated during the iterations,
   * which is one step behind `x` whenever the last iteration ended with an accepted step; no
   * extra evaluations are made. If `true`, the Jacobian is evaluated at `x` after the
   * iterations have ended, unless it already was, at the cost of `n` evaluations of `f` (or
   * one call to `jac`). This extra evaluation is not subject to `maxFunEvals`. An exact
   * Jacobian is useful, e.g., to estimate the covariance of the solution.
   * @default false
   */
  jacFinal?: boolean;
}

/**
 * Minimizes the sum of squares of a vector function, `0.5*||f(x)||²`, using a trust-region
 * Levenberg-Marquardt algorithm.
 *
 * The residual vector `f(x)` has `m >= n` components. The method uses derivative information,
 * supplied through `jac` or approximated by finite differences.
 *
 * This is a MINPACK-style implementation (Moré, 1978): each step solves the damped
 * least-squares problem `min ||J*p + f||² + par*||D*p||²` with the Levenberg-Marquardt
 * parameter `par` chosen so that `||D*p||` fits inside a trust region whose radius is
 * adapted from the agreement between actual and predicted reductions. The Jacobian is
 * factored by a pivoted QR decomposition ({@link Matrix.qrPivot}), so the cost per iteration
 * is `O(m*n²)` and the Jacobian is never squared. Rank-deficient Jacobians are handled
 * gracefully.
 *
 * A trial point at which `f` returns non-finite values (`NaN` or `±Infinity`) is treated as a
 * rejected step and the trust region is reduced. If `f(x0)` or a Jacobian is not finite, the
 * solver returns a failure result (`success: false`).
 *
 * Results are validated against `scipy.optimize.leastsq` (MINPACK). See the
 * [comparison table](https://github.com/HugoMVale/numerics-js/blob/main/benchmarks/optimize/levenbergMarquardt.scipy.benchmark.md).
 *
 * **References**
 *
 * - J.J. Moré, "The Levenberg-Marquardt algorithm: implementation and
 *   theory", in Numerical Analysis, Lecture Notes in Mathematics 630,
 *   Springer, 1978, pp. 105-116.
 * - J.J. Moré, B.S. Garbow, K.E. Hillstrom, "User guide for MINPACK-1",
 *   Argonne National Laboratory, ANL-80-74, 1980.
 *
 * @param f Residual function, returning `m >= n` components.
 * @param x0 Initial guess for the minimizer: a `Vector`, a plain `number[]` or a `Float64Array`.
 * @param options Configuration options for the solver.
 * @returns A `LevenbergMarquardtResult` with the solution and diagnostics.
 * @throws {TypeError} If `x0` or `options.sclx` is not a `VectorLike`.
 * @throws {RangeError} If `x0` has no components, if `f(x0)` has fewer
 * components than `x0` (or `f` later returns a different number of
 * components), if `tolf`, `tolx`, `tolg` or `trustLenFactor` are not
 * valid, if `maxFunEvals` or `maxIter` are not strictly positive integers
 * (`maxIter` may also be `Infinity`), if `sclx` does not have the size of
 * `x0`, or if `jac` does not return an `m`×`n` matrix.
 * @throws {Error} If `sclx` contains a value that is not strictly positive.
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { levenbergMarquardt } from 'numerics-js/optimize';
 *
 * // Fit `y = a*exp(-b*t)` to noisy data by minimizing the residuals.
 * const t = [0, 1, 2, 3, 4, 5];
 * const y = [3.02, 1.81, 1.12, 0.66, 0.41, 0.24];
 * const residuals = (p: Vector) =>
 *     Vector.from(t.map((ti, i) => p.get(0) * Math.exp(-p.get(1) * ti) - y[i]));
 * const result = levenbergMarquardt(residuals, [1, 1]);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   method: 'Levenberg-Marquardt (Trust region, Adaptive scaling)',
 *   success: true,
 *   message: 'Both the actual and predicted relative reductions in ||f||² are at most tolf.',
 *   evaluationsFunction: 23,
 *   evaluationsJacobian: 0,
 *   iterations: 7,
 *   x: Vector {
 *     data: Float64Array(2) [ 3.016152409051229, 0.5027330972776392 ]
 *   },
 *   fx: Vector {
 *     data: Float64Array(6) [
 *       -0.0038475909487711846,
 *       0.014395838987910414,
 *       -0.01646817338272344,
 *       0.007499051649247357,
 *       -0.006246367158769306,
 *       0.004220565751683336
 *     ]
 *   },
 *   cost: 0.0003031554622114309,
 *   Jx: Matrix {
 *     _rows: 6,
 *     _cols: 2,
 *     data: Float64Array(12) [
 *       1,
 *       0,
 *       0.6048753122268176,
 *       -1.8243959695100784,
 *       0.3658741414182455,
 *       -2.207064151763916,
 *       0.2213082338240245,
 *       -2.0024979040026665,
 *       0.13386388853339576,
 *       -1.6150153689086437,
 *       0.0809709609297824,
 *       -1.2211036495864391
 *     ]
 *   }
 * }
 * ```
 */
export function levenbergMarquardt(
  f: (x: Vector) => Vector,
  x0: VectorLike,
  options: LevenbergMarquardtOptions = {}
): LevenbergMarquardtResult {
  // Owned copy: it may be returned as the result, so it must never alias the caller's data.
  let x = Vector.asVector(x0, true);
  const n = x.size;
  if (n === 0) {
    throw new RangeError('levenbergMarquardt: x0 must have at least one component');
  }

  const tolf = options.tolf ?? SQRT_EPS;
  const tolx = options.tolx ?? SQRT_EPS;
  const tolg = options.tolg ?? 0;
  for (const [name, value] of [
    ['tolf', tolf],
    ['tolx', tolx],
    ['tolg', tolg]
  ] as const) {
    if (!(value >= 0)) {
      throw new RangeError(
        `levenbergMarquardt: ${name} must be a nonnegative number, got ${value}`
      );
    }
  }
  const maxFunEvals = options.maxFunEvals ?? 100 * (n + 1);
  if (!(Number.isInteger(maxFunEvals) && maxFunEvals >= 1)) {
    throw new RangeError(
      `levenbergMarquardt: maxFunEvals must be a finite strictly positive integer, got ${maxFunEvals}`
    );
  }
  const maxIter = options.maxIter ?? Infinity;
  if (!(maxIter === Infinity || (Number.isInteger(maxIter) && maxIter >= 1))) {
    throw new RangeError(
      `levenbergMarquardt: maxIter must be a strictly positive integer or Infinity, got ${maxIter}`
    );
  }
  const factor = options.trustLenFactor ?? 100;
  if (!(factor > 0 && Number.isFinite(factor))) {
    throw new RangeError(
      `levenbergMarquardt: trustLenFactor must be a finite strictly positive number, got ${factor}`
    );
  }

  let userSclx: Vector | undefined;
  if (options.sclx !== undefined) {
    userSclx = Vector.asVector(options.sclx, true);
    if (userSclx.size !== n) {
      throw new RangeError(
        `levenbergMarquardt: sclx must have ${n} components, got ${userSclx.size}`
      );
    }
    for (let i = 0; i < n; i++) {
      if (!(userSclx.get(i) > 0))
        throw new Error('levenbergMarquardt: sclx must contain strictly positive values');
    }
  }
  const sclx = userSclx ?? scaleVector(x);

  const method = `Levenberg-Marquardt (Trust region, ${userSclx ? 'Fixed' : 'Adaptive'} scaling)`;

  let nFev = 0;
  let nJev = 0;
  let m = -1;

  // All evaluations of `f` (including those made by finite differences) go
  // through this wrapper, which counts them and checks the output size.
  const countedF = (xv: Vector): Vector => {
    nFev += 1;
    const r = f(xv);
    if (m >= 0 && r.size !== m) {
      throw new RangeError(`levenbergMarquardt: f must return ${m} components, got ${r.size}`);
    }
    return r;
  };

  let fvec = countedF(x);
  m = fvec.size;
  if (m < n) {
    throw new RangeError(
      `levenbergMarquardt: f(x0) must have at least ${n} components (as many as x0), got ${m}`
    );
  }

  let J: Matrix | null = null;

  const failure = (message: string): LevenbergMarquardtResult => ({
    method,
    success: false,
    message,
    evaluationsFunction: nFev,
    evaluationsJacobian: nJev,
    iterations: 0,
    x,
    fx: fvec,
    cost: 0.5 * enorm(fvec.data) ** 2,
    Jx: J
  });

  if (!fvec.allFinite()) {
    return failure('f(x0) contains non-finite values.');
  }

  // Evaluates the Jacobian at `xv` (with `fx = f(xv)`), either with the user-supplied `jac`
  // or by forward finite differences.
  const evaluateJacobian = (xv: Vector, fx: Vector): Matrix => {
    if (options.jac) {
      const Jv = options.jac(xv);
      nJev += 1;
      if (Jv.rows !== m || Jv.cols !== n) {
        throw new RangeError(
          `levenbergMarquardt: jac must return a ${m}x${n} matrix, got ${Jv.rows}x${Jv.cols}`
        );
      }
      return Jv;
    }
    return jacobianForward(countedF, xv, { fx, sclx, epsf: options.epsf });
  };

  // Jacobian at x0, optionally checked against finite differences.
  J = evaluateJacobian(x, fvec);
  if (!J.allFinite()) {
    return failure('Jacobian at x0 contains non-finite values.');
  }
  if (options.jac && (options.jacCheck ?? true)) {
    const Jfd = jacobianForward(countedF, x, { fx: fvec, sclx, epsf: options.epsf });
    const epsfEff = options.epsf !== undefined ? Math.max(options.epsf, EPS) : EPS;
    const tol = 1e2 * Math.sqrt(epsfEff);
    if (!J.allClose(Jfd, tol, tol)) {
      return failure(
        'User-provided Jacobian `jac` does not match finite-difference approximation.'
      );
    }
  }
  let Jnext: Matrix | null = J;

  // Working storage, reused across iterations.
  const dec: QRPivotDecomposition = {
    QR: new Matrix(m, n),
    tau: new Vector(n),
    perm: new Array<number>(n)
  };
  const qtbFull = new Vector(m);
  const rbuf = new Float64Array(n * n);
  const qtf = new Float64Array(n);
  const acnorm = new Float64Array(n);
  const diag = new Float64Array(n);
  const step = new Float64Array(n);
  const sdiag = new Float64Array(n);
  const w1 = new Float64Array(n);
  const w2 = new Float64Array(n);
  const w3 = new Float64Array(n);

  let fnorm = enorm(fvec.data);
  const fnorm0 = fnorm;
  let xnorm = 0;
  let delta = 0;
  let par = 0;
  let gnorm: number;
  let iterations = 0;
  let info = 0;
  // Whether `x` has changed since `J` was evaluated.
  let jacStale = false;

  outer: for (;;) {
    // Jacobian at the current iterate.
    if (Jnext) {
      J = Jnext;
      Jnext = null;
    } else {
      J = evaluateJacobian(x, fvec);
      if (!J.allFinite()) {
        return {
          ...failure('Jacobian at the current iterate contains non-finite values.'),
          iterations
        };
      }
    }
    jacStale = false;

    // Pivoted QR decomposition of the Jacobian, and norms of its columns.
    J.qrPivot(dec);
    acnorm.fill(0);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) acnorm[j] += J.data[i * n + j] ** 2;
    }
    for (let j = 0; j < n; j++) acnorm[j] = Math.sqrt(acnorm[j]);

    // On the first iteration, set the scaling and the initial step bound.
    if (iterations === 0) {
      for (let j = 0; j < n; j++) {
        diag[j] = userSclx ? userSclx.get(j) : acnorm[j] === 0 ? 1 : acnorm[j];
        w1[j] = diag[j] * x.get(j);
      }
      xnorm = enorm(w1, n);
      delta = factor * xnorm;
      if (delta === 0) delta = factor;
    }

    // First n components of Q^T f.
    Matrix.qrPivotMulQt(dec, fvec, qtbFull);
    for (let j = 0; j < n; j++) qtf[j] = qtbFull.data[j];

    // Norm of the scaled gradient.
    gnorm = 0;
    if (fnorm !== 0) {
      for (let j = 0; j < n; j++) {
        const l = dec.perm[j];
        if (acnorm[l] !== 0) {
          let sum = 0;
          for (let i = 0; i <= j; i++) sum += dec.QR.data[i * n + j] * (qtf[i] / fnorm);
          gnorm = Math.max(gnorm, Math.abs(sum / acnorm[l]));
        }
      }
    }
    if (gnorm <= tolg) {
      info = 4;
      break;
    }

    // Rescale if necessary.
    if (!userSclx) {
      for (let j = 0; j < n; j++) diag[j] = Math.max(diag[j], acnorm[j]);
    }

    // The upper triangle of R, as the working matrix of lmpar.
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) rbuf[i * n + j] = dec.QR.data[i * n + j];
    }

    // Inner loop: repeat until a step is accepted.
    let ratio: number;
    do {
      // Levenberg-Marquardt parameter and step.
      par = lmpar(n, rbuf, dec.perm, diag, qtf, delta, par, step, sdiag, w1, w2, w3);
      const xTrial = new Vector(n);
      for (let j = 0; j < n; j++) {
        step[j] = -step[j];
        xTrial.data[j] = x.data[j] + step[j];
        w1[j] = diag[j] * step[j];
      }
      const pnorm = enorm(w1, n);

      // On the first iteration, adjust the initial step bound.
      if (iterations === 0) delta = Math.min(delta, pnorm);

      // Function at the trial point.
      const fTrial = countedF(xTrial);
      const fnorm1 = enorm(fTrial.data);

      // Scaled actual reduction. If the norm is not finite, the comparison fails, so
      // `actred = -1` and the step is rejected.
      let actred = -1;
      if (P1 * fnorm1 < fnorm) {
        const d1 = fnorm1 / fnorm;
        actred = 1 - d1 * d1;
      }

      // Scaled predicted reduction and scaled directional derivative.
      w1.fill(0);
      for (let j = 0; j < n; j++) {
        const temp = step[dec.perm[j]];
        for (let i = 0; i <= j; i++) w1[i] += rbuf[i * n + j] * temp;
      }
      const temp1 = enorm(w1, n) / fnorm;
      const temp2 = (Math.sqrt(par) * pnorm) / fnorm;
      const prered = temp1 * temp1 + (temp2 * temp2) / P5;
      const dirder = -(temp1 * temp1 + temp2 * temp2);

      // Ratio of the actual to the predicted reduction.
      ratio = prered !== 0 ? actred / prered : 0;

      // Update the step bound.
      if (ratio <= P25) {
        let temp = actred >= 0 ? P5 : (P5 * dirder) / (dirder + P5 * actred);
        if (P1 * fnorm1 >= fnorm || temp < P1) temp = P1;
        delta = temp * Math.min(delta, pnorm / P1);
        par /= temp;
      } else if (par === 0 || ratio >= P75) {
        delta = pnorm / P5;
        par = P5 * par;
      }

      // Successful iteration: update x, f and their norms.
      if (ratio >= P0001) {
        x = xTrial;
        fvec = fTrial;
        for (let j = 0; j < n; j++) w1[j] = diag[j] * x.data[j];
        xnorm = enorm(w1, n);
        fnorm = fnorm1;
        iterations += 1;
        jacStale = true;
      }

      // Tests for convergence.
      const smallReductions = Math.abs(actred) <= tolf && prered <= tolf && P5 * ratio <= 1;
      if (smallReductions) info = 1;
      if (delta <= tolx * xnorm) info = 2;
      if (smallReductions && info === 2) info = 3;
      if (info !== 0) break outer;

      // Tests for termination and stringent tolerances.
      if (nFev >= maxFunEvals) info = 5;
      if (Math.abs(actred) <= EPS && prered <= EPS && P5 * ratio <= 1) info = 6;
      if (delta <= EPS * xnorm) info = 7;
      if (gnorm <= EPS) info = 8;
      if (info !== 0) break outer;
      if (ratio >= P0001 && iterations >= maxIter) {
        info = 9;
        break outer;
      }
    } while (ratio < P0001);
  }

  // Jacobian at the returned solution, unless the last one was evaluated there already.
  if (options.jacFinal && jacStale) {
    J = evaluateJacobian(x, fvec);
  }

  let success = true;
  let message: string;
  switch (info) {
    case 1:
      message = 'Both the actual and predicted relative reductions in ||f||² are at most tolf.';
      break;
    case 2:
      message =
        'The trust-region radius is at most tolx*||D*x||: `x` may be an approximate minimizer, ' +
        'but it is also possible that the algorithm is making slow progress or that `tolx` is too large.';
      break;
    case 3:
      message =
        'Both the actual and predicted relative reductions in ||f||² are at most tolf, ' +
        'and the trust-region radius is at most tolx*||D*x||.';
      break;
    case 4:
      message =
        'The cosine of the angle between f and every column of the Jacobian is at most tolg.';
      break;
    case 5:
      success = false;
      message = `Maximum number of function evaluations (${maxFunEvals}) reached.`;
      break;
    case 6:
    case 7:
    case 8: {
      // The tolerances are below what machine precision allows, so no further progress is
      // possible. This is a success only if the residual is already negligible, i.e. the
      // problem has been solved to machine precision (MINPACK itself reports these as failures).
      success = fnorm <= SQRT_EPS * fnorm0;
      const reason =
        info === 6
          ? 'no further reduction in ||f||² is possible (tolf is too small)'
          : info === 7
            ? 'no further improvement in `x` is possible (tolx is too small)'
            : 'f is orthogonal to the columns of the Jacobian to machine precision (tolg is too small)';
      message = success
        ? `Converged to machine precision: ${reason}.`
        : `Stopped at machine precision with a non-negligible residual: ${reason}.`;
      break;
    }
    default:
      success = false;
      message = `Maximum number of iterations (${maxIter}) reached.`;
  }

  return {
    method,
    success,
    message,
    evaluationsFunction: nFev,
    evaluationsJacobian: nJev,
    iterations,
    x,
    fx: fvec,
    cost: 0.5 * enorm(fvec.data) ** 2,
    Jx: J
  };
}
