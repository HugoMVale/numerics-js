import { Vector, type VectorLike } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import { gradientForward, hessianForward, jacobianForward, scaleVector } from '../numdiff.js';
import { lineSearch } from '../roots/lineSearch.js';
import { dogleg } from '../roots/dogleg.js';
import { relativeStepLength } from '../roots/common.js';
import { validateMaxIter, validateTolerance } from '../roots/validation.js';
import type { GlobalMethod, GlobalStepContext, GlobalStepResult } from '../roots/types.js';

const EPS = Number.EPSILON;
const SQRT_EPS = Math.sqrt(EPS);

/** Number of consecutive maximum-length global steps after which the iteration is abandoned. */
const MAX_CONSECUTIVE_MAX_STEPS = 5;

/** Form in which the BFGS update of the Hessian approximation is carried out. */
export type BfgsMethod = 'factored' | 'unfactored' | 'inverse';

/** Options for {@link quasiNewton}. */
export interface QuasiNewtonOptions {
  /**
   * Tolerance for the scaled step size. The algorithm terminates when the scaled
   * distance between two successive iterates `||Δx/max(x, 1/sclx)||∞` is below this
   * threshold. If too large, the algorithm may terminate prematurely. A value on the order
   * of `eps^(2/3)` is typically recommended. Must be finite and strictly positive.
   * @default 1e-10
   */
  tolx?: number;
  /**
   * Tolerance for the scaled gradient. This is the main convergence criterion: the
   * algorithm terminates when `max_i |∇f(x)_i| * max(|x_i|, 1/sclx_i) / max(|f(x)|, 1/sclf)`
   * is below this threshold. If too large, the algorithm may terminate prematurely. A value
   * on the order of `eps^(1/3)` is typically recommended. Must be finite and non-negative.
   * @default 1e-5
   */
  tolg?: number;
  /**
   * Positive scaling factors for the components of `x`. Ideally chosen so that `sclx*x` is
   * of order 1 near the solution for all components. A `Vector`, a plain `number[]` or a
   * `Float64Array`, with one factor per component of `x`. By default, the scaling is
   * determined from `x0` via {@link scaleVector}.
   */
  sclx?: VectorLike;
  /**
   * Positive scaling factor for the function values. Ideally chosen so that `sclf*f(x)` is
   * of order 1 across the domain of interest. It enters the scaled-gradient criterion
   * through `max(|f(x)|, 1/sclf)`; if too small a value is provided, the algorithm may
   * terminate prematurely. Must be finite and strictly positive.
   * @default 1
   */
  sclf?: number;
  /**
   * Maximum number of outer quasi-Newton iterations. Must be a finite, strictly positive
   * integer.
   * @default 100
   */
  maxIter?: number;
  /**
   * Factor determining the maximum allowable scaled step length `||sclx*Δx||₂` for global
   * methods: the limit is `maxLenFactor` times the larger of `||sclx*x0||₂` and
   * `||sclx||₂`. Prevents steps that would overflow, leave the domain of interest, or
   * diverge. Must be strictly positive (`Infinity` removes the limit).
   * @default 1e3
   */
  maxLenFactor?: number;
  /**
   * Initial trust-region radius for the dogleg global method. The value is capped at the
   * maximum step length; if omitted or non-positive, the length of the initial Cauchy step
   * (along the scaled gradient) is used. Ignored by the other global methods.
   */
  trustLen?: number;
  /**
   * Machine precision of the function values, used to choose the step of the
   * finite-difference derivatives. If the number of reliable base-10 digits in `f`'s
   * results is `n`, then `epsf` is approximately `10^(-n)`. Values below the machine
   * precision of the 64-bit floating-point type are raised to it. Must be finite and
   * non-negative.
   * @default Number.EPSILON
   */
  epsf?: number;
  /**
   * Global strategy to improve convergence from remote starting points. With
   * `'line-search'`, the search direction is the quasi-Newton step and the step length is
   * found by backtracking until the Armijo condition holds. With `'dogleg'`, a trust-region
   * double-dogleg strategy computes both the direction and the length of the step. With
   * `null`, no global strategy is used and the full quasi-Newton step is taken at each
   * iteration.
   * @default 'dogleg'
   */
  globalMethod?: GlobalMethod;
  /**
   * Whether to update the Hessian approximation with the BFGS positive-definite secant
   * formula. If `false`, the Hessian is evaluated anew at each iteration, using `hess` or
   * finite differences. If `true`, the BFGS update avoids the cost of a full evaluation; it
   * typically increases the number of iterations, but decreases the total number of function
   * evaluations when the Hessian would otherwise be approximated by finite differences.
   * By default, the BFGS update is used if no `hess` is provided, and is not used otherwise.
   * @default hess === undefined
   */
  bfgsUpdate?: boolean;
  /**
   * Form in which the BFGS update is carried out. In exact arithmetic all forms produce the
   * same result, but their cost and numerical stability differ. The `'factored'` form updates
   * the Cholesky factor of the Hessian (`O(n^3)` per update). The `'unfactored'` form updates
   * the Hessian and refactors it at each iteration. The `'inverse'` form updates the inverse
   * Hessian and has the lowest overhead. The dogleg global method requires the Cholesky factor
   * of the Hessian, so with `globalMethod: 'dogleg'` the `'factored'` form is always used,
   * whatever the value of this option (the `method` reported in the result shows the form
   * actually used). Ignored if `bfgsUpdate` is `false`.
   * @default globalMethod === 'dogleg' ? 'factored' : 'inverse'
   */
  bfgsMethod?: BfgsMethod;
  /**
   * Function to compute the gradient of `f`, which must return a vector with as many
   * components as `x`. The returned vector is copied, so a buffer may be reused between
   * calls. By default, the gradient is approximated using forward finite differences
   * ({@link gradientForward}). In that case, setting `epsf` appropriately is essential.
   */
  grad?: (x: Vector) => Vector;
  /**
   * Function to compute the Hessian of `f`, which must return an `n`×`n` matrix. By default,
   * the Hessian is approximated using forward finite differences, from `grad` if available
   * ({@link jacobianForward}) or from `f` otherwise ({@link hessianForward}). In that case,
   * setting `epsf` appropriately is essential.
   */
  hess?: (x: Vector) => Matrix;
  /**
   * Initial Hessian approximation at `x0`, expected to be symmetric positive definite (it is
   * symmetrized and, if needed, perturbed to make it so). By default, if `bfgsUpdate` is
   * `false` or `hess` is provided, the initial Hessian is computed with `hess` or finite
   * differences; otherwise it is the identity matrix. `H0` is an approximation by design
   * and is never checked against finite differences.
   */
  H0?: Matrix;
  /**
   * If `true` and `grad` is provided, `grad(x0)` is checked against a forward
   * finite-difference approximation, to help catch errors in the user-provided gradient.
   * The comparison is made on the scaled gradient (the quantity entering `tolg`), with a
   * tolerance of `1000*sqrt(epsf)`, which is loose enough to accommodate the truncation error
   * of the forward differences for functions with large curvature, but still catches wrong
   * signs, factors and missing terms. The check costs `n` extra function evaluations; disable
   * it for badly scaled problems.
   * @default true
   */
  gradCheck?: boolean;
  /**
   * Optional callback invoked after each completed quasi-Newton iteration as
   * `callback(nIter, x, fx, gx) -> {stop, success}`, where `nIter` is the 1-based iteration
   * number (as reported in `iterations`), `x` the iterate reached at the end of that
   * iteration, `fx` the corresponding `f(x)` and `gx` the gradient at `x`. None of the
   * vectors should be mutated by the callback. If a global step is rejected, the callback
   * still receives the unchanged iterate. The callback is not called for the initial guess.
   *
   * The callback is called before the built-in convergence checks, so it takes precedence
   * over them. If `stop` is `true`, the iteration is terminated and `x` and `fx` are
   * returned; `success` then decides whether the result is reported as successful.
   */
  callback?: (
    nIter: number,
    x: Vector,
    fx: number,
    gx: Vector
  ) => { stop: boolean; success: boolean };
}

/** Result of a {@link quasiNewton} minimization. */
export interface QuasiNewtonResult {
  /** Descriptive method name, including the global strategy and BFGS form used. */
  method: string;
  /** Whether the solver met a success termination criterion; see `message`. */
  success: boolean;
  /** Message describing the reason for termination. */
  message: string;
  /** Number of function evaluations, including those made by finite differences. */
  evaluations: number;
  /** Number of user-supplied gradient evaluations (`0` if `grad` is not given). */
  evaluationsGradient: number;
  /** Number of user-supplied Hessian evaluations (`0` if `hess` is not given). */
  evaluationsHessian: number;
  /**
   * Number of outer quasi-Newton iterations performed. An iteration during which the solver
   * stops (convergence, callback, or failure) is counted, even if it did not move `x`.
   */
  iterations: number;
  /** The solution vector (or best estimate, if not converged). */
  x: Vector;
  /** Function value at `x`. */
  fx: number;
  /** Gradient at `x`, or `null` if it could not be evaluated. */
  gx: Vector | null;
  /**
   * Last evaluated or estimated Hessian matrix (may be one iteration behind `x`). With the
   * factored BFGS form it is reconstructed as `L*Lᵀ`; with the inverse form it is `null`
   * (see `Hinv`).
   */
  Hx: Matrix | null;
  /** Last estimated inverse Hessian (only with the `'inverse'` BFGS form, otherwise `null`). */
  Hinv: Matrix | null;
  /**
   * Last estimated lower-triangular Cholesky factor `L` of the Hessian, `H = L*Lᵀ` (only with
   * the `'factored'` BFGS form, otherwise `null`). The diagonal of `L` is not necessarily
   * positive.
   */
  L: Matrix | null;
}

/**
 * Finds a local minimum of a multivariate function using a quasi-Newton method with optional
 * global strategies.
 *
 * This solver follows Dennis and Schnabel (1996). The user can choose how the Hessian is
 * obtained (analytically, by finite differences, or by BFGS secant updates in factored,
 * unfactored or inverse form) and the global strategy used to improve convergence from remote
 * starting points (backtracking line search or double-dogleg trust region). The gradient and
 * the Hessian are approximated by forward finite differences unless provided.
 *
 * The default settings (BFGS update of the Cholesky factor of the Hessian, dogleg global
 * strategy, finite-difference gradient) balance reliability and the number of function
 * evaluations when only `f` is available. If analytic derivatives are available, provide
 * `grad` and, for difficult problems, `hess`.
 *
 * > [!Note]
 * > The method is a local one: it converges to a stationary point near the initial guess, not
 * > necessarily to the global minimum. A good initial guess and appropriate scaling factors
 * > (`sclx`, `sclf`) are an essential part of solving the problem. With finite-difference
 * > derivatives, setting `epsf` to the actual precision of `f` is also essential.
 *
 * Reaching the `tolx` step-size criterion is *not* reported as success, because it can also
 * be met by slow progress far from a minimizer; inspect `message` and `gx` in that case.
 *
 * Non-finite values (`NaN` or `±Infinity`) are handled as follows: if `f(x0)`, the gradient or
 * the Hessian at `x0` is not finite, the solver returns a failure result (`success: false`);
 * the same happens if the gradient or Hessian is not finite at a later iterate. If `f` is not
 * finite at a trial point, the global strategies reject the step and backtrack, whereas with
 * `globalMethod: null` the solver returns a failure result.
 *
 * **References**
 *
 * - J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *   Optimization and Nonlinear Equations", SIAM, 1996.
 *
 * @param f Function to minimize. It must map `n` unknowns to a scalar.
 * @param x0 Initial guess for the minimum: a `Vector`, a plain `number[]` or a `Float64Array`
 * with finite values. If no user-defined `sclx` is provided, the scaling factors are determined
 * from this value.
 * @param options Configuration options for the solver; see {@link QuasiNewtonOptions}.
 * @returns A `QuasiNewtonResult` with the solution and diagnostics.
 * @throws {TypeError} If `x0` or `options.sclx` is not a `VectorLike`.
 * @throws {RangeError} If `x0` has no components or non-finite values, if `sclx` does not have
 * as many components as `x0` or contains a value that is not strictly positive, if `maxIter`
 * is not a finite strictly positive integer, if `tolx`, `tolg`, `sclf`, `maxLenFactor`,
 * `trustLen` or `epsf` is outside its documented range, if `globalMethod` is not
 * `'line-search'`, `'dogleg'` or `null`, if `bfgsMethod` is not `'factored'`, `'unfactored'` or
 * `'inverse'`, if `H0` is not an `n`×`n` matrix, or if `grad` or `hess` returns an object of
 * the wrong size (at any iteration).
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { quasiNewton } from 'numerics-js/optimize';
 *
 * // Minimum of the Rosenbrock function `f(x, y) = (1 - x)^2 + 100*(y - x^2)^2`.
 * const f = (x: Vector) => (1 - x.get(0)) ** 2 + 100 * (x.get(1) - x.get(0) ** 2) ** 2;
 * const result = quasiNewton(f, [-1.0, -1.0]);
 * console.log(result.method);
 * console.log(result.success, result.iterations, result.evaluations);
 * console.log(result.x.toString());
 * ```
 *
 * Output:
 * ```text
 * Quasi-Newton (Global: Dogleg, BFGS: Factored)
 * true 28 98
 * Vector(0.9999960755461739, 0.9999921383296332)
 * ```
 */
export function quasiNewton(
  f: (x: Vector) => number,
  x0: VectorLike,
  options: QuasiNewtonOptions = {}
): QuasiNewtonResult {
  // Owned copy: it becomes the first iterate, which may be returned, so it must never alias
  // the caller's data.
  const x0v = Vector.asVector(x0, true);
  const n = x0v.size;
  if (n === 0) {
    throw new RangeError('quasiNewton: x0 must have at least one component');
  }
  if (!x0v.allFinite()) {
    throw new RangeError('quasiNewton: x0 must contain only finite values');
  }

  // Option ranges. Non-finite or non-positive values would otherwise make the solver loop
  // forever (e.g. `tolx <= 0` or `NaN` radii in dogleg) or fail with obscure errors.
  if (options.maxIter !== undefined) validateMaxIter('quasiNewton', options.maxIter);
  validateTolerance('quasiNewton', 'tolx', options.tolx, true);
  validateTolerance('quasiNewton', 'tolg', options.tolg);
  validateTolerance('quasiNewton', 'sclf', options.sclf, true);
  validateTolerance('quasiNewton', 'epsf', options.epsf);
  if (options.maxLenFactor !== undefined && !(options.maxLenFactor > 0)) {
    throw new RangeError(
      `quasiNewton: maxLenFactor must be strictly positive, got ${options.maxLenFactor}`
    );
  }
  if (options.trustLen !== undefined && Number.isNaN(options.trustLen)) {
    throw new RangeError('quasiNewton: trustLen must not be NaN');
  }

  // Scaling factors are only read, so they are not copied.
  const sclxUser = options.sclx === undefined ? undefined : Vector.asVector(options.sclx);
  if (sclxUser !== undefined) {
    if (sclxUser.size !== n) {
      throw new RangeError(`quasiNewton: sclx must have ${n} components, got ${sclxUser.size}`);
    }
    if (!sclxUser.data.every((value) => value > 0)) {
      throw new RangeError('quasiNewton: sclx must contain strictly positive values');
    }
  }

  const globalMethod: GlobalMethod =
    options.globalMethod === undefined ? 'dogleg' : options.globalMethod;
  if (globalMethod !== null && globalMethod !== 'line-search' && globalMethod !== 'dogleg') {
    throw new RangeError(`quasiNewton: unknown globalMethod '${String(globalMethod)}'`);
  }
  const bfgsMethodUser = options.bfgsMethod;
  if (
    bfgsMethodUser !== undefined &&
    bfgsMethodUser !== 'factored' &&
    bfgsMethodUser !== 'unfactored' &&
    bfgsMethodUser !== 'inverse'
  ) {
    throw new RangeError(`quasiNewton: unknown bfgsMethod '${String(bfgsMethodUser)}'`);
  }

  const H0 = options.H0?.copy();
  if (H0 && (H0.rows !== n || H0.cols !== n)) {
    throw new RangeError(
      `quasiNewton: H0 must have dimensions ${n}x${n}, got ${H0.rows}x${H0.cols}`
    );
  }

  const sclx = sclxUser ?? scaleVector(x0v);
  const sclf = options.sclf ?? 1;
  const tolx = options.tolx ?? 1e-10;
  const tolg = options.tolg ?? 1e-5;
  const maxIter = options.maxIter ?? 100;
  const maxLenFactor = options.maxLenFactor ?? 1e3;
  const gradCheck = options.gradCheck ?? true;
  const { grad, hess, epsf } = options;

  // BFGS options. The dogleg strategy needs the Cholesky factor of the Hessian.
  const bfgsUpdate = options.bfgsUpdate ?? hess === undefined;
  let bfgsMethod: BfgsMethod | null = null;
  if (bfgsUpdate) {
    bfgsMethod = globalMethod === 'dogleg' ? 'factored' : (bfgsMethodUser ?? 'inverse');
  }

  // Method name for the result.
  const methodOptions: string[] = [`Global: ${titleCase(globalMethod ?? 'none')}`];
  if (bfgsMethod) methodOptions.push(`BFGS: ${titleCase(bfgsMethod)}`);
  const method = `Quasi-Newton (${methodOptions.join(', ')})`;

  let nFev = 0;
  let nGev = 0;
  let nHev = 0;

  // All evaluations of `f` (including those made by finite differences and by the global
  // strategies) go through this wrapper, which counts them.
  const countedF = (x: Vector): number => {
    nFev += 1;
    return f(x);
  };

  const failure = (
    message: string,
    iterations: number,
    x: Vector,
    fx: number,
    gx: Vector | null,
    Hx: Matrix | null = null
  ): QuasiNewtonResult => ({
    method,
    success: false,
    message,
    evaluations: nFev,
    evaluationsGradient: nGev,
    evaluationsHessian: nHev,
    iterations,
    x,
    fx,
    gx,
    Hx,
    Hinv: null,
    L: null
  });

  // Gradient: user-supplied (validated and copied, so that buffers can be reused by the caller
  // without aliasing `gc` and `gp`) or forward finite differences.
  const callGrad = (x: Vector, at: string): Vector => {
    const g = Vector.asVector(grad!(x), true);
    nGev += 1;
    if (g.size !== n) {
      throw new RangeError(`quasiNewton: grad(${at}) must have ${n} components, got ${g.size}`);
    }
    return g;
  };
  const evaluateGradient = (x: Vector, fx: number, at: string): Vector =>
    grad ? callGrad(x, at) : gradientForward(countedF, x, { fx, sclx, epsf });

  // Hessian: user-supplied, or forward finite differences of `grad` or of `f`.
  const evaluateHessian = (x: Vector, fx: number, gx: Vector, at: string): Matrix => {
    if (hess) {
      const H = hess(x);
      nHev += 1;
      if (H.rows !== n || H.cols !== n) {
        throw new RangeError(
          `quasiNewton: hess(${at}) must have dimensions ${n}x${n}, got ${H.rows}x${H.cols}`
        );
      }
      return H;
    }
    if (grad) {
      return jacobianForward((v) => callGrad(v, at), x, { fx: gx, sclx, epsf });
    }
    return hessianForward(countedF, x, { fx, sclx, epsf });
  };

  let xc = x0v;
  let fc = countedF(xc);

  if (!Number.isFinite(fc)) {
    return failure('f(x0) is not finite.', 0, xc, fc, null);
  }

  // Gradient at x0.
  let gc = evaluateGradient(xc, fc, 'x0');
  if (!gc.allFinite()) {
    return failure('Gradient at x0 contains non-finite values.', 0, xc, fc, gc);
  }

  // Check `grad(x0)` against a finite-difference approximation.
  if (grad && gradCheck) {
    const gfd = gradientForward(countedF, xc, { fx: fc, sclx, epsf });
    const epsfEff = epsf !== undefined ? Math.max(epsf, EPS) : EPS;
    const tol = 1e3 * Math.sqrt(epsfEff);
    const fScale = Math.max(Math.abs(fc), 1 / sclf);
    for (let i = 0; i < n; i++) {
      // Scaled quantities, as in the convergence criterion.
      const scale = Math.max(Math.abs(xc.data[i]), 1 / sclx.data[i]) / fScale;
      const gi = Math.abs(gc.data[i]) * scale;
      const diff = Math.abs(gc.data[i] - gfd.data[i]) * scale;
      if (!(diff <= tol * Math.max(1, gi))) {
        return failure(
          'User-provided gradient `grad` does not match finite-difference approximation.',
          0,
          xc,
          fc,
          gc
        );
      }
    }
  }

  // Check initial solution with a tight tolerance.
  if (gradientCondition(xc, fc, gc, sclx, sclf, 1e-3 * tolg)) {
    return {
      method,
      success: true,
      message: 'Scaled gradient of the initial guess is at most 1e-3*`tolg`.',
      evaluations: nFev,
      evaluationsGradient: nGev,
      evaluationsHessian: nHev,
      iterations: 0,
      x: xc,
      fx: fc,
      gx: gc,
      Hx: null,
      Hinv: null,
      L: null
    };
  }

  // Hessian approximation at x0. `H` stays `null` if the BFGS update starts from the identity.
  let H: Matrix | null = null;
  let L: Matrix | null = null;
  let Hinv: Matrix | null = null;
  if (H0) {
    if (!H0.allFinite()) {
      return failure('H0 contains non-finite values.', 0, xc, fc, gc, H0);
    }
    H = H0;
  } else if (bfgsUpdate && !hess) {
    if (bfgsMethod === 'unfactored') {
      H = Matrix.identity(n);
    } else if (bfgsMethod === 'factored') {
      L = Matrix.identity(n);
    } else {
      Hinv = Matrix.identity(n);
    }
  } else {
    H = evaluateHessian(xc, fc, gc, 'x0');
    if (!H.allFinite()) {
      return failure('Hessian at x0 contains non-finite values.', 0, xc, fc, gc, H);
    }
  }

  // Factorize or invert the initial Hessian if the BFGS update requires it.
  if (bfgsUpdate && H) {
    try {
      if (bfgsMethod === 'unfactored') {
        H = makeHessianSpd(H).H;
      } else if (bfgsMethod === 'factored') {
        L = makeHessianSpd(H).L;
        H = null;
      } else {
        Hinv = symmetrize(H).inverse();
        H = null;
      }
    } catch (e) {
      const what =
        bfgsMethod === 'inverse'
          ? 'Inverse of initial Hessian failed'
          : 'Cholesky factorization of initial Hessian failed';
      return failure(`${what}: ${errorMessage(e)}.`, 0, xc, fc, gc, H);
    }
  }

  // Maximum step length for global methods.
  const maxLen = maxLenFactor * Math.max(sclx.mult(x0v).norm(), sclx.norm());

  // Objective for the global methods, which are shared with the root finders and thus expect a
  // `[scalar, vector]` pair: here the vector part is not used.
  const noVector = new Vector(0);
  const fN = (x: Vector): [number, Vector] => [countedF(x), noVector];
  const noMatrix = Matrix.identity(1);

  let consecutiveMaxSteps = 0;
  let trustLen = options.trustLen === undefined ? 0 : Math.min(options.trustLen, maxLen);
  let nIter: number;
  let message = '';
  let success = false;
  let terminated = false;

  for (nIter = 1; nIter <= maxIter; nIter++) {
    // Ensure the Hessian is positive definite and get its Cholesky factor.
    if (!bfgsUpdate) {
      try {
        ({ H, L } = makeHessianSpd(H!));
      } catch (e) {
        message = `Perturbed Cholesky factorization of Hessian failed: ${errorMessage(e)}.`;
        break;
      }
    }

    // Compute the quasi-Newton step.
    let p: Vector;
    try {
      if (bfgsUpdate && bfgsMethod === 'inverse') {
        // p = - H⁻¹.gc
        p = Hinv!.mulVec(gc).mult(-1);
      } else if (bfgsUpdate && bfgsMethod === 'unfactored') {
        // H.p = - gc
        L = H!.cholesky();
        p = L.choleskySolve(gc).mult(-1);
      } else {
        // (L.Lᵀ).p = - gc
        p = L!.choleskySolve(gc).mult(-1);
      }
    } catch (e) {
      message = `Solve for Newton step failed: ${errorMessage(e)}.`;
      break;
    }
    if (!p.allFinite()) {
      message = 'Newton step contains non-finite values.';
      break;
    }

    // Compute the actual x step.
    const ctx: GlobalStepContext = {
      fN,
      p,
      xc,
      fc,
      gc,
      R: globalMethod === 'dogleg' ? L!.transpose() : noMatrix,
      tolx,
      sclx,
      maxLen,
      trustLen
    };
    let step: GlobalStepResult;
    if (globalMethod === null) {
      // No global strategy: take the full quasi-Newton step as-is.
      const xpFull = xc.add(p);
      step = {
        success: true,
        wasMaxStep: true,
        xp: xpFull,
        fp: countedF(xpFull),
        Fp: noVector,
        trustLen
      };
    } else if (globalMethod === 'line-search') {
      step = lineSearch(ctx);
    } else {
      step = dogleg(ctx);
    }

    consecutiveMaxSteps = step.wasMaxStep ? consecutiveMaxSteps + 1 : 0;
    trustLen = step.trustLen;

    // A step is rejected if the strategy failed or if it returned a non-finite objective value.
    // In both cases the current iterate, its function value and its gradient are kept
    // (strategies do not guarantee meaningful values on failure).
    const nonFiniteStep = step.success && !Number.isFinite(step.fp);
    const rejected = !step.success || nonFiniteStep;
    const xp = rejected ? xc : step.xp;
    const fp = rejected ? fc : step.fp;
    let gp = gc;
    if (!rejected) {
      gp = evaluateGradient(xp, fp, 'x');
      if (!gp.allFinite()) {
        return failure(
          'Gradient at the current iterate contains non-finite values.',
          nIter,
          xp,
          fp,
          gp,
          currentHessian(bfgsUpdate, bfgsMethod, H, L)
        );
      }
    }

    // User callback, called before the built-in checks so that it takes precedence.
    const cb = options.callback?.(nIter, xp, fp, gp);

    // Check termination and convergence conditions.
    let stop: boolean;
    if (cb?.stop) {
      message = 'Terminated by user callback.';
      success = cb.success;
      stop = true;
    } else if (nonFiniteStep) {
      message =
        'f returned non-finite values at the proposed step. Consider using a global method (`globalMethod`).';
      stop = true;
    } else if (!step.success) {
      message =
        'Last global step failed to decrease f(x) sufficiently. Either `x` is an approximate ' +
        'local minimizer and no more accuracy is possible, or the finite-difference gradient ' +
        'approximation is too inaccurate, or `tolx` is too large.';
      stop = true;
    } else if (gradientCondition(xp, fp, gp, sclx, sclf, tolg)) {
      message = 'Scaled gradient is at most `tolg`.';
      success = true;
      stop = true;
    } else if (relativeStepLength(xp.sub(xc), xp, sclx) <= tolx) {
      message =
        'Scaled step size ||Δx/max(x, 1/sclx)||∞ is at most `tolx`. `x` may be an approximate ' +
        'local minimizer, but it is also possible that the algorithm is making slow progress ' +
        'and is not near a minimizer, or that `tolx` is too large.';
      stop = true;
    } else if (globalMethod && consecutiveMaxSteps >= MAX_CONSECUTIVE_MAX_STEPS) {
      message =
        `Maximum number (${MAX_CONSECUTIVE_MAX_STEPS}) of consecutive steps of length \`maxLen\` reached. ` +
        'Perhaps f(x) is unbounded below, or f(x) has a finite asymptote in some direction, or ' +
        '`maxLen` is too small.';
      stop = true;
    } else {
      stop = false;
    }

    if (stop) {
      xc = xp;
      fc = fp;
      gc = gp;
      terminated = true;
      break;
    }

    // Update the Hessian approximation.
    if (bfgsUpdate) {
      if (bfgsMethod === 'factored') {
        L = updateBfgsFactored(xc, xp, gc, gp, L!, grad !== undefined);
      } else if (bfgsMethod === 'unfactored') {
        H = updateBfgsUnfactored(xc, xp, gc, gp, H!, grad !== undefined);
      } else {
        Hinv = updateBfgsInverse(xc, xp, gc, gp, Hinv!, grad !== undefined);
      }
    } else {
      H = evaluateHessian(xp, fp, gp, 'x');
      if (!H.allFinite()) {
        return failure(
          'Hessian at the current iterate contains non-finite values.',
          nIter,
          xp,
          fp,
          gp,
          H
        );
      }
    }

    // Next iteration will start at xp.
    xc = xp;
    fc = fp;
    gc = gp;
  }

  if (!terminated && message === '') {
    message = `Maximum number of iterations (${maxIter}) reached.`;
  }

  return {
    method,
    success,
    message,
    evaluations: nFev,
    evaluationsGradient: nGev,
    evaluationsHessian: nHev,
    iterations: Math.min(nIter, maxIter),
    x: xc,
    fx: fc,
    gx: gc,
    Hx: currentHessian(bfgsUpdate, bfgsMethod, H, L),
    Hinv: bfgsMethod === 'inverse' ? Hinv : null,
    L: bfgsMethod === 'factored' ? L : null
  };
}

/**
 * Checks whether the scaled gradient is within tolerance:
 * `max_i |g_i| * max(|x_i|, 1/sclx_i) / max(|f|, 1/sclf) <= tolg`. `NaN` never satisfies it.
 *
 * @internal
 *
 * @param x Current value of the variable vector.
 * @param fx Function value at `x`.
 * @param g Gradient at `x`.
 * @param sclx Scaling factors for `x`.
 * @param sclf Scaling factor for the function value.
 * @param tolg Tolerance.
 * @returns Whether the condition is satisfied.
 */
export function gradientCondition(
  x: Vector,
  fx: number,
  g: Vector,
  sclx: Vector,
  sclf: number,
  tolg: number
): boolean {
  const fScale = Math.max(Math.abs(fx), 1 / sclf);
  let max = 0;
  for (let i = 0; i < x.size; i++) {
    const sclg = Math.max(Math.abs(x.data[i]), 1 / sclx.data[i]) / fScale;
    const v = Math.abs(g.data[i]) * sclg;
    if (Number.isNaN(v)) return false;
    if (v > max) max = v;
  }
  return max <= tolg;
}

/** Hessian estimate to report in the result, given the Hessian representation in use. */
function currentHessian(
  bfgsUpdate: boolean,
  bfgsMethod: BfgsMethod | null,
  H: Matrix | null,
  L: Matrix | null
): Matrix | null {
  if (!bfgsUpdate || bfgsMethod === 'unfactored') return H;
  if (bfgsMethod === 'factored' && L) return L.matmul(L.transpose());
  return null;
}

/**
 * Updates the Hessian approximation using the BFGS positive-definite secant formula,
 * `H ← H - (H s sᵀ H)/(sᵀ H s) + (y yᵀ)/(yᵀ s)`, where `s = xp - xc` and `y = gp - gc`.
 *
 * The update is skipped if the curvature condition `yᵀs > sqrt(eps)*||s||*||y||` is not met, or if
 * it would be negligible. Mutates `Hc` in place and returns it.
 *
 * **References**
 *
 * - J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *   Optimization and Nonlinear Equations", SIAM, 1996, p. 355.
 *
 * @internal
 *
 * @param xc Current value of the variable vector.
 * @param xp Next value of the variable vector.
 * @param gc Current gradient vector.
 * @param gp Next gradient vector.
 * @param Hc Current symmetric positive-definite Hessian approximation. Mutated in place.
 * @param gradAnalytic Whether the gradient is analytic (otherwise it is a finite-difference
 * approximation, and the threshold below which the update is negligible is larger).
 * @returns The updated Hessian approximation (`Hc` itself).
 */
export function updateBfgsUnfactored(
  xc: Vector,
  xp: Vector,
  gc: Vector,
  gp: Vector,
  Hc: Matrix,
  gradAnalytic: boolean
): Matrix {
  const s = xp.sub(xc);
  const y = gp.sub(gc);

  // Curvature condition: yᵀ.s > 0
  const ys = y.dot(s);
  if (!(ys > 0) || ys < SQRT_EPS * s.norm() * y.norm()) return Hc;

  // Skip the update if it is small enough to be ignored.
  const tol = gradAnalytic ? EPS : SQRT_EPS;
  const Hs = Hc.mulVec(s);
  if (negligible(y, Hs, gc, gp, tol)) return Hc;

  const sHs = s.dot(Hs);
  const n = s.size;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      Hc.setUnchecked(
        i,
        j,
        Hc.getUnchecked(i, j) + (y.data[i] * y.data[j]) / ys - (Hs.data[i] * Hs.data[j]) / sHs
      );
    }
  }
  return symmetrizeSelf(Hc);
}

/**
 * Updates the lower-triangular Cholesky factor `L` of the Hessian approximation using the BFGS
 * positive-definite secant formula, with `J = L + (y - α L Lᵀ s)(α Lᵀ s)ᵀ/(yᵀ s)` and
 * `α = (yᵀ s / sᵀ L Lᵀ s)^(1/2)`, where `s = xp - xc` and `y = gp - gc`. The triangular form
 * is restored by a QR rank-1 update of `Lᵀ`, which does not keep the diagonal of `L` positive.
 *
 * The update is skipped if the curvature condition `yᵀs > sqrt(eps)*||s||*||y||` is not met, or if
 * it would be negligible.
 *
 * **References**
 *
 * - J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *   Optimization and Nonlinear Equations", SIAM, 1996, p. 356.
 *
 * @internal
 *
 * @param xc Current value of the variable vector.
 * @param xp Next value of the variable vector.
 * @param gc Current gradient vector.
 * @param gp Next gradient vector.
 * @param Lc Current lower-triangular factor of the Hessian approximation. Not modified.
 * @param gradAnalytic Whether the gradient is analytic (see {@link updateBfgsUnfactored}).
 * @returns The updated factor (a new matrix, or `Lc` itself if the update was skipped).
 */
export function updateBfgsFactored(
  xc: Vector,
  xp: Vector,
  gc: Vector,
  gp: Vector,
  Lc: Matrix,
  gradAnalytic: boolean
): Matrix {
  const s = xp.sub(xc);
  const y = gp.sub(gc);

  const ys = y.dot(s);
  if (!(ys > 0) || ys < SQRT_EPS * s.norm() * y.norm()) return Lc;

  const LTs = Lc.transpose().mulVec(s);
  const sHs = LTs.dot(LTs);
  if (!(sHs > 0)) return Lc;

  const alpha = Math.sqrt(ys / sHs); // α = √(yᵀ.s / sᵀ.L.Lᵀ.s)

  const Hs = Lc.mulVec(LTs); // L.Lᵀ.s = H.s

  const tol = gradAnalytic ? EPS : SQRT_EPS;
  if (negligible(y, Hs, gc, gp, tol)) return Lc;

  // u = y - α.L.Lᵀ.s
  const u = y.sub(Hs.mult(alpha));

  // t = Lᵀ.s / √(yᵀ.s * sᵀ.L.Lᵀ.s)
  const t = LTs.mult(1 / Math.sqrt(ys * sHs));

  // QR rank-1 update: Jᵀ = Lᵀ + t.uᵀ = Q.R, hence H⁺ = J.Jᵀ = Rᵀ.R.
  const n = Lc.rows;
  const qr = Matrix.qrUpdateSelf({ Q: Matrix.identity(n), R: Lc.transpose() }, t, u);
  return qr.R.transpose();
}

/**
 * Updates the inverse Hessian approximation `B = H⁻¹` using the expanded Sherman-Morrison
 * formula, `B ← B + ((sᵀy + yᵀBy)(s sᵀ))/(sᵀy)² - (B y sᵀ + s yᵀ B)/(sᵀy)`, where
 * `s = xp - xc` and `y = gp - gc`.
 *
 * The update is skipped if the curvature condition `yᵀs > sqrt(eps)*||s||*||y||` is not met, or if
 * it would be negligible. Mutates `Bc` in place and returns it.
 *
 * @internal
 *
 * @param xc Current value of the variable vector.
 * @param xp Next value of the variable vector.
 * @param gc Current gradient vector.
 * @param gp Next gradient vector.
 * @param Bc Current inverse Hessian approximation. Mutated in place.
 * @param gradAnalytic Whether the gradient is analytic (see {@link updateBfgsUnfactored}).
 * @returns The updated inverse Hessian approximation (`Bc` itself).
 */
export function updateBfgsInverse(
  xc: Vector,
  xp: Vector,
  gc: Vector,
  gp: Vector,
  Bc: Matrix,
  gradAnalytic: boolean
): Matrix {
  const s = xp.sub(xc);
  const y = gp.sub(gc);

  // Curvature condition: yᵀ.s > 0
  const ys = y.dot(s);
  if (!(ys > 0) || ys < SQRT_EPS * s.norm() * y.norm()) return Bc;

  // Skip the update if it is small enough to be ignored.
  const By = Bc.mulVec(y);
  const tol = gradAnalytic ? EPS : SQRT_EPS;
  let isSmall = true;
  for (let i = 0; i < s.size; i++) {
    const scale = Math.max(1, Math.abs(s.data[i]), Math.abs(By.data[i]));
    if (!(Math.abs(s.data[i] - By.data[i]) < tol * scale)) {
      isSmall = false;
      break;
    }
  }
  if (isSmall) return Bc;

  const yBy = y.dot(By);
  const c = (ys + yBy) / (ys * ys);
  const n = s.size;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      Bc.setUnchecked(
        i,
        j,
        Bc.getUnchecked(i, j) +
          c * s.data[i] * s.data[j] -
          (By.data[i] * s.data[j] + s.data[i] * By.data[j]) / ys
      );
    }
  }
  return symmetrizeSelf(Bc);
}

/** Whether `|y - Hs| < tol*(|gc| + |gp|)` holds for every component. */
function negligible(y: Vector, Hs: Vector, gc: Vector, gp: Vector, tol: number): boolean {
  for (let i = 0; i < y.size; i++) {
    if (!(Math.abs(y.data[i] - Hs.data[i]) < tol * (Math.abs(gc.data[i]) + Math.abs(gp.data[i])))) {
      return false;
    }
  }
  return true;
}

/**
 * Makes a (square) Hessian matrix positive definite by adding a scaled identity matrix, and
 * computes its Cholesky factor.
 *
 * The matrix is first symmetrized (by averaging it with its transpose). Then a perturbation is
 * added to the diagonal so that all diagonal elements are positive and sufficiently large
 * compared to the off-diagonal ones. Finally, a progressively larger multiple of the identity
 * (starting from `eps*max(||H||∞, 1)` and growing by a factor of `10` per attempt) is added until
 * the Cholesky factorization succeeds. The input is not modified.
 *
 * **References**
 *
 * - J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
 *   Optimization and Nonlinear Equations", SIAM, 1996, A5.5.1.
 *
 * @internal
 *
 * @param Hin Hessian matrix to be made positive definite.
 * @param maxIter Maximum number of perturbation attempts, besides the unperturbed one.
 * @returns The perturbed Hessian `H`, its lower-triangular Cholesky factor `L`, and the total
 * perturbation added to the diagonal.
 * @throws {Error} If the Cholesky factorization still fails after `maxIter` attempts.
 */
export function makeHessianSpd(
  Hin: Matrix,
  maxIter = 10
): { H: Matrix; L: Matrix; perturbation: number } {
  const n = Hin.rows;
  const H = symmetrize(Hin);

  // Ensure all diagonal elements are positive and sufficiently large compared to the
  // off-diagonal elements, to improve the chances of the Hessian being SPD.
  let maxDiag = -Infinity;
  let minDiag = Infinity;
  let maxOff = 0;
  for (let i = 0; i < n; i++) {
    const d = H.getUnchecked(i, i);
    if (d > maxDiag) maxDiag = d;
    if (d < minDiag) minDiag = d;
    for (let j = 0; j < n; j++) {
      if (j !== i) maxOff = Math.max(maxOff, Math.abs(H.getUnchecked(i, j)));
    }
  }
  const maxPosDiag = Math.max(maxDiag, 0);

  let mu = 0;
  if (minDiag <= SQRT_EPS * maxPosDiag) {
    mu = 2 * (maxPosDiag - minDiag) * SQRT_EPS - minDiag;
    maxDiag += mu;
  }

  if (maxOff * (1 + 2 * SQRT_EPS) > maxDiag) {
    mu += maxOff - maxDiag + 2 * SQRT_EPS * maxOff;
    maxDiag = maxOff * (1 + 2 * SQRT_EPS);
  }

  if (maxDiag === 0) {
    mu = 1;
    maxDiag = 1;
  }

  if (mu > 0) {
    for (let i = 0; i < n; i++) H.setUnchecked(i, i, H.getUnchecked(i, i) + mu);
  }

  // Perturb the Hessian by adding a scaled identity matrix until it becomes SPD.
  let perturb = 0;
  const perturb0 = EPS * Math.max(H.normInf(), 1);
  const Hp = H.copy();

  for (let attempt = 0; attempt <= maxIter; attempt++) {
    for (let i = 0; i < n; i++) Hp.setUnchecked(i, i, H.getUnchecked(i, i) + perturb);

    try {
      const L = Hp.cholesky();
      return { H: Hp, L, perturbation: perturb + mu };
    } catch {
      perturb = Math.max(perturb0, 10 * perturb);
    }
  }

  throw new Error(
    `Perturbed Cholesky failed after total perturbation=${(mu + perturb).toExponential(3)}`
  );
}

// -----------------------------------------------------------------
// Small helpers.
// -----------------------------------------------------------------

/** Returns `(M + Mᵀ)/2` as a new matrix. */
function symmetrize(M: Matrix): Matrix {
  return symmetrizeSelf(M.copy());
}

/** Replaces `M` by `(M + Mᵀ)/2` in place and returns it. */
function symmetrizeSelf(M: Matrix): Matrix {
  const n = M.rows;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const avg = 0.5 * (M.getUnchecked(i, j) + M.getUnchecked(j, i));
      M.setUnchecked(i, j, avg);
      M.setUnchecked(j, i, avg);
    }
  }
  return M;
}

/** Message of a caught error. */
function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Title-cases a hyphen/space-separated label, e.g. `'line-search' -> 'Line-Search'`. */
function titleCase(s: string): string {
  return s.replace(/(^|[\s-])([a-z])/g, (_m, sep: string, ch: string) => sep + ch.toUpperCase());
}
