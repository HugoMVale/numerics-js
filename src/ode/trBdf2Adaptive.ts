import { Vector, type VectorLike } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import type { LUDecomposition } from '../linalg/Matrix.js';
import { BandedMatrix } from '../linalg/BandedMatrix.js';
import type { BandedLUDecomposition } from '../linalg/BandedMatrix.js';
import { typeName } from '../linalg/checks.js';
import { clip } from '../math.js';
import { jacobianForward, jacobianForwardBanded } from '../numdiff.js';
import {
  emptyIntervalResult,
  SolutionRecorder,
  resolveAdaptiveControl,
  validateAdaptiveOptions
} from './adaptiveCommon.js';
import { OdeDenseOutput, StepPolynomial } from './denseOutput.js';
import { EventDetector, validateEvents } from './events.js';
import { estimateInitialStep } from './stepControl.js';
import type {
  BandedJacobianFunction,
  DerivativeFunction,
  JacobianFunction,
  OdeAdaptiveOptions,
  OdeResult
} from './types.js';

// -----------------------------------------------------------------
// TR-BDF2 (Bank et al., 1985), in the three-stage DIRK form and with the implementation
// choices recommended by Hosea & Shampine (1996), "Analysis and implementation of TR-BDF2".
//
// With scaled derivatives z = h f, one step from (t_n, y_n) to t_n + h has the stages
//   1. z_n = h f(t_n, y_n)                                 (explicit, c = 0)
//   2. z_g = h f(t_n + GAMMA h, y_n + D z_n + D z_g)        (implicit, c = GAMMA)
//   3. z_1 = h f(t_n + h, y_n + W z_n + W z_g + D z_1)      (implicit, c = 1)
// and y_{n+1} = y_n + W z_n + W z_g + D z_1. Both implicit stages share the Newton iteration
// matrix I - h D J, so one LU factorization serves the whole step.
// -----------------------------------------------------------------
const GAMMA = 2 - Math.SQRT2;
const D = GAMMA / 2;
const W = Math.SQRT2 / 4;

// Stage-3 initial guess: derivative at t_n + h of the cubic Hermite interpolant of
// (y_n, z_n, y_g, z_g), extrapolated beyond its first sub-interval.
const GUESS_ZN = 1.5 + Math.SQRT2;
const GUESS_ZG = 2.5 + 2 * Math.SQRT2;
const GUESS_DY = 6 + 4.5 * Math.SQRT2;

// Local truncation error: y(x + h) - y_{n+1} = LTE_CONST * h^3 * y'''(x) + O(h^4).
const LTE_CONST = -(3 * GAMMA * GAMMA - 4 * GAMMA + 2) / (12 * (2 - GAMMA));

// Weights (b_hat - b) of the embedded third-order companion, so that the raw error estimate is
// est = E_N z_n + E_G z_g + E_1 z_1. They vanish for exact quadratic solutions
// (E_N + E_G + E_1 = 0 and GAMMA E_G + E_1 = 0) and reproduce LTE_CONST on cubic ones.
const E_1 = (2 * LTE_CONST) / (1 - GAMMA);
const E_G = -E_1 / GAMMA;
const E_N = (E_1 * (1 - GAMMA)) / GAMMA;

/**
 * Newton termination threshold `tau = kappa * epsilon`, with `kappa = 1/2` as
 * recommended by Hosea & Shampine and `epsilon = 1` in the weighted norm
 * `atol + rtol |y|`.
 */
const NEWTON_KAPPA = 0.5;
/**
 * Maximum number of Newton iterations per stage (a standard safeguard, not
 * part of the paper).
 */
const MAX_NEWTON_ITERATIONS = 4;
/** Step-size multiplier after a Newton failure with a current Jacobian. */
const NEWTON_FAILURE_SHRINK = 0.5;
/**
 * Accepted-step growth factors below this keep the step size, so the LU
 * factors stay valid.
 */
const MIN_STEP_CHANGE = 1.2;

/** Exponent of the step-size controller: the local error is `O(h^3)`, so `h_new = h err^(-1/3)`. */
const ERROR_EXPONENT = -1 / 3;

/**
 * Degree of the continuous extension of a TR-BDF2 step; see
 * {@link trBdf2AdaptiveDenseCoefficients}.
 */
const DENSE_DEGREE = 3;

/**
 * Computes the coefficients of the continuous extension of an accepted
 * TR-BDF2 step, without evaluating `f`.
 *
 * @internal
 *
 * The extension is the cubic
 * `p(x) = y_n + z_n x + c_2 x^2 + c_3 x^3`, with `x = (t - t_n) / h`, that
 * matches the step's starting state `y_n` and starting slope `z_n = h f_n`,
 * the intermediate state `y_g` at `x = GAMMA`, and the final state `y_{n+1}`
 * at `x = 1`. It reproduces both end states up to rounding and has error
 * `O(h^3)`, like the method itself. The end-point derivative is deliberately
 * not used: for stiff problems it carries `h * lambda` times the end-state
 * error, which would make a Hermite cubic much less accurate than the step.
 *
 * @param y State `y_n` at the start of the step. Not mutated.
 * @param yNew Accepted state `y_{n+1}` at the end of the step. Not mutated.
 * @param yG Intermediate state `y_g` at `x = GAMMA`. Not mutated.
 * @param zN Scaled first-stage derivative `z_n = h f_n` of the step. Not mutated.
 * @param out Array of length `4 * y.size` that receives the coefficients as
 * the four blocks `y_n, z_n, c_2, c_3`, one after the other, each of length
 * `y.size`. It is overwritten and must not alias the inputs' data.
 */
export function trBdf2AdaptiveDenseCoefficients(
  y: Vector,
  yNew: Vector,
  yG: Vector,
  zN: Vector,
  out: Float64Array
): void {
  const yd = y.data;
  const nd = yNew.data;
  const gd = yG.data;
  const zd = zN.data;
  const dim = yd.length;
  const g2 = GAMMA * GAMMA;
  for (let i = 0; i < dim; i++) {
    // p(GAMMA) = y_g and p(1) = y_{n+1} determine c2 and c3.
    const a = gd[i] - yd[i] - zd[i] * GAMMA;
    const b = nd[i] - yd[i] - zd[i];
    const c3 = (a - b * g2) / (g2 * (GAMMA - 1));
    out[i] = yd[i];
    out[dim + i] = zd[i];
    out[2 * dim + i] = b - c3;
    out[3 * dim + i] = c3;
  }
}

/**
 * Optional tuning parameters for {@link trBdf2Adaptive}.
 */
export interface TrBdf2AdaptiveOptions extends OdeAdaptiveOptions {
  /**
   * Minimum permitted step-size magnitude after a rejected step or a Newton failure.
   * @default 1e-12
   */
  hMin?: number;
  /**
   * Maximum number of attempted steps, including rejected steps and steps
   * abandoned because the Newton iteration failed.
   * @default 1e5
   */
  maxSteps?: number;
  /**
   * Optional analytic Jacobian `∂f/∂y`. Without `kl` and `ku` it is a {@link JacobianFunction}
   * that fills a dense `n x n` {@link Matrix}; with them it is a {@link BandedJacobianFunction}
   * that fills a {@link BandedMatrix}. Because the two types cannot be told apart at run time,
   * a callback written for one must not be combined with the other layout. When omitted, the
   * Jacobian is approximated by forward finite differences, which costs `n + 1` derivative
   * evaluations per refresh, where `n` is the number of state components, or
   * `min(kl + ku + 1, n) + 1` in banded mode.
   */
  jac?: JacobianFunction | BandedJacobianFunction;
  /**
   * Number of subdiagonals of the Jacobian `∂f/∂y`, which switches the solver to banded mode:
   * `∂f_i/∂y_j = 0` for `i - j > kl`. Must be given together with `ku`, as a non-negative
   * integer; a value of `n` or more is clamped to `n - 1`. See {@link TrBdf2AdaptiveOptions.ku}.
   * @default undefined (dense Jacobian)
   */
  kl?: number;
  /**
   * Number of superdiagonals of the Jacobian `∂f/∂y`: `∂f_i/∂y_j = 0` for `j - i > ku`. Must be
   * given together with `kl`, as a non-negative integer; a value of `n` or more is clamped to
   * `n - 1`.
   *
   * In banded mode the Jacobian, the iteration matrix `I - (1 - sqrt(2)/2) h J` and its LU
   * factors are held in LAPACK band storage ({@link BandedMatrix}), so the memory is
   * `O(n (kl + ku))` instead of `O(n^2)` and each Newton iteration costs `O(n (kl + ku))`
   * instead of `O(n^2)`. The results are those of the dense mode up to rounding. A finite-difference
   * Jacobian is obtained with {@link jacobianForwardBanded}, by perturbing columns that are
   * `kl + ku + 1` apart together. If `kl + ku >= n - 1` the band covers the whole matrix: the
   * solver does not fall back to dense storage but runs banded on the full band, which gives the
   * same results with no saving.
   *
   * **The band is a promise.** It is not verified. A function that depends on a state
   * component outside the declared band makes the finite-difference Jacobian wrong without
   * any error, and an analytic one cannot represent the entry. Periodic boundary conditions,
   * such as a ring of cells, add entries in the corners of the Jacobian and are therefore not
   * banded; declaring the band for them is wrong, and the dense mode must be used.
   * @default undefined (dense Jacobian)
   */
  ku?: number;
}

/**
 * Integrates the stiff or non-stiff system `dy/dt = f(t, y)` from `t0` to `tEnd` using
 * the adaptive TR-BDF2 method, following the implementation recommendations of
 * Hosea & Shampine (1996).
 *
 * TR-BDF2 is a two-implicit-stage, second-order Runge-Kutta method: a trapezoidal-rule
 * step to an intermediate time, followed by a second-order backward differentiation
 * step to the end of the step. It is A-stable and L-stable, so very stiff transient
 * components are damped instead of ringing, and both stages share one Newton iteration
 * matrix `I - (1 - sqrt(2)/2) h J`. Being second order, it is best suited to stiff
 * problems at moderate tolerances; for non-stiff problems, {@link rungeKuttaAdaptive}
 * is more efficient.
 *
 * Zero crossings of user-supplied event functions can be located with the
 * `events` option, with the same semantics as in {@link rungeKuttaAdaptive}.
 * After each accepted step the event functions are evaluated, and a sign
 * change triggers a root search on the step's dense output, a cubic through
 * the step's start state and slope, its intermediate stage state and its end
 * state, which costs no further evaluations of `f`. Event times are therefore
 * as accurate as this interpolant, which is second order accurate like the
 * method. On stiff problems its error can exceed the step's end-point error by
 * a factor that shrinks with the step size, so tighten the tolerances if very
 * accurate event times are needed. Non-terminal events do not alter the
 * integration. A terminal event ends it at the event time, which becomes the
 * last recorded point. Only sign changes between accepted steps are seen, so
 * use `hMax` if two crossings can occur within a step. An event function that
 * is exactly zero at `t0` is not reported.
 *
 * With the `denseOutput` option, the solution can also be evaluated at any time
 * within the integration interval, not only at the accepted steps; see
 * {@link OdeDenseOutput}. The dense output is the piecewise cubic formed by the
 * same step interpolants, which is continuous across steps and second order
 * accurate like the method, with error `O(h^3)`. It costs no further
 * evaluations of `f` and does not change the steps. Between steps its accuracy
 * is not controlled by `atol` and `rtol`, but is generally close to that of the
 * step points; on stiff problems it can exceed it by a factor that shrinks
 * with the step size, as for event times. If the integration stops early,
 * because of a terminal event or a failure, it covers the part that was
 * solved.
 *
 * **Banded Jacobians.** Systems from the method of lines on a 1D or 2D grid have a Jacobian that
 * is zero outside a narrow band around the diagonal. With the options `kl` and `ku` (the numbers
 * of sub- and superdiagonals) the Jacobian, the iteration matrix and its LU factors are kept in
 * LAPACK band storage ({@link BandedMatrix}), which cuts memory from `O(n^2)` to
 * `O(n (kl + ku))` and each Newton iteration from `O(n^2)` to `O(n (kl + ku))`. An analytic
 * Jacobian is then a {@link BandedJacobianFunction}; otherwise it is estimated with
 * {@link jacobianForwardBanded}, at `min(kl + ku + 1, n) + 1` evaluations of `f` per refresh
 * instead of `n + 1`. The same steps are taken as in dense mode, up to rounding. Without `kl` and
 * `ku` nothing changes. On a 1D grid `kl = ku = 1`; on an `m x m` grid numbered row by row with a
 * five-point stencil `kl = ku = m`. The band is a promise that is not verified, and the solver
 * is never faster than the band allows: with `kl + ku >= n - 1` it still runs banded, on the full
 * band, with no saving. **Periodic boundary conditions** (a ring of cells, as in a periodic
 * Brusselator) couple the first and the last cell and add entries in the corners of the Jacobian:
 * such a Jacobian is not banded, and declaring a band for it gives wrong results, so use the dense
 * mode for it.
 *
 * Step-size underflow and exceeding `maxSteps` do not throw: they return an unsuccessful result
 * (`success === false`) with the reason in `message`.
 *
 * **References**
 * - Bank, R. E., Coughran, W. M., Fichtner, W., Grosse, E. W., Rose, D. J., & Smith, R. K.
 *   (1985). "Transient simulation of silicon devices and circuits." IEEE Transactions on
 *   Computer-Aided Design of Integrated Circuits and Systems, 4(4), 436–451.
 * - Hosea, M. E., & Shampine, L. F. (1996). "Analysis and implementation of TR-BDF2."
 *   Applied Numerical Mathematics, 20(1–2), 21–37.
 *
 * @param f Derivative function `dy/dt = f(t, y)`. Must write its result into
 * the supplied `dydt` buffer rather than allocating.
 * @param t0 Starting time. Must be finite.
 * @param tEnd Final integration time. Must be finite. May be less than `t0` to integrate
 * backward; the solver selects the required direction automatically.
 * @param y0 Initial state at `t0`: a `Vector`, a plain `number[]` or a `Float64Array`. It
 * must have at least one component, all of them finite, and is copied internally, so it is
 * never mutated.
 * @param options Optional tuning parameters; see {@link TrBdf2AdaptiveOptions}.
 * @returns An {@link OdeResult}: `t`, the accepted time points, and `y`, an
 * `n x y0.size` matrix whose row `i` is the state at `t.get(i)`. Row 0 is
 * `y0`; on success, the last row is at `tEnd`. `evaluations` is the number of calls to
 * `f`, including those spent on finite-difference Jacobians;
 * `evaluationsJacobian` is the number of Jacobian refreshes and
 * `decompositions` the number of LU factorizations. `method` is `'trbdf2'`.
 * The `success` flag indicates whether the integration completed, i.e. reached
 * `tEnd` or was stopped by a terminal event, and `message` provides additional
 * status information. `events` is present when the `events` option was given,
 * and so is `denseOutput` when the `denseOutput` option was.
 * @throws {RangeError} If the state has no components or a non-finite component, `t0` or `tEnd`
 * is not finite, tolerances are invalid, a step-size or controller limit is invalid, an event
 * option is invalid, only one of `kl` and `ku` is given, or one of them is not a non-negative
 * integer. A banded Jacobian callback that writes outside the band also throws a `RangeError`
 * (from `BandedMatrix.set`).
 * @throws {TypeError} If `y0` is not a `VectorLike`, if an event function is not a
 * function, or if `kl` or `ku` is not a number.
 *
 * @example
 * ```ts
 * import { trBdf2Adaptive, type DerivativeFunction } from 'numerics-js/ode';
 *
 * // Stiff relaxation towards cos(t): dy/dt = -1000 (y - cos(t)), y(0) = 0
 * const f: DerivativeFunction = (t, y, out) => {
 *     out.data[0] = -1000 * (y.data[0] - Math.cos(t));
 *     return out;
 * };
 * const result = trBdf2Adaptive(f, 0, 1, [0]);
 * console.log({
 *     success: result.success,
 *     steps: result.t.size - 1,
 *     evaluations: result.evaluations,
 *     yEnd: result.y.get(result.y.rows - 1, 0),
 * });
 * ```
 *
 * Output:
 * ```text
 * { success: true, steps: 21, evaluations: 84, yEnd: 0.5411713864049918 }
 * ```
 *
 * @example
 * ```ts
 * import {
 *     trBdf2Adaptive,
 *     type DerivativeFunction,
 *     type JacobianFunction,
 * } from 'numerics-js/ode';
 *
 * // Same problem with tighter tolerances and an analytic Jacobian, df/dy = -1000.
 * const f: DerivativeFunction = (t, y, out) => {
 *     out.data[0] = -1000 * (y.data[0] - Math.cos(t));
 *     return out;
 * };
 * const jac: JacobianFunction = (_t, _y, J) => J.set(0, 0, -1000);
 * const result = trBdf2Adaptive(f, 0, 1, [0], {
 *     atol: 1e-8,
 *     rtol: 1e-6,
 *     jac,
 * });
 * console.log({
 *     success: result.success,
 *     evaluations: result.evaluations,
 *     evaluationsJacobian: result.evaluationsJacobian,
 *     decompositions: result.decompositions,
 *     yEnd: result.y.get(result.y.rows - 1, 0),
 * });
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   evaluations: 652,
 *   evaluationsJacobian: 1,
 *   decompositions: 37,
 *   yEnd: 0.5411437997526156
 * }
 * ```
 *
 * @example
 * ```ts
 * import {
 *     trBdf2Adaptive,
 *     type DerivativeFunction,
 *     type OdeEvent,
 * } from 'numerics-js/ode';
 *
 * // Stiff relaxation towards cos(t): dy/dt = -1000 (y - cos(t)), y(0) = 0.
 * // Locate the first time the solution crosses zero, i.e. y(t) = 0.
 * const f: DerivativeFunction = (t, y, out) => {
 *     out.data[0] = -1000 * (y.data[0] - Math.cos(t));
 *     return out;
 * };
 * const zero: OdeEvent = {
 *     fn: (t, y) => y.data[0],
 *     direction: 'falling',
 *     terminal: true,
 * };
 * const result = trBdf2Adaptive(f, 0, 3, [0], {
 *     atol: 1e-8,
 *     rtol: 1e-8,
 *     events: [zero],
 * });
 * const [hit] = result.events!;
 * console.log({
 *     message: result.message,
 *     hitTime: hit.t,
 *     hitDirection: hit.direction,
 *     steps: result.t.size - 1,
 *     evaluations: result.evaluations,
 * });
 * ```
 *
 * Output:
 * ```text
 * {
 *   message: 'Integration stopped by a terminal event.',
 *   hitTime: 1.571796323420337,
 *   hitDirection: 'falling',
 *   steps: 577,
 *   evaluations: 2355
 * }
 * ```
 *
 * @example
 * ```ts
 * import {
 *     trBdf2Adaptive,
 *     type DerivativeFunction,
 *     type OdeEvent,
 * } from 'numerics-js/ode';
 *
 * // Simple harmonic oscillator: x'' = -x, starting at x = 1. Stop at the
 * // first downward zero crossing.
 * const f: DerivativeFunction = (_t, y, out) => {
 *     out.data[0] = y.data[1];
 *     out.data[1] = -y.data[0];
 *     return out;
 * };
 * const zeroCrossing: OdeEvent = {
 *     fn: (_t, y) => y.data[0],
 *     direction: 'falling',
 *     terminal: true,
 * };
 * const result = trBdf2Adaptive(f, 0, 2 * Math.PI, [1, 0], {
 *     atol: 1e-9,
 *     rtol: 1e-9,
 *     events: [zeroCrossing],
 * });
 * const [event] = result.events!;
 * console.log({
 *     eventTime: event.t,
 *     eventVelocity: event.y.data[1],
 *     expectedTime: Math.PI / 2,
 * });
 * ```
 *
 * Output:
 * ```text
 * {
 *   eventTime: 1.5707968774578687,
 *   eventVelocity: -0.9999999998524634,
 *   expectedTime: 1.5707963267948966
 * }
 * ```
 *
 * @example
 * ```ts
 * import { trBdf2Adaptive, type DerivativeFunction } from 'numerics-js/ode';
 *
 * // Stiff relaxation towards cos(t): dy/dt = -1000 (y - cos(t)), y(0) = 0.
 * // Evaluate the solution on a uniform grid, independent of the steps taken,
 * // and compare it with the exact one.
 * const lambda = 1000;
 * const f: DerivativeFunction = (t, y, out) => {
 *     out.data[0] = -lambda * (y.data[0] - Math.cos(t));
 *     return out;
 * };
 * const exact = (t: number): number =>
 *     (
 *         lambda ** 2 * Math.cos(t) +
 *         lambda * Math.sin(t) -
 *         lambda ** 2 * Math.exp(-lambda * t)
 *     ) / (lambda ** 2 + 1);
 * const result = trBdf2Adaptive(f, 0, 1, [0], {
 *     atol: 1e-8,
 *     rtol: 1e-8,
 *     denseOutput: true,
 * });
 * const times = Array.from({ length: 101 }, (_, i) => i / 100);
 * const y = result.denseOutput!.eval(times);
 * let maxError = 0;
 * times.forEach((t, i) => {
 *     maxError = Math.max(maxError, Math.abs(y.get(i, 0) - exact(t)));
 * });
 * console.log({ steps: result.t.size - 1, points: times.length, maxError });
 * ```
 *
 * Output:
 * ```text
 * { steps: 533, points: 101, maxError: 9.004891221575662e-8 }
 * ```
 *
 * @example
 * ```ts
 * import {
 *     trBdf2Adaptive,
 *     type BandedJacobianFunction,
 *     type DerivativeFunction,
 * } from 'numerics-js/ode';
 *
 * // Heat equation u_t = 0.01 u_xx on (0, 1) with u(0) = u(1) = 0 and u(x, 0) = sin(pi x),
 * // discretized on n interior points. The Jacobian is tridiagonal, so kl = ku = 1 and the
 * // solver works in band storage: O(n) memory and work per Newton iteration instead of O(n^2).
 * const n = 200;
 * const nu = 0.01 * (n + 1) ** 2;
 * const f: DerivativeFunction = (_t, y, out) => {
 *     for (let i = 0; i < n; i++) {
 *         const left = i > 0 ? y.data[i - 1] : 0;
 *         const right = i < n - 1 ? y.data[i + 1] : 0;
 *         out.data[i] = nu * (left - 2 * y.data[i] + right);
 *     }
 *     return out;
 * };
 * const jac: BandedJacobianFunction = (_t, _y, J) => {
 *     for (let i = 0; i < n; i++) {
 *         J.set(i, i, -2 * nu);
 *         if (i > 0) J.set(i, i - 1, nu);
 *         if (i < n - 1) J.set(i, i + 1, nu);
 *     }
 *     return J;
 * };
 * const x = Array.from({ length: n }, (_, i) => (i + 1) / (n + 1));
 * const result = trBdf2Adaptive(f, 0, 1, x.map((xi) => Math.sin(Math.PI * xi)), {
 *     atol: 1e-8,
 *     rtol: 1e-6,
 *     kl: 1,
 *     ku: 1,
 *     jac,
 * });
 * const decay = Math.exp(-0.01 * Math.PI ** 2);
 * let maxError = 0;
 * x.forEach((xi, i) => {
 *     const exact = decay * Math.sin(Math.PI * xi);
 *     maxError = Math.max(maxError, Math.abs(result.y.get(result.y.rows - 1, i) - exact));
 * });
 * console.log({
 *     success: result.success,
 *     steps: result.t.size - 1,
 *     evaluations: result.evaluations,
 *     decompositions: result.decompositions,
 *     maxError,
 * });
 * ```
 *
 * Output:
 * ```text
 * {
 *   success: true,
 *   steps: 6,
 *   evaluations: 23,
 *   decompositions: 4,
 *   maxError: 3.3846555091976427e-7
 * }
 * ```
 */
export function trBdf2Adaptive(
  f: DerivativeFunction,
  t0: number,
  tEnd: number,
  y0: VectorLike,
  options: TrBdf2AdaptiveOptions = {}
): OdeResult {
  const control = resolveAdaptiveControl(options);
  const { atol, rtol, h0, hMax, hMin, maxSteps, safety, minScale, maxScale } = control;
  const { jac, events, denseOutput = false } = options;
  const { kl: klOption, ku: kuOption } = options;
  // Owned copy: it becomes the working state, so the caller's data is never mutated.
  const y0v = Vector.asVector(y0, true);
  const dim = y0v.size;
  validateAdaptiveOptions('trBdf2Adaptive', t0, tEnd, y0v, control);
  if (events !== undefined) validateEvents(events, 'trBdf2Adaptive');
  const banded = validateBandwidth(klOption, kuOption);
  const kl = banded ? Math.min(klOption as number, dim - 1) : 0;
  const ku = banded ? Math.min(kuOption as number, dim - 1) : 0;

  if (tEnd === t0) return emptyIntervalResult('trbdf2', t0, y0v, events, denseOutput);

  let evaluations = 0;
  let evaluationsJacobian = 0;
  let decompositions = 0;

  const dir = Math.sign(tEnd - t0) as 1 | -1;

  // Every derivative evaluation, including those inside finite-difference Jacobians, is counted.
  const fc: DerivativeFunction = (t, y, out) => {
    evaluations++;
    return f(t, y, out);
  };

  // Working storage, allocated once.
  let y = y0v;
  let yNew = new Vector(dim);
  const dY = new Vector(dim); // z_n / h: the first-stage derivative carried over from the previous step
  const zN = new Vector(dim); // z_n = h * dY
  const zG = new Vector(dim); // converged stage-2 scaled derivative
  const z1 = new Vector(dim); // converged stage-3 scaled derivative
  const zIter = new Vector(dim); // Newton iterate
  const yG = new Vector(dim); // intermediate state at t_n + GAMMA h
  const base = new Vector(dim); // stage state not depending on the unknown z
  const yStage = new Vector(dim); // stage state passed to f
  const fStage = new Vector(dim);
  const fJac = new Vector(dim); // fresh f(t_n, y_n) for finite-difference Jacobians
  const resid = new Vector(dim); // Newton right-hand side, later the raw error estimate
  const scale = new Vector(dim); // weights atol + rtol |y_n|
  const delta = new Vector(dim); // Newton correction, later the filtered error estimate
  // Storage of the factorization of I - D h J, reused by every call to `lu`. Dense mode only: in
  // banded mode no n x n array is ever allocated.
  const lu: LUDecomposition = {
    L: new Matrix(banded ? 1 : dim, banded ? 1 : dim),
    U: new Matrix(banded ? 1 : dim, banded ? 1 : dim),
    perm: new Array<number>(banded ? 1 : dim),
    sign: 1
  };
  const jacBuf = jac !== undefined && !banded ? new Matrix(dim, dim) : null;
  // Banded mode: the iteration matrix I - D h J, its factorization (the first one is of the
  // identity, only to allocate the storage that every later `lu(out)` reuses), and the buffer of
  // an analytic Jacobian.
  const iterBand = banded ? new BandedMatrix(dim, kl, ku) : null;
  if (iterBand !== null) {
    for (let j = 0; j < dim; j++) iterBand.data[ku + j * iterBand.ldab] = 1;
  }
  const luBand: BandedLUDecomposition | null = iterBand !== null ? iterBand.lu() : null;
  const jacBand = jac !== undefined && banded ? new BandedMatrix(dim, kl, ku) : null;

  let t = t0;

  // Jacobian and factorization state. `J` is assigned inside closures, so it is declared through an
  // assertion to keep TypeScript from narrowing it to `null`.
  let J = null as Matrix | null;
  let JB = null as BandedMatrix | null; // banded mode: the Jacobian, in band storage
  let luValid = false; // whether `lu` holds the factorization for step size `luH`
  let luH = NaN; // step size for which `lu` was built
  let jacCurrent = false; // whether J was evaluated at (t_n, y_n)

  let h: number;
  if (h0 !== undefined) {
    fc(t0, y, dY);
    h = dir * Math.abs(h0);
  } else {
    h = estimateInitialStep(fc, t0, y, dir, atol, rtol, 2, dY, fStage, yStage);
  }
  h = dir * Math.min(Math.abs(h), hMax, Math.abs(tEnd - t0));

  const refreshJacobian = (): void => {
    if (banded) {
      if (jacBand !== null && jac !== undefined) {
        jacBand.data.fill(0);
        (jac as BandedJacobianFunction)(t, y, jacBand);
        JB = jacBand;
      } else {
        fc(t, y, fJac);
        JB = jacobianForwardBanded((x) => fc(t, x, fStage), y, { kl, ku, fx: fJac });
      }
    } else if (jacBuf !== null && jac !== undefined) {
      jacBuf.fill(0);
      (jac as JacobianFunction)(t, y, jacBuf);
      J = jacBuf;
    } else {
      // The carried-over first-stage derivative is not accurate enough to be differenced,
      // so a fresh f(t_n, y_n) is evaluated.
      fc(t, y, fJac);
      J = jacobianForward((x) => fc(t, x, fStage), y, { fx: fJac });
    }
    evaluationsJacobian++;
    jacCurrent = true;
    luValid = false;
  };

  // Factors I - D*h*J into `lu`, in place. Returns false if the matrix is exactly singular or
  // is not a finite number (`lu` propagates NaN instead of failing, and rejects infinite entries).
  const factorize = (hStep: number): boolean => {
    if (banded) return factorizeBanded(hStep);
    const n = dim;
    const md = lu.U.data;
    const jd = (J as Matrix).data;
    const c = D * hStep;
    let hasNaN = false;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = (i === j ? 1 : 0) - c * jd[i * n + j];
        md[i * n + j] = v;
        if (v !== v) hasNaN = true;
      }
    }
    if (hasNaN) {
      luValid = false;
      return false;
    }
    try {
      lu.U.lu(lu);
    } catch {
      // Deliberately any error: a singular iteration matrix (`Error`) and one with an infinite
      // entry (`RangeError`) both mean that this step size is unusable, so the step is retried
      // with a smaller one.
      luValid = false;
      return false;
    }
    decompositions++;
    luH = hStep;
    luValid = true;
    return true;
  };

  // Banded counterpart of `factorize`: builds I - D*h*J straight in band storage (the J band and
  // the iteration band have the same shape, so entry k of one is entry k of the other) and factors
  // it into the preallocated `luBand`. The NaN scan is needed because the banded `lu` propagates
  // NaN instead of failing; the catch-all covers a zero pivot (`Error`) and an infinite entry
  // (`RangeError`), exactly as in the dense path.
  const factorizeBanded = (hStep: number): boolean => {
    const md = (iterBand as BandedMatrix).data;
    const jd = (JB as BandedMatrix).data;
    const ldab = kl + ku + 1;
    const c = D * hStep;
    let hasNaN = false;
    for (let j = 0; j < dim; j++) {
      const col = j * ldab;
      // Rows of column j inside the matrix; the unused corners of the storage stay 0.
      const k0 = Math.max(0, ku - j);
      const k1 = Math.min(ldab - 1, ku + dim - 1 - j);
      for (let k = k0; k <= k1; k++) {
        const v = (k === ku ? 1 : 0) - c * jd[col + k];
        md[col + k] = v;
        if (v !== v) hasNaN = true;
      }
    }
    if (hasNaN) {
      luValid = false;
      return false;
    }
    try {
      (iterBand as BandedMatrix).lu(luBand as BandedLUDecomposition);
    } catch {
      luValid = false;
      return false;
    }
    decompositions++;
    luH = hStep;
    luValid = true;
    return true;
  };

  // Solves (I - D h J) x = b with the current factorization, dense or banded.
  const solveIteration = (b: Vector, out: Vector): Vector =>
    banded
      ? BandedMatrix.solveLU(luBand as BandedLUDecomposition, b, out)
      : Matrix.solveLU(lu, b, out);

  // Solves z - h f(tStage, base + D z) = 0 for z by simplified Newton iteration, starting from
  // the current content of `zIter`. Each iteration costs one evaluation of f and ends with the
  // updated z, which is used as the stage value without a further evaluation. Returns the number
  // of iterations used, or 0 on failure.
  const newton = (tStage: number, hStep: number): number => {
    const zd = zIter.data;
    const bd = base.data;
    const ysd = yStage.data;
    const rd = resid.data;
    const sd = scale.data;
    let normOld = 0;

    for (let k = 0; k < MAX_NEWTON_ITERATIONS; k++) {
      for (let i = 0; i < dim; i++) ysd[i] = bd[i] + D * zd[i];
      fc(tStage, yStage, fStage);
      const fd = fStage.data;
      for (let i = 0; i < dim; i++) rd[i] = hStep * fd[i] - zd[i];
      solveIteration(resid, delta);
      const dd = delta.data;

      let sumSq = 0;
      for (let i = 0; i < dim; i++) {
        const e = dd[i] / sd[i];
        sumSq += e * e;
      }
      const norm = Math.sqrt(sumSq / dim);
      if (!Number.isFinite(norm)) return 0;

      for (let i = 0; i < dim; i++) zd[i] += dd[i];
      if (norm < NEWTON_KAPPA) return k + 1;

      if (k > 0) {
        const rate = norm / normOld;
        if (
          rate >= 1 ||
          (Math.pow(rate, MAX_NEWTON_ITERATIONS - k) / (1 - rate)) * norm > NEWTON_KAPPA
        )
          return 0;
      }
      normOld = norm;
    }
    return 0;
  };

  const detector = events === undefined ? null : new EventDetector(events, t0, y);
  // Continuous extension of the accepted step under evaluation. It is built lazily, on first use,
  // from the loop variables as they are at that time, so it must be used before the state swap,
  // which would discard the step's start state.
  const stepPolynomial = new StepPolynomial(dim, DENSE_DEGREE, (c) =>
    trBdf2AdaptiveDenseCoefficients(y, yNew, yG, zN, c)
  );
  const dense = denseOutput ? new OdeDenseOutput(t0, y) : null;

  const recorder = new SolutionRecorder(t0, y);
  const makeResult = (success: boolean, message: string): OdeResult => {
    const result = recorder.result({
      method: 'trbdf2',
      success,
      message,
      evaluations,
      evaluationsJacobian,
      decompositions
    });
    if (detector !== null) result.events = detector.occurrences;
    if (dense !== null) result.denseOutput = dense;
    return result;
  };

  const underflowMessage = (hNext: number, reason: string): string =>
    `trBdf2Adaptive: step size underflowed below hMin (${hMin}) near t=${t}` +
    (Number.isNaN(hNext) ? ' (step size became NaN)' : '') +
    `; ${reason}`;

  let lastRejected = false;
  let steps = 0;
  while (t !== tEnd) {
    if (steps >= maxSteps) {
      return makeResult(
        false,
        `trBdf2Adaptive: exceeded maxSteps (${maxSteps}) without reaching tEnd.`
      );
    }
    steps++;

    const remaining = tEnd - t;
    let isFinalStep = false;
    if (Math.abs(h) >= Math.abs(remaining)) {
      h = remaining;
      isFinalStep = true;
    }

    if (J === null && JB === null) refreshJacobian();
    let ok = luValid && h === luH ? true : factorize(h);

    if (ok) {
      const yd = y.data;
      const znd = zN.data;
      const zgd = zG.data;
      const zid = zIter.data;
      const bd = base.data;
      const ygd = yG.data;
      const sd = scale.data;
      const dyd = dY.data;
      for (let i = 0; i < dim; i++) {
        sd[i] = atol + rtol * Math.abs(yd[i]);
        znd[i] = h * dyd[i]; // smoothed first stage, rescaled to the current step size
      }

      // Stage 2 (t_n + GAMMA h): initial guess z_n.
      const tG = t + GAMMA * h;
      for (let i = 0; i < dim; i++) {
        bd[i] = yd[i] + D * znd[i];
        zid[i] = znd[i];
      }
      ok = newton(tG, h) > 0;
      if (ok) {
        zG.set(zid);

        // Stage 3 (t_n + h): initial guess from the extrapolated Hermite cubic.
        for (let i = 0; i < dim; i++) {
          ygd[i] = yd[i] + D * znd[i] + D * zgd[i];
          bd[i] = yd[i] + W * znd[i] + W * zgd[i];
          zid[i] = GUESS_ZN * znd[i] + GUESS_ZG * zgd[i] - GUESS_DY * (ygd[i] - yd[i]);
        }
        ok = newton(t + h, h) > 0;
        if (ok) {
          z1.set(zid);
          const nd = yNew.data;
          const z1d = z1.data;
          for (let i = 0; i < dim; i++) nd[i] = bd[i] + D * z1d[i];
        }
      }
    }

    if (!ok) {
      if (!jacCurrent) {
        // A stale Jacobian may be the culprit: refresh it and retry with the same step.
        refreshJacobian();
        continue;
      }
      const hNext = h * NEWTON_FAILURE_SHRINK;
      if (!(Math.abs(hNext) >= hMin)) {
        return makeResult(
          false,
          underflowMessage(
            hNext,
            'the Newton iteration failed to converge; the problem may be singular, or the derivative function may return non-finite values.'
          )
        );
      }
      h = hNext;
      lastRejected = true;
      continue;
    }

    // Local error estimate Est, solving (I - h D J) Est = est with the factored matrix.
    {
      const ed = resid.data;
      const znd = zN.data;
      const zgd = zG.data;
      const z1d = z1.data;
      for (let i = 0; i < dim; i++) ed[i] = E_N * znd[i] + E_G * zgd[i] + E_1 * z1d[i];
    }
    const filtered = solveIteration(resid, delta);
    let sumSq = 0;
    {
      const yd = y.data;
      const nd = yNew.data;
      const fd = filtered.data;
      for (let i = 0; i < dim; i++) {
        const e = fd[i] / (atol + rtol * Math.max(Math.abs(yd[i]), Math.abs(nd[i])));
        sumSq += e * e;
      }
    }
    let err = Math.sqrt(sumSq / dim);
    if (!Number.isFinite(err)) err = Infinity;

    if (err <= 1) {
      const tNew = isFinalStep ? tEnd : t + h;
      stepPolynomial.reset(t, h);
      if (detector !== null) {
        // Before the state swap, which would discard the step's start state.
        const stop = detector.check(tNew, yNew, stepPolynomial.evaluate);
        if (stop !== null) {
          dense?.append(stop.t, stepPolynomial);
          recorder.push(stop.t, stop.y);
          return makeResult(true, 'Integration stopped by a terminal event.');
        }
      }
      dense?.append(tNew, stepPolynomial);
      t = tNew;
      [y, yNew] = [yNew, y];
      recorder.push(t, y);

      // Smoothed first stage for the next step: the last stage, rescaled by the new step size.
      const z1d = z1.data;
      const dyd = dY.data;
      for (let i = 0; i < dim; i++) dyd[i] = z1d[i] / h;
      jacCurrent = false;

      let factor =
        err === 0 ? maxScale : clip(safety * Math.pow(err, ERROR_EXPONENT), minScale, maxScale);
      if (lastRejected) factor = Math.min(factor, 1);
      if (factor < MIN_STEP_CHANGE) factor = 1;
      h = dir * Math.min(Math.abs(h) * factor, hMax);
      lastRejected = false;
    } else {
      const shrink = clip(safety * Math.pow(err, ERROR_EXPONENT), minScale, maxScale);
      const hNext = h * shrink;
      if (!(Math.abs(hNext) >= hMin)) {
        return makeResult(
          false,
          underflowMessage(hNext, 'atol/rtol may be too tight for the problem.')
        );
      }
      h = hNext;
      lastRejected = true;
    }
  }

  return makeResult(true, 'Integration successful.');
}

/**
 * Validates the `kl` and `ku` options of {@link trBdf2Adaptive}.
 * @returns Whether banded mode is requested.
 * @throws {RangeError} If only one of them is given, or one is not a non-negative integer.
 * @throws {TypeError} If one is not a number.
 */
function validateBandwidth(kl: unknown, ku: unknown): boolean {
  if (kl === undefined && ku === undefined) return false;
  if (kl === undefined || ku === undefined) {
    throw new RangeError('trBdf2Adaptive: kl and ku must be given together.');
  }
  for (const [name, value] of [
    ['kl', kl],
    ['ku', ku]
  ] as const) {
    if (typeof value !== 'number') {
      throw new TypeError(`trBdf2Adaptive: ${name} must be a number, got ${typeName(value)}.`);
    }
    if (!Number.isInteger(value) || value < 0) {
      throw new RangeError(`trBdf2Adaptive: ${name} must be a non-negative integer, got ${value}.`);
    }
  }
  return true;
}
