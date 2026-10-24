import { Vector, type VectorLike } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import type { LUDecomposition } from '../linalg/Matrix.js';
import { clip } from '../math.js';
import { jacobianForward } from '../numdiff.js';
import { OdeDenseOutput, StepPolynomial } from './denseOutput.js';
import { EventDetector, validateEvents } from './events.js';
import { estimateInitialStep } from './stepControl.js';
import type { DerivativeFunction, JacobianFunction, OdeEvent, OdeResult } from './types.js';

// -----------------------------------------------------------------
// TR-BDF2 (Bank et al., 1985), in the three-stage DIRK form and with the implementation
// choices recommended by Hosea & Shampine (1996), "Analysis and implementation of TR-BDF2".
//
// With scaled derivatives z = h f, one step from (x_n, y_n) to x_n + h has the stages
//   1. z_n = h f(x_n, y_n)                                  (explicit, c = 0)
//   2. z_g = h f(x_n + GAMMA h, y_n + D z_n + D z_g)         (implicit, c = GAMMA)
//   3. z_1 = h f(x_n + h, y_n + W z_n + W z_g + D z_1)       (implicit, c = 1)
// and y_{n+1} = y_n + W z_n + W z_g + D z_1. Both implicit stages share the Newton iteration
// matrix I - h D J, so one LU factorization serves the whole step.
// -----------------------------------------------------------------
const GAMMA = 2 - Math.SQRT2;
const D = GAMMA / 2;
const W = Math.SQRT2 / 4;

// Stage-3 initial guess: derivative at x_n + h of the cubic Hermite interpolant of
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

/**
 * Degree of the continuous extension of a TR-BDF2 step; see
 * {@link trBdf2AdaptiveDenseCoefficients}.
 */
const DENSE_DEGREE = 3;

/**
 * Computes the coefficients of the continuous extension of an accepted
 * TR-BDF2 step, without evaluating `f`.
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
export interface TrBdf2AdaptiveOptions {
  /**
   * Absolute error tolerance per state component.
   * @default 1e-6
   */
  atol?: number;
  /**
   * Relative error tolerance per state component.
   * @default 1e-3
   */
  rtol?: number;
  /**
   * Optional initial step-size magnitude. Its sign is ignored because the
   * integration direction follows `tEnd - t0`. Estimated automatically when
   * omitted.
   */
  h0?: number;
  /**
   * Maximum permitted step-size magnitude.
   * @default Infinity
   */
  hMax?: number;
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
   * Safety factor applied when scaling a step after error estimation. Must
   * be in `(0, 1]`.
   * @default 0.9
   */
  safety?: number;
  /**
   * Smallest allowed multiplier for the next step size.
   * @default 0.2
   */
  minScale?: number;
  /**
   * Largest allowed multiplier for the next step size.
   * @default 10
   */
  maxScale?: number;
  /**
   * Optional analytic Jacobian `∂f/∂y`; see {@link JacobianFunction}. When omitted,
   * the Jacobian is approximated by forward finite differences, which costs `n + 1`
   * derivative evaluations per refresh, where `n` is the number of state components.
   */
  jac?: JacobianFunction;
  /**
   * Events to detect along the solution; see {@link OdeEvent}. Occurrences
   * are located inside each accepted step from the step's dense output and
   * reported in {@link OdeResult.events}.
   * Detection does not change the step sequence. If omitted, no events are evaluated.
   */
  events?: OdeEvent[];
  /**
   * Whether to also return the dense output, a continuous approximation of
   * the whole solution that can be evaluated at any time in the solved
   * interval; see {@link OdeResult.denseOutput}. It is built from the step
   * interpolants, which also serve the `events` option, at no further
   * evaluations of `f`, and does not change the step sequence. It stores a
   * few vectors per accepted step.
   * @default false
   */
  denseOutput?: boolean;
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
 * An event function that is exactly zero at `t0` is not reported.
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
 * @param f Derivative function `dy/dt = f(t, y)`. Must write its result into
 * the supplied `dydt` buffer rather than allocating.
 * @param t0 Starting time.
 * @param tEnd Final integration time. May be less than `t0` to integrate
 * backward; the solver selects the required direction automatically.
 * @param y0 Initial state at `t0`: a `Vector`, a plain `number[]` or a `Float64Array`. It
 * must have at least one component and is copied internally, so it is never mutated.
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
 * @throws {RangeError} If the state has no components, tolerances are invalid, a
 * step-size or controller limit is invalid, or an event option is invalid.
 * @throws {TypeError} If `y0` is not a `VectorLike`, or if an event function is not a
 * function.
 * Step-size underflow and exceeding `maxSteps` return an unsuccessful result
 * with the reason in `message`.
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
 * {
 *  success: true,
 *  steps: 21,
 *  evaluations: 84,
 *  yEnd: 0.5411713864049918
 * }
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
 *  success: true,
 *  evaluations: 652,
 *  evaluationsJacobian: 1,
 *  decompositions: 37,
 *  yEnd: 0.5411437997526206
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
 *  message: 'Integration stopped by a terminal event.',
 *  hitTime: 1.5717963234203387,
 *  hitDirection: 'falling',
 *  steps: 577,
 *  evaluations: 2355
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
 */
export function trBdf2Adaptive(
  f: DerivativeFunction,
  t0: number,
  tEnd: number,
  y0: VectorLike,
  options: TrBdf2AdaptiveOptions = {}
): OdeResult {
  const {
    atol = 1e-6,
    rtol = 1e-3,
    h0,
    hMax = Infinity,
    hMin = 1e-12,
    maxSteps = 1e5,
    safety = 0.9,
    minScale = 0.2,
    maxScale = 10,
    jac,
    events,
    denseOutput = false
  } = options;
  // Owned copy: it becomes the working state, so the caller's data is never mutated.
  const y0v = Vector.asVector(y0, true);
  const dim = y0v.size;
  if (dim <= 0)
    throw new RangeError(
      `trBdf2Adaptive: y0 must have at least one component, got dimension ${dim}.`
    );
  if (!Number.isFinite(atol) || !Number.isFinite(rtol) || atol < 0 || rtol < 0) {
    throw new RangeError(
      `trBdf2Adaptive: atol (${atol}) and rtol (${rtol}) must be finite and non-negative.`
    );
  }
  if (atol === 0 && rtol === 0)
    throw new RangeError(
      'trBdf2Adaptive: atol and rtol cannot both be 0; no step could ever satisfy the tolerance.'
    );
  if (h0 !== undefined && h0 === 0) throw new RangeError('trBdf2Adaptive: h0 must be nonzero.');
  if (!(hMin > 0)) throw new RangeError(`trBdf2Adaptive: hMin (${hMin}) must be positive.`);
  if (!(hMax > 0)) throw new RangeError(`trBdf2Adaptive: hMax (${hMax}) must be positive.`);
  if (hMin > hMax)
    throw new RangeError(`trBdf2Adaptive: hMin (${hMin}) must not exceed hMax (${hMax}).`);
  if (!Number.isInteger(maxSteps) || maxSteps <= 0) {
    throw new RangeError(`trBdf2Adaptive: maxSteps (${maxSteps}) must be a positive integer.`);
  }
  if (!(safety > 0 && safety <= 1))
    throw new RangeError(`trBdf2Adaptive: safety (${safety}) must be in (0, 1].`);
  if (!(minScale > 0) || minScale > maxScale) {
    throw new RangeError(
      `trBdf2Adaptive: minScale (${minScale}) must be positive and not exceed maxScale (${maxScale}).`
    );
  }
  if (events !== undefined) validateEvents(events, 'trBdf2Adaptive');

  let evaluations = 0;
  let evaluationsJacobian = 0;
  let decompositions = 0;

  if (tEnd === t0) {
    const tVec = new Vector(1);
    tVec.data[0] = t0;
    const yMat = new Matrix(1, dim);
    yMat.setRow(0, y0v.data);
    const trivial: OdeResult = {
      method: 'trbdf2',
      success: true,
      message: 'Integration successful.',
      evaluations,
      evaluationsJacobian,
      decompositions,
      t: tVec,
      y: yMat
    };
    if (events !== undefined) trivial.events = [];
    if (denseOutput) trivial.denseOutput = new OdeDenseOutput(t0, y0v);
    return trivial;
  }

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
  const yG = new Vector(dim); // intermediate state at x_n + GAMMA h
  const base = new Vector(dim); // stage state not depending on the unknown z
  const yStage = new Vector(dim); // stage state passed to f
  const fStage = new Vector(dim);
  const fJac = new Vector(dim); // fresh f(t_n, y_n) for finite-difference Jacobians
  const resid = new Vector(dim); // Newton right-hand side, later the raw error estimate
  const scale = new Vector(dim); // weights atol + rtol |y_n|
  const delta = new Vector(dim); // Newton correction, later the filtered error estimate
  // Storage of the factorization of I - D h J, reused by every call to `lu`.
  const lu: LUDecomposition = {
    L: new Matrix(dim, dim),
    U: new Matrix(dim, dim),
    perm: new Array<number>(dim),
    sign: 1
  };
  const jacBuf = jac !== undefined ? new Matrix(dim, dim) : null;

  let t = t0;

  // Jacobian and factorization state. `J` is assigned inside closures, so it is declared through an
  // assertion to keep TypeScript from narrowing it to `null`.
  let J = null as Matrix | null;
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
    if (jacBuf !== null && jac !== undefined) {
      jacBuf.fill(0);
      jac(t, y, jacBuf);
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

  // Factors I - D*h*J into `lu`, in place. Returns false if the matrix is exactly singular.
  const factorize = (hStep: number): boolean => {
    const n = dim;
    const md = lu.U.data;
    const jd = (J as Matrix).data;
    const c = D * hStep;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) md[i * n + j] = (i === j ? 1 : 0) - c * jd[i * n + j];
    }
    try {
      lu.U.lu(lu);
    } catch {
      luValid = false;
      return false;
    }
    decompositions++;
    luH = hStep;
    luValid = true;
    return true;
  };

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
      Matrix.luSolve(lu, resid, delta);
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

  const tBuf: number[] = [t0];
  const yBuf: Float64Array[] = [y.data.slice()];
  const makeResult = (success: boolean, message: string): OdeResult => {
    const tVec = new Vector(tBuf.length);
    tVec.set(tBuf);
    const yMat = new Matrix(tBuf.length, dim);
    for (let i = 0; i < tBuf.length; i++) {
      yMat.setRow(i, yBuf[i]);
    }
    const result: OdeResult = {
      method: 'trbdf2',
      success,
      message,
      evaluations,
      evaluationsJacobian,
      decompositions,
      t: tVec,
      y: yMat
    };
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

    if (J === null) refreshJacobian();
    let ok = luValid && h === luH ? true : factorize(h);

    let iterations1: number;
    let iterations2: number;
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

      // Stage 2 (x_n + GAMMA h): initial guess z_n.
      const tG = t + GAMMA * h;
      for (let i = 0; i < dim; i++) {
        bd[i] = yd[i] + D * znd[i];
        zid[i] = znd[i];
      }
      iterations1 = newton(tG, h);
      ok = iterations1 > 0;
      if (ok) {
        zG.set(zid);

        // Stage 3 (x_n + h): initial guess from the extrapolated Hermite cubic.
        for (let i = 0; i < dim; i++) {
          ygd[i] = yd[i] + D * znd[i] + D * zgd[i];
          bd[i] = yd[i] + W * znd[i] + W * zgd[i];
          zid[i] = GUESS_ZN * znd[i] + GUESS_ZG * zgd[i] - GUESS_DY * (ygd[i] - yd[i]);
        }
        iterations2 = newton(t + h, h);
        ok = iterations2 > 0;
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
    const filtered = Matrix.luSolve(lu, resid, delta);
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
          tBuf.push(stop.t);
          yBuf.push(stop.y.data.slice());
          return makeResult(true, 'Integration stopped by a terminal event.');
        }
      }
      dense?.append(tNew, stepPolynomial);
      t = tNew;
      [y, yNew] = [yNew, y];
      tBuf.push(t);
      yBuf.push(y.data.slice());

      // Smoothed first stage for the next step: the last stage, rescaled by the new step size.
      const z1d = z1.data;
      const dyd = dY.data;
      for (let i = 0; i < dim; i++) dyd[i] = z1d[i] / h;
      jacCurrent = false;

      let factor = err === 0 ? maxScale : clip(safety * Math.pow(err, -1 / 3), minScale, maxScale);
      if (lastRejected) factor = Math.min(factor, 1);
      if (factor < MIN_STEP_CHANGE) factor = 1;
      h = dir * Math.min(Math.abs(h) * factor, hMax);
      lastRejected = false;
    } else {
      const shrink = clip(safety * Math.pow(err, -1 / 3), minScale, maxScale);
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
