import { Vector } from '../linalg/Vector.js';
import { Matrix } from '../linalg/Matrix.js';
import { clip } from '../math.js';
import { EventDetector, validateEvents, type StepInterpolant } from './events.js';
import { estimateInitialStep } from './stepControl.js';
import type {
    DerivativeFunction,
    OdeEvent,
    OdeResult,
    RungeKuttaAdaptiveMethod,
} from './types.js';

/**
 * Reusable intermediate vectors for adaptive Runge-Kutta integration steps.
 *
 * The workspace must contain vectors with the same dimension as the state
 * passed to {@link rungeKuttaAdaptiveStep}. The solver uses `k1` and the final
 * stage (`k4` for `'rk23'`, `k7` for `'rk45'`) to reuse first-same-as-last
 * derivatives between accepted steps.
 */
export interface AdaptiveScratch {
    /** First stage derivative; may contain the derivative reused from the previous accepted step. */
    k1: Vector;
    /** Second stage derivative. */
    k2: Vector;
    /** Third stage derivative. */
    k3: Vector;
    /** Fourth stage derivative and the FSAL derivative for `'rk23'`. */
    k4: Vector;
    /** Fifth stage derivative, used by `'rk45'`. */
    k5: Vector;
    /** Sixth stage derivative, used by `'rk45'`. */
    k6: Vector;
    /** Seventh stage derivative and the FSAL derivative for `'rk45'`. */
    k7: Vector;
    /** Temporary state vector used to evaluate intermediate stages. */
    yTemp: Vector;
}

/**
 * Allocates a scratch workspace of vectors reused across every stage.
 */
function makeAdaptiveScratch(dim: number): AdaptiveScratch {
    return {
        k1: new Vector(dim),
        k2: new Vector(dim),
        k3: new Vector(dim),
        k4: new Vector(dim),
        k5: new Vector(dim),
        k6: new Vector(dim),
        k7: new Vector(dim),
        yTemp: new Vector(dim),
    };
}

/**
 * Advances the state by one attempted step of the chosen adaptive method.
 *
 * This computes the higher-order solution only; it does not accept or reject
 * the step. Callers that need adaptive control should use
 * {@link rungeKuttaAdaptive}. The embedded lower-order solution remains
 * implicit in `scratch` and is used internally for error estimation.
 *
 * @param method Embedded Runge-Kutta pair: `'rk23'` for Bogacki-Shampine 3(2),
 * or `'rk45'` for Dormand-Prince 5(4).
 * @param f Derivative function `dy/dt = f(t, y)`. It must write into the
 * provided `dydt` buffer rather than allocate a result.
 * @param t Time at which `y` is the state.
 * @param y Current state vector. This function does not mutate it.
 * @param h Attempted step size; a negative value integrates backward in time.
 * @param out Vector that receives the higher-order state at `t + h`, mutated
 * in place. It must have the same dimension as `y`.
 * @param scratch Reusable intermediate-stage workspace whose vectors must have
 * the same dimension as `y`.
 * @param k1Ready Whether `scratch.k1` already contains `f(t, y)`, usually from
 * the first-same-as-last stage of a prior accepted step. Defaults to `false`.
 * @returns `out`, for chaining.
 * @throws {RangeError} If `method` is not a recognized adaptive method.
 */
export function rungeKuttaAdaptiveStep(
    method: RungeKuttaAdaptiveMethod,
    f: DerivativeFunction,
    t: number,
    y: Vector,
    h: number,
    out: Vector,
    scratch: AdaptiveScratch,
    k1Ready: boolean = false
): Vector {
    const { k1, k2, k3, k4, k5, k6, k7, yTemp } = scratch;

    if (!k1Ready) f(t, y, k1);

    switch (method) {
        case 'rk23': {
            // Bogacki-Shampine 3(2)
            yTemp.set(y.data).addScaled(k1, h * 0.5);
            f(t + 0.5 * h, yTemp, k2);

            yTemp.set(y.data).addScaled(k2, h * 0.75);
            f(t + 0.75 * h, yTemp, k3);

            out.set(y.data)
                .addScaled(k1, h * (2 / 9))
                .addScaled(k2, h * (1 / 3))
                .addScaled(k3, h * (4 / 9));

            f(t + h, out, k4); // FSAL stage
            break;
        }

        case 'rk45': {
            // Dormand-Prince 5(4)
            yTemp.set(y.data).addScaled(k1, h * (1 / 5));
            f(t + (1 / 5) * h, yTemp, k2);

            yTemp.set(y.data).addScaled(k1, h * (3 / 40)).addScaled(k2, h * (9 / 40));
            f(t + (3 / 10) * h, yTemp, k3);

            yTemp.set(y.data).addScaled(k1, h * (44 / 45)).addScaled(k2, h * (-56 / 15)).addScaled(k3, h * (32 / 9));
            f(t + (4 / 5) * h, yTemp, k4);

            yTemp.set(y.data).addScaled(k1, h * (19372 / 6561)).addScaled(k2, h * (-25360 / 2187)).addScaled(k3, h * (64448 / 6561)).addScaled(k4, h * (-212 / 729));
            f(t + (8 / 9) * h, yTemp, k5);

            yTemp.set(y.data).addScaled(k1, h * (9017 / 3168)).addScaled(k2, h * (-355 / 33)).addScaled(k3, h * (46732 / 5247)).addScaled(k4, h * (49 / 176)).addScaled(k5, h * (-5103 / 18656));
            f(t + h, yTemp, k6); // c6 = 1

            out.set(y.data)
                .addScaled(k1, h * (35 / 384))
                .addScaled(k3, h * (500 / 1113))
                .addScaled(k4, h * (125 / 192))
                .addScaled(k5, h * (-2187 / 6784))
                .addScaled(k6, h * (11 / 84));

            f(t + h, out, k7); // FSAL stage
            break;
        }

        default:
            throw new RangeError(`rungeKuttaAdaptiveStep: unknown method "${method satisfies never}"`);
    }
    return out;
}

// Dense-output coefficients (rows: stages k1..ks; columns: powers x^1..x^p of the relative position
// x = (t - t_n) / h in the step), from Bogacki & Shampine (cubic) and Shampine (1986) (quartic).
const DENSE_RK23 = [
    [1, -4 / 3, 5 / 9],
    [0, 1, -2 / 3],
    [0, 4 / 3, -8 / 9],
    [0, -1, 1],
];
const DENSE_RK45 = [
    [1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
    [0, 0, 0, 0],
    [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
    [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
    [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
    [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
    [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
];

/**
 * Evaluates the continuous extension of an attempted step inside the step, without evaluating `f`.
 *
 * Uses the stages left in `scratch` by {@link rungeKuttaAdaptiveStep}: a cubic polynomial for
 * `'rk23'` (error `O(h^4)`) and a quartic one for `'rk45'` (error `O(h^5)`). The interpolant
 * reproduces `y` at `x = 0` and the step's end state at `x = 1` up to rounding.
 *
 * @param method Embedded Runge-Kutta pair, as passed to {@link rungeKuttaAdaptiveStep}.
 * @param y State at the start of the step, as passed to {@link rungeKuttaAdaptiveStep}.
 * @param h Step size used by {@link rungeKuttaAdaptiveStep}.
 * @param scratch Workspace holding the stages of that step; all stages must be intact, so the
 * first-same-as-last copy into `k1` must not have happened yet.
 * @param x Relative position in the step, `(t - t_n) / h`; `[0, 1]` is inside the step.
 * @param out Vector that receives the interpolated state, mutated in place. It must not alias `y`.
 * @returns `out`, for chaining.
 */
export function rungeKuttaAdaptiveDense(
    method: RungeKuttaAdaptiveMethod,
    y: Vector,
    h: number,
    scratch: AdaptiveScratch,
    x: number,
    out: Vector
): Vector {
    const { k1, k2, k3, k4, k5, k6, k7 } = scratch;
    const coefficients = method === 'rk23' ? DENSE_RK23 : DENSE_RK45;
    const stages = method === 'rk23' ? [k1, k2, k3, k4] : [k1, k2, k3, k4, k5, k6, k7];
    out.set(y.data);
    for (let i = 0; i < stages.length; i++) {
        const row = coefficients[i];
        let weight = 0;
        let power = 1;
        for (let j = 0; j < row.length; j++) {
            power *= x;
            weight += row[j] * power;
        }
        if (weight !== 0) out.addScaled(stages[i], h * weight);
    }
    return out;
}

/**
 * Computes the weighted RMS local-error norm for an attempted step.
 */
function adaptiveErrorNorm(
    method: RungeKuttaAdaptiveMethod,
    y: Vector,
    out: Vector,
    h: number,
    scratch: AdaptiveScratch,
    atol: number,
    rtol: number
): number {
    const dim = y.size;
    const yd = y.data;
    const od = out.data;
    let sumSq = 0;

    switch (method) {
        case 'rk23': {
            const { k1, k2, k3, k4 } = scratch;
            const k1d = k1.data, k2d = k2.data, k3d = k3.data, k4d = k4.data;
            for (let i = 0; i < dim; i++) {
                const errI = h * ((-5 / 72) * k1d[i] + (1 / 12) * k2d[i] + (1 / 9) * k3d[i] - (1 / 8) * k4d[i]);
                const scale = atol + rtol * Math.max(Math.abs(yd[i]), Math.abs(od[i]));
                const e = errI / scale;
                sumSq += e * e;
            }
            break;
        }

        case 'rk45': {
            const { k1, k3, k4, k5, k6, k7 } = scratch;
            const k1d = k1.data, k3d = k3.data, k4d = k4.data, k5d = k5.data, k6d = k6.data, k7d = k7.data;
            for (let i = 0; i < dim; i++) {
                const errI = h * ((71 / 57600) * k1d[i] - (71 / 16695) * k3d[i] + (71 / 1920) * k4d[i] - (17253 / 339200) * k5d[i] + (22 / 525) * k6d[i] - (1 / 40) * k7d[i]);
                const scale = atol + rtol * Math.max(Math.abs(yd[i]), Math.abs(od[i]));
                const e = errI / scale;
                sumSq += e * e;
            }
            break;
        }

        default:
            throw new RangeError(`adaptiveErrorNorm: unknown method "${method satisfies never}"`);
    }
    return Math.sqrt(sumSq / dim);
}

/**
 * Optional tuning parameters for {@link rungeKuttaAdaptive}.
 */
export interface RungeKuttaAdaptiveOptions {
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
     * Minimum permitted step-size magnitude after a rejected step.
     * @default 1e-12
     */
    hMin?: number;
    /**
     * Maximum number of attempted steps, including rejected steps.
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
     * Events to detect along the solution; see {@link OdeEvent}. Occurrences are located inside
     * each accepted step from the step's dense output and reported in {@link OdeResult.events}.
     * Detection does not change the step sequence. If omitted, no events are evaluated.
     */
    events?: OdeEvent[];
}

/**
 * Integrates `dy/dt = f(t, y)` from `t0` to `tEnd` using adaptive step size control.
 *
 * Uses Bogacki-Shampine 3(2) (`'rk23'`) or Dormand-Prince 5(4) (`'rk45'`) with
 * a weighted RMS error norm. Accepted steps satisfy the requested absolute and
 * relative tolerances; rejected steps are retried with a smaller step. The
 * initial step is estimated when `h0` is omitted, and the final step is
 * shortened so the last recorded time is exactly `tEnd`.
 *
 * Zero crossings of user-supplied event functions can be located with the `events` option.
 * After each accepted step the event functions are evaluated, and a sign change triggers a 
 * root search on the step's dense output (a cubic for `'rk23'`, a quartic for `'rk45'`), 
 * which costs no further evaluations of `f`. Event times are therefore as accurate as the 
 * step's interpolant, comparable to the integration tolerances. Non-terminal events do not 
 * alter the integration. A terminal event ends it at the event time, which becomes the last 
 * recorded point. Only sign changes between accepted steps are seen, so use `hMax` if two 
 * crossings can occur within a step. An event function that is exactly zero at `t0` is not 
 * reported.
 *
 * Results are validated against the corresponding `scipy.integrate.solve_ivp`
 * `RK23` and `RK45` methods. With the same `atol` and `rtol`, the number of function 
 * calls is essentially identical to SciPy's across the benchmark problems;
 * see the [comparison table](https://github.com/HugoMVale/numerics-js/blob/main/benchmarks/ode/rungeKuttaAdaptive.scipy.benchmark.md).
 *
 * @param method Embedded Runge-Kutta pair: `'rk23'` for Bogacki-Shampine 3(2),
 * or `'rk45'` for Dormand-Prince 5(4).
 * @param f Derivative function `dy/dt = f(t, y)`. Must write its result into
 * the supplied `dydt` buffer rather than allocating.
 * @param t0 Starting time.
 * @param tEnd Final integration time. May be less than `t0` to integrate
 * backward; the solver selects the required direction automatically.
 * @param y0 Initial state at `t0`. It must have at least one component and is
 * copied internally, so it is never mutated.
 * @param options Optional tuning parameters; see {@link RungeKuttaAdaptiveOptions}.
 * @returns An {@link OdeResult}: `t`, the accepted time points, and `y`, an
 * `n x y0.size` matrix whose row `i` is the state at `t.get(i)`. Row 0 is
 * `y0`; on success, the last row is at `tEnd`. `evaluations` is the number of calls to `f`,
 * and `method` echoes `method`. The `success` flag indicates whether the integration completed,
 * i.e. reached `tEnd` or was stopped by a terminal event, and `message` provides additional 
 * status information. `events` is present when the `events` option was given.
 * @throws {RangeError} If the state has no components, tolerances are invalid,
 * a step-size or controller limit is invalid, an event option is invalid, or `method` is not
 * recognized when `t0 !== tEnd`.
 * @throws {TypeError} If an event function is not a function.
 * Step-size underflow and exceeding `maxSteps` return an unsuccessful result
 * with the reason in `message`.
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { rungeKuttaAdaptive, type DerivativeFunction } from 'numerics-js/ode';
 *
 * // Exponential decay: dy/dt = -y, y(0) = 1 -> y(t) = e^-t
 * const f: DerivativeFunction = (t, y, out) => out.set(y.data).multSelf(-1);
 * const result = rungeKuttaAdaptive('rk45', f, 0, 1, new Vector([1]));
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   method: 'rk45',
 *   success: true,
 *   message: 'Integration successful.',
 *   evaluations: 14,
 *   evaluationsJacobian: 0,
 *   decompositions: 0,
 *   t: Vector { data: Float64Array(3) [ 0, 0.14680437989650819, 1 ] },
 *   y: Matrix {
 *     _rows: 3,
 *     _cols: 1,
 *     data: Float64Array(3) [ 1, 0.863462874659396, 0.3680228572282582 ]
 *   }
 * }
 * ```
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { rungeKuttaAdaptive, type DerivativeFunction } from 'numerics-js/ode';
 *
 * // Same integration, with explicit tolerances and an initial step size.
 * const f: DerivativeFunction = (t, y, out) => out.set(y.data).multSelf(-1);
 * const result = rungeKuttaAdaptive('rk45', f, 0, 1, new Vector([1]), {
 *     atol: 1e-8,
 *     rtol: 1e-8,
 *     h0: 0.01,
 * });
 * console.log(result.y.get(result.y.rows - 1, 0));
 * ```
 *
 * Output:
 * ```text
 * 0.3678794431779414
 * ```
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { rungeKuttaAdaptive, type DerivativeFunction, type OdeEvent } from 'numerics-js/ode';
 *
 * // A ball dropped from 1 m. State: [height, velocity]. Stop when it reaches the ground.
 * const f: DerivativeFunction = (t, y, out) => {
 *     out.data[0] = y.data[1];
 *     out.data[1] = -9.81;
 *     return out;
 * };
 * const ground: OdeEvent = { fn: (t, y) => y.data[0], direction: 'falling', terminal: true };
 * const result = rungeKuttaAdaptive('rk45', f, 0, 5, new Vector([1, 0]), { events: [ground] });
 * const [hit] = result.events!;
 * console.log({
 *     message: result.message,
 *     hitTime: hit.t,
 *     hitHeight: hit.y.data[0],
 *     expectedTime: Math.sqrt(2 / 9.81),
 * });
 * ```
 *
 * Output:
 * ```text
 * { 
 *  message: 'Integration stopped by a terminal event.', 
 *  hitTime: 0.45152364098573017, 
 *  hitHeight: -1.1102230246251565e-16, 
 *  expectedTime: 0.4515236409857309 
 * }
 * ```
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { rungeKuttaAdaptive, type DerivativeFunction, type OdeEvent } from 'numerics-js/ode';
 *
 * // Simple harmonic oscillator: x'' = -x, starting at x = 1. Stop at the first downward zero crossing.
 * const oscillator: DerivativeFunction = (_t, y, out) => {
 *     out.data[0] = y.data[1];
 *     out.data[1] = -y.data[0];
 *     return out;
 * };
 * const zeroCrossing: OdeEvent = {
 *     fn: (_t, y) => y.data[0],
 *     direction: 'falling',
 *     terminal: true,
 * };
 * const result = rungeKuttaAdaptive('rk45', oscillator, 0, 2 * Math.PI, new Vector([1, 0]), {
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
 *  eventTime: 1.5707963266771703, 
 *  eventVelocity: -0.9999999993753101, 
 *  expectedTime: 1.5707963267948966
 * }
 * ```
 */
export function rungeKuttaAdaptive(
    method: RungeKuttaAdaptiveMethod,
    f: DerivativeFunction,
    t0: number,
    tEnd: number,
    y0: Vector,
    options: RungeKuttaAdaptiveOptions = {}
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
        events,
    } = options;
    const dim = y0.size;
    if (dim <= 0) throw new RangeError(`rungeKuttaAdaptive: y0 must have at least one component, got dimension ${dim}.`);
    if (!Number.isFinite(atol) || !Number.isFinite(rtol) || atol < 0 || rtol < 0) {
        throw new RangeError(`rungeKuttaAdaptive: atol (${atol}) and rtol (${rtol}) must be finite and non-negative.`);
    }
    if (atol === 0 && rtol === 0) throw new RangeError('rungeKuttaAdaptive: atol and rtol cannot both be 0; no step could ever satisfy the tolerance.');
    if (h0 !== undefined && h0 === 0) throw new RangeError('rungeKuttaAdaptive: h0 must be nonzero.');
    if (!(hMin > 0)) throw new RangeError(`rungeKuttaAdaptive: hMin (${hMin}) must be positive.`);
    if (!(hMax > 0)) throw new RangeError(`rungeKuttaAdaptive: hMax (${hMax}) must be positive.`);
    if (hMin > hMax) throw new RangeError(`rungeKuttaAdaptive: hMin (${hMin}) must not exceed hMax (${hMax}).`);
    if (!Number.isInteger(maxSteps) || maxSteps <= 0) {
        throw new RangeError(`rungeKuttaAdaptive: maxSteps (${maxSteps}) must be a positive integer.`);
    }
    if (!(safety > 0 && safety <= 1)) throw new RangeError(`rungeKuttaAdaptive: safety (${safety}) must be in (0, 1].`);
    if (!(minScale > 0) || minScale > maxScale) {
        throw new RangeError(`rungeKuttaAdaptive: minScale (${minScale}) must be positive and not exceed maxScale (${maxScale}).`);
    }
    if (events !== undefined) validateEvents(events, 'rungeKuttaAdaptive');

    if (tEnd === t0) {
        const tVec = new Vector(1);
        tVec.data[0] = t0;
        const yMat = new Matrix(1, dim);
        yMat.setRow(0, y0.data);
        const trivial: OdeResult = {
            method: method,
            success: true,
            message: 'Integration successful.',
            evaluations: 0,
            evaluationsJacobian: 0,
            decompositions: 0,
            t: tVec,
            y: yMat,
        };
        if (events !== undefined) trivial.events = [];
        return trivial;
    }

    const dir = Math.sign(tEnd - t0) as 1 | -1;
    const scratch = makeAdaptiveScratch(dim);

    let methodOrder: number;
    let errExp: number;
    let fsalSource: Vector;
    let stagesPerAttempt: number;
    switch (method) {
        case 'rk23':
            methodOrder = 3;
            errExp = -1 / 3;
            fsalSource = scratch.k4;
            stagesPerAttempt = 4;
            break;

        case 'rk45':
            methodOrder = 5;
            errExp = -1 / 5;
            fsalSource = scratch.k7;
            stagesPerAttempt = 7;
            break;

        default:
            throw new RangeError(`rungeKuttaAdaptive: unknown method "${method satisfies never}"`);
    }

    let t = t0;
    let y = y0.copy();
    let next = new Vector(dim);
    let evaluations = h0 === undefined ? 2 : 0;

    let k1Ready = false;
    let h: number;
    if (h0 !== undefined) {
        h = dir * Math.abs(h0);
    } else {
        h = estimateInitialStep(f, t0, y, dir, atol, rtol, methodOrder, scratch.k1, scratch.k2, scratch.yTemp);
        k1Ready = true;
    }
    h = dir * Math.min(Math.abs(h), hMax, Math.abs(tEnd - t0));

    const detector = events === undefined ? null : new EventDetector(events, t0, y);
    // Reads the loop variables at call time, so it always describes the step under evaluation.
    const dense: StepInterpolant = (tq, out) => rungeKuttaAdaptiveDense(method, y, h, scratch, (tq - t) / h, out);

    const tBuf: number[] = [t0];
    const yBuf: Float64Array[] = [y.data.slice()];
    const makeResult = (success: boolean, message: string): OdeResult => {
        const tVec = new Vector(tBuf.length);
        tVec.set(tBuf);
        const yMat = new Matrix(tBuf.length, dim);
        for (let i = 0; i < tBuf.length; i++) {
            yMat.setRow(i, yBuf[i]);
        }
        const result: OdeResult = { method, success, message, evaluations, evaluationsJacobian: 0, decompositions: 0, t: tVec, y: yMat };
        if (detector !== null) result.events = detector.occurrences;
        return result;
    };

    let steps = 0;
    while (t !== tEnd) {
        const remaining = tEnd - t;
        let isFinalStep = false;
        if (Math.abs(h) >= Math.abs(remaining)) {
            h = remaining;
            isFinalStep = true;
        }

        rungeKuttaAdaptiveStep(method, f, t, y, h, next, scratch, k1Ready);
        evaluations += stagesPerAttempt - (k1Ready ? 1 : 0);
        k1Ready = true;

        const err = adaptiveErrorNorm(method, y, next, h, scratch, atol, rtol);

        if (err <= 1) {
            const tNew = isFinalStep ? tEnd : t + h;
            if (detector !== null) {
                // Before the state swap and the FSAL copy, which would destroy the step's stages.
                const stop = detector.check(tNew, next, dense);
                if (stop !== null) {
                    tBuf.push(stop.t);
                    yBuf.push(stop.y.data.slice());
                    return makeResult(true, 'Integration stopped by a terminal event.');
                }
            }
            t = tNew;
            [y, next] = [next, y];
            tBuf.push(t);
            yBuf.push(y.data.slice());

            scratch.k1.set(fsalSource.data);

            const growth = err === 0 ? maxScale : clip(safety * Math.pow(err, errExp), minScale, maxScale);
            h = dir * Math.min(Math.abs(h) * growth, hMax);
        } else {
            const shrink = clip(safety * Math.pow(err, errExp), minScale, maxScale);
            const hNext = h * shrink;
            if (!(Math.abs(hNext) >= hMin)) {
                const message =
                    `rungeKuttaAdaptive: step size underflowed below hMin (${hMin}) near t=${t}` +
                    (Number.isNaN(hNext) ? ' (step size became NaN)' : '') +
                    `; the problem may be stiff, or atol/rtol too tight.`;
                return makeResult(false, message);
            }
            h = hNext;
        }

        steps++;
        if (t !== tEnd && steps >= maxSteps) {
            return makeResult(false, `rungeKuttaAdaptive: exceeded maxSteps (${maxSteps}) without reaching tEnd.`);
        }
    }

    return makeResult(true, 'Integration successful.');
}