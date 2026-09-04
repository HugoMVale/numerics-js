import { Array1D } from '../array/array1d.js';
import { Matrix } from '../array/Matrix.js';
import { clip } from '../math.js';
import type { DerivativeFunction, OdeResult, RungeKuttaAdaptiveMethod } from './types.js';

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
    k1: Array1D;
    /** Second stage derivative. */
    k2: Array1D;
    /** Third stage derivative. */
    k3: Array1D;
    /** Fourth stage derivative and the FSAL derivative for `'rk23'`. */
    k4: Array1D;
    /** Fifth stage derivative, used by `'rk45'`. */
    k5: Array1D;
    /** Sixth stage derivative, used by `'rk45'`. */
    k6: Array1D;
    /** Seventh stage derivative and the FSAL derivative for `'rk45'`. */
    k7: Array1D;
    /** Temporary state vector used to evaluate intermediate stages. */
    yTemp: Array1D;
}

/**
 * Allocates a scratch workspace of vectors reused across every stage.
 */
function makeAdaptiveScratch(dim: number): AdaptiveScratch {
    return {
        k1: new Array1D(dim),
        k2: new Array1D(dim),
        k3: new Array1D(dim),
        k4: new Array1D(dim),
        k5: new Array1D(dim),
        k6: new Array1D(dim),
        k7: new Array1D(dim),
        yTemp: new Array1D(dim),
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
    y: Array1D,
    h: number,
    out: Array1D,
    scratch: AdaptiveScratch,
    k1Ready: boolean = false
): Array1D {
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

/**
 * Computes the weighted RMS local-error norm for an attempted step.
 */
function adaptiveErrorNorm(
    method: RungeKuttaAdaptiveMethod,
    y: Array1D,
    out: Array1D,
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
 * Estimates a reasonable initial step size using Hairer, Norsett & Wanner's algorithm.
 */
function estimateInitialStep(
    f: DerivativeFunction,
    t0: number,
    y0: Array1D,
    dir: 1 | -1,
    atol: number,
    rtol: number,
    order: number,
    scratch: AdaptiveScratch
): number {
    const dim = y0.size;
    const f0 = scratch.k1;
    f(t0, y0, f0);

    const yd = y0.data;
    const f0d = f0.data;

    let d0Sq = 0;
    let d1Sq = 0;
    for (let i = 0; i < dim; i++) {
        const scale = atol + rtol * Math.abs(yd[i]);
        d0Sq += (yd[i] / scale) ** 2;
        d1Sq += (f0d[i] / scale) ** 2;
    }
    const d0 = Math.sqrt(d0Sq / dim);
    const d1 = Math.sqrt(d1Sq / dim);

    let h0 = d0 < 1e-5 || d1 < 1e-5 ? 1e-6 : 0.01 * (d0 / d1);

    const y1 = scratch.yTemp;
    const y1d = y1.data;
    for (let i = 0; i < dim; i++) {
        y1d[i] = yd[i] + dir * h0 * f0d[i];
    }
    const f1 = scratch.k2;
    f(t0 + dir * h0, y1, f1);
    const f1d = f1.data;

    let d2Sq = 0;
    for (let i = 0; i < dim; i++) {
        const scale = atol + rtol * Math.abs(yd[i]);
        d2Sq += ((f1d[i] - f0d[i]) / scale) ** 2;
    }
    const d2 = Math.sqrt(d2Sq / dim) / h0;

    let h1: number;
    if (Math.max(d1, d2) <= 1e-15) {
        h1 = Math.max(1e-6, h0 * 1e-3);
    } else {
        h1 = Math.pow(0.01 / Math.max(d1, d2), 1 / (order + 1));
    }

    return dir * Math.min(100 * h0, h1);
}

/**
 * Optional tuning parameters for {@link rungeKuttaAdaptive}.
 */
export interface RungeKuttaAdaptiveOptions {
    /** Absolute error tolerance per state component. Defaults to `1e-6`. */
    atol?: number;
    /** Relative error tolerance per state component. Defaults to `1e-3`. */
    rtol?: number;
    /**
     * Optional initial step-size magnitude. Its sign is ignored because the
     * integration direction follows `tEnd - t0`. Estimated automatically when
     * omitted.
     */
    h0?: number;
    /** Maximum permitted step-size magnitude. Defaults to `Infinity`. */
    hMax?: number;
    /**
     * Minimum permitted step-size magnitude after a rejected step. Defaults
     * to `1e-12`.
     */
    hMin?: number;
    /**
     * Maximum number of attempted steps, including rejected steps. Defaults
     * to `1e5`.
     */
    maxSteps?: number;
    /**
     * Safety factor applied when scaling a step after error estimation. Must
     * be in `(0, 1]`; defaults to `0.9`.
     */
    safety?: number;
    /** Smallest allowed multiplier for the next step size. Defaults to `0.2`. */
    minScale?: number;
    /** Largest allowed multiplier for the next step size. Defaults to `10`. */
    maxScale?: number;
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
 * Results are validated against the corresponding `scipy.integrate.solve_ivp`
 * `RK23` and `RK45` methods (see `tests/ode/rungeKuttaAdaptive.scipy.test.ts`).
 * With the same `atol` and `rtol`, the number of function calls is essentially 
 * identical to SciPy's across the benchmark problems;
 * see `tests/ode/rungeKuttaAdaptive.scipy.benchmark.md` for the comparison
 * table.
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
 * and `method` echoes `method`. The `success` flag indicates whether `tEnd` was
 * reached, and `message` provides additional status information.
 * @throws {RangeError} If the state has no components, tolerances are invalid,
 * a step-size or controller limit is invalid, or `method` is not recognized.
 * Step-size underflow and exceeding `maxSteps` return an unsuccessful result
 * with the reason in `message`.
 *
 * @example
 * ```ts
 * // Exponential decay: dy/dt = -y, y(0) = 1 -> y(t) = e^-t
 * const f: DerivativeFunction = (t, y, out) => out.set(y.data).multSelf(-1);
 * const result = rungeKuttaAdaptive('rk45', f, 0, 1, new Array1D([1]));
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
 *   t: Array1D [ 0, 0.14680437989650819, 1 ],
 *   y: Matrix [[1], [0.863462874659396], [0.3680228572282582]],
 * }
 * ```
 *
 * @example
 * ```ts
 * // Same integration, with explicit tolerances and an initial step size.
 * const result = rungeKuttaAdaptive('rk45', f, 0, 1, new Array1D([1]), {
 *     atol: 1e-8,
 *     rtol: 1e-8,
 *     h0: 0.01,
 * });
 * ```
 */
export function rungeKuttaAdaptive(
    method: RungeKuttaAdaptiveMethod,
    f: DerivativeFunction,
    t0: number,
    tEnd: number,
    y0: Array1D,
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
    } = options;
    const dim = y0.size;
    if (dim <= 0) throw new RangeError(`rungeKuttaAdaptive: y0 must have at least one component, got dimension ${dim}.`);
    if (atol < 0 || rtol < 0) throw new RangeError(`rungeKuttaAdaptive: atol (${atol}) and rtol (${rtol}) must be non-negative.`);
    if (atol === 0 && rtol === 0) throw new RangeError('rungeKuttaAdaptive: atol and rtol cannot both be 0; no step could ever satisfy the tolerance.');
    if (h0 !== undefined && h0 === 0) throw new RangeError('rungeKuttaAdaptive: h0 must be nonzero.');
    if (!(hMin > 0)) throw new RangeError(`rungeKuttaAdaptive: hMin (${hMin}) must be positive.`);
    if (!(hMax > 0)) throw new RangeError(`rungeKuttaAdaptive: hMax (${hMax}) must be positive.`);
    if (hMin > hMax) throw new RangeError(`rungeKuttaAdaptive: hMin (${hMin}) must not exceed hMax (${hMax}).`);
    if (!(maxSteps > 0)) throw new RangeError(`rungeKuttaAdaptive: maxSteps (${maxSteps}) must be positive.`);
    if (!(safety > 0 && safety <= 1)) throw new RangeError(`rungeKuttaAdaptive: safety (${safety}) must be in (0, 1].`);
    if (!(minScale > 0) || minScale > maxScale) {
        throw new RangeError(`rungeKuttaAdaptive: minScale (${minScale}) must be positive and not exceed maxScale (${maxScale}).`);
    }

    if (tEnd === t0) {
        const tVec = new Array1D(1);
        tVec.data[0] = t0;
        const yMat = new Matrix(1, dim);
        yMat.setRow(0, y0.data);
        return {
            method: method,
            success: true,
            message: 'Integration successful.',
            evaluations: 0,
            t: tVec,
            y: yMat,
        };
    }

    const dir = Math.sign(tEnd - t0) as 1 | -1;
    const scratch = makeAdaptiveScratch(dim);

    let methodOrder: number;
    let errExp: number;
    let fsalSource: Array1D;
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
    let next = new Array1D(dim);
    let evaluations = h0 === undefined ? 2 : 0;

    let k1Ready = false;
    let h: number;
    if (h0 !== undefined) {
        h = dir * Math.abs(h0);
    } else {
        h = estimateInitialStep(f, t0, y, dir, atol, rtol, methodOrder, scratch);
        k1Ready = true;
    }
    h = dir * Math.min(Math.abs(h), hMax, Math.abs(tEnd - t0));

    const tBuf: number[] = [t0];
    const yBuf: Float64Array[] = [y.data.slice()];
    const makeResult = (success: boolean, message: string): OdeResult => {
        const tVec = new Array1D(tBuf.length);
        tVec.set(tBuf);
        const yMat = new Matrix(tBuf.length, dim);
        for (let i = 0; i < tBuf.length; i++) {
            yMat.setRow(i, yBuf[i]);
        }
        return { method, success, message, evaluations, t: tVec, y: yMat };
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
            t = isFinalStep ? tEnd : t + h;
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
        if (steps > maxSteps) {
            return makeResult(false, `rungeKuttaAdaptive: exceeded maxSteps (${maxSteps}) without reaching tEnd.`);
        }
    }

    return makeResult(true, 'Integration successful.');
}