import { Array1D } from '../array/array1d';
import { Array2D } from '../array/array2d';
import type { DerivativeFunction, OdeResult, RungeKuttaFixedMethod } from './types';

export type { RungeKuttaFixedMethod as RungeKuttaMethod } from './types';

/**
 * Reusable intermediate vectors for a Runge-Kutta integration step.
 *
 * @internal
 */
export interface RungeKuttaScratch {
    k1: Array1D;
    k2: Array1D;
    k3: Array1D;
    k4: Array1D;
    yTemp: Array1D;
}

/**
 * Allocates a scratch workspace of vectors reused across every stage of
 * an integration, ensuring `odeIntegrate` only allocates once per output state.
 */
function makeScratch(dim: number): RungeKuttaScratch {
    return {
        k1: new Array1D(dim),
        k2: new Array1D(dim),
        k3: new Array1D(dim),
        k4: new Array1D(dim),
        yTemp: new Array1D(dim),
    };
}

function stagesPerStep(method: RungeKuttaFixedMethod): number {
    switch (method) {
        case 'euler':
            return 1;
        case 'midpoint':
        case 'trapezoid':
            return 2;
        case 'rk4':
            return 4;
        default:
            throw new RangeError(`rungeKuttaFixed: unknown method "${method satisfies never}"`);
    }
}

/**
 * Advances the state by one step using the requested explicit Runge-Kutta method.
 *
 * @param method Which Runge-Kutta scheme to use: `'euler'` (order 1), `'midpoint'`
 * or `'trapezoid'` (order 2), or `'rk4'` (order 4, the classic 4-stage method).
 * @param f Derivative function `dy/dt = f(t, y)`. Must write its result into the
 * `dydt` buffer it's given (the third argument) rather than allocating; each stage
 * call reuses one of `scratch`'s vectors as that buffer.
 * @param t The current time, i.e. the time at which `y` is the state.
 * @param y The current state vector. Read-only: never mutated by this function,
 * so it's safe to reuse across repeated calls.
 * @param h The step size (may be negative to step backwards in time).
 * @param out Vector that receives the state at `t + h`, mutated in place.
 * Must have the same dimension as `y`. May safely alias `y` (e.g. `out === y`),
 * since every branch fully overwrites `out` from `y`'s original values before
 * returning.
 * @param scratch Reusable workspace for the intermediate `k1..k4` and `yTemp`
 * vectors, so repeated calls (e.g. from `rungeKuttaFixed`) don't allocate on every
 * step. Defaults to a freshly allocated workspace sized to `y.size` when omitted;
 * pass one explicitly when calling this in a loop.
 * @returns `out`, for chaining.
 * @throws {RangeError} If `method` is not one of the recognized `RungeKuttaMethod` values.
 *
 * @example
 * ```ts
 * // Exponential decay: dy/dt = -y
 * const f: DerivativeFunction = (t, y, out) => out.set(y.data).multSelf(-1);
 * const out = new Array1D(1);
 * rungeKuttaStep('rk4', f, 0, new Array1D([1]), 0.25, out);
 * console.log(out);
 * ```
 *
 * Output:
 * ```text
 * Array1D [ 0.7788085937500001 ]
 * ```
 */
export function rungeKuttaStep(
    method: RungeKuttaFixedMethod,
    f: DerivativeFunction,
    t: number,
    y: Array1D,
    h: number,
    out: Array1D,
    scratch: RungeKuttaScratch = makeScratch(y.size)
): Array1D {
    const { k1, k2, k3, k4, yTemp } = scratch;
    const half = h / 2;

    switch (method) {
        case 'euler':
            f(t, y, k1);
            out.set(y.data).addScaled(k1, h);
            break;

        case 'midpoint':
            f(t, y, k1);
            yTemp.set(y.data).addScaled(k1, half);
            f(t + half, yTemp, k2);

            out.set(y.data).addScaled(k2, h);
            break;

        case 'trapezoid':
            f(t, y, k1);
            yTemp.set(y.data).addScaled(k1, h);
            f(t + h, yTemp, k2);

            out.set(y.data)
                .addScaled(k1, half)
                .addScaled(k2, half);
            break;

        case 'rk4':

            f(t, y, k1);

            yTemp.set(y.data).addScaled(k1, half);
            f(t + half, yTemp, k2);

            yTemp.set(y.data).addScaled(k2, half);
            f(t + half, yTemp, k3);

            yTemp.set(y.data).addScaled(k3, h);
            f(t + h, yTemp, k4);

            out.set(y.data)
                .addScaled(k1, h / 6)
                .addScaled(k2, h / 3)
                .addScaled(k3, h / 3)
                .addScaled(k4, h / 6);
            break;
        default:
            throw new RangeError(`rungeKuttaStep: unknown method "${method satisfies never}"`);
    }

    return out;
}

/**
 * Integrates `dy/dt = f(t, y)` from `t0` to `tEnd` using a constant step size `h`
 * and the specified numerical method, recording the state at every step.
 *
 * If `h` does not evenly divide `tEnd - t0`, a final shortened step is taken so the
 * last recorded time is always exactly `tEnd`.
 *
 * @param method Which Runge-Kutta scheme to use: `'euler'` (order 1), `'midpoint'`
 * or `'trapezoid'` (order 2), or `'rk4'` (order 4, the classic 4-stage method).
 * @param f Derivative function `dy/dt = f(t, y)`. Must write its result into the
 * `dydt` buffer it's given (the third argument) rather than allocating.
 * @param t0 The starting time.
 * @param tEnd The time to integrate to. May be less than `t0` to integrate
 * backwards, as long as the sign of `h` matches the direction of travel.
 * @param y0 The initial state at `t0`. Must have at least one component; copied
 * internally, so the caller's vector is left unchanged.
 * @param h The fixed step size. Must be nonzero, and (when `t0 !== tEnd`) its sign
 * must match the direction from `t0` to `tEnd` (positive to integrate forward,
 * negative to integrate backward).
 * @returns An {@link OdeResult}: `t`, the recorded time points (length `n`), and
 * `y`, an `n x y0.size` matrix whose row `i` is the state at `t.get(i)`. Row 0 is
 * always `y0` and the last row is always the state at `tEnd`. `method` echoes the
 * `method` argument, and `evaluations` is the number of calls to `f`.
 * @throws {RangeError} If `h` is `0`, if the sign of `h` doesn't match the direction
 * from `t0` to `tEnd`, if `y0` has dimension `0`, or if `method` is not one of the
 * recognized `RungeKuttaMethod` values.
 *
 * @example
 * ```ts
 * // Exponential decay: dy/dt = -y, y(0) = 1 -> y(t) = e^-t
 * const f: DerivativeFunction = (t, y, out) => out.set(y.data).multSelf(-1);
 * const result = rungeKuttaFixed('rk4', f, 0, 1, new Array1D([1]), 0.25);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   t: Array1D [ 0, 0.25, 0.5, 0.75, 1 ],
 *   y: Array2D [[1], [0.7788085937500001], [0.6065428256988528], [0.472380765131675], [0.36789419940674883]],
 *   evaluations: 16,
 *   method: 'rk4'
 * }
 * ```
 */
export function rungeKuttaFixed(
    method: RungeKuttaFixedMethod,
    f: DerivativeFunction,
    t0: number,
    tEnd: number,
    y0: Array1D,
    h: number
): OdeResult {
    if (h === 0) throw new RangeError('Step size h must be nonzero.');
    const totalSpan = tEnd - t0;
    if (totalSpan !== 0 && Math.sign(h) !== Math.sign(totalSpan)) {
        throw new RangeError(
            `Sign of h (${h}) must match the direction from t0 (${t0}) to tEnd (${tEnd}).`
        );
    }

    const dim = y0.size;
    if (dim <= 0) {
        throw new RangeError(`rungeKuttaFixed: y0 must have at least one component, got dimension ${dim}.`);
    }
    const scratch = makeScratch(dim);

    const EPS = 1e-9;
    const nFull = Math.floor(Math.abs(totalSpan / h) + EPS);
    const remainder = totalSpan - nFull * h;
    const hasPartialStep = Math.abs(remainder) > EPS * Math.abs(h || 1);

    const nRecorded = nFull + (hasPartialStep ? 1 : 0);
    const nTimes = nRecorded + 1;
    const evaluationsPerStep = stagesPerStep(method);
    const tVec = new Array1D(nTimes);
    const yMat = new Array2D(nTimes, dim);
    let evaluations = 0;

    let t = t0;
    let y = y0.copy();
    let next = new Array1D(dim);
    tVec.data[0] = t0;
    yMat.setRow(0, y.data);

    for (let i = 1; i <= nFull; i++) {
        rungeKuttaStep(method, f, t, y, h, next, scratch);
        evaluations += evaluationsPerStep;
        t = t0 + i * h;
        tVec.data[i] = t;
        yMat.setRow(i, next.data);
        [y, next] = [next, y];
    }

    if (hasPartialStep) {
        rungeKuttaStep(method, f, t, y, remainder, next, scratch);
        evaluations += evaluationsPerStep;
        tVec.data[nRecorded] = tEnd;
        yMat.setRow(nRecorded, next.data);
    }

    return { t: tVec, y: yMat, evaluations, method };
}