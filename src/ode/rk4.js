import { Array1D, Array2D } from '../index.js';

/**
 * Allocates a scratch workspace of vectors reused across every RK4 stage of
 * an integration, so `rk4Integrate` only allocates once per *recorded*
 * output state instead of once per internal stage evaluation.
 * @param {number} dim
 * @returns {{k1: Array1D, k2: Array1D, k3: Array1D, k4: Array1D, yTemp: Array1D}}
 */
function makeScratch(dim) {
    return {
        k1: new Array1D(dim),
        k2: new Array1D(dim),
        k3: new Array1D(dim),
        k4: new Array1D(dim),
        yTemp: new Array1D(dim),
    };
}

/**
 * Advances the state by one step of the classic explicit 4th-order
 * Runge-Kutta method (RK4):
 *
 *   k1 = f(t,        y)
 *   k2 = f(t + h/2,  y + h/2 * k1)
 *   k3 = f(t + h/2,  y + h/2 * k2)
 *   k4 = f(t + h,    y + h   * k3)
 *   y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
 *
 * Zero-allocation: `f` must write the derivative into the `dydt` vector it's
 * given (rather than returning a new one), and both the RK stages (`k1..k4`,
 * `yTemp`) and the result (`out`) are caller-supplied and reused, so calling
 * this in a loop with the same `scratch`/`out` does not allocate.
 *
 * @param {(t: number, y: Array1D, dydt: Array1D) => Array1D} f - Derivative function
 *   dy/dt = f(t, y). Must write into and return `dydt`; must not mutate `y`.
 * @param {number} t - Current time.
 * @param {Array1D} y - Current state. Read-only; not mutated by this function.
 * @param {number} h - Step size (may be negative to step backward).
 * @param {Array1D} out - Array1D to overwrite with the state at `t + h`. May not
 *   alias `y` (the algorithm reads `y` throughout the step). Returned for
 *   convenience.
 * @param {{k1: Array1D, k2: Array1D, k3: Array1D, k4: Array1D, yTemp: Array1D}} [scratch] -
 *   Reusable stage buffers, same `dim` as `y`. If omitted, a fresh workspace
 *   is allocated for this call only — fine for one-off use, but pass an
 *   explicit workspace (see `makeScratch`, used internally by
 *   `rk4Integrate`) to avoid allocating on every call in a loop.
 * @returns {Array1D} `out`, set to the state at `t + h`.
 */
export function rk4Step(f, t, y, h, out, scratch = makeScratch(y.dim)) {
    const { k1, k2, k3, k4, yTemp } = scratch;
    const half = h / 2;

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

    return out;
}

/**
 * Integrates `dy/dt = f(t, y)` with classic RK4 from `t0` to `tEnd` using a
 * constant step size `h`, recording the state at every step.
 *
 * `h` need not divide `tEnd - t0` evenly: full-size steps of `h` are taken
 * for as long as they fit, and a single shorter final step is taken to land
 * exactly on `tEnd` (so the last recorded time is always exactly `tEnd`,
 * never an overshoot).
 *
 * Allocation: one scratch workspace and two state vectors reused for the whole
 * call, plus the returned outputs (`t` and `y`). `f` must not allocate either
 * (see `rk4Step`); use `wrapAllocatingDerivative` to adapt a `(t, y) => Array1D`
 * style function if you don't need the allocation-free hot path.
 *
 * @param {(t: number, y: Array1D, dydt: Array1D) => Array1D} f - Derivative function,
 *   writing into and returning `dydt`.
 * @param {number} t0 - Initial time.
 * @param {number} tEnd - Final time. May be less than `t0` for backward
 *   integration, in which case `h` must be negative.
 * @param {Array1D} y0 - Initial state. Not mutated.
 * @param {number} h - Step size. Sign must match the direction from `t0` to
 *   `tEnd` (positive if `tEnd > t0`, negative if `tEnd < t0`).
 * @returns {{t: Array1D, y: Array2D}} `t` holds the recorded times (component
 *   `i` is the time of row `i + 1` in `y`), with `t.data[0] === t0` and
 *   `t.data[t.dim - 1] === tEnd` exactly. `y` is a `(numTimes x stateDim)`
 *   matrix where row `i + 1` stores the state at time `t.data[i]`.
 */
export function rk4Integrate(f, t0, tEnd, y0, h) {
    if (h === 0) throw new RangeError('Step size h must be nonzero.');
    const totalSpan = tEnd - t0;
    if (totalSpan !== 0 && Math.sign(h) !== Math.sign(totalSpan)) {
        throw new RangeError(
            `Sign of h (${h}) must match the direction from t0 (${t0}) to tEnd (${tEnd}).`
        );
    }

    const dim = y0.dim;
    const scratch = makeScratch(dim);

    // Floating-point tolerance so that e.g. (1 - 0) / 0.1 counts as exactly
    // 10 full steps rather than 9 (with a needless tiny final step) or 10
    // (with a step that overshoots tEnd).
    const EPS = 1e-9;
    const nFull = Math.floor(Math.abs(totalSpan / h) + EPS);
    const remainder = totalSpan - nFull * h;
    const hasPartialStep = Math.abs(remainder) > EPS * Math.abs(h || 1);

    const nRecorded = nFull + (hasPartialStep ? 1 : 0);
    const nTimes = nRecorded + 1;
    const tVec = new Array1D(nTimes);
    const yMat = new Array2D(nTimes, dim);

    let t = t0;
    let y = y0.copy();
    let next = new Array1D(dim);
    tVec.data[0] = t0;
    yMat.setRow(1, y.data);

    for (let i = 1; i <= nFull; i++) {
        rk4Step(f, t, y, h, next, scratch);
        t = t0 + i * h;
        tVec.data[i] = t;
        yMat.setRow(i + 1, next.data);
        [y, next] = [next, y];
    }

    if (hasPartialStep) {
        rk4Step(f, t, y, remainder, next, scratch);
        tVec.data[nRecorded] = tEnd;
        yMat.setRow(nRecorded + 1, next.data);
    }

    return { t: tVec, y: yMat };
}

/**
 * Adapts a convenience-style derivative function that allocates and returns
 * a new `Array1D` — `(t, y) => Array1D` — into the writes-into-`dydt` style that
 * `rk4Step`/`rk4Integrate` require.
 *
 * This restores the simpler authoring style at the cost of the allocation
 * it causes: the wrapped function still allocates a new `Array1D` internally
 * on every call, it's just copied into `dydt` afterward. Prefer writing `f`
 * directly against `dydt` (see the example in the module docs / tests) when
 * the allocation-free hot path matters.
 *
 * @param {(t: number, y: Array1D) => Array1D} f
 * @returns {(t: number, y: Array1D, dydt: Array1D) => Array1D}
 */
export function wrapAllocatingDerivative(f) {
    return (t, y, dydt) => dydt.set(f(t, y).data);
}