import { Array1D } from '../array/array1d';
import { Array2D } from '../array/array2d';
import { clip } from '../misc';
import type { DerivativeFunction, OdeResult } from './types';

export interface DP54Scratch {
    k1: Array1D;
    k2: Array1D;
    k3: Array1D;
    k4: Array1D;
    k5: Array1D;
    k6: Array1D;
    k7: Array1D;
    yTemp: Array1D;
}

export interface DP54Options {
    /** Absolute error tolerance. Default: 1e-6 */
    atol?: number;
    /** Relative error tolerance. Default: 1e-3 */
    rtol?: number;
    /** Initial step size magnitude. Estimated automatically if omitted. */
    h0?: number;
    /** Maximum step size magnitude. Default: Infinity */
    hMax?: number;
    /** Minimum step size magnitude before giving up. Default: 1e-12 */
    hMin?: number;
    /** Maximum number of attempted steps before throwing. Default: 1e5 */
    maxSteps?: number;
    /** Safety factor applied to optimal step-size prediction. Default: 0.9 */
    safety?: number;
    /** Smallest allowed per-step shrink factor. Default: 0.2 */
    minScale?: number;
    /** Largest allowed per-step growth factor. Default: 10 */
    maxScale?: number;
}

export type DP54Result = OdeResult;

/**
 * Classic Dormand-Prince 5(4) Butcher tableau (Dormand & Prince, 1980).
 * Coefficients as used by e.g. MATLAB's ode45 / SciPy's RK45.
 *
 *   c | a
 *   0        |
 *   1/5      | 1/5
 *   3/10     | 3/40        9/40
 *   4/5      | 44/45       -56/15      32/9
 *   8/9      | 19372/6561  -25360/2187 64448/6561  -212/729
 *   1        | 9017/3168   -355/33     46732/5247  49/176      -5103/18656
 *   1        | 35/384      0           500/1113    125/192     -2187/6784   11/84
 *   ---------+--------------------------------------------------------------------
 *   5th ord. | 35/384      0           500/1113    125/192     -2187/6784   11/84
 *   4th ord. | 5179/57600  0           7571/16695  393/640     -92097/339200 187/2100  1/40
 *
 * The 7th stage shares its node (c7 = 1) and weights (b = a7) with the 5th-order
 * solution, which is the FSAL ("First Same As Last") property: k7 = f(t+h, y5)
 * is simultaneously the final stage of this step *and* the first stage (k1) of
 * the next step, so an accepted step only costs 6 new derivative evaluations
 * instead of 7.
 */
const C2 = 1 / 5;
const C3 = 3 / 10;
const C4 = 4 / 5;
const C5 = 8 / 9;
// C6 = 1, C7 = 1 (elided below; used directly as `h`).

const A21 = 1 / 5;

const A31 = 3 / 40;
const A32 = 9 / 40;

const A41 = 44 / 45;
const A42 = -56 / 15;
const A43 = 32 / 9;

const A51 = 19372 / 6561;
const A52 = -25360 / 2187;
const A53 = 64448 / 6561;
const A54 = -212 / 729;

const A61 = 9017 / 3168;
const A62 = -355 / 33;
const A63 = 46732 / 5247;
const A64 = 49 / 176;
const A65 = -5103 / 18656;

// 5th-order (propagating) solution weights. These equal the stage-7 (a7j)
// coefficients by construction (FSAL); b2 = b7 = 0.
const B1 = 35 / 384;
const B3 = 500 / 1113;
const B4 = 125 / 192;
const B5 = -2187 / 6784;
const B6 = 11 / 84;

// Error-estimate weights e_i = b_i - b*_i (5th order minus embedded 4th
// order), i.e. the coefficients such that h * sum(e_i * k_i) is the local
// error estimate directly, without ever forming the 4th-order solution.
const E1 = 71 / 57600;
const E3 = -71 / 16695;
const E4 = 71 / 1920;
const E5 = -17253 / 339200;
const E6 = 22 / 525;
const E7 = -1 / 40;

const ERROR_EXPONENT = -1 / 5; // local error is O(h^5); controller uses 1/(order+1) = 1/5

/**
 * Allocates a scratch workspace of vectors reused across every stage of a
 * DP54 step (and across every step of an integration), so neither
 * `dp54Step` nor `dp54Integrate` allocates on the hot per-stage path.
 */
function makeScratch(dim: number): DP54Scratch {
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
 * Advances the state by one attempted step of the Dormand-Prince 5(4)
 * method, writing the 5th-order (propagating) solution into `out`.
 *
 * This computes a *proposed* step only; it does not know about, or enforce,
 * any error tolerance. Pair it with `dp54ErrorNorm` to decide whether to
 * accept the step (this is exactly what `dp54Integrate` does), or call it
 * directly if you want to implement your own step-size control.
 *
 * Zero-allocation: `f` must write into the `dydt` vector it's given, and the
 * stage buffers (`k1..k7`, `yTemp`) plus the result (`out`) are all
 * caller-supplied and reused, so calling this in a loop with the same
 * `scratch`/`out` does not allocate.
 *
 * FSAL reuse: `f(t, y)` (stage 1) is identical to `f(t_prev+h_prev, y)`
 * (stage 7) of the *previous* accepted step. If you've copied that stage
 * into `scratch.k1` yourself (`dp54Integrate` does this internally), pass
 * `k1Ready = true` to skip recomputing it. After this call, `scratch.k7`
 * holds `f(t + h, out)` — copy it into `scratch.k1` before the next call to
 * carry the FSAL saving forward.
 *
 * @param f Derivative function dy/dt = f(t, y). Must write into and return `dydt`; must not mutate `y`.
 * @param t Current time.
 * @param y Current state. Read-only; not mutated by this function.
 * @param h Attempted step size (may be negative to step backward).
 * @param out Array1D to overwrite with the proposed 5th-order state
 *   at `t + h`. May not alias `y` or any scratch buffer.
 * @param scratch Reusable stage buffers, same `dim` as `y`. If omitted, a fresh workspace is allocated
 *   for this call only (see `makeScratch`, used internally by `dp54Integrate`).
 * @param k1Ready If true, assumes `scratch.k1` already holds `f(t, y)` (e.g. carried over via FSAL) and skips recomputing it.
 * @returns `out`, set to the proposed state at `t + h`.
 */
export function dp54Step(
    f: DerivativeFunction,
    t: number,
    y: Array1D,
    h: number,
    out: Array1D,
    scratch: DP54Scratch = makeScratch(y.size),
    k1Ready: boolean = false
): Array1D {
    const { k1, k2, k3, k4, k5, k6, k7, yTemp } = scratch;

    if (!k1Ready) f(t, y, k1);

    yTemp.set(y.data).addScaled(k1, h * A21);
    f(t + C2 * h, yTemp, k2);

    yTemp.set(y.data).addScaled(k1, h * A31).addScaled(k2, h * A32);
    f(t + C3 * h, yTemp, k3);

    yTemp.set(y.data).addScaled(k1, h * A41).addScaled(k2, h * A42).addScaled(k3, h * A43);
    f(t + C4 * h, yTemp, k4);

    yTemp
        .set(y.data)
        .addScaled(k1, h * A51)
        .addScaled(k2, h * A52)
        .addScaled(k3, h * A53)
        .addScaled(k4, h * A54);
    f(t + C5 * h, yTemp, k5);

    yTemp
        .set(y.data)
        .addScaled(k1, h * A61)
        .addScaled(k2, h * A62)
        .addScaled(k3, h * A63)
        .addScaled(k4, h * A64)
        .addScaled(k5, h * A65);
    f(t + h, yTemp, k6); // c6 = 1

    out.set(y.data)
        .addScaled(k1, h * B1)
        .addScaled(k3, h * B3)
        .addScaled(k4, h * B4)
        .addScaled(k5, h * B5)
        .addScaled(k6, h * B6);
    // b2 = 0, so k2 does not contribute to the propagating solution.

    f(t + h, out, k7); // FSAL stage; c7 = 1, and b = a7 so this doubles as k7's contribution (weight 0)

    return out;
}

/**
 * Computes the weighted RMS local-error norm for a step already taken by
 * `dp54Step`, using the embedded-pair error weights and the stage values
 * left behind in `scratch` (`k1, k3, k4, k5, k6, k7`; `k2`'s error weight is
 * zero). A result `<= 1` means the step meets the requested tolerances.
 *
 * Mixed absolute/relative error control: componentwise scale is
 * `atol + rtol * max(|y_i|, |out_i|)`.
 */
function dp54ErrorNorm(
    y: Array1D,
    out: Array1D,
    h: number,
    scratch: DP54Scratch,
    atol: number,
    rtol: number
): number {
    const { k1, k3, k4, k5, k6, k7 } = scratch;
    const dim = y.size;
    const yd = y.data;
    const od = out.data;
    const k1d = k1.data;
    const k3d = k3.data;
    const k4d = k4.data;
    const k5d = k5.data;
    const k6d = k6.data;
    const k7d = k7.data;

    let sumSq = 0;
    for (let i = 0; i < dim; i++) {
        const errI = h * (E1 * k1d[i] + E3 * k3d[i] + E4 * k4d[i] + E5 * k5d[i] + E6 * k6d[i] + E7 * k7d[i]);
        const scale = atol + rtol * Math.max(Math.abs(yd[i]), Math.abs(od[i]));
        const e = errI / scale;
        sumSq += e * e;
    }
    return Math.sqrt(sumSq / dim);
}

/**
 * Estimates a reasonable initial step size (Hairer, Norsett & Wanner's
 * algorithm, "Solving ODEs I", II.4), so `dp54Integrate` doesn't need one
 * supplied by the caller. As a side effect, leaves `f(t0, y0)` in
 * `scratch.k1` (the caller can then skip recomputing it via `k1Ready`).
 *
 * Reuses `scratch.k2` and `scratch.yTemp` as throwaway buffers; both are
 * overwritten again on the first real step, so this is safe to call right
 * before the main integration loop.
 */
function estimateInitialStep(
    f: DerivativeFunction,
    t0: number,
    y0: Array1D,
    dir: 1 | -1,
    atol: number,
    rtol: number,
    scratch: DP54Scratch
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

    const order = 5;
    let h1: number;
    if (Math.max(d1, d2) <= 1e-15) {
        h1 = Math.max(1e-6, h0 * 1e-3);
    } else {
        h1 = Math.pow(0.01 / Math.max(d1, d2), 1 / (order + 1));
    }

    return dir * Math.min(100 * h0, h1);
}

/**
 * Integrates `dy/dt = f(t, y)` from `t0` to `tEnd` with the adaptive-step
 * Dormand-Prince 5(4) method (a.k.a. DOPRI5 / RK45 / the method behind
 * MATLAB's `ode45`), recording the state at every *accepted* step.
 *
 * @param f Derivative function, writing into and returning `dydt`.
 * @param t0 Initial time.
 * @param tEnd Final time. May be less than `t0` for backward integration; direction is inferred automatically.
 * @param y0 Initial state. Not mutated.
 * @param options Configuration options for tolerances and step sizes.
 * @returns Object containing `t` (recorded times) and `y` (state matrix).
 * @throws {RangeError} If `h` underflows below `hMin`.
 * @throws {Error} If `maxSteps` attempted steps elapse without reaching `tEnd`.
 */
export function dp54Integrate(
    f: DerivativeFunction,
    t0: number,
    tEnd: number,
    y0: Array1D,
    options: DP54Options = {}
): DP54Result {
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

    if (tEnd === t0) {
        const tVec = new Array1D(1);
        tVec.data[0] = t0;
        const yMat = new Array2D(1, dim);
        yMat.setRow(0, y0.data);
        return { t: tVec, y: yMat };
    }

    const dir = Math.sign(tEnd - t0) as 1 | -1;
    const scratch = makeScratch(dim);

    let t = t0;
    let y = y0.copy();
    let next = new Array1D(dim);

    let k1Ready = false;
    let h: number;
    if (h0 !== undefined) {
        h = dir * Math.abs(h0);
    } else {
        h = estimateInitialStep(f, t0, y, dir, atol, rtol, scratch);
        k1Ready = true; // estimateInitialStep left f(t0, y0) in scratch.k1
    }
    h = dir * Math.min(Math.abs(h), hMax, Math.abs(tEnd - t0));

    const tBuf: number[] = [t0];
    const yBuf: Float64Array[] = [y.data.slice()];

    let steps = 0;
    while (t !== tEnd) {
        const remaining = tEnd - t;
        let isFinalStep = false;
        if (Math.abs(h) >= Math.abs(remaining)) {
            h = remaining;
            isFinalStep = true;
        }

        dp54Step(f, t, y, h, next, scratch, k1Ready);
        k1Ready = true; // scratch.k1 now holds f(t, y) for the *current* (t, y), win or lose below

        const err = dp54ErrorNorm(y, next, h, scratch, atol, rtol);

        if (err <= 1) {
            t = isFinalStep ? tEnd : t + h;
            [y, next] = [next, y];
            tBuf.push(t);
            yBuf.push(y.data.slice());

            scratch.k1.set(scratch.k7.data); // FSAL: carry f(t, y) forward as next step's k1

            const growth = err === 0 ? maxScale : clip(safety * Math.pow(err, ERROR_EXPONENT), minScale, maxScale);
            h = dir * Math.min(Math.abs(h) * growth, hMax);
        } else {
            const shrink = clip(safety * Math.pow(err, ERROR_EXPONENT), minScale, maxScale);
            const hNext = h * shrink;
            if (!(Math.abs(hNext) >= hMin)) {
                throw new RangeError(
                    `dp54Integrate: step size underflowed below hMin (${hMin}) near t=${t}` +
                    (Number.isNaN(hNext) ? ' (step size became NaN)' : '') +
                    `; the problem may be stiff, or atol/rtol too tight for DP54.`
                );
            }
            h = hNext;
        }

        steps++;
        if (steps > maxSteps) {
            throw new Error(`dp54Integrate: exceeded maxSteps (${maxSteps}) without reaching tEnd.`);
        }
    }

    const nTimes = tBuf.length;
    const tVec = new Array1D(nTimes);
    tVec.set(tBuf);
    const yMat = new Array2D(nTimes, dim);
    for (let i = 0; i < nTimes; i++) {
        yMat.setRow(i, yBuf[i]);
    }

    return { t: tVec, y: yMat };
}