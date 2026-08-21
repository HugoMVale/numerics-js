import { Array1D } from '../array/array1d';
import { Array2D } from '../array/array2d';
import type { DerivativeFn, ODEIntegrateResult } from './types';

export type { DerivativeFn } from './types';

export interface RK4Scratch {
    k1: Array1D;
    k2: Array1D;
    k3: Array1D;
    k4: Array1D;
    yTemp: Array1D;
}

export type RK4Result = ODEIntegrateResult;

/**
 * Allocates a scratch workspace of vectors reused across every RK4 stage of
 * an integration, so `rk4Integrate` only allocates once per *recorded*
 * output state instead of once per internal stage evaluation.
 */
function makeScratch(dim: number): RK4Scratch {
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
 * Runge-Kutta method (RK4).
 */
export function rk4Step(
    f: DerivativeFn,
    t: number,
    y: Array1D,
    h: number,
    out: Array1D,
    scratch: RK4Scratch = makeScratch(y.dim)
): Array1D {
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
 */
export function rk4Integrate(
    f: DerivativeFn,
    t0: number,
    tEnd: number,
    y0: Array1D,
    h: number
): RK4Result {
    if (h === 0) throw new RangeError('Step size h must be nonzero.');
    const totalSpan = tEnd - t0;
    if (totalSpan !== 0 && Math.sign(h) !== Math.sign(totalSpan)) {
        throw new RangeError(
            `Sign of h (${h}) must match the direction from t0 (${t0}) to tEnd (${tEnd}).`
        );
    }

    const dim = y0.dim;
    const scratch = makeScratch(dim);

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
