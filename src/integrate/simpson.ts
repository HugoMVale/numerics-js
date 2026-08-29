import { Array1D } from '../array/array1d.js';
import { trapezoid } from './trapezoid.js';

/**
 * Integrates a parabola through 3 points, `y0, y1, y2`, spaced `h0` and
 * `h1` apart respectively, over the full span `[0, h0 + h1]`. Reduces to
 * the classic `(h/3)(y0 + 4y1 + y2)` rule when `h0 === h1`; exact for
 * uneven spacing too, and exact for cubics as well as quadratics (the
 * symmetric 2-interval span cancels odd-degree error terms).
 */
function simpsonPair(y0: number, y1: number, y2: number, h0: number, h1: number): number {
    return ((h0 + h1) / 6) * ((2 - h1 / h0) * y0 + ((h0 + h1) * (h0 + h1)) / (h0 * h1) * y1 + (2 - h0 / h1) * y2);
}

/**
 * Correction term for a single trailing interval when the interval count
 * is odd, per the composite Simpson's rule for irregularly spaced data
 * (Cartwright, 2017; see the "Composite Simpson's rule for irregularly
 * spaced data" section of Wikipedia's "Simpson's rule" article).
 *
 * Fits a quadratic through the last 3 points and integrates it only over
 * the final sub-interval `[x_{N-1}, x_N]`, rather than the full
 * symmetric 2-interval span `simpsonPair` uses. Because that span is
 * asymmetric, this correction is exact for quadratics but, unlike
 * `simpsonPair`, not for cubics.
 *
 * @param fN2 - Value 2 points before the end (`f_{N-2}`).
 * @param fN1 - Value 1 point before the end (`f_{N-1}`).
 * @param fN - The final value (`f_N`).
 * @param hPrev - Width of the second-to-last interval (`h_{N-2}`).
 * @param hLast - Width of the last interval (`h_{N-1}`).
 */
function trailingCorrection(fN2: number, fN1: number, fN: number, hPrev: number, hLast: number): number {
    const alpha = (2 * hLast * hLast + 3 * hLast * hPrev) / (6 * (hPrev + hLast));
    const beta = (hLast * hLast + 3 * hLast * hPrev) / (6 * hPrev);
    const eta = (hLast * hLast * hLast) / (6 * hPrev * (hPrev + hLast));
    return alpha * fN + beta * fN1 - eta * fN2;
}

/**
 * Integrates `y` using the composite Simpson's rule.
 *
 * Works pairwise over consecutive intervals, fitting a parabola through
 * each group of 3 points and integrating it over its full 2-interval
 * span. This is exact for any cubic (not just quadratic), and supports
 * irregular spacing when `x` is given, reducing to the familiar
 * `(h/3)(y0 + 4y1 + y2)` rule when spacing is uniform.
 *
 * Simpson's rule needs pairs of intervals, so it only applies directly
 * when the number of intervals (`size - 1`) is even. When it's odd,
 * there's one interval left over after pairing the rest up; that
 * interval is handled with the trailing correction described in
 * `trailingCorrection` above, which supports arbitrary spacing but is
 * only exact for quadratics rather than cubics.
 *
 * @param y - Sample values to integrate. Must have `size >= 0`.
 * @param x - Optional sample locations, same `size` as `y`. If omitted,
 * uniform spacing of `dx` is assumed.
 * @param dx - Spacing between samples, used only when `x` is omitted.
 * Defaults to `1`.
 * @returns The approximate definite integral of `y`. Returns `0` if
 * `y.size` is `0` or `1`. Falls back to the trapezoidal rule if
 * `y.size` is `2`, since Simpson's rule needs at least 3 points.
 * @throws {RangeError} If `x` is provided and `x.size !== y.size`.
 *
 * @example
 * ```ts
 * // Integrate y = x^2 at non-uniform sample locations.
 * const x = Array1D.from([0, 1, 3]);
 * const y = Array1D.from([0, 1, 9]);
 * const integral = simpson(y, x);
 * console.log(`integral = ${integral}`);
 * ```
 *
 * Output:
 * ```text
 * integral = 9
 * ```
 */
export function simpson(y: Array1D, x?: Array1D, dx: number = 1): number {
    const n = y.size;
    if (n < 2) return 0;

    if (x !== undefined && x.size !== n) {
        throw new RangeError(`simpson: x and y must have the same size: ${x.size} vs ${n}`);
    }

    // Simpson's rule needs at least 3 points to fit a parabola; with only
    // 2, trapz is the best we can do.
    if (n === 2) return trapezoid(y, x, dx);

    const nIntervals = n - 1;
    const nPairs = Math.floor(nIntervals / 2);

    let sum = 0;
    for (let k = 0; k < nPairs; k++) {
        const i0 = 2 * k;
        const i1 = i0 + 1;
        const i2 = i0 + 2;
        const h0 = x !== undefined ? x.data[i1] - x.data[i0] : dx;
        const h1 = x !== undefined ? x.data[i2] - x.data[i1] : dx;
        sum += simpsonPair(y.data[i0], y.data[i1], y.data[i2], h0, h1);
    }

    if (nIntervals % 2 === 1) {
        const hPrev = x !== undefined ? x.data[n - 2] - x.data[n - 3] : dx;
        const hLast = x !== undefined ? x.data[n - 1] - x.data[n - 2] : dx;
        sum += trailingCorrection(y.data[n - 3], y.data[n - 2], y.data[n - 1], hPrev, hLast);
    }

    return sum;
}