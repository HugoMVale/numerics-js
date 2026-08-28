import { Array1D } from '../array/array1d.js';
import { trapezoid } from './trapezoid.js';

/**
 * Integrates `y` using the composite Simpson's rule.
 *
 * Works pairwise over consecutive intervals, fitting a parabola through
 * each group of 3 points. This formula is exact for any quadratic, and
 * reduces to the familiar `(h/3)(y0 + 4y1 + y2)` rule when spacing is
 * uniform, while still supporting irregular spacing when `x` is given.
 *
 * If the number of intervals (`size - 1`) is odd, there's one interval
 * left over after pairing the rest up. That final interval is handled
 * with a single trapezoidal step, matching the common convention used
 * by `numpy`/`scipy` for even-length inputs.
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

    // Simpson's rule needs at least 3 points; with only 2, trapezoid is the
    // best we can do.
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

        const y0 = y.data[i0];
        const y1 = y.data[i1];
        const y2 = y.data[i2];

        sum += ((h0 + h1) / 6) * ((2 - h1 / h0) * y0 + ((h0 + h1) * (h0 + h1)) / (h0 * h1) * y1 + (2 - h0 / h1) * y2);
    }

    // Odd number of intervals: one interval is left dangling at the end.
    // Cover it with a plain trapezoidal step.
    if (nIntervals % 2 === 1) {
        const h = x !== undefined ? x.data[n - 1] - x.data[n - 2] : dx;
        sum += (h * (y.data[n - 2] + y.data[n - 1])) / 2;
    }

    return sum;
}
