import { Array1D } from '../array/array1d.js';

/**
 * Integrates `y` using the composite trapezoidal rule.
 *
 * If `x` is provided, the spacing between samples is taken from `x`
 * (which need not be uniform). Otherwise, samples are assumed to be
 * spaced `dx` apart.
 *
 * @param y - Sample values to integrate. Must have `size >= 0`.
 * @param x - Optional sample locations, same `size` as `y`. If omitted,
 * uniform spacing of `dx` is assumed.
 * @param dx - Spacing between samples, used only when `x` is omitted.
 * Defaults to `1`.
 * @returns The approximate definite integral of `y`. Returns `0` if
 * `y.size` is `0` or `1` (a single point, or no points, has no area).
 * @throws {RangeError} If `x` is provided and `x.size !== y.size`.
 *
 * @example
 * ```ts
 * const y = Array1D.from([0, 1, 2, 3, 4]);
 * const integral = trapezoid(y);
 * console.log(`integral = ${integral}`);
 * ```
 *
 * Output:
 * ```text
 * integral = 8
 * ```
 */
export function trapezoid(y: Array1D, x?: Array1D, dx: number = 1): number {
    const n = y.size;
    if (n < 2) return 0;

    if (x !== undefined) {
        if (x.size !== n) {
            throw new RangeError(`trapz: x and y must have the same size: ${x.size} vs ${n}`);
        }
        let sum = 0;
        for (let i = 0; i < n - 1; i++) {
            sum += (x.data[i + 1] - x.data[i]) * (y.data[i] + y.data[i + 1]);
        }
        return sum / 2;
    }

    // Uniform spacing: sum all samples, then subtract half of each
    // endpoint (since those only contribute one trapezoid, not two).
    let sum = y.sum() - (y.data[0] + y.data[n - 1]) / 2;
    return sum * dx;
}