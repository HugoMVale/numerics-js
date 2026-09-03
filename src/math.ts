/**
 * Restricts a number to the inclusive range between `lo` and `hi`.
 *
 * @param x The number to clip.
 * @param low The lower bound.
 * @param high The upper bound.
 * @returns The clipped number.
 *
 * @example
 * ```ts
 * import { clip } from 'numerics-js/math';
 *
 * const value = clip(12, 0, 10);
 * console.log(value);
 * ```
 *
 * Output:
 * ```text
 * 10
 * ```
 */
export function clip(x: number, low: number, high: number): number {
    return x < low ? low : x > high ? high : x;
}


/**
 * Returns a value with the magnitude of `magnitude` and the sign of `sign`.
 *
 * Mirrors the semantics of `Math.copysign` in other languages (e.g. Python's
 * `math.copysign`, NumPy's `np.copysign`): a `sign` of zero is treated as
 * positive.
 *
 * @param magnitude Value whose absolute value is used.
 * @param sign Value whose sign is used.
 * @returns `|magnitude|` with the sign of `sign`.
 *
 * @example
 * ```ts
 * import { copysign } from 'numerics-js/math';
 *
 * const value = copysign(3.5, -1);
 * console.log(value);
 * ```
 *
 * Output:
 * ```text
 * -3.5
 * ```
 */
export function copysign(magnitude: number, sign: number): number {
    return sign < 0 ? -Math.abs(magnitude) : Math.abs(magnitude);
}


/**
 * Checks whether two numbers are approximately equal, mirroring numpy.isclose:
 * |a - b| <= atol + rtol * |b|.
 *
 * @param a First value.
 * @param b Second value (reference value for the relative tolerance term).
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @returns True if a and b are within tolerance of each other.
 *
 * @example
 * ```ts
 * import { isClose } from 'numerics-js/math';
 *
 * const result = isClose(Math.PI, 3.14159, 1e-5);
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * true
 * ```
 */
export function isClose(a: number, b: number, atol: number, rtol: number = 1e-5): boolean {
    return Math.abs(a - b) <= atol + rtol * Math.abs(b);
}