/**
 * Returns a value with the magnitude of `magnitude` and the sign of `sign`.
 *
 * Follows `copysign` in C, Python (`math.copysign`) and NumPy, and the Fortran `SIGN` intrinsic
 * used by LAPACK: the sign of `-0` is negative and that of `+0` is positive. The one difference
 * is a `NaN` `sign`, which is treated as positive instead of having its sign bit inspected. A
 * `NaN` `magnitude` is returned as `NaN`.
 *
 * @param magnitude Value whose absolute value is used.
 * @param sign Value whose sign is used.
 * @returns `|magnitude|` with the sign of `sign`.
 *
 * @example
 * ```ts
 * import { copysign } from 'numerics-js/math';
 *
 * console.log(copysign(3.5, -1), copysign(-3.5, 2), copysign(3.5, -0));
 * ```
 *
 * Output:
 * ```text
 * -3.5 3.5 -3.5
 * ```
 */
export function copysign(magnitude: number, sign: number): number {
  const negative = sign < 0 || (sign === 0 && 1 / sign < 0);
  return negative ? -Math.abs(magnitude) : Math.abs(magnitude);
}
