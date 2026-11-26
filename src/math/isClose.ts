/**
 * Tolerances of {@link isClose}, shared by `Vector.isClose`, `Matrix.isClose`, their `allClose`
 * counterparts and `Vec3.allClose`.
 */
export interface CloseOptions {
  /**
   * Absolute tolerance.
   * @default 1e-8
   */
  atol?: number;
  /**
   * Relative tolerance, applied to the magnitude of the reference value.
   * @default 1e-5
   */
  rtol?: number;
}

/** Default absolute tolerance of the closeness tests. @internal */
export const DEFAULT_ATOL = 1e-8;

/** Default relative tolerance of the closeness tests. @internal */
export const DEFAULT_RTOL = 1e-5;

/**
 * Tolerance test for a single pair of numbers with already resolved tolerances: the one
 * implementation behind `isClose`, `ArrayND.isClose`/`allClose` and `Vec3.allClose`, so that every
 * closeness comparison in the library behaves identically.
 *
 * `a` is close to `b` if `a === b` or, for finite `b`, if `|a - b| <= atol + rtol * |b|`. As in
 * `numpy.isclose`, equal infinities are therefore close (`|Infinity - Infinity|` is `NaN`) and an
 * infinite `b` is close to nothing but itself (its tolerance would otherwise be infinite). NaN is
 * never close to anything, including another NaN. The test is not symmetric: `rtol` scales `|b|`
 * only.
 * @param a The value being tested.
 * @param b The reference value.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance, applied to `|b|`.
 * @returns `true` if `a` is close to `b`.
 * @internal
 */
export function isCloseScalar(a: number, b: number, atol: number, rtol: number): boolean {
  return a === b || (Number.isFinite(b) && Math.abs(a - b) <= atol + rtol * Math.abs(b));
}

/**
 * Checks whether two numbers are approximately equal, as `numpy.isclose` does:
 * `|a - b| <= atol + rtol * |b|`.
 *
 * The test is **not symmetric**: `rtol` scales the reference value `b` only. Exactly equal values
 * are always close, including equal infinities, while an infinite `b` is close to nothing but
 * itself. `NaN` is never close to anything, including another `NaN`. The tolerances are not
 * validated, like the other arguments of the scalar functions; `Vector.isClose` and
 * `Matrix.isClose` do validate them.
 *
 * @param a The value being tested.
 * @param b The reference value, which also scales the relative tolerance term.
 * @param options Tolerances `atol` and `rtol`, see {@link CloseOptions}.
 * @returns `true` if `a` is close to `b`.
 *
 * @example
 * ```ts
 * import { isClose } from 'numerics-js/math';
 *
 * console.log(isClose(Math.PI, 3.14159, { atol: 1e-5 }));
 * console.log(isClose(1, Infinity));
 * ```
 *
 * Output:
 * ```text
 * true
 * false
 * ```
 */
export function isClose(a: number, b: number, options?: CloseOptions): boolean {
  return isCloseScalar(a, b, options?.atol ?? DEFAULT_ATOL, options?.rtol ?? DEFAULT_RTOL);
}
