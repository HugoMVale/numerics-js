/**
 * Tolerance test for a single pair of numbers, shared by `ArrayND` and `Vec3`
 * so that every closeness comparison in the module behaves identically.
 *
 * `a` is close to `b` if `a === b` or, for finite `b`, if
 * `|a - b| <= atol + rtol * |b|`. As in `numpy.isclose`, equal infinities are
 * therefore close (`|Infinity - Infinity|` is `NaN`) and an infinite `b` is close
 * to nothing but itself (its tolerance would otherwise be infinite). NaN is never
 * close to anything, including another NaN. The test is not symmetric: `rtol`
 * scales `|b|` only.
 * @param a The value being tested.
 * @param b The reference value.
 * @param rtol Relative tolerance, applied to `|b|`.
 * @param atol Absolute tolerance.
 * @returns `true` if `a` is close to `b`.
 * @internal
 */
export function isCloseScalar(a: number, b: number, rtol: number, atol: number): boolean {
  return a === b || (Number.isFinite(b) && Math.abs(a - b) <= atol + rtol * Math.abs(b));
}
