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

/**
 * Smallest sum of squares for which `Math.sqrt(sum)` is still accurate: below
 * it the squares start to lose precision (and eventually underflow to 0).
 */
const NORM_SQ_MIN = 1e-250;

/**
 * Euclidean norm of `a`, or of `a - b` if `b` is given, computed with scaling so
 * that it neither overflows nor underflows unless the true result does.
 * One pass to find the largest magnitude and one to accumulate.
 * @param a The components.
 * @param b Optional second operand, of the same length as `a`.
 * @returns The norm: `NaN` if any component is `NaN`, `Infinity` if any is infinite.
 * @internal
 */
function scaledNorm(a: ArrayLike<number>, b?: ArrayLike<number>): number {
  const n = a.length;
  let max = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.abs(b === undefined ? a[i] : a[i] - b[i]);
    if (v > max) max = v;
    else if (v !== v) return NaN;
  }
  if (max === 0 || max === Infinity) return max;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const r = (b === undefined ? a[i] : a[i] - b[i]) / max;
    sum += r * r;
  }
  return max * Math.sqrt(sum);
}

/**
 * Turns a plain sum of squares into the Euclidean norm. In the normal range
 * this is just `Math.sqrt(sumSq)` (bit-identical to the naive formula); if the
 * sum overflowed or lost precision to underflow, the norm is recomputed with
 * scaling, so that e.g. `[3e200, 4e200]` has norm `5e200`, not `Infinity`.
 * @param sumSq The sum of squares of the components, computed naively.
 * @param a The components the sum was computed from.
 * @param b Optional second operand, if the components are `a - b`.
 * @returns The norm.
 * @internal
 */
export function sqrtOrScaled(sumSq: number, a: ArrayLike<number>, b?: ArrayLike<number>): number {
  if (sumSq >= NORM_SQ_MIN && sumSq < Infinity) return Math.sqrt(sumSq);
  return sumSq !== sumSq ? sumSq : scaledNorm(a, b);
}

/**
 * Euclidean norm of three components, with the same range guarantees as
 * `sqrtOrScaled`.
 * @param sumSq The sum of squares of `x`, `y` and `z`, computed naively.
 * @param x First component.
 * @param y Second component.
 * @param z Third component.
 * @returns The norm.
 * @internal
 */
export function sqrtOrHypot(sumSq: number, x: number, y: number, z: number): number {
  if (sumSq >= NORM_SQ_MIN && sumSq < Infinity) return Math.sqrt(sumSq);
  return sumSq !== sumSq ? sumSq : Math.hypot(x, y, z);
}
