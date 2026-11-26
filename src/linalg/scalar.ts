/**
 * Smallest sum of squares for which `Math.sqrt(sum)` is still accurate: below
 * it the squares start to lose precision (and eventually underflow to 0).
 */
const NORM_SQ_MIN = 1e-250;

/** Smallest positive normal double; below it a reciprocal loses precision. @internal */
export const MIN_NORMAL = 2.2250738585072014e-308;

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

/**
 * Validates the options of `isClose`/`allClose` when they are given.
 * @param options The options argument.
 * @param who Full name of the public method (`Vector.isClose`), for the error message.
 * @throws {TypeError} If `options` is not an object (for example a number: the old positional
 * `(rtol, atol)` form), or `atol`/`rtol` is not a number.
 * @throws {RangeError} If `atol` or `rtol` is negative or `NaN`.
 * @internal
 */
export function checkCloseOptions(options: unknown, who: string): void {
  if (options === undefined) return;
  if (options === null || typeof options !== 'object') {
    const got = options === null ? 'null' : typeof options;
    throw new TypeError(`${who}: options must be an object like { atol, rtol }, got ${got}.`);
  }
  for (const name of ['atol', 'rtol'] as const) {
    const v = (options as Record<string, unknown>)[name];
    if (v === undefined) continue;
    if (typeof v !== 'number')
      throw new TypeError(`${who}: ${name} must be a number, got ${typeof v}.`);
    if (!(v >= 0)) throw new RangeError(`${who}: ${name} must be >= 0, got ${v}.`);
  }
}
