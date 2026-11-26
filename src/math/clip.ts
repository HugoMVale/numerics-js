/**
 * Restricts a number to the inclusive range `[min, max]`.
 *
 * The bounds are not validated, unlike in `Vector.clip` and `Matrix.clip`, so that the function
 * stays cheap in inner loops: if `min > max`, values below `min` give `min` and all others give
 * `max`. A `NaN` value is returned unchanged.
 *
 * @param x The number to clip.
 * @param min The lower bound.
 * @param max The upper bound.
 * @returns `min` if `x < min`, `max` if `x > max`, otherwise `x`.
 *
 * @example
 * ```ts
 * import { clip } from 'numerics-js/math';
 *
 * console.log(clip(12, 0, 10), clip(-3, 0, 10), clip(4, 0, 10));
 * ```
 *
 * Output:
 * ```text
 * 10 0 4
 * ```
 */
export function clip(x: number, min: number, max: number): number {
  return x < min ? min : x > max ? max : x;
}
