import { Vector, type VectorLike } from '../linalg/Vector.js';

/**
 * Integrates `y` using the composite trapezoidal rule.
 *
 * If `x` is provided, the spacing between samples is taken from `x`
 * (which need not be uniform). Otherwise, samples are assumed to be
 * spaced `dx` apart.
 *
 * @param y Sample values `y` to integrate: a `Vector`, a plain `number[]` or a `Float64Array`.
 * May be empty.
 * @param x Optional sample locations, same forms and same length as `y`. If omitted,
 * uniform spacing of `dx` is assumed.
 * @param dx Spacing between samples, used only when `x` is omitted.
 * Defaults to `1`.
 * @returns The approximate definite integral of `y`. Returns `0` if
 * `y.size` is `0` or `1` (a single point, or no points, has no area).
 * @throws {TypeError} If `y` or `x` is not a `VectorLike`.
 * @throws {RangeError} If `x` is provided and has a different length than `y`.
 *
 * @example
 * ```ts
 * import { Vector } from 'numerics-js/linalg';
 * import { trapezoid } from 'numerics-js/integrate';
 *
 * const y = Vector.from([0, 1, 2, 3, 4]);
 * const integral = trapezoid(y);
 * console.log(`integral = ${integral}`);
 * ```
 *
 * Output:
 * ```text
 * integral = 8
 * ```
 */
export function trapezoid(y: VectorLike, x?: VectorLike, dx: number = 1): number {
  const yv = Vector.asVector(y);
  const xv = x === undefined ? undefined : Vector.asVector(x);
  const n = yv.size;
  if (n < 2) return 0;

  if (xv !== undefined) {
    if (xv.size !== n) {
      throw new RangeError(`trapezoid: x and y must have the same size: ${xv.size} vs ${n}`);
    }
    let sum = 0;
    for (let i = 0; i < n - 1; i++) {
      sum += (xv.data[i + 1] - xv.data[i]) * (yv.data[i] + yv.data[i + 1]);
    }
    return sum / 2;
  }

  // Uniform spacing: sum all samples, then subtract half of each
  // endpoint (since those only contribute one trapezoid, not two).
  const sum = yv.sum() - (yv.data[0] + yv.data[n - 1]) / 2;
  return sum * dx;
}
