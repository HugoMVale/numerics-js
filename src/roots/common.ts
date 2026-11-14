import type { Vector } from '../linalg/Vector.js';

/**
 * Sufficient-decrease parameter of the Armijo condition, `f(x + s) ≤ f(x) + ARMIJO_ALPHA*∇f(x)·s`,
 * shared by the global strategies (Dennis and Schnabel, 1996, A6.3.1 and A6.4.5).
 *
 * @internal
 */
export const ARMIJO_ALPHA = 1e-4;

/**
 * Relative length of a step, `max_i |s_i| / max(|x_i|, 1/sclx_i)`, used by the global strategies
 * to decide whether a step is too small to make progress. `NaN` components propagate.
 *
 * @internal
 *
 * @param s Step.
 * @param x Point the relative length is measured against (`xc` or `xp`, as the caller sees fit).
 * @param sclx Positive scaling factors for `x`.
 * @returns The relative step length.
 */
export function relativeStepLength(s: Vector, x: Vector, sclx: Vector): number {
  let len = 0;
  for (let i = 0; i < s.size; i++) {
    len = Math.max(len, Math.abs(s.data[i]) / Math.max(Math.abs(x.data[i]), 1 / sclx.data[i]));
  }
  return len;
}
