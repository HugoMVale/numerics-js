import type { Vector } from '../linalg/Vector.js';
import type { DerivativeFunction } from './types.js';

/**
 * Estimates a reasonable initial step size using Hairer, Norsett & Wanner's algorithm.
 *
 * Two derivative evaluations are performed: `f(t0, y0)` and one at an explicit
 * Euler trial point. On return, `f0` holds `f(t0, y0)`, so callers can reuse it
 * as the first derivative of their first step.
 *
 * @param f Derivative function `dy/dt = f(t, y)`, writing into its `dydt` argument.
 * @param t0 Starting time.
 * @param y0 Initial state at `t0`. Not mutated.
 * @param dir Integration direction: `1` forward, `-1` backward.
 * @param atol Absolute error tolerance.
 * @param rtol Relative error tolerance.
 * @param order Order of the method that will be used, which sets the step-size exponent `1 / (order + 1)`.
 * @param f0 Work vector of the same dimension as `y0`; receives `f(t0, y0)`.
 * @param f1 Work vector of the same dimension as `y0`; receives the derivative at the Euler trial point.
 * @param yTemp Work vector of the same dimension as `y0`; receives the Euler trial state.
 * @returns The signed initial step size, with the sign of `dir`.
 */
export function estimateInitialStep(
  f: DerivativeFunction,
  t0: number,
  y0: Vector,
  dir: 1 | -1,
  atol: number,
  rtol: number,
  order: number,
  f0: Vector,
  f1: Vector,
  yTemp: Vector
): number {
  const dim = y0.size;
  f(t0, y0, f0);

  const yd = y0.data;
  const f0d = f0.data;

  let d0Sq = 0;
  let d1Sq = 0;
  for (let i = 0; i < dim; i++) {
    const scale = atol + rtol * Math.abs(yd[i]);
    d0Sq += (yd[i] / scale) ** 2;
    d1Sq += (f0d[i] / scale) ** 2;
  }
  const d0 = Math.sqrt(d0Sq / dim);
  const d1 = Math.sqrt(d1Sq / dim);

  const h0 = d0 < 1e-5 || d1 < 1e-5 ? 1e-6 : 0.01 * (d0 / d1);

  const y1d = yTemp.data;
  for (let i = 0; i < dim; i++) {
    y1d[i] = yd[i] + dir * h0 * f0d[i];
  }
  f(t0 + dir * h0, yTemp, f1);
  const f1d = f1.data;

  let d2Sq = 0;
  for (let i = 0; i < dim; i++) {
    const scale = atol + rtol * Math.abs(yd[i]);
    d2Sq += ((f1d[i] - f0d[i]) / scale) ** 2;
  }
  const d2 = Math.sqrt(d2Sq / dim) / h0;

  let h1: number;
  if (Math.max(d1, d2) <= 1e-15) {
    h1 = Math.max(1e-6, h0 * 1e-3);
  } else {
    h1 = Math.pow(0.01 / Math.max(d1, d2), 1 / (order + 1));
  }

  return dir * Math.min(100 * h0, h1);
}
