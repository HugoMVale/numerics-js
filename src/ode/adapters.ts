import type { Vector } from '../linalg/Vector.js';
import type { AllocatingDerivativeFunction, DerivativeFunction } from './types.js';

/**
 * Adapts a convenience-style derivative function that allocates and returns
 * a new `Vector` - `(t, y) => Vector` - into the writes-into-`dydt` style that
 * ODE steppers/integrators in this package use.
 *
 * The wrapped function allocates on every derivative evaluation. Prefer a
 * {@link DerivativeFunction} that writes directly into its output buffer in
 * performance-sensitive integrations.
 *
 * @param f Derivative function `dy/dt = f(t, y)` that returns a newly allocated
 * `Vector`. Its result must have the same dimension as `y`.
 * @returns A {@link DerivativeFunction} that copies `f`'s result into the
 * supplied `dydt` buffer and returns that buffer.
 *
 * @example
 * ```ts
 * // Exponential decay: dy/dt = -y
 * const f = wrapAllocatingDerivative((t, y) => y.copy().multSelf(-1));
 * const result = dormandPrince45(f, 0, 1, new Vector([1]));
 * console.log(result);
 * ```
 *
 * Output:
 * ```text
 * {
 *   t: Vector [ 0, 0.14680437989650819, 1 ],
 *   y: Matrix [[1], [0.863462874659396], [0.3680228572282582]],
 *   method: 'dp54'
 * }
 * ```
 */
export function wrapAllocatingDerivative(
    f: AllocatingDerivativeFunction
): DerivativeFunction {
    return (t: number, y: Vector, dydt: Vector): Vector =>
        dydt.set(f(t, y).data);
}